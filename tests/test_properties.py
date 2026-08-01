"""Property-based tests for the pure functions with algebraic invariants.

Why these four and not more: the two most expensive bugs in outmem's
history were both round-trip failures, and both shipped because the
example-based tests happened not to contain the one input that broke.

- ``serialize_wiki_page`` corrupted values on the way out — a tag written
  ``007`` came back as ``7``, ``12:30`` as ``750`` — because YAML 1.1
  resolves those scalars on load. That is ``parse(serialize(x)) != x``.
- ``normalize_document_key`` stripped one extension rather than
  stripping until stable, so ``report.csv.md`` produced a key that
  re-normalised to something else. That is
  ``normalize(normalize(x)) != normalize(x)`` — and the refusal message
  told operators to pass that key back as ``--as``, which would then
  silently fail to link the two versions it was suggested to link.

Both are properties a generator finds mechanically and a hand-written
example finds only if you already suspected the bug.

Deliberately **not** property-tested: anything touching git, SQLite or the
filesystem. Hypothesis would be slow there, and the value in those paths
is integration rather than algebra — ``tests/test_supersession.py`` covers
them with scenarios instead. The one shape that would justify a
``RuleBasedStateMachine`` later is the registry's "at most one live row
per document_key" invariant across register/supersede/gc/backfill.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from outmem.exceptions import OutmemError
from outmem.frontmatter import (
    WikiFrontmatter,
    parse_wiki_page,
    serialize_wiki_page,
)
from outmem.slug import relpath_to_slug, slug_to_relpath, validate_slug
from outmem.sources import (
    ALLOWED_EXTENSIONS,
    SHA_PREFIX_LEN,
    candidate_document_key,
    derive_document_key,
    normalize_document_key,
    propose_document_keys,
)

# Hypothesis picks its own examples, so a bad seed would otherwise make an
# unrelated PR's CI fail. Deterministic runs keep a red build meaningful;
# `--hypothesis-seed=random` still explores when you want it to.
# 60 keeps the whole file under ~25s. The value curve is steep early and
# flat after that for properties this small, and the hazard inputs that
# motivated the file are sampled explicitly rather than left to chance.
settings.register_profile("outmem", derandomize=True, max_examples=60)
settings.load_profile("outmem")

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Built from primitives rather than `from_regex` — same grammar
# (`[a-z0-9]+(-[a-z0-9]+)*`), several times cheaper to generate.
_word = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=6)
_segment = st.lists(_word, min_size=1, max_size=3).map("-".join)
slugs = st.lists(_segment, min_size=1, max_size=4).map(":".join)
shas = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)

# Deliberately hostile: the YAML 1.1 scalars that broke serialisation
# (`007`, `12:30`, `yes`, `~`) plus arbitrary text.
_yaml_hazards = st.sampled_from(
    ["007", "12:30", "yes", "no", "on", "off", "~", "null", "3:1", "1_000", "0x1f", "-"]
)
_texts = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc"), blacklist_characters="\x7f"),
    min_size=1,
    max_size=40,
).filter(lambda s: s.strip() == s and s.strip() != "")
tag_like = st.one_of(_yaml_hazards, _segment, _texts)

# A key part that is not itself a source extension, so generated keys stay
# meaningful after normalisation strips extensions.
_key_part = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_."),
    min_size=1,
    max_size=12,
).filter(lambda s: s.strip("./") != "")
document_keys = st.lists(_key_part, min_size=1, max_size=4).map("/".join)


# ---------------------------------------------------------------------------
# 1. normalize_document_key — idempotence
# ---------------------------------------------------------------------------


class TestDocumentKeyNormalisation:
    @given(document_keys)
    def test_is_idempotent(self, raw: str) -> None:
        """The refusal message prints a key and tells the operator to pass
        it back as `--as`. A key that normalises to something else would
        silently fail to link the versions it was suggested to link."""
        once = normalize_document_key(raw)
        assert normalize_document_key(once) == once

    @given(document_keys)
    def test_output_carries_no_source_extension(self, raw: str) -> None:
        """An extension is a property of the file, not the document — a
        pipeline switching .md to .txt must not start a new identity.

        A key that is *nothing but* a file type names no document, so it
        is refused rather than stored with the extension still on it.
        """
        try:
            key = normalize_document_key(raw)
        except OutmemError:
            assert raw.strip("./").lower() in {e.lstrip(".") for e in ALLOWED_EXTENSIONS}
            return
        assert not any(key.endswith(ext) for ext in ALLOWED_EXTENSIONS)

    @given(document_keys)
    def test_output_is_normalised_shape(self, raw: str) -> None:
        key = normalize_document_key(raw)
        assert key == key.lower()
        assert not key.startswith("/") and not key.endswith("/")
        assert "//" not in key

    @given(document_keys, st.sampled_from(sorted(ALLOWED_EXTENSIONS)))
    def test_format_change_keeps_the_identity(self, raw: str, ext: str) -> None:
        """The whole point: the same document exported as a different
        source type is the same document."""
        base = normalize_document_key(raw)
        assume(base)
        assert normalize_document_key(base + ext) == base

    @given(
        st.lists(
            st.text(alphabet=st.characters(min_codepoint=48, max_codepoint=122),
                    min_size=1, max_size=10).filter(lambda s: s.strip("./") != ""),
            min_size=1,
            max_size=3,
        ).map("/".join)
    )
    def test_case_folding_is_not_a_new_identity(self, raw: str) -> None:
        """ASCII only on purpose. Unicode case folding is not a round trip
        — U+0149 uppercases to *two* characters — which is a property of
        Unicode, not a defect here. Keys stay case-insensitive where that
        actually means anything.
        """
        assert normalize_document_key(raw.upper()) == normalize_document_key(raw)


# ---------------------------------------------------------------------------
# 2. serialize_wiki_page / parse_wiki_page — round-trip
# ---------------------------------------------------------------------------


class TestFrontmatterRoundTrip:
    @given(
        title=_texts,
        slug=slugs,
        tags=st.lists(tag_like, max_size=4),
        aliases=st.lists(_segment, max_size=3),
        body=st.text(max_size=60).filter(lambda s: "---" not in s),
    )
    @settings(max_examples=150)
    def test_values_survive_the_write(
        self, title: str, slug: str, tags: list[str], aliases: list[str], body: str
    ) -> None:
        """`007` came back as `7` and `12:30` as `750`, because YAML 1.1
        resolves those scalars on load and the value was then persisted.
        A page that loses a tag this way silently drops out of the index."""
        original = WikiFrontmatter(
            title=title, slug=slug, tags=list(tags), aliases=list(aliases)
        )
        reparsed, _body = parse_wiki_page(serialize_wiki_page(original, body))
        assert reparsed.title == title
        assert reparsed.slug == slug
        assert reparsed.tags == tags
        assert reparsed.aliases == aliases

    @given(
        title=_texts,
        slug=slugs,
        tags=st.lists(tag_like, max_size=4),
        body=st.text(max_size=60).filter(lambda s: "---" not in s),
    )
    def test_second_write_is_a_no_op(
        self, title: str, slug: str, tags: list[str], body: str
    ) -> None:
        """Serialisation must reach a fixed point, or the pre-commit hook
        rewrites files forever and every commit carries phantom churn."""
        fm = WikiFrontmatter(title=title, slug=slug, tags=list(tags))
        once = serialize_wiki_page(fm, body)
        twice = serialize_wiki_page(*parse_wiki_page(once))
        assert twice == once

    @given(
        title=_texts,
        slug=slugs,
        body=st.text(max_size=60).filter(lambda s: "---" not in s),
    )
    def test_body_survives_intact(self, title: str, slug: str, body: str) -> None:
        _fm, reparsed = parse_wiki_page(
            serialize_wiki_page(WikiFrontmatter(title=title, slug=slug), body)
        )
        assert reparsed == body.lstrip("\n")


# ---------------------------------------------------------------------------
# 3. slug <-> relpath — round-trip
# ---------------------------------------------------------------------------


class TestSlugPathRoundTrip:
    @given(slugs)
    def test_slug_survives_the_path(self, slug: str) -> None:
        """The path is authoritative for addressing, so every slug the
        grammar accepts has to survive the trip through disk and back."""
        assert relpath_to_slug(slug_to_relpath(slug)) == slug

    @given(slugs)
    def test_derived_slug_is_valid(self, slug: str) -> None:
        """`list_slugs()` derives from the path — what it advertises must
        be what `read()` accepts, or the agent burns a call finding out."""
        validate_slug(relpath_to_slug(slug_to_relpath(slug)))

    @given(slugs)
    def test_path_stays_inside_the_pages_tree(self, slug: str) -> None:
        rel = slug_to_relpath(slug)
        assert not rel.is_absolute()
        assert ".." not in rel.parts
        assert rel.suffix == ".md"

    @given(slugs, slugs)
    def test_distinct_slugs_get_distinct_paths(self, a: str, b: str) -> None:
        assume(a != b)
        assert slug_to_relpath(a) != slug_to_relpath(b)


# ---------------------------------------------------------------------------
# 4. candidate_document_key — the two readers of one rule agree
# ---------------------------------------------------------------------------


class TestCandidateKeyAgreement:
    @given(
        into=st.lists(_segment, max_size=2).map("/".join),
        name=_segment,
        ext=st.sampled_from(sorted(ALLOWED_EXTENSIONS)),
        sha=shas,
    )
    def test_backfill_proposes_what_a_reingest_would_derive(
        self, into: str, name: str, ext: str, sha: str
    ) -> None:
        """`sources backfill` and the ingest-time refusal are two readers
        of one rule. If they disagree, backfill proposes an identity that
        a re-ingest would not derive, and the two halves of one invariant
        drift apart silently."""
        rel_path = "/".join(p for p in (into, sha[:SHA_PREFIX_LEN], name + ext) if p)
        derived = derive_document_key(rel_path, sha)
        assert derived is not None  # the sha segment is present and verified
        assert derived == candidate_document_key(rel_path, sha)

    @given(
        rel_path=st.lists(_segment, min_size=1, max_size=3).map("/".join),
        sha=shas,
    )
    def test_a_row_without_a_sha_segment_still_gets_a_candidate(
        self, rel_path: str, sha: str
    ) -> None:
        """Pre-hash-dir rows exist in wikis older than the layout change;
        their own path is the candidate. `candidate_document_key` must
        never return None, or backfill would skip them silently."""
        assume(derive_document_key(rel_path, sha) is None)
        assert candidate_document_key(rel_path, sha) == normalize_document_key(rel_path)

    @given(
        into=st.lists(_segment, max_size=2).map("/".join),
        name=_segment,
        ext=st.sampled_from(sorted(ALLOWED_EXTENSIONS)),
        sha=shas,
    )
    def test_candidates_are_normalised(
        self, into: str, name: str, ext: str, sha: str
    ) -> None:
        rel_path = "/".join(p for p in (into, sha[:SHA_PREFIX_LEN], name + ext) if p)
        key = candidate_document_key(rel_path, sha)
        assert normalize_document_key(key) == key

    @given(
        rel_paths=st.lists(
            st.tuples(
                st.lists(_segment, min_size=1, max_size=3).map("/".join),
                shas,
            ),
            min_size=1,
            max_size=6,
            unique_by=lambda t: t[0],
        )
    )
    def test_every_unkeyed_row_appears_in_exactly_one_group(
        self, rel_paths: list[tuple[str, str]]
    ) -> None:
        """Backfill must partition the un-keyed rows: a row missing from
        every group is a row that silently never gets an identity."""
        from outmem.sources import SourceEntry, SourceRegistry

        registry = SourceRegistry(sources_dir=Path("/nonexistent"))
        for rel_path, sha in rel_paths:
            registry.entries[rel_path] = SourceEntry(
                rel_path=rel_path,
                sha256=sha,
                registered_at=None,  # type: ignore[arg-type]  # unread by grouping
                size_bytes=0,
            )
        grouped = [r for cand in propose_document_keys(registry) for r in cand.rows]
        assert sorted(grouped) == sorted(rel for rel, _ in rel_paths)
