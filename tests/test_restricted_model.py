"""The label model — labels, grants, and the two access predicates.

Pure policy, no disk. Everything the store later enforces is decided
here, so these are the tests that say what the rules *are*; the store
tests say only that each code path calls them.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from outmem.config import _config_from_dict, load_yaml_config
from outmem.restricted import (
    DENY,
    DENY_SET,
    Grants,
    LabelError,
    RestrictedSettings,
    normalise_labels,
    settings_from_dict,
    validate_label,
    visible,
    writable,
)


class TestLabelGrammar:
    def test_a_plain_label(self) -> None:
        assert validate_label("hr") == "hr"

    def test_hyphens_are_allowed(self) -> None:
        assert validate_label("legal-privileged") == "legal-privileged"

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert validate_label("  hr  ") == "hr"

    @pytest.mark.parametrize(
        "bad", ["", "   ", "HR", "hr:payroll", "hr payroll", "-hr", "hr-", "hr--x", "hr/x"]
    )
    def test_rejected(self, bad: str) -> None:
        with pytest.raises(LabelError):
            validate_label(bad)

    def test_uppercase_is_rejected_not_folded(self) -> None:
        """Case-folding is the one normalisation that can *widen* access:
        it would silently map a label onto a different one that somebody
        already holds. Reject instead, at the moment it is typed."""
        with pytest.raises(LabelError, match="lowercase"):
            validate_label("HR")

    def test_non_string_is_rejected(self) -> None:
        with pytest.raises(LabelError):
            validate_label(3)  # type: ignore[arg-type]

    def test_the_deny_sentinel_can_never_be_declared(self) -> None:
        """Everything about fail-closed rests on this: DENY is not a legal
        label, so it cannot be declared, granted, or put in a mode — and
        `labels ⊆ mode` is therefore false for every mode."""
        with pytest.raises(LabelError):
            validate_label(DENY)


class TestNormaliseLabels:
    def test_none_is_open(self) -> None:
        assert normalise_labels(None) == frozenset()

    def test_duplicates_collapse(self) -> None:
        assert normalise_labels(["hr", "hr"]) == frozenset({"hr"})

    def test_a_bare_string_is_refused(self) -> None:
        """`normalise_labels("hr")` would otherwise iterate characters and
        produce {'h', 'r'} — two labels nobody holds, so the content
        vanishes and the mistake looks like it worked."""
        with pytest.raises(LabelError, match=r"\['hr'\]"):
            normalise_labels("hr")


class TestGrants:
    def test_default_is_open_only(self) -> None:
        g = Grants()
        assert g.read == frozenset()
        assert not g.may_read({"hr"})
        assert g.may_read(set())

    def test_write_without_read_is_invalid(self) -> None:
        with pytest.raises(LabelError, match="write without read"):
            Grants(read=frozenset(), write=frozenset({"hr"}))

    def test_declassify_without_write_is_invalid(self) -> None:
        with pytest.raises(LabelError, match="declassify without write"):
            Grants(read=frozenset({"hr"}), declassify=frozenset({"hr"}))

    def test_nesting_is_checked_at_construction_not_at_use(self) -> None:
        """A bad grant must fail in the application's auth code, loudly,
        not widen quietly at whichever call site happens to consult it."""
        with pytest.raises(LabelError):
            Grants(read=frozenset({"hr"}), write=frozenset({"hr", "legal"}))

    def test_writer_helper(self) -> None:
        g = Grants.writer("hr")
        assert g.may_read({"hr"}) and g.may_write({"hr"})
        assert not g.may_declassify({"hr"})

    def test_reader_helper_cannot_write(self) -> None:
        assert not Grants.reader("hr").may_write({"hr"})

    def test_labels_are_validated(self) -> None:
        with pytest.raises(LabelError):
            Grants.reader("HR")

    def test_grants_are_hashable_and_frozen(self) -> None:
        """Grants are handed across a process boundary by the calling
        application and consulted at every write. Frozen means a caller
        cannot widen one after the view was built from it; hashable
        means two equal entitlements are interchangeable."""
        assert hash(Grants.reader("hr")) == hash(Grants.reader("hr"))
        with pytest.raises(FrozenInstanceError):
            Grants().read = frozenset({"hr"})  # type: ignore[misc]


class TestVisibility:
    """§3.1 — visible iff labels ⊆ mode."""

    def test_open_content_is_visible_in_every_mode(self) -> None:
        assert visible(set(), set())
        assert visible(set(), {"hr"})
        assert visible(set(), {"hr", "legal"})

    def test_restricted_content_is_hidden_from_the_empty_mode(self) -> None:
        assert not visible({"hr"}, set())

    def test_the_matching_mode_sees_it(self) -> None:
        assert visible({"hr"}, {"hr"})

    def test_a_mode_sees_open_content_too_not_just_its_own(self) -> None:
        """Mode {hr} is not an HR-only session. The unsafe direction is
        read-HR-then-write-open; reading open while writing HR is a write
        *up*. An HR-only session would also leave the agent composing HR
        pages with no access to the company glossary."""
        assert visible(set(), {"hr"})

    def test_multi_label_content_needs_every_label(self) -> None:
        assert not visible({"hr", "legal"}, {"hr"})
        assert visible({"hr", "legal"}, {"hr", "legal"})

    def test_labels_are_flat_no_prefix_implication(self) -> None:
        """`hr` does not contain `hr-payroll`; they are unrelated strings."""
        assert not visible({"hr-payroll"}, {"hr"})

    def test_deny_is_invisible_to_every_mode(self) -> None:
        for mode in (set(), {"hr"}, {"hr", "legal", "board"}):
            assert not visible(DENY_SET, mode)


class TestWritability:
    """§3.2 — writable iff labels == mode, with write held on the mode."""

    def test_open_write_from_the_open_mode(self) -> None:
        assert writable(set(), set(), Grants())

    def test_no_write_down(self) -> None:
        """In mode {hr} you may not touch an open page — that is exactly
        the path that carries restricted facts into open content."""
        assert not writable(set(), {"hr"}, Grants.writer("hr"))

    def test_no_write_up_either(self) -> None:
        """Holding `write hr` does not let you edit an HR page from an
        open session; you must start a session in mode {hr}."""
        assert not writable({"hr"}, set(), Grants.writer("hr"))

    def test_exact_match_with_the_grant_succeeds(self) -> None:
        assert writable({"hr"}, {"hr"}, Grants.writer("hr"))

    def test_read_grant_alone_is_not_enough(self) -> None:
        assert not writable({"hr"}, {"hr"}, Grants.reader("hr"))

    def test_every_label_in_the_mode_needs_the_grant(self) -> None:
        g = Grants(
            read=frozenset({"hr", "legal"}), write=frozenset({"hr"})
        )
        assert not writable({"hr", "legal"}, {"hr", "legal"}, g)


class TestPathRules:
    def test_a_namespace_glob_matches_children(self) -> None:
        s = RestrictedSettings(labels=frozenset({"hr"}), paths={"hr:*": frozenset({"hr"})})
        assert s.labels_for_slug("hr:severance") == frozenset({"hr"})
        assert s.labels_for_slug("hr:pay:bands") == frozenset({"hr"})

    def test_it_also_covers_the_namespace_root_page(self) -> None:
        """`fnmatch("hr", "hr:*")` is False, so the root page `hr` would
        be the one *open* page in a restricted namespace. "Everything
        under hr" is what the rule means."""
        s = RestrictedSettings(labels=frozenset({"hr"}), paths={"hr:*": frozenset({"hr"})})
        assert s.labels_for_slug("hr") == frozenset({"hr"})

    def test_it_does_not_match_a_similarly_named_namespace(self) -> None:
        s = RestrictedSettings(labels=frozenset({"hr"}), paths={"hr:*": frozenset({"hr"})})
        assert s.labels_for_slug("hrx:notes") == frozenset()
        assert s.labels_for_slug("chr:notes") == frozenset()

    def test_unrelated_slugs_stay_open(self) -> None:
        s = RestrictedSettings(labels=frozenset({"hr"}), paths={"hr:*": frozenset({"hr"})})
        assert s.labels_for_slug("clinical:sepsis") == frozenset()

    def test_overlapping_rules_union(self) -> None:
        s = RestrictedSettings(
            labels=frozenset({"hr", "legal"}),
            paths={"hr:*": frozenset({"hr"}), "*:sealed": frozenset({"legal"})},
        )
        assert s.labels_for_slug("hr:sealed") == frozenset({"hr", "legal"})

    def test_source_rules_use_the_slash_form(self) -> None:
        s = RestrictedSettings(
            labels=frozenset({"hr"}), sources={"hr/*": frozenset({"hr"})}
        )
        assert s.labels_for_source("hr/severance-plan.md") == frozenset({"hr"})
        assert s.labels_for_source("hr") == frozenset({"hr"})
        assert s.labels_for_source("policy/handbook.md") == frozenset()

    def test_matching_is_case_sensitive(self) -> None:
        s = RestrictedSettings(labels=frozenset({"hr"}), paths={"hr:*": frozenset({"hr"})})
        assert s.labels_for_slug("HR:x") == frozenset()


class TestDeclaredSet:
    def test_enabled_tracks_the_declared_labels(self) -> None:
        assert not RestrictedSettings().enabled
        assert RestrictedSettings(labels=frozenset({"hr"})).enabled

    def test_undeclared_labels_are_reported(self) -> None:
        s = RestrictedSettings(labels=frozenset({"hr"}))
        assert s.undeclared({"hr", "legal"}) == frozenset({"legal"})

    def test_deny_is_not_reported_as_undeclared(self) -> None:
        """It is already the fail-closed outcome; reporting it too would
        turn one lint finding into two."""
        assert RestrictedSettings().undeclared(DENY_SET) == frozenset()

    def test_check_declared_raises_with_the_declared_set_in_the_message(self) -> None:
        s = RestrictedSettings(labels=frozenset({"hr"}))
        with pytest.raises(LabelError, match="Declared labels are: hr"):
            s.check_declared({"legal"})

    def test_resolve_keeps_declared_labels(self) -> None:
        s = RestrictedSettings(labels=frozenset({"hr"}))
        assert s.resolve({"hr"}) == frozenset({"hr"})

    def test_resolve_collapses_an_undeclared_label_to_deny(self) -> None:
        """Not merely "keep it, nobody holds it": that works by accident
        only until someone adds the label to config later and thereby
        publishes the item to that compartment retroactively."""
        s = RestrictedSettings(labels=frozenset({"hr"}))
        assert s.resolve({"legal"}) == DENY_SET

    def test_one_undeclared_label_taints_the_whole_set(self) -> None:
        s = RestrictedSettings(labels=frozenset({"hr"}))
        assert s.resolve({"hr", "typo"}) == DENY_SET

    def test_resolve_leaves_open_content_open(self) -> None:
        assert RestrictedSettings(labels=frozenset({"hr"})).resolve(set()) == frozenset()


class TestSettingsFromDict:
    def test_absent_block_is_disabled(self) -> None:
        assert not settings_from_dict(None).enabled

    def test_a_full_block(self) -> None:
        s = settings_from_dict(
            {
                "labels": ["hr", "legal"],
                "paths": {"hr:*": ["hr"]},
                "sources": {"hr/*": ["hr"]},
            }
        )
        assert s.labels == frozenset({"hr", "legal"})
        assert s.labels_for_slug("hr:x") == frozenset({"hr"})
        assert s.labels_for_source("hr/x.md") == frozenset({"hr"})

    def test_a_rule_naming_an_undeclared_label_is_refused(self) -> None:
        """A rule that looks right and restricts a namespace to a
        compartment nobody can hold makes the whole namespace invisible."""
        with pytest.raises(LabelError, match="unknown restriction"):
            settings_from_dict({"labels": ["hr"], "paths": {"x:*": ["legal"]}})

    def test_a_rule_with_no_labels_is_refused(self) -> None:
        with pytest.raises(LabelError, match="assigns no labels"):
            settings_from_dict({"labels": ["hr"], "paths": {"hr:*": []}})

    def test_a_non_mapping_block_is_refused(self) -> None:
        with pytest.raises(LabelError):
            settings_from_dict(["hr"])

    def test_a_non_mapping_rule_block_is_refused(self) -> None:
        with pytest.raises(LabelError):
            settings_from_dict({"labels": ["hr"], "paths": ["hr:*"]})


class TestConfigIntegration:
    def test_the_block_lands_on_the_config(self) -> None:
        cfg = _config_from_dict({"restricted": {"labels": ["hr"]}})
        assert cfg.restricted.enabled
        assert cfg.restricted.labels == frozenset({"hr"})

    def test_absent_block_leaves_it_disabled(self) -> None:
        assert not _config_from_dict({"model": "x"}).restricted.enabled

    def test_it_is_not_swallowed_into_extra(self) -> None:
        cfg = _config_from_dict({"restricted": {"labels": ["hr"]}})
        assert "restricted" not in cfg.extra

    def test_a_malformed_block_raises_rather_than_defaulting(self, tmp_path) -> None:
        """Every other block degrades a feature when it is wrong. This one
        degrades a boundary, so the forgiving-load contract does not apply."""
        (tmp_path / "config.yaml").write_text(
            "restricted:\n  labels: [hr]\n  paths:\n    'x:*': [nope]\n"
        )
        with pytest.raises(LabelError):
            load_yaml_config(tmp_path)

    def test_unparseable_yaml_mentioning_restricted_is_fatal(self, tmp_path) -> None:
        """The hole the forgiving contract would otherwise leave: a syntax
        error anywhere in the file drops the whole config, and a wiki with
        restrictions opens with them silently off."""
        (tmp_path / "config.yaml").write_text(
            "model: x\nrestricted:\n  labels: [hr]\n  bad: [unclosed\n"
        )
        with pytest.raises(LabelError, match="silently disabled"):
            load_yaml_config(tmp_path)

    def test_unparseable_yaml_without_restrictions_is_still_forgiven(
        self, tmp_path
    ) -> None:
        """The existing contract survives for every wiki that has no
        restricted content — which is all of them today."""
        (tmp_path / "config.yaml").write_text("model: x\nbad: [unclosed\n")
        assert load_yaml_config(tmp_path).model  # defaults, no raise

    def test_a_commented_out_block_does_not_make_a_typo_fatal(
        self, tmp_path
    ) -> None:
        (tmp_path / "config.yaml").write_text(
            "# restricted:\n#   labels: [hr]\nbad: [unclosed\n"
        )
        load_yaml_config(tmp_path)
