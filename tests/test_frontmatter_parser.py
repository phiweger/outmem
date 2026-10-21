"""Which YAML parser ``outmem.frontmatter`` runs, as opposed to what it produces.

``tests/test_frontmatter.py`` runs under both parsers and pins that a page
means the same thing under either. What it cannot see is which one ran:
the results agree, so a stray ``yaml.safe_load`` or bare ``yaml.compose``
would put the pure-Python parser back on every page load, twelve times
slower over a real corpus, and every behavioural test would stay green.
"""

from __future__ import annotations

import pytest
import yaml
from yaml.reader import Reader

from outmem import frontmatter
from outmem.frontmatter import parse_wiki_page, repair_wiki_page
from tests.conftest import needs_libyaml

# Between them these take every parse path in the module: the load, the
# compose that recovers `tags` and `aliases` as written, and the repair's
# probe and its verifying load.
_PAGE = "---\ntitle: T\nslug: t\ntags: [icd, 007]\naliases: [12:30]\n---\n\nbody\n"
_BROKEN = "---\ntitle: Influenza (Teil 1): Erkrankungen\nslug: t\n---\n\nbody\n"


@needs_libyaml
def test_libyaml_is_picked_when_pyyaml_has_it() -> None:
    assert frontmatter._LOADER is yaml.CSafeLoader


def test_pyyaml_s_own_parser_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PyYAML built without libyaml has no ``CSafeLoader`` at all."""
    monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    assert frontmatter._fastest_safe_loader() is yaml.SafeLoader


@pytest.mark.parametrize(
    "parser", ["python", pytest.param("libyaml", marks=needs_libyaml)]
)
def test_every_parse_goes_through_the_picked_parser(
    parser: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the pure-Python parser builds a ``yaml.reader.Reader``.

    The ``python`` case is the control: it shows the spy sees these
    parses, so an empty list under libyaml means none of them fell back
    to the slow parser, not that the spy missed them.
    """
    loader = yaml.SafeLoader if parser == "python" else yaml.CSafeLoader
    monkeypatch.setattr(frontmatter, "_LOADER", loader)
    readers: list[object] = []
    real_init = Reader.__init__

    def counting_init(self: Reader, stream: object) -> None:
        readers.append(stream)
        real_init(self, stream)

    monkeypatch.setattr(Reader, "__init__", counting_init)

    fm, _body = parse_wiki_page(_PAGE)
    assert fm.tags == ["icd", "007"]
    assert fm.aliases == ["12:30"]
    assert repair_wiki_page(_PAGE) is None
    assert repair_wiki_page(_BROKEN) is not None

    if parser == "python":
        assert readers
    else:
        assert readers == []
