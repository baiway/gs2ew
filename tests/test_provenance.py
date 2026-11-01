"""Recording provenance: code versions, input decks and source identity."""

from __future__ import annotations

import subprocess

import pytest
import xarray as xr

from gs2ew import provenance
from gs2ew.provenance import UNKNOWN, CodeVersion

from conftest import SYNTHETIC_DECK, SYNTHETIC_GS2_HASH, make_dataset


class TestGS2Version:
    def test_clean_build(self, ds):
        v = provenance.gs2_version(ds)
        assert v.version == "8.2.1-2129-g5f2ef277b"
        assert v.commit == SYNTHETIC_GS2_HASH
        assert v.branch == "fix/kinetic-energy-transfer-prefactor"
        assert v.dirty is False

    def test_modified_build_is_dirty(self):
        assert provenance.gs2_version(make_dataset(gs2_state="modified")).dirty is True

    def test_no_metadata(self):
        v = provenance.gs2_version(make_dataset(metadata=False))
        assert (v.version, v.commit, v.branch, v.dirty) == (
            UNKNOWN,
            UNKNOWN,
            UNKNOWN,
            None,
        )
        assert v.commit_url == ""

    def test_commit_url_uses_bitbucket_layout(self, ds):
        assert provenance.gs2_version(ds).commit_url == (
            f"https://bitbucket.org/gyrokinetics/gs2/commits/{SYNTHETIC_GS2_HASH}"
        )


class TestCodeVersionAttrs:
    def test_github_commit_url(self):
        v = CodeVersion("gs2ew", provenance.GS2EW_REPOSITORY, commit="abc123")
        assert v.commit_url == "https://github.com/baiway/gs2ew/commit/abc123"

    @pytest.mark.parametrize(
        "dirty, expected", [(True, "true"), (False, "false"), (None, UNKNOWN)]
    )
    def test_dirty_flag_is_a_string(self, dirty, expected):
        v = CodeVersion("gs2", provenance.GS2_REPOSITORY, dirty=dirty)
        assert v.attrs()["gs2_dirty"] == expected

    def test_all_attrs_are_strings(self, ds):
        attrs = provenance.gs2_version(ds).attrs()
        assert all(isinstance(v, str) for v in attrs.values())
        assert set(attrs) == {
            "gs2_version",
            "gs2_commit",
            "gs2_branch",
            "gs2_dirty",
            "gs2_repository",
            "gs2_commit_url",
        }


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def checkout(tmp_path):
    """A throwaway git checkout laid out like gs2ew."""
    (tmp_path / "src" / "gs2ew").mkdir(parents=True)
    (tmp_path / "src" / "gs2ew" / "__init__.py").write_text("")
    (tmp_path / "README.md").write_text("readme\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


class TestGitState:
    def test_clean(self, checkout):
        commit, branch, dirty = provenance._git_state(checkout)
        assert len(commit) == 40
        assert branch
        assert dirty is False

    def test_source_edit_is_dirty(self, checkout):
        (checkout / "src" / "gs2ew" / "__init__.py").write_text("x = 1\n")
        assert provenance._git_state(checkout)[2] is True

    def test_new_source_file_is_dirty(self, checkout):
        (checkout / "src" / "gs2ew" / "new.py").write_text("")
        assert provenance._git_state(checkout)[2] is True

    def test_docs_edit_is_not_dirty(self, checkout):
        """Only changes that can alter a result count."""
        (checkout / "README.md").write_text("edited\n")
        (checkout / "scratch.txt").write_text("")
        assert provenance._git_state(checkout)[2] is False

    def test_not_a_repository(self, tmp_path):
        assert provenance._git_state(tmp_path) is None


class TestGS2ewVersion:
    def test_vcs_install_is_pinned_and_clean(self):
        direct_url = {
            "url": "https://github.com/baiway/gs2ew.git",
            "vcs_info": {
                "vcs": "git",
                "commit_id": "0123456789abcdef0123456789abcdef01234567",
                "requested_revision": "v0.2.0",
            },
        }
        v = provenance._from_vcs_install(direct_url, "0.2.0")
        assert v.commit == "0123456789abcdef0123456789abcdef01234567"
        assert v.branch == "v0.2.0"
        assert v.dirty is False

    def test_editable_root(self):
        direct_url = {
            "url": "file:///Users/someone/gs2%20ew",
            "dir_info": {"editable": True},
        }
        assert str(provenance._editable_root(direct_url)) == "/Users/someone/gs2 ew"

    def test_local_snapshot_install_is_not_trusted(self):
        """A non-editable directory install can't be tied to a git state."""
        direct_url = {"url": "file:///Users/someone/gs2ew", "dir_info": {}}
        assert provenance._editable_root(direct_url) is None

    def test_this_checkout(self):
        """The test suite runs from an editable checkout, so git is asked."""
        v = provenance.gs2ew_version()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert v.commit == head
        assert v.dirty in (True, False)
        assert v.version != UNKNOWN


def test_dependency_versions():
    versions = provenance.dependency_versions()
    assert {"python", "numpy", "xarray", "matplotlib", "netCDF4"} <= set(versions)
    assert versions["numpy"] != UNKNOWN


class TestInputDeck:
    def test_embedded_deck(self, ds):
        assert provenance.input_deck(ds) == SYNTHETIC_DECK

    def test_used_deck_has_defaults_filled_in(self, ds):
        used = provenance.input_deck(ds, used=True)
        assert used.startswith(SYNTHETIC_DECK.rstrip("\n"))
        assert "layout = 'lxyes'" in used

    def test_absent(self):
        assert provenance.input_deck(make_dataset(metadata=False)) is None

    def test_matches_file_on_disk(self, ds, tmp_path):
        deck = tmp_path / "run.in"
        # Trailing whitespace, blank lines and tabs (which GS2 stores as single
        # spaces) are not differences.
        deck.write_text(
            SYNTHETIC_DECK.replace("delt = 0.05", "delt = 0.05   ").replace(
                " nstep", "\tnstep"
            )
            + "\n\n"
        )
        assert provenance.deck_matches(provenance.input_deck(ds), deck)

    def test_edited_file_on_disk(self, ds, tmp_path):
        deck = tmp_path / "run.in"
        deck.write_text(SYNTHETIC_DECK.replace("nstep = 1000", "nstep = 2000"))
        assert not provenance.deck_matches(provenance.input_deck(ds), deck)


    def test_restarted_file_on_disk(self, ds, tmp_path):
        deck = tmp_path / "run.in"
        deck.write_text(
            SYNTHETIC_DECK.replace("nstep = 1000", "nstep = 1000\n delt_option = 'check_restart'")
            + "&init_g_knobs\n ginit_option = 'many'\n/\n"
        )
        assert not provenance.deck_matches(provenance.input_deck(ds), deck)
        assert provenance.deck_restarted(provenance.input_deck(ds), deck)

    def test_edited_is_not_restarted(self, ds, tmp_path):
        deck = tmp_path / "run.in"
        deck.write_text(SYNTHETIC_DECK.replace("nstep = 1000", "nstep = 2000"))
        assert not provenance.deck_restarted(provenance.input_deck(ds), deck)


class TestSourceIdentity:
    def test_from_file(self, ds, tmp_path):
        path = tmp_path / "run.out.nc"
        ds.to_netcdf(path)
        with xr.open_dataset(path) as reopened:
            ident = provenance.source_identity(reopened)
        assert ident["source_path"] == str(path.resolve())
        assert int(ident["source_size_bytes"]) == path.stat().st_size
        assert ident["source_id"] == "00000000-0000-4000-8000-000000000000"
        assert ident["source_run_command"] == "gs2 synthetic.in"

    def test_in_memory(self):
        ident = provenance.source_identity(make_dataset(metadata=False))
        assert set(ident.values()) == {UNKNOWN}


@pytest.mark.slow
def test_real_files_degrade_gracefully(real_file):
    """Every real output yields a full record, with "unknown" where GS2 kept less."""
    with xr.open_dataset(real_file) as ds:
        version = provenance.gs2_version(ds)
        identity = provenance.source_identity(ds)
        deck = provenance.input_deck(ds)

    assert all(isinstance(v, str) for v in version.attrs().values())
    assert identity["source_path"] == str(real_file.resolve())
    assert int(identity["source_size_bytes"]) > 0

    sibling = real_file.with_name(real_file.name.replace(".out.nc", ".in"))
    if deck is not None and sibling.exists():
        assert provenance.deck_matches(deck, sibling)


@pytest.mark.slow
@pytest.mark.parametrize(
    "key, version, dirty",
    [
        ("nonlinear-8.2.1-2129", "8.2.1-2129-g5f2ef277b", False),
        ("linear-8.2.1-2130-modified", "8.2.1-2130-g0a2c8f2e6", True),
        ("reduced-8.2.0", "8.2.0-2-ge29e9ab8", None),
        ("no-metadata", UNKNOWN, None),
    ],
)
def test_real_gs2_versions(key, version, dirty):
    from conftest import REAL_FILES

    path = REAL_FILES[key]
    if not path.exists():
        pytest.skip(f"GS2 output not found at {path}")
    with xr.open_dataset(path) as ds:
        v = provenance.gs2_version(ds)
    assert (v.version, v.dirty) == (version, dirty)
