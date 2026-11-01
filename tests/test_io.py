"""Locating a GS2 run on disk."""

from __future__ import annotations

import pytest

from gs2ew.io import find_run


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


class TestFindRun:
    def test_run_directory(self, tmp_path):
        out = _touch(tmp_path / "cbc.out.nc")
        deck = _touch(tmp_path / "cbc.in")
        run = find_run(tmp_path)
        assert run.out_nc == out.resolve()
        assert run.input_file == deck.resolve()
        assert run.run_dir == tmp_path.resolve()
        assert run.name == "cbc"

    def test_output_file(self, tmp_path):
        out = _touch(tmp_path / "cbc.out.nc")
        assert find_run(out).out_nc == out.resolve()

    def test_dotted_run_name(self, tmp_path):
        """GS2 run names can themselves contain dots."""
        out = _touch(tmp_path / "cbc.early.out.nc")
        _touch(tmp_path / "cbc.early.in")
        run = find_run(out)
        assert run.name == "cbc.early"
        assert run.input_file.name == "cbc.early.in"

    def test_missing_input_deck(self, tmp_path):
        _touch(tmp_path / "cbc.out.nc")
        assert find_run(tmp_path).input_file is None

    def test_restart_and_other_files_ignored(self, tmp_path):
        _touch(tmp_path / "cbc.out.nc")
        _touch(tmp_path / "cbc.restart.nc.0")
        _touch(tmp_path / "cbc.used_inputs.in")
        assert find_run(tmp_path).name == "cbc"

    def test_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No"):
            find_run(tmp_path)

    def test_ambiguous_directory_lists_candidates(self, tmp_path):
        _touch(tmp_path / "a.out.nc")
        _touch(tmp_path / "a.early.out.nc")
        with pytest.raises(ValueError, match="a.early.out.nc"):
            find_run(tmp_path)

    def test_not_an_output_file(self, tmp_path):
        with pytest.raises(ValueError, match="neither"):
            find_run(_touch(tmp_path / "cbc.in"))

    def test_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_run(tmp_path / "nope")
