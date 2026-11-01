"""Parameter scans: writing, submitting and following them."""

from __future__ import annotations

import json
import os
import stat

import f90nml
import pytest

from gs2ew import cli, scan

TEMPLATE = """\
&kt_grids_box_parameters
  nx = 192 ! radial
  ny = 128
/
&theta_grid_parameters
  ntheta = 64
  akappa = 1.0
  tri = 0.0
  tripri = 0.0
/
&knobs
  beta = 0.0 ! Electrostatic
  avail_cpu_time = 86400
/
&fields_knobs
  response_dir = 'response'
  read_response = .false.
/
&init_g_knobs
  restart_dir = "restart"
/
&species_parameters_1
  tprim = 6.0
/
&species_parameters_2
  tprim = 2.0
/
"""


@pytest.fixture
def template(tmp_path):
    path = tmp_path / "base.in"
    path.write_text(TEMPLATE)
    return path


def changes(template, run_input):
    """The values in `run_input` that differ from `template`, as namelist.key."""
    a, b = f90nml.read(template), f90nml.read(run_input)
    return {
        f"{group}.{key}": b[group][key]
        for group in b
        for key in b[group]
        if a.get(group, {}).get(key) != b[group][key]
    }


class TestPoints:
    def test_product(self):
        points = scan.product({"a": [1, 2], "b": [3, 4]})
        assert points == [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]

    def test_zipped(self):
        assert scan.zipped({"a": [1, 2], "b": [3, 4]}) == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]

    def test_zipped_needs_equal_lengths(self):
        with pytest.raises(ValueError, match="equally many"):
            scan.zipped({"a": [1, 2], "b": [3]})

    def test_one_at_a_time_starts_from_the_template(self):
        points = scan.one_at_a_time({"a": [1, 2], "b": [3]})
        assert points == [{}, {"a": 1}, {"a": 2}, {"b": 3}]

    def test_combine(self):
        shaping = scan.zipped({"tri": [0.1, 0.2], "tripri": [0.3, 0.4]})
        points = scan.combine(shaping, scan.product({"beta": [0.0, 0.01]}))
        assert len(points) == 4
        assert points[1] == {"tri": 0.1, "tripri": 0.3, "beta": 0.01}


class TestResolveKey:
    deck = f90nml.reads(TEMPLATE)

    def test_bare(self):
        assert scan.resolve_key(self.deck, "ntheta") == "theta_grid_parameters.ntheta"

    def test_qualified_may_add_a_value(self):
        assert scan.resolve_key(self.deck, "le_grids_knobs.ngauss") == "le_grids_knobs.ngauss"

    def test_ambiguous(self):
        with pytest.raises(ValueError, match="species_parameters_2.tprim"):
            scan.resolve_key(self.deck, "tprim")

    def test_missing(self):
        with pytest.raises(ValueError, match="not in the template"):
            scan.resolve_key(self.deck, "negrid")


class TestWrite:
    def test_runs_differ_only_in_the_scanned_values(self, template, tmp_path):
        points = scan.one_at_a_time({"ntheta": [32], "nx": [256]})
        run_dirs = scan.write(template, points, tmp_path / "scan")

        assert [d.name for d in run_dirs] == ["base", "ntheta_32", "nx_256"]
        assert changes(template, run_dirs[0] / "base.in") == {}
        assert changes(template, run_dirs[1] / "ntheta_32.in") == {
            "theta_grid_parameters.ntheta": 32
        }
        assert changes(template, run_dirs[2] / "nx_256.in") == {
            "kt_grids_box_parameters.nx": 256
        }

    def test_comments_are_kept(self, template, tmp_path):
        (run_dir,) = scan.write(template, [{"nx": 256}], tmp_path / "scan")
        assert "nx = 256 ! radial" in (run_dir / "nx_256.in").read_text()

    def test_values_take_the_template_type(self, template, tmp_path):
        (run_dir,) = scan.write(template, [{"beta": 1, "ntheta": 32.0}], tmp_path / "scan")
        deck = f90nml.read(run_dir / f"{run_dir.name}.in")
        assert isinstance(deck["knobs"]["beta"], float)
        assert isinstance(deck["theta_grid_parameters"]["ntheta"], int)

    def test_names(self, template, tmp_path):
        points = [
            {"beta": 0.01, "tri": -0.5},
            {"species_parameters_1.tprim": 6.9, "species_parameters_2.tprim": 2.5},
        ]
        names = [d.name for d in scan.write(template, points, tmp_path / "scan")]
        assert names == ["beta_0p01-tri_-0p5", "tprim1_6p9-tprim2_2p5"]

    def test_points_sharing_a_name(self, template, tmp_path):
        with pytest.raises(ValueError, match="share a directory"):
            scan.write(template, [{"nx": 256}, {"nx": 256.0}], tmp_path / "scan")

    def test_directories_and_job_script(self, template, tmp_path):
        (run_dir,) = scan.write(template, [{"nx": 256}], tmp_path / "scan", machine="viking")
        assert (run_dir / "restart").is_dir()
        assert (run_dir / "response").is_dir()
        assert (run_dir / scan.JOB_SCRIPT).read_text() == (
            scan.TEMPLATES_DIR / "viking.slurm"
        ).read_text()

    def test_no_job_script(self, template, tmp_path):
        (run_dir,) = scan.write(template, [{"nx": 256}], tmp_path / "scan", machine=None)
        assert not (run_dir / scan.JOB_SCRIPT).exists()

    def test_unknown_machine(self, template, tmp_path):
        with pytest.raises(ValueError, match="Unknown machine"):
            scan.write(template, [{"nx": 256}], tmp_path / "scan", machine="frontier")

    def test_manifest(self, template, tmp_path):
        directory = tmp_path / "scan"
        scan.write(template, [{"nx": 256}], directory)
        manifest = scan.load(directory)
        assert manifest["template"] == str(template.resolve())
        assert manifest["runs"] == [
            {"name": "nx_256", "values": {"kt_grids_box_parameters.nx": 256}, "derived": {}}
        ]
        assert "gs2ew_commit" in manifest
        assert (directory / scan.TEMPLATE_INPUT).read_text() == TEMPLATE

    def test_refuses_an_existing_scan(self, template, tmp_path):
        directory = tmp_path / "scan"
        scan.write(template, [{"nx": 256}], directory)
        with pytest.raises(FileExistsError):
            scan.write(template, [{"nx": 128}], directory)

    def test_the_template_is_run_once(self, template, tmp_path, capsys):
        points = scan.one_at_a_time({"ntheta": [32, 64], "akappa": [1.0, 1.5]})
        run_dirs = scan.write(template, points, tmp_path / "scan")
        assert [d.name for d in run_dirs] == ["base", "ntheta_32", "akappa_1p5"]
        assert "theta_grid_parameters.ntheta = 64" in capsys.readouterr().out

    def test_without_the_template_every_point_runs(self, template, tmp_path):
        run_dirs = scan.write(template, scan.product({"ntheta": [32, 64]}), tmp_path / "scan")
        assert [d.name for d in run_dirs] == ["ntheta_32", "ntheta_64"]

    def test_derive(self, template, tmp_path):
        def tripri(point, deck):
            return {"tripri": 2 * deck["theta_grid_parameters"]["tri"]}

        (run_dir,) = scan.write(template, [{"tri": 0.25}], tmp_path / "scan", derive=tripri)
        assert changes(template, run_dir / "tri_0p25.in") == {
            "theta_grid_parameters.tri": 0.25,
            "theta_grid_parameters.tripri": 0.5,
        }
        assert scan.load(tmp_path / "scan")["runs"][0]["derived"] == {
            "theta_grid_parameters.tripri": 0.5
        }

    def test_derive_may_not_change_a_scanned_value(self, template, tmp_path):
        with pytest.raises(ValueError, match="scanned"):
            scan.write(
                template, [{"tri": 0.25}], tmp_path / "scan", derive=lambda p, d: {"tri": 0}
            )


class TestWarnings:
    def test_read_response(self, template, tmp_path, capsys):
        template.write_text(TEMPLATE.replace("read_response = .false.", "read_response = .true."))
        scan.write(template, [{"nx": 256}], tmp_path / "scan")
        assert "read_response" in capsys.readouterr().out

    def test_beta_without_auto_set_beta_prime(self, template, tmp_path, capsys):
        scan.write(template, [{"beta": 0.01}], tmp_path / "scan")
        assert "auto_set_beta_prime" in capsys.readouterr().out

    def test_explicit_beta_prime_overrides_auto_set(self, template, tmp_path, capsys):
        text = TEMPLATE.replace("  tri = 0.0\n", "  tri = 0.0\n  auto_set_beta_prime = .true.\n")
        text += "&theta_grid_eik_knobs\n  beta_prime_input = 0.0\n/\n"
        template.write_text(text)
        scan.write(template, [{"beta": 0.01}], tmp_path / "scan")
        assert "beta_prime_input" in capsys.readouterr().out

    def test_auto_set_beta_prime(self, template, tmp_path, capsys):
        text = TEMPLATE.replace("  tri = 0.0\n", "  tri = 0.0\n  auto_set_beta_prime = .true.\n")
        template.write_text(text)
        scan.write(template, [{"beta": 0.01}], tmp_path / "scan")
        assert "Warning" not in capsys.readouterr().out

    def test_resolution_needs_no_pressure_gradient(self, template, tmp_path, capsys):
        scan.write(template, [{"nx": 256}], tmp_path / "scan")
        assert "Warning" not in capsys.readouterr().out


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    """Fake sbatch and squeue on PATH. sbatch logs its arguments and prints
    a job ID; squeue reports job 1001 as running."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch.log"
    scripts = {
        "sbatch": f'#!/bin/sh\necho "$PWD $*" >> {log}\necho $((1000 + $(wc -l < {log})))\n',
        "squeue": '#!/bin/sh\necho "1001 RUNNING"\n',
    }
    for name, text in scripts.items():
        path = bin_dir / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


class TestSubmit:
    def test_submits_from_each_run_directory(self, template, tmp_path, slurm):
        directory = tmp_path / "scan"
        scan.write(template, [{"nx": 128}, {"nx": 256}], directory)
        outcomes = scan.submit(directory, ["--time=02:00:00"])

        assert outcomes == [("nx_128", "job 1001"), ("nx_256", "job 1002")]
        first = slurm.read_text().splitlines()[0]
        assert first == (
            f"{directory / 'nx_128'} --parsable --job-name=scan-nx_128 "
            "--time=02:00:00 job.slurm nx_128.in"
        )
        assert "nx_256,1002," in (directory / scan.JOBS).read_text()

    def test_skips_submitted_and_finished_runs(self, template, tmp_path, slurm):
        directory = tmp_path / "scan"
        scan.write(template, [{"nx": 128}, {"nx": 256}], directory)
        scan.submit(directory, [], dry_run=False)
        (directory / "nx_256" / "nx_256.out.nc").touch()
        (directory / "jobs.csv").write_text("run,job,submitted\n")

        outcomes = dict(scan.submit(directory))
        assert outcomes["nx_256"] == "skipped: has output"
        assert outcomes["nx_128"] == "job 1003"
        assert dict(scan.submit(directory))["nx_128"] == "skipped: submitted as job 1003"

    def test_dry_run(self, template, tmp_path, slurm):
        directory = tmp_path / "scan"
        scan.write(template, [{"nx": 128}], directory)
        ((_, command),) = scan.submit(directory, dry_run=True)
        assert "sbatch --parsable --job-name=scan-nx_128 job.slurm nx_128.in" in command
        assert not slurm.exists()
        assert not (directory / scan.JOBS).exists()

    def test_warns_if_slurm_stops_gs2_first(self, template, tmp_path, slurm, capsys):
        directory = tmp_path / "scan"
        scan.write(template, [{"nx": 128}], directory)
        scan.submit(directory, ["--time=02:00:00"], dry_run=True)
        assert "avail_cpu_time is 86400 s" in capsys.readouterr().out
        scan.write(template, [{"nx": 128}, {"nx": 256}], tmp_path / "two")
        scan.submit(tmp_path / "two", ["--time=02:00:00"], dry_run=True)
        assert capsys.readouterr().out.count("Warning") == 1
        # The ARCHER2 template asks for 24 hours, which is enough.
        scan.submit(directory, dry_run=True)
        assert "Warning" not in capsys.readouterr().out

    def test_without_a_job_script(self, template, tmp_path):
        directory = tmp_path / "scan"
        scan.write(template, [{"nx": 128}], directory, machine=None)
        with pytest.raises(ValueError, match="without a job script"):
            scan.submit(directory)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("90", 90 * 60),
        ("10:30", 10 * 60 + 30),
        ("02:00:00", 7200),
        ("1-12", 36 * 3600),
        ("00-24:00:00", 86400),
        ("1-00:30", 86400 + 1800),
    ],
)
def test_slurm_time_limits(text, seconds):
    assert scan._slurm_seconds(text) == seconds


def test_status(template, tmp_path, slurm):
    directory = tmp_path / "scan"
    scan.write(template, scan.product({"nx": [128, 192, 256, 384]}), directory)
    scan.submit(directory)  # jobs 1001, 1002, 1003, 1004
    (directory / "nx_192" / "nx_192.exit_reason").write_text(
        " -1 GS2 internal time limit exceeded.\n  1 nstep timesteps completed.\n"
    )
    (directory / "jobs.csv").write_text(
        "run,job,submitted\nnx_128,1001,x\nnx_192,1002,x\nnx_256,1003,x\n"
    )
    assert scan.status(directory) == [
        ("nx_128", "job 1001: running"),
        ("nx_192", "finished: -1 GS2 internal time limit exceeded.; 1 nstep timesteps completed."),
        ("nx_256", "job 1003: ended without an exit reason"),
        ("nx_384", "not submitted"),
    ]


class TestCli:
    def test_write_submit_status(self, template, tmp_path, slurm, capsys):
        directory = tmp_path / "scan"
        cli.main([
            "scan", "write", str(template), str(directory),
            "--vary", "ntheta=32,96", "--vary", "nx=256",
        ])
        assert [r["name"] for r in scan.load(directory)["runs"]] == [
            "base", "ntheta_32", "ntheta_96", "nx_256"
        ]

        assert cli.main(["scan", "submit", str(directory), "--", "--time=1-00:00:00"]) == 0
        assert "--time=1-00:00:00 job.slurm base.in" in slurm.read_text()

        capsys.readouterr()
        cli.main(["scan", "status", str(directory)])
        assert "job 1001: running" in capsys.readouterr().out

    def test_dry_run_before_sbatch_args(self, template, tmp_path, slurm, capsys):
        directory = tmp_path / "scan"
        cli.main(["scan", "write", str(template), str(directory), "--vary", "nx=128"])
        capsys.readouterr()
        cli.main(["scan", "submit", str(directory), "--dry-run", "--", "--time=1-00:00:00"])
        assert "--time=1-00:00:00 job.slurm nx_128.in" in capsys.readouterr().out
        assert not slurm.exists()

    def test_sbatch_args_need_a_separator(self, template, tmp_path):
        directory = tmp_path / "scan"
        cli.main(["scan", "write", str(template), str(directory), "--vary", "nx=128"])
        with pytest.raises(SystemExit):
            cli.main(["scan", "submit", str(directory), "--time=1-00:00:00"])

    def test_values_are_typed(self, template, tmp_path):
        directory = tmp_path / "scan"
        cli.main([
            "scan", "write", str(template), str(directory), "--machine", "none", "--mode", "product",
            "--vary", "species_parameters_1.tprim=6.9", "--vary", "le_grids_knobs.ngauss=8",
            "--vary", "fields_knobs.read_response=.false.",
        ])
        (run,) = json.loads((directory / scan.MANIFEST).read_text())["runs"]
        assert run["values"] == {
            "species_parameters_1.tprim": 6.9,
            "le_grids_knobs.ngauss": 8,
            "fields_knobs.read_response": False,
        }

    def test_bad_key_is_a_usage_error(self, template, tmp_path):
        with pytest.raises(SystemExit, match="several namelists"):
            cli.main(["scan", "write", str(template), str(tmp_path / "s"), "--vary", "tprim=1"])
