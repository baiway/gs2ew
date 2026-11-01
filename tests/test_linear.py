"""Linear scans: writing and submitting them, and their spectra."""

from __future__ import annotations

import json
import os
import stat

import f90nml
import numpy as np
import pytest
import xarray as xr

from conftest import make_linear_dataset
from gs2ew import cli, linear, scan
from gs2ew.plots import linear as linear_plots

#: A nonlinear input file, as a linear scan is often written from.
TEMPLATE = """\
&nonlinear_terms_knobs
  nonlinear_mode = 'on'
/
&kt_grids_knobs
  grid_option = 'box'
/
&kt_grids_box_parameters
  ny = 13
  y0 = -0.1
/
&theta_grid_parameters
  ntheta = 32
  nperiod = 1
/
&knobs
  beta = 0.0 ! Electrostatic
  delt = 0.05
  max_sim_time = 600.0
  avail_cpu_time = 86400
  margin_cpu_time = 1800
/
&fields_knobs
  dump_response = .true.
  response_dir = 'response'
/
&gs2_diagnostics_knobs
  write_fields = .true.
  omegatol = -0.001
  save_for_restart = .true.
  write_free_energy_transfer = .true.
  write_kinetic_energy_transfer = .true.
/
"""


@pytest.fixture
def template(tmp_path):
    path = tmp_path / "nonlinear.in"
    path.write_text(TEMPLATE)
    return path


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    """A fake sbatch on PATH that logs its arguments and prints a job ID."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch.log"
    path = bin_dir / "sbatch"
    path.write_text(f'#!/bin/sh\necho "$PWD $*" >> {log}\necho $((1000 + $(wc -l < {log})))\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


class TestKy:
    def test_list(self):
        assert linear.ky_values("0.3,0.1,0.3") == [0.1, 0.3]

    def test_log_spaced(self):
        assert linear.ky_values("0.1:1:3") == [0.1, 0.316, 1.0]
        assert linear.ky_values("0.2:5:1") == [0.2]

    @pytest.mark.parametrize("spec", ["0:1:3", "1:0.1:3", "0.1:1", "-0.1", "a,b", ""])
    def test_bad(self, spec):
        with pytest.raises(ValueError):
            linear.ky_values(spec)

    @pytest.mark.parametrize(
        ("box", "step"),
        [({"y0": -0.1}, 0.1), ({"y0": 5.0}, 0.2), ({"ly": 2 * np.pi / 0.25}, 0.25), ({}, 0.5)],
    )
    def test_box(self, box, step):
        deck = f90nml.Namelist({"kt_grids_box_parameters": {"ny": 13, **box}})
        assert linear.box_ky(deck) == pytest.approx([step * n for n in range(1, 5)])

    def test_no_box(self):
        with pytest.raises(ValueError, match="no kt_grids_box_parameters"):
            linear.box_ky(f90nml.Namelist({}))


class TestSettings:
    def test_from_a_nonlinear_template(self, template):
        settings = linear.linear_settings(f90nml.read(template))
        assert settings["nonlinear_terms_knobs.nonlinear_mode"] == "off"
        assert settings["kt_grids_knobs.grid_option"] == "single"
        assert settings["kt_grids_single_parameters.theta0"] == 0.0
        # 25 a/v_th at delt = 0.05.
        assert settings["gs2_diagnostics_knobs.navg"] == 500
        # The template's negative omegatol would never be met.
        assert settings["gs2_diagnostics_knobs.omegatol"] == linear.DEFAULT_OMEGATOL
        for name in (
            "gs2_diagnostics_knobs.save_for_restart",
            "gs2_diagnostics_knobs.write_free_energy_transfer",
            "gs2_diagnostics_knobs.write_kinetic_energy_transfer",
            "fields_knobs.dump_response",
        ):
            assert settings[name] is False

    def test_changes_only_what_it_must(self):
        settings = linear.linear_settings(
            f90nml.Namelist({"knobs": {"delt": 0.1}}), omega_window=10.0, theta0=0.5
        )
        assert settings == {
            "kt_grids_knobs.grid_option": "single",
            "kt_grids_single_parameters.theta0": 0.5,
            "gs2_diagnostics_knobs.write_omega": True,
            "gs2_diagnostics_knobs.write_omavg": True,
            "gs2_diagnostics_knobs.write_fields": True,
            "gs2_diagnostics_knobs.exit_when_converged": True,
            "gs2_diagnostics_knobs.navg": 100,
        }


class TestWrite:
    def test_a_run_for_each_point_and_ky(self, template, tmp_path):
        directory = tmp_path / "lin"
        runs = linear.write(
            template, scan.product({"beta": [0.0, 0.01]}), [0.1, 0.3], directory,
            settings={"nperiod": 3},
        )
        assert [r.name for r in runs] == [
            "beta_0-aky_0p1", "beta_0-aky_0p3", "beta_0p01-aky_0p1", "beta_0p01-aky_0p3",
        ]
        deck = f90nml.read(directory / "beta_0p01-aky_0p3" / "beta_0p01-aky_0p3.in")
        assert deck["kt_grids_single_parameters"]["aky"] == 0.3
        assert deck["knobs"]["beta"] == 0.01
        assert deck["theta_grid_parameters"]["nperiod"] == 3
        assert deck["nonlinear_terms_knobs"]["nonlinear_mode"] == "off"
        assert "! Electrostatic" in (directory / "beta_0-aky_0p1" / "beta_0-aky_0p1.in").read_text()

        manifest = scan.load(directory)
        assert manifest["linear"] == {"ky": [0.1, 0.3], "omega_window": 25.0, "theta0": 0.0}
        assert manifest["settings"]["theta_grid_parameters.nperiod"] == 3
        assert manifest["runs"][3]["values"] == {"knobs.beta": 0.01, linear.KY_KEY: 0.3}
        assert (directory / scan.JOB_SCRIPT).read_text().startswith("#!/bin/bash")
        assert linear.scanned_key(directory) == "knobs.beta"

    def test_the_template_alone(self, template, tmp_path):
        runs = linear.write(template, [{}], [0.2], tmp_path / "lin", machine=None)
        assert [r.name for r in runs] == ["aky_0p2"]
        assert not (tmp_path / "lin" / scan.JOB_SCRIPT).exists()
        assert linear.scanned_key(tmp_path / "lin") is None

    def test_runs_the_template_values_once(self, template, tmp_path):
        points = scan.one_at_a_time({"beta": [0.0, 0.01]})
        runs = linear.write(template, points, [0.2], tmp_path / "lin")
        assert [r.name for r in runs] == ["aky_0p2", "beta_0p01-aky_0p2"]

    def test_ky_is_not_a_point(self, template, tmp_path):
        with pytest.raises(ValueError, match="ky is given"):
            linear.write(template, [{"aky": 0.2}], [0.2], tmp_path / "lin")

    def test_warns_of_too_short_a_run(self, template, tmp_path, capsys):
        linear.write(template, [{}], [0.2], tmp_path / "lin", max_sim_time=50.0)
        assert "max_sim_time = 50" in capsys.readouterr().out

    def test_scan_submit_refuses_it(self, template, tmp_path):
        linear.write(template, [{}], [0.2], tmp_path / "lin")
        with pytest.raises(ValueError, match="gs2ew linear submit"):
            scan.submit(tmp_path / "lin", dry_run=True)

    def test_cli(self, template, tmp_path, capsys):
        directory = tmp_path / "lin"
        assert cli.main([
            "linear", "write", str(template), str(directory),
            "--vary", "beta=0.01", "--ky", "0.5", "--box-ky", "--set", "nperiod=2",
        ]) == 0
        assert "ky: 0.1, 0.2, 0.3, 0.4, 0.5" in capsys.readouterr().out
        assert len(scan.load(directory)["runs"]) == 10


class TestSubmit:
    def scan(self, template, tmp_path):
        directory = tmp_path / "lin"
        linear.write(template, scan.product({"beta": [0.0, 0.01]}), [0.1, 0.2, 0.4], directory)
        return directory

    def test_deals_runs_to_jobs_by_ky(self, template, tmp_path, slurm):
        directory = self.scan(template, tmp_path)
        outcomes = dict(
            linear.submit(directory, jobs=2, cores_per_run=4, sbatch_args=["--ntasks=8", "--time=00:20:00"])
        )
        assert outcomes["beta_0-aky_0p1"].startswith("job 1001 (runs_00.list: 3 runs in 2 batches of 4 cores")
        assert (directory / "runs_00.list").read_text().splitlines() == [
            "beta_0-aky_0p1/beta_0-aky_0p1.in",
            "beta_0-aky_0p2/beta_0-aky_0p2.in",
            "beta_0-aky_0p4/beta_0-aky_0p4.in",
        ]
        first, second = slurm.read_text().splitlines()
        assert first == (
            f"{directory} --parsable --job-name=lin-runs_00 --ntasks=8 "
            "--time=00:20:00 job.slurm runs_00.list --nbatch 2"
        )
        assert second.endswith("runs_01.list --nbatch 2")
        # Two runs to a batch share the 20 minutes, less the start-up.
        knobs = f90nml.read(directory / "beta_0-aky_0p1" / "beta_0-aky_0p1.in")["knobs"]
        assert knobs["avail_cpu_time"] == 570
        assert knobs["margin_cpu_time"] == 28
        assert "beta_0-aky_0p1,1001,," not in (directory / scan.JOBS).read_text()
        assert "beta_0-aky_0p1,1001," in (directory / scan.JOBS).read_text()

    def test_submits_only_whats_left(self, template, tmp_path, slurm):
        directory = self.scan(template, tmp_path)
        linear.submit(directory, sbatch_args=["--ntasks=128"])
        (directory / "runs_00.list").unlink()
        (directory / scan.JOBS).write_text("run,job,submitted,cores\nbeta_0-aky_0p1,1001,,4\n")
        (directory / "beta_0-aky_0p2" / "beta_0-aky_0p2.out.nc").touch()
        outcomes = dict(linear.submit(directory, sbatch_args=["--ntasks=128"]))
        assert outcomes["beta_0-aky_0p1"].startswith("skipped")
        assert outcomes["beta_0-aky_0p2"].startswith("skipped")
        assert outcomes["beta_0-aky_0p4"].startswith("job 1002 (runs_00.list: 4 runs in 4 batches of 32 cores")

    def test_dry_run_changes_nothing(self, template, tmp_path, slurm):
        directory = self.scan(template, tmp_path)
        before = (directory / "beta_0-aky_0p1" / "beta_0-aky_0p1.in").read_text()
        outcomes = linear.submit(directory, sbatch_args=["--ntasks=8", "--time=00:20:00"], dry_run=True)
        assert "job.slurm runs_00.list --nbatch 2" in outcomes[0][1]
        assert not slurm.exists()
        assert not list(directory.glob("*.list"))
        assert (directory / "beta_0-aky_0p1" / "beta_0-aky_0p1.in").read_text() == before

    def test_needs_a_core_count(self, template, tmp_path):
        directory = self.scan(template, tmp_path)
        (directory / scan.JOB_SCRIPT).write_text("#!/bin/bash\n")
        with pytest.raises(ValueError, match="how many cores"):
            linear.submit(directory, dry_run=True)
        with pytest.raises(ValueError, match="can't give a run 8"):
            linear.submit(directory, cores_per_run=8, sbatch_args=["--ntasks=4"], dry_run=True)

    def test_refuses_a_scan_that_isnt_linear(self, tmp_path):
        (tmp_path / "base.in").write_text("&knobs\n beta = 0.0\n/\n")
        scan.write(tmp_path / "base.in", [{"beta": 0.01}], tmp_path / "scan")
        with pytest.raises(ValueError, match="not a linear scan"):
            linear.submit(tmp_path / "scan", dry_run=True)


class TestStopped:
    def test_from_the_output(self):
        assert linear.stopped(make_linear_dataset(t_end=20.0)) == "converged"
        assert linear.stopped(make_linear_dataset(t_end=99.97)) == "reached max_sim_time"

    def test_from_the_exit_reason(self, tmp_path):
        path = tmp_path / "run.out.nc"
        make_linear_dataset(t_end=20.0).to_netcdf(path)
        with xr.open_dataset(path) as ds:
            assert linear.stopped(ds) == "converged (no .exit_reason)"
            # A converged run says it completed its steps; that doesn't count.
            (tmp_path / "run.exit_reason").write_text("  1 nstep timesteps completed.\n")
            assert linear.stopped(ds) == "converged"
            (tmp_path / "run.exit_reason").write_text(" -1 GS2 internal time limit exceeded.\n")
            assert linear.stopped(ds) == "reached avail_cpu_time"


def spectrum_runs(**kwargs):
    """Linear runs at two betas and three ky, peaking at ky = 0.3; the
    highest ky at beta = 0.01 ran out of time."""
    return [
        make_linear_dataset(
            ky=ky,
            beta=beta,
            gamma=gamma * (1 - 20 * beta),
            omega=ky,
            t_end=100.0 if (beta, ky) == (0.01, 1.0) else 20.0,
            **kwargs,
        )
        for beta in (0.0, 0.01)
        for ky, gamma in ((0.1, 0.1), (0.3, 0.2), (1.0, 0.05))
    ]


class TestSpectrum:
    def test_by_value_and_ky(self):
        spectrum = linear_plots.compute_linear_spectrum(spectrum_runs(), "beta")
        assert spectrum.attrs["parameter"] == "knobs.beta"
        assert list(spectrum["value"].values) == [0.0, 0.01]
        assert list(spectrum["ky"].values) == [0.1, 0.3, 1.0]
        assert spectrum["gamma"].values[0] == pytest.approx([0.1, 0.2, 0.05])
        assert spectrum["omega"].values[1] == pytest.approx([0.1, 0.3, 1.0])
        assert list(spectrum["converged"].values[1]) == [1.0, 1.0, 0.0]
        assert spectrum["stopped"].values[1, 2] == "reached max_sim_time"

    def test_one_line_without_vary(self):
        spectrum = linear_plots.compute_linear_spectrum(spectrum_runs()[:3])
        assert spectrum.sizes["value"] == 1
        assert spectrum.attrs["parameter"] == ""

    def test_runs_must_differ_only_in_ky(self):
        with pytest.raises(ValueError, match="differ only in ky"):
            linear_plots.compute_linear_spectrum(spectrum_runs())

    def test_needs_the_frequency(self):
        ds = make_linear_dataset().drop_vars("omega_average")
        with pytest.raises(ValueError, match="write_omavg"):
            linear_plots.compute_linear_spectrum([ds])

    def test_gamma_max_leaves_out_unconverged_runs(self):
        runs = spectrum_runs()
        runs[5] = make_linear_dataset(ky=1.0, beta=0.01, gamma=0.9, t_end=100.0)
        reduced = linear_plots.compute_linear_gamma_max(runs, "beta")
        assert reduced["gamma_max"].values == pytest.approx([0.2, 0.16])
        assert list(reduced["ky_max"].values) == [0.3, 0.3]
        assert list(reduced["unconverged"].values) == [0, 1]

    def test_gamma_max_needs_numbers(self):
        with pytest.raises(ValueError, match="doesn't take numbers"):
            linear_plots.compute_linear_gamma_max(spectrum_runs()[:3], "grid_option")

    def test_summary(self):
        spectrum = linear_plots.compute_linear_spectrum(spectrum_runs(), "beta")
        lines = linear.summary(spectrum)
        assert lines[0] == "beta = 0: gamma_max = 0.2 at ky = 0.3."
        assert "25% of the peak" in lines[1]
        assert lines[2] == "beta = 0.01: gamma_max = 0.16 at ky = 0.3; 1 run(s) not converged."
        peaked = linear_plots.compute_linear_spectrum(spectrum_runs()[1:3], None)
        assert "at the lowest ky" in linear.summary(peaked)[1]


class TestModes:
    def test_turned_and_scaled(self):
        reduced = linear_plots.compute_linear_modes(spectrum_runs(), "beta")
        assert list(reduced["ky"].values) == [0.3, 0.3]
        phi = reduced["phi"].values[0]
        peak = np.nanargmax(np.hypot(phi[:, 0], phi[:, 1]))
        assert phi[peak] == pytest.approx([1.0, 0.0])
        apar = reduced["apar"].values[0, :, 0] + 1j * reduced["apar"].values[0, :, 1]
        assert np.abs(apar).max() == pytest.approx(1.0)
        assert apar == pytest.approx(-apar[::-1])

    def test_at_a_ky(self):
        reduced = linear_plots.compute_linear_modes(spectrum_runs(), "beta", ky=0.9)
        assert list(reduced["ky"].values) == [1.0, 1.0]
        assert list(reduced["converged"].values) == [1.0, 0.0]

    def test_runs_of_different_nperiod(self):
        runs = [make_linear_dataset(nperiod=n) for n in (1, 2)]
        reduced = linear_plots.compute_linear_modes(runs, "nperiod")
        assert reduced.sizes["point"] == 33
        assert np.isnan(reduced["theta"].values[0, 17:]).all()


class TestPlot:
    def test_cli(self, tmp_path, capsys):
        directory = tmp_path / "runs"
        for ds in spectrum_runs():
            name = f"b{ds['ky'].item()}-{ds.attrs['id']}".replace(".", "p")
            (directory / name).mkdir(parents=True)
            ds.to_netcdf(directory / name / f"{name}.out.nc")
            (directory / name / f"{name}.exit_reason").write_text("  1 nstep timesteps completed.\n")
        out = tmp_path / "out"
        assert cli.main(["linear", "plot", str(directory), "--vary", "beta", "-o", str(out)]) == 0
        printed = capsys.readouterr().out
        assert "beta = 0.01: gamma_max = 0.16 at ky = 0.3; 1 run(s) not converged." in printed
        assert {p.name for p in out.glob("*.png")} == {
            "linear_beta_spectrum.png", "linear_beta_gamma_max.png", "linear_beta_modes.png",
        }
        bundle = xr.open_dataset(out / "linear_beta_spectrum.nc")
        assert len(json.loads(bundle.attrs["gs2ew_sources"])) == 6
        bundle.close()
