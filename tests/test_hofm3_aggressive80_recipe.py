"""No-data smoke tests for the aggressive HOFM3 80-epoch training recipe.

Tests call --dry-run only and never access/checkpoint/train MovieLens 32M.
"""
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_hofm3_aggressive80.sh"


def _shell(*arguments):
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=str(ROOT), text=True, capture_output=True, check=False,
    )


def _flags(stdout):
    command_line = next(line.removeprefix("Command: ")
                        for line in stdout.splitlines()
                        if line.startswith("Command: "))
    return shlex.split(command_line)


def test_recipe_has_valid_bash_syntax():
    process = subprocess.run(["bash", "-n", str(SCRIPT)], text=True,
                             capture_output=True, check=False)
    assert process.returncode == 0, process.stderr


def test_dry_run_is_80_epochs_and_aggressively_regularized():
    run = _shell("--dry-run", "--label", "experiment80_ci")
    assert run.returncode == 0, run.stderr
    assert "DRY RUN ONLY" in run.stdout
    argv = _flags(run.stdout)
    assert argv[:5] == ["python", "-u", "-m", "src.hofm_experiments", "run"]
    assert "--force-retrain" in argv
    assert argv[argv.index("--run-label") + 1] == "experiment80_ci"
    assert argv[argv.index("--epochs") + 1] == "80"
    assert argv[argv.index("--min-epochs") + 1] == "80"
    assert argv[argv.index("--steps-per-epoch") + 1] == "800"
    assert argv[argv.index("--patience") + 1] == "999"
    assert argv[argv.index("--max-candidates") + 1] == "0"
    # Regression guard: all penalties must exceed the earlier strongreg run.
    penalty_minimums = {
        "--weight-decay": 0.08,
        "--third-l2": 0.0005,
        "--bcd-dropout": 0.15,
        "--factor-l2": 0.0025,
        "--content-l2": 0.0005,
        "--bcd-l1": 0.001,
        "--bcd-l2": 0.002,
    }
    for flag, baseline in penalty_minimums.items():
        assert float(argv[argv.index(flag) + 1]) > baseline, flag
    assert "--resume" not in argv
    assert "--registry-db" not in argv


@pytest.mark.parametrize("label", ["bad-label", "bad/name", "bad name", "fast_bcd",
                                  "bcd_hofm3_research"])
def test_invalid_and_reserved_labels_rejected_without_training(label):
    run = _shell("--dry-run", "--label", label)
    assert run.returncode != 0
    assert "Command:" not in run.stdout


def test_help_exits_without_training():
    run = _shell("--help")
    assert run.returncode == 0
    assert "80 epochs" in run.stdout
