"""Data-free CLI checks for the HOFM4 80-epoch training script."""
from pathlib import Path
import shlex
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_hofm4_jointbcd.sh"


def execute(*args):
    return subprocess.run(["bash", str(SCRIPT), *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_script_bash_syntax():
    process = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr


def test_hofm4_dry_run_unique_model_and_80_epochs():
    result = execute("--dry-run", "--label", "hofm4_test_dryrun")
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    line = next(row.removeprefix("Command: ") for row in result.stdout.splitlines()
                if row.startswith("Command: "))
    tokens = shlex.split(line)
    assert tokens[:5] == ["python", "-u", "-m", "src.hofm_experiments", "run"]
    assert tokens[tokens.index("--model-id") + 1] == "hofm4_jointbcd"
    assert tokens[tokens.index("--script") + 1] == "fast_unified_hofm4.py"
    assert tokens[tokens.index("--run-label") + 1] == "hofm4_test_dryrun"
    assert tokens[tokens.index("--epochs") + 1] == "80"
    assert tokens[tokens.index("--min-epochs") + 1] == "80"
    assert tokens[tokens.index("--fourth-rank") + 1] == "24"
    assert tokens[tokens.index("--fourth-strength") + 1] == "1.0"
    assert "--force-retrain" in tokens
    assert "--resume" not in tokens


@pytest.mark.parametrize("invalid", ["bad label", "bad/label", "bad-label", ""])
def test_invalid_labels_rejected(invalid):
    result = execute("--dry-run", "--label", invalid)
    assert result.returncode != 0


def test_self_test_cli_documented_in_training_command():
    result = execute("--help")
    assert result.returncode == 0
    assert "train_hofm4_jointbcd.sh" in result.stdout

def test_hofm4_python_sources_compile_without_torch_import():
    for relative in ("src/hofm4_interaction.py", "fast_unified_hofm4.py"):
        path = ROOT / relative
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
