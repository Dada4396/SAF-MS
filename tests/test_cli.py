import subprocess
import sys


def test_all_scripts_expose_help():
    for script in ["train.py", "evaluate.py", "encode.py", "decode.py"]:
        result = subprocess.run(
            [sys.executable, f"scripts/{script}", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_synthetic_smoke_example_runs():
    result = subprocess.run(
        [sys.executable, "examples/smoke_test.py"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "exact_round_trip=True" in result.stdout
