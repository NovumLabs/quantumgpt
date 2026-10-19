"""Offline tests for safe pipeline discovery; no training stage is launched."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PipelineCliTests(unittest.TestCase):
    def test_list_command_succeeds_and_separates_8k_track(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "pipeline_runner.py"), "--list"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prepare-stories-gpt2", result.stdout)
        self.assertIn("prepare-data-8k", result.stdout)
        self.assertIn("train_tiny_shakespeare_gpt.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
