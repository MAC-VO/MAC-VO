import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import yaml
from Utility.Config import load_config

ROOT = Path(__file__).resolve().parents[2]


class ProvenanceTests(unittest.TestCase):
    def test_constructor_failure_records_failed_status(self):
        _, config = load_config(ROOT/"Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml")
        config["Odometry"]["optimizer"]["args"]["parallel"] = True
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp, "invalid.yaml")
            cfg.write_text(yaml.safe_dump(config))
            result = subprocess.run([
                sys.executable, "MACVO.py", "--odom", str(cfg),
                "--data", "Config/Sequence/WindowICP_test_fixture.yaml",
                "--seq_to", "3", "--resultRoot", tmp, "--noeval",
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            meta = list(Path(tmp).glob("*/*/run_provenance.json"))
            self.assertEqual(len(meta), 1, result.stderr)
            self.assertEqual(json.loads(meta[0].read_text())["status"], "failed")
            saved_config = yaml.safe_load(
                Path(meta[0].parent, "config.yaml").read_text()
            )
            self.assertIn("Preprocess", saved_config)


if __name__ == "__main__":
    unittest.main()
