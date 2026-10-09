import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build/orb_bow/macvo_orb_bow"


def tiny_vocabulary(path):
    zero = " ".join(["0"] * 32)
    full = " ".join(["255"] * 32)
    path.write_text(
        f"2 1 0 0\n0 1 {zero} 1.0\n0 1 {full} 1.0",
        encoding="utf-8",
    )


def parse_candidate(field):
    source, score, raw_count, ratio_count, mutual_count, payload = field.split(
        ":", 5
    )
    matches = [] if not payload else [
        [float(value) for value in item.split(",")]
        for item in payload.split(";")
    ]
    return {
        "source": int(source),
        "score": float(score),
        "raw_count": int(raw_count),
        "ratio_count": int(ratio_count),
        "mutual_count": int(mutual_count),
        "matches": np.asarray(matches, dtype=np.float64).reshape(-1, 5),
    }


class ORBBoWSidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env["ORB_SLAM3_ROOT"] = "/home/zzh/ORB_SLAM3"
        subprocess.run(
            [str(ROOT / "Scripts/Build/build_orb_bow_sidecar.sh")],
            cwd=ROOT,
            env=env,
            check=True,
        )

    def test_handshake_query_temporal_filter_and_malformed_command(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            vocabulary = directory / "tiny_vocabulary.txt"
            tiny_vocabulary(vocabulary)
            rng = np.random.default_rng(9)
            image = (rng.random((128, 128)) * 255).astype(np.uint8)
            image_path = directory / "image.png"
            cv2.imwrite(str(image_path), image)
            process = subprocess.Popen(
                [str(BUILD), "--vocabulary", str(vocabulary)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert process.stdin is not None and process.stdout is not None
            ready = process.stdout.readline().strip().split("\t")
            self.assertEqual(ready[0], "READY")
            self.assertEqual(ready[1], "2")

            process.stdin.write(f"QUERY\t1\t0\t{image_path}\t30\t3\t0.8\t50\n")
            process.stdin.flush()
            first = process.stdout.readline().strip().split("\t")
            self.assertEqual(first[:4], ["RESULT", "1", "0", "0"])

            process.stdin.write(f"QUERY\t2\t40\t{image_path}\t30\t3\t0.8\t50\n")
            process.stdin.flush()
            second = process.stdout.readline().strip().split("\t")
            self.assertEqual(second[:4], ["RESULT", "2", "40", "1"])
            candidate = parse_candidate(second[4])
            self.assertEqual(candidate["source"], 0)
            self.assertGreater(candidate["score"], 0)
            self.assertGreater(candidate["mutual_count"], 0)

            process.stdin.write("BAD\n")
            process.stdin.flush()
            self.assertTrue(process.stdout.readline().startswith("ERROR\t"))

            process.stdin.write("STOP\n")
            process.stdin.flush()
            self.assertEqual(process.stdout.readline().strip(), "BYE")
            self.assertEqual(process.wait(timeout=5), 0)
            process.stdin.close()
            process.stdout.close()
            assert process.stderr is not None
            process.stderr.close()

    def test_returns_bounded_mutual_matches_for_translated_image(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            vocabulary = directory / "tiny_vocabulary.txt"
            tiny_vocabulary(vocabulary)
            rng = np.random.default_rng(17)
            source = (rng.random((240, 320)) * 255).astype(np.uint8)
            target = cv2.warpAffine(
                source,
                np.asarray([[1, 0, 5], [0, 1, 3]], dtype=np.float32),
                (source.shape[1], source.shape[0]),
            )
            source_path = directory / "source.png"
            target_path = directory / "target.png"
            cv2.imwrite(str(source_path), source)
            cv2.imwrite(str(target_path), target)
            process = subprocess.Popen(
                [str(BUILD), "--vocabulary", str(vocabulary)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert process.stdin is not None and process.stdout is not None
            self.assertEqual(process.stdout.readline().split("\t")[1], "2")
            process.stdin.write(
                f"QUERY\t1\t0\t{source_path}\t30\t3\t0.8\t50\n"
            )
            process.stdin.flush()
            self.assertEqual(
                process.stdout.readline().strip().split("\t")[:4],
                ["RESULT", "1", "0", "0"],
            )
            process.stdin.write(
                f"QUERY\t2\t40\t{target_path}\t30\t3\t0.8\t50\n"
            )
            process.stdin.flush()
            response = process.stdout.readline().strip().split("\t")
            candidate = parse_candidate(response[4])
            process.stdin.write("STOP\n")
            process.stdin.flush()
            self.assertEqual(process.stdout.readline().strip(), "BYE")
            self.assertEqual(process.wait(timeout=5), 0)
            process.stdin.close()
            process.stdout.close()
            assert process.stderr is not None
            process.stderr.close()

        self.assertGreaterEqual(candidate["mutual_count"], 20)
        self.assertLessEqual(candidate["mutual_count"], 50)
        displacement = candidate["matches"][:, 2:4] - candidate["matches"][:, :2]
        median = np.median(displacement, axis=0)
        self.assertAlmostEqual(float(median[0]), 5.0, delta=1.5)
        self.assertAlmostEqual(float(median[1]), 3.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
