import os
from pathlib import Path
import stat
import tempfile
import textwrap
import time
import unittest

from Module.LoopClosure.ORBBoW import (
    ORBLoopCandidateProvider,
    filter_loop_candidates,
    parse_candidate_field,
)


def fake_sidecar(path, version=2, malformed=False):
    response = (
        "RESULT\\t{request}\\t{target}\\t1\\tbad"
        if malformed else
        "RESULT\\t{request}\\t{target}\\t2"
        "\\t0:0.9:120:80:2:10,20,12,22,4;30,40,31,41,8"
        "\\t5:0.8:90:50:1:4,5,6,7,12"
    )
    source = textwrap.dedent("""\
        #!/usr/bin/env python3
        import sys
        print('READY\\tVERSION\\tfixture\\tfake', flush=True)
        for line in sys.stdin:
            fields=line.rstrip('\\n').split('\\t')
            if fields[0]=='STOP':
                print('BYE', flush=True)
                break
            if fields[0]=='QUERY':
                request,target=fields[1],fields[2]
                print(f'RESPONSE', flush=True)
            else:
                print('ERROR\\tbad command', flush=True)
    """).replace("VERSION", str(version)).replace("RESPONSE", response)
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class ORBBoWProviderTests(unittest.TestCase):
    def test_provider_validates_handshake_and_returns_async_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            executable = directory / "sidecar.py"
            vocabulary = directory / "vocab.txt"
            image = directory / "left.png"
            fake_sidecar(executable)
            vocabulary.write_text("fixture")
            image.write_bytes(b"fixture")
            provider = ORBLoopCandidateProvider(
                executable=executable,
                vocabulary=vocabulary,
                min_temporal_gap=30,
                top_k=3,
            )
            request = provider.submit(frame_id=40, left_image_path=image)
            deadline = time.time() + 2
            result = None
            while result is None and time.time() < deadline:
                responses = provider.poll()
                result = responses[0] if responses else None
                time.sleep(0.01)
            provider.close()

        self.assertTrue(provider.handshake["protocol_version"] == 2)
        self.assertEqual(result.request_id, request)
        self.assertEqual(result.target, 40)
        self.assertEqual([item.source for item in result.candidates], [0, 5])
        candidate = result.candidates[0]
        self.assertEqual(candidate.raw_knn_matches, 120)
        self.assertEqual(candidate.ratio_matches, 80)
        self.assertEqual(candidate.mutual_matches, 2)
        self.assertEqual(candidate.matches.shape, (2, 5))
        self.assertEqual(
            candidate.matches[0].tolist(), [10.0, 20.0, 12.0, 22.0, 4.0]
        )

    def test_candidate_parser_rejects_declared_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "match count"):
            parse_candidate_field("0:0.9:120:80:2:10,20,12,22,4", rank=0)

    def test_candidate_parser_rejects_nonfinite_coordinates(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            parse_candidate_field("0:0.9:120:80:1:nan,20,12,22,4", rank=0)

    def test_candidate_parser_rejects_negative_distance(self):
        with self.assertRaisesRegex(ValueError, "distance"):
            parse_candidate_field("0:0.9:120:80:1:10,20,12,22,-1", rank=0)

    def test_sidecar_start_failure_disables_provider(self):
        provider = ORBLoopCandidateProvider(
            executable=Path("/missing/sidecar"),
            vocabulary=Path("/missing/vocab"),
            min_temporal_gap=30,
            top_k=3,
        )

        self.assertFalse(provider.enabled)
        self.assertIn("missing", provider.status["reason"])

    def test_protocol_v1_handshake_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            executable = directory / "sidecar.py"
            vocabulary = directory / "vocab.txt"
            fake_sidecar(executable, version=1)
            vocabulary.write_text("fixture")

            provider = ORBLoopCandidateProvider(
                executable=executable,
                vocabulary=vocabulary,
                min_temporal_gap=30,
                top_k=3,
            )

            self.assertFalse(provider.enabled)
            self.assertIn("handshake", provider.status["reason"])

    def test_malformed_response_disables_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            executable = directory / "sidecar.py"
            vocabulary = directory / "vocab.txt"
            image = directory / "left.png"
            fake_sidecar(executable, malformed=True)
            vocabulary.write_text("fixture")
            image.write_bytes(b"fixture")
            provider = ORBLoopCandidateProvider(
                executable=executable,
                vocabulary=vocabulary,
                min_temporal_gap=30,
                top_k=3,
            )

            provider.submit(frame_id=40, left_image_path=image)
            deadline = time.time() + 2
            while provider.status.get("status") != "error" and time.time() < deadline:
                time.sleep(0.01)

            self.assertEqual(provider.status["status"], "error")
            self.assertFalse(provider.enabled)
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                provider.submit(frame_id=45, left_image_path=image)
            provider.close()

    def test_filter_removes_existing_pending_and_nms_near_pairs(self):
        candidates = [(0, 0.9), (5, 0.8), (100, 0.7)]

        selected, reasons = filter_loop_candidates(
            target=200,
            candidates=candidates,
            existing_pairs={(0, 200)},
            pending_pairs={(5, 200)},
            rejected_pairs={(95, 195)},
            nms_frames=10,
            maximum=1,
        )

        self.assertEqual(selected, [])
        self.assertEqual(reasons["existing"], 1)
        self.assertEqual(reasons["pending"], 1)
        self.assertEqual(reasons["nms"] , 1)


if __name__ == "__main__":
    unittest.main()
