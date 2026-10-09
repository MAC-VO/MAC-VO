"""Asynchronous adapter for the persistent ORB-BoW candidate sidecar."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import queue
import subprocess
import threading

import numpy as np


@dataclass(frozen=True)
class ORBLoopCandidate:
    source: int
    score: float
    rank: int
    raw_knn_matches: int = 0
    ratio_matches: int = 0
    matches: np.ndarray = field(
        default_factory=lambda: np.empty((0, 5), dtype=np.float64)
    )

    @property
    def mutual_matches(self):
        return len(self.matches)


@dataclass(frozen=True)
class ORBLoopCandidateBatch:
    request_id: int
    target: int
    candidates: tuple[ORBLoopCandidate, ...]


def parse_candidate_field(value, *, rank):
    fields = value.split(":", 5)
    if len(fields) != 6:
        raise ValueError("malformed ORB candidate field")
    source = int(fields[0])
    score = float(fields[1])
    raw_knn_matches = int(fields[2])
    ratio_matches = int(fields[3])
    mutual_matches = int(fields[4])
    if source < 0 or not math.isfinite(score):
        raise ValueError("ORB candidate metadata must be finite and nonnegative")
    if min(raw_knn_matches, ratio_matches, mutual_matches) < 0:
        raise ValueError("ORB match counts must be nonnegative")
    matches = []
    if fields[5]:
        for encoded in fields[5].split(";"):
            match = [float(component) for component in encoded.split(",")]
            if len(match) != 5:
                raise ValueError("malformed ORB match")
            if not all(math.isfinite(component) for component in match):
                raise ValueError("ORB match values must be finite")
            if match[4] < 0:
                raise ValueError("ORB descriptor distance must be nonnegative")
            matches.append(match)
    if len(matches) != mutual_matches:
        raise ValueError("ORB match count does not match payload")
    array = np.asarray(matches, dtype=np.float64).reshape(-1, 5)
    array.setflags(write=False)
    return ORBLoopCandidate(
        source=source,
        score=score,
        rank=rank,
        raw_knn_matches=raw_knn_matches,
        ratio_matches=ratio_matches,
        matches=array,
    )


def filter_loop_candidates(
    *,
    target,
    candidates,
    existing_pairs,
    pending_pairs,
    rejected_pairs,
    nms_frames,
    maximum,
):
    selected = []
    reasons = {"existing": 0, "pending": 0, "nms": 0}
    for rank, item in enumerate(candidates):
        candidate = (
            item if isinstance(item, ORBLoopCandidate)
            else ORBLoopCandidate(item[0], float(item[1]), rank)
        )
        source, score = candidate.source, candidate.score
        pair = (source, target)
        if pair in existing_pairs:
            reasons["existing"] += 1
            continue
        if pair in pending_pairs:
            reasons["pending"] += 1
            continue
        if any(
            abs(source - old_source) <= nms_frames
            and abs(target - old_target) <= nms_frames
            for old_source, old_target in rejected_pairs | existing_pairs
        ):
            reasons["nms"] += 1
            continue
        selected.append(candidate)
        if len(selected) >= maximum:
            break
    return selected, reasons


class ORBLoopCandidateProvider:
    def __init__(
        self,
        *,
        executable,
        vocabulary,
        min_temporal_gap,
        top_k,
        ratio_test=0.80,
        max_matches=300,
        queue_size=8,
    ):
        self.executable = Path(executable)
        self.vocabulary = Path(vocabulary)
        self.min_temporal_gap = min_temporal_gap
        self.top_k = top_k
        self.ratio_test = float(ratio_test)
        self.max_matches = int(max_matches)
        if not 0 < self.ratio_test < 1:
            raise ValueError("ORB ratio test must be within (0, 1)")
        if self.max_matches < 1:
            raise ValueError("ORB maximum matches must be positive")
        self._responses = queue.Queue(maxsize=queue_size)
        self._write_lock = threading.Lock()
        self._next_request = 0
        self._pending = {}
        self._closed = False
        self.process = None
        self.reader = None
        self.handshake = {}
        self.status = {"status": "disabled", "reason": "not started"}
        if not self.executable.is_file() or not self.vocabulary.is_file():
            self.status = {
                "status": "disabled",
                "reason": "missing sidecar executable or vocabulary",
            }
            return
        try:
            self.process = subprocess.Popen(
                [
                    str(self.executable),
                    "--vocabulary",
                    str(self.vocabulary),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert self.process.stdout is not None
            fields = self.process.stdout.readline().strip().split("\t")
            if len(fields) != 4 or fields[0] != "READY" or fields[1] != "2":
                raise RuntimeError("invalid ORB sidecar handshake")
            self.handshake = {
                "protocol_version": int(fields[1]),
                "vocabulary_sha256": fields[2],
                "opencv_version": fields[3],
            }
            self.status = {"status": "ready"}
            self.reader = threading.Thread(
                target=self._read_responses,
                name="ORBBoWReader",
                daemon=True,
            )
            self.reader.start()
        except Exception as error:
            self.status = {"status": "disabled", "reason": str(error)}
            if self.process is not None:
                self.process.terminate()
                self.process.wait(timeout=5)
                self._close_process_streams()
            self.process = None

    def _close_process_streams(self):
        if self.process is None:
            return
        for stream in (
            self.process.stdin, self.process.stdout, self.process.stderr
        ):
            if stream is not None and not stream.closed:
                stream.close()

    @property
    def enabled(self):
        return (
            self.status.get("status") == "ready"
            and self.process is not None
            and self.process.poll() is None
        )

    def _read_responses(self):
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            fields = line.strip().split("\t")
            if not fields or fields[0] == "BYE":
                break
            if fields[0] == "ERROR":
                self.status = {
                    "status": "error",
                    "reason": "\t".join(fields[1:]),
                }
                continue
            try:
                if len(fields) < 4 or fields[0] != "RESULT":
                    raise ValueError("malformed sidecar response")
                request_id = int(fields[1])
                target = int(fields[2])
                count = int(fields[3])
                if request_id not in self._pending:
                    raise ValueError("unknown or duplicate request ID")
                expected_target = self._pending.pop(request_id)
                if target != expected_target or len(fields[4:]) != count:
                    raise ValueError("sidecar result metadata mismatch")
                candidates = []
                for rank, field in enumerate(fields[4:]):
                    candidates.append(parse_candidate_field(field, rank=rank))
                response = ORBLoopCandidateBatch(
                    request_id, target, tuple(candidates)
                )
                try:
                    self._responses.put_nowait(response)
                except queue.Full:
                    self.status = {
                        "status": "degraded",
                        "reason": "candidate response queue full",
                    }
            except Exception as error:
                self.status = {"status": "error", "reason": str(error)}
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                break
        if not self._closed and self.status.get("status") == "ready":
            self.status = {
                "status": "disabled",
                "reason": "ORB sidecar exited",
            }

    def submit(self, *, frame_id, left_image_path):
        if not self.enabled:
            raise RuntimeError("ORB candidate provider is disabled")
        request_id = self._next_request
        self._next_request += 1
        self._pending[request_id] = frame_id
        assert self.process is not None and self.process.stdin is not None
        command = (
            f"QUERY\t{request_id}\t{frame_id}\t{Path(left_image_path)}\t"
            f"{self.min_temporal_gap}\t{self.top_k}\t{self.ratio_test}\t"
            f"{self.max_matches}\n"
        )
        with self._write_lock:
            self.process.stdin.write(command)
            self.process.stdin.flush()
        return request_id

    def poll(self):
        responses = []
        while True:
            try:
                responses.append(self._responses.get_nowait())
            except queue.Empty:
                return responses

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.enabled:
            assert self.process is not None and self.process.stdin is not None
            with self._write_lock:
                self.process.stdin.write("STOP\n")
                self.process.stdin.flush()
            self.process.wait(timeout=5)
        elif self.process is not None and self.process.poll() is None:
            self.process.terminate()
        if self.reader is not None:
            self.reader.join(timeout=5)
        self._close_process_streams()
