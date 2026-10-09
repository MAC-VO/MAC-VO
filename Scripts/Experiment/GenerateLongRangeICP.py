"""Generate sparse gap-5 and gap-10 ICP factors from a saved WindowMACVO run."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from DataLoader import SequenceBase, StereoFrame, smart_transform
from Module.Optimization.FactorArchive import (
    load_factor_archive,
    save_factor_archive,
)
from Module.Optimization.LongRangeICP import generate_long_range_archive
from Odometry.WindowMACVO import WindowMACVO
from Utility.Sandbox import Sandbox


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=Path, required=True)
    parser.add_argument(
        "--output", default="long_factors_gap5_10.npz"
    )
    return parser


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_index(value):
    return None if isinstance(value, SimpleNamespace) else value


def main(argv=None):
    args = build_parser().parse_args(argv)
    space = Sandbox.load(args.space)
    cfg = space.config
    if not hasattr(cfg, "Preprocess"):
        raise ValueError(
            "Saved result config does not contain Preprocess; rerun online odometry"
        )
    source_path = space.path("global_factors.npz")
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source factor archive: {source_path}")
    source = load_factor_archive(source_path)
    sequence = smart_transform(
        SequenceBase[StereoFrame].instantiate(
            cfg.Data.args.type, cfg.Data.args.args
        ).clip(
            optional_index(cfg.Data.start_idx),
            optional_index(cfg.Data.end_idx),
        ),
        cfg.Preprocess,
    )
    if len(sequence) != len(source.initial_sensor_poses):
        raise ValueError(
            "Saved sequence range does not match global_factors.npz"
        )

    system = WindowMACVO.from_config(cfg)
    result = generate_long_range_archive(
        sequence,
        source,
        frontend=system.Frontend,
        selector=system.KeypointSelector,
        covariance_model=system.ObsCovModel,
        num_point=system.num_point,
        min_num_point=system.min_num_point,
        edge_width=system.edge_width,
        match_cov_default=system.match_cov_default,
        device=system.device,
        metadata={
            "source_space": str(space.folder.resolve()),
            "source_factor_sha256": file_sha256(source_path),
        },
    )
    output_path = space.path(args.output)
    save_factor_archive(output_path, result.archive)
    diagnostics_path = output_path.with_suffix(".json")
    diagnostics_path.write_text(
        json.dumps(
            result.diagnostics | {
                "archive": output_path.name,
                "archive_sha256": file_sha256(output_path),
                "source_factor_sha256": file_sha256(source_path),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(output_path)
    return 0 if result.diagnostics["status"] == "generated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
