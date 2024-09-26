import json
import os
import types
from pathlib import Path

import yaml
from typing import Any
from typing_extensions import NamedTuple
from types import SimpleNamespace
from yacs.config import CfgNode as CN

from Utility.PrettyPrint import Logger


class LoadFrom(NamedTuple):
    path: Path


class IncludeLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]  #type: ignore[reportAttributeAccessIssue]
        super(IncludeLoader, self).__init__(stream)

    def flatten_sequence(self, node):
        arr = self.construct_sequence(node)
        acc_arr = []
        for subarr in arr:
            if isinstance(subarr, list):
                acc_arr.extend(subarr)
            else:
                acc_arr.append(subarr)
        return acc_arr

    def include(self, node):
        filename = os.path.join(self._root, str(self.construct_scalar(node)))

        if not Path(filename).exists():
            raise ValueError(
                f"Tried to include file {filename} to *.yaml, but cannot find it."
            )

        with open(filename, "r") as f:
            return yaml.load(f, IncludeLoader)

    def include_dataset(self, node):
        # Basically, this inflates a compressed representation of dataset into full form of dataset config.
        filename = os.path.join(self._root, str(self.construct_scalar(node)))
        if not Path(filename).exists():
            raise ValueError(
                f"Tried to include file {filename} to *.yaml, but cannot find it."
            )

        with open(filename, "r") as f:
            assert (
                "#*IS_DATASET_TYPE*#" in f.readline()
            ), "!include_dataset can only be applied to *.yaml file with special header #*IS_DATASET_TYPE*#"

        with open(filename, "r") as f:
            compressed_datacfg = yaml.load(f, IncludeLoader)

        data_root = compressed_datacfg["dataroot"]
        shared_data = {
            k: compressed_datacfg[k]
            for k in compressed_datacfg.keys()
            if k not in {"roots", "begin_idx", "end_idx"}
        }
        inflate_data = [
            shared_data | {"root": data_root + cfg["root"], "name": cfg["name"]}
            for cfg in compressed_datacfg["roots"]
        ]

        return [
            {
                "begin_idx": compressed_datacfg["begin_idx"],
                "end_idx": compressed_datacfg["end_idx"],
                "args": args,
            }
            for args in inflate_data
        ]


IncludeLoader.add_constructor("!include", IncludeLoader.include)
IncludeLoader.add_constructor("!include_dataset", IncludeLoader.include_dataset)
IncludeLoader.add_constructor("!flatten_seq", IncludeLoader.flatten_sequence)


def __build_dynamic_config(spec: dict[str, Any] | list | LoadFrom):
    if isinstance(spec, LoadFrom):
        if not spec.path.exists():
            Logger.write(
                "fatal", f"Unable to build dynamic config: {spec.path} does not exists!"
            )
        assert spec.path.exists()

        with open(spec.path, "r") as f:
            data = yaml.load(f, IncludeLoader)
        return data

    if isinstance(spec, dict):
        for key, value in spec.items():
            spec[key] = __build_dynamic_config(value)
    elif isinstance(spec, list):
        for idx in range(len(spec)):
            spec[idx] = __build_dynamic_config(spec[idx])
    return spec


def build_dynamic_config(spec: dict[str, Any] | list | LoadFrom) -> tuple[types.SimpleNamespace, dict[str, Any]]:
    cfg = __build_dynamic_config(spec)
    return asNamespace(cfg), cfg    #type: ignore


def load_config(path: Path):
    assert path.exists()
    with open(path, "r") as f:
        data = yaml.load(f, IncludeLoader)
    return asNamespace(data), data


def namespace_to_cfgnode(ns: SimpleNamespace):
    """
    Design for Flowformer. Flowformer uses yacs.config.CfgNode as config container.
    """
    cfg = CN()
    for key, value in vars(ns).items():
        if isinstance(value, SimpleNamespace):
            cfg[key] = namespace_to_cfgnode(value)
        else:
            cfg[key] = value
    return cfg


def dump_config(path: Path, cfg: dict):
    assert path.parent.exists()
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def asNamespace(dictionary) -> types.SimpleNamespace:
    def load_object(obj):
        if isinstance(obj, dict):
            for k in obj:
                if obj[k] is None:
                    obj[k] = types.SimpleNamespace()
            return types.SimpleNamespace(**obj)
        return obj
    return json.loads(json.dumps(dictionary), object_hook=load_object)
