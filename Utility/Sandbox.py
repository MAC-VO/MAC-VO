import atexit
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import yaml

from Utility.Config import load_config
from Utility.PrettyPrint import Logger


class YAMLFileProxy:
    def __init__(self, file_name: Path):
        self.file_path = file_name
        
        # Transparent Config Cache
        if not self.file_path.exists():
            self.__file_cache = SimpleNamespace()
        else:
            self.__file_cache, _ = load_config(self.file_path)
    
    @property
    def data(self) -> SimpleNamespace:
        return self.__file_cache

    @data.setter
    def data(self, value):
        self.__file_cache = value
        with open(self.file_path, "w") as file:
            yaml.safe_dump(self.__file_cache, file)


class SandboxFile:
    def __init__(self, root_path: Path, name: str, mode: str):
        self.file = Path(root_path, name)
        self.mode = mode
        self.fp = None

    def __enter__(self):
        self.fp = open(self.file, self.mode)
        return self.fp

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fp is None:
            return
    
        self.fp.close()
        if exc_type is not None:
            Logger.write("warn", f"File {self.file} with mode {self.mode} fails.",)
        return True


class Sandbox:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        if not self.folder.exists():
            self.folder.mkdir(parents=True)

        # Transparent Config Cache
        self.config_proxy = YAMLFileProxy(Path(self.folder, "config.yaml"))

    @classmethod
    def create(cls, project_root: Path, project_name: str):
        timestr = cls.__get_curr_time()
        try: gitver = cls.__get_git_version()
        except: gitver = "NOT_AVAILABLE"
        cmd = cls.__get_sys_command()

        box = cls(Path(project_root, project_name, timestr))
        with box.open("metadata.yaml", "w") as f:
            yaml.dump({"time": timestr, "git_version": gitver, "command": cmd}, f)
        return box

    @classmethod
    def load(cls, root: Path | str):
        if isinstance(root, str):
            root_path = Path(root)
        else:
            root_path = root
        if not root_path.exists():
            raise FileNotFoundError(f"Unable to load sandbox from {root_path}")
        return Sandbox(root_path)

    def path(self, name: str | Path) -> Path:
        target_path = Path(self.folder, name).parent
        if not target_path.exists():
            target_path.mkdir(parents=True)
        return Path(self.folder, name)

    def open(self, name: str, mode: str) -> SandboxFile:
        target_path = self.path(name)
        if not target_path.parent.exists():
            target_path.parent.mkdir(parents=True)
        return SandboxFile(self.folder, name, mode)

    def path_folder(self, name: str) -> Path:
        target_path = self.path(Path(self.folder, name))
        if not target_path.exists():
            target_path.mkdir(parents=True)
        return target_path

    def new_child(self, name: str) -> "Sandbox":
        subbox = Sandbox.create(self.folder, name)
        with self.open("children.txt", "a") as f:
            f.write(str(subbox.folder.relative_to(self.folder).as_posix()) + "\n")
        return subbox

    def get_children(self) -> list["Sandbox"]:
        if not Path(self.folder, "children.txt").exists():
            return []

        with self.open("children.txt", "r") as f:
            lines = f.read().strip().split("\n")
        return [Sandbox.load(Path(self.folder, Path(subbox_path))) for subbox_path in lines]

    def get_leaves(self) -> list["Sandbox"]:
        children = self.get_children()
        if len(children) == 0:
            return [self]
        result = []
        for child in children:
            result.extend(child.get_leaves())
        return result

    def set_autoremove(self):
        Logger.write(
            "warn", f"Sandbox at '{str(self.folder)}' is set to be auto-removed."
        )

        def autophagy():
            try:
                shutil.rmtree(str(self.folder))
                
                Logger.write(
                    "warn", f"[bold red]Sandbox at {str(self.folder)} is auto-removed.[/]", marked=True
                )
            except Exception as _:
                Logger.write(
                    "warn", f"Failed to auto-remove sandbox at {str(self.folder)}"
                )

        atexit.register(autophagy)

    @staticmethod
    def __get_sys_command():
        return " ".join(sys.orig_argv)

    @staticmethod
    def __get_curr_time():
        time_str = datetime.now().strftime("%m_%d_%H%M%S")
        return time_str

    @staticmethod
    def __get_git_version():
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )

    @property
    def config(self) -> SimpleNamespace:
        return self.config_proxy.data

    @config.setter
    def config(self, value):
        self.config_proxy.data = value
