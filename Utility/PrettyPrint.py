import logging
import os
import sys
from typing import Any, Literal, Callable

from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from rich.table import Table
from rich.text import Text
from rich.live import Live
from tqdm import tqdm

IS_TERMINAL_MODE = False
TERMINAL_WIDTH = 150
try:
    TERMINAL_WIDTH = os.get_terminal_size().columns
    IS_TERMINAL_MODE = False
except OSError:
    pass

GlobalConsole = Console(width=TERMINAL_WIDTH)
# install(console=GlobalConsole)


#
def on_breakpoint():
    global IS_TERMINAL_MODE
    IS_TERMINAL_MODE = False
    import pdb; pdb.set_trace(header="Press <S> to step to the actual breakpoint location.")
sys.breakpointhook = on_breakpoint
#

def print_as_table(headers: list[str], rows: list[list[Any]], title=None, sort_rows: None | Callable[[list[Any],], Any]=None):
    def cvt2str(i) -> str:
        if isinstance(i, float): return str(round(i, 4))
        if i is None: return ""
        return str(i)
    
    if sort_rows is not None: rows.sort(key=sort_rows)
    
    table = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD)
    for hdr in headers:
        table.add_column(hdr, justify="left")
    for row in rows:
        table.add_row(*[cvt2str(i) for i in row])
    GlobalConsole.print(table)


class ColoredTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        if IS_TERMINAL_MODE:
            self.rich_container = Text()
            self.rich_displayer = Live(self.rich_container, console=GlobalConsole, auto_refresh=False, vertical_overflow="crop")
            self.rich_displayer.start()
        else:
            self.rich_displayer = None
        super().__init__(
            *args, colour="yellow", ascii=False, ncols=GlobalConsole.width - 10, **kwargs
        )

    def close(self, *args, **kwargs):
        if hasattr(self, "n") and self.n < self.total:
            self.colour = "red"
            self.desc = "❌" + self.desc
        else:
            self.colour = "#35aca4"
            self.desc = "✅" + self.desc 
        super().close(*args, **kwargs)
        
        if self.rich_displayer: self.rich_displayer.stop()
    
    def display(self, msg: str | None = None, pos: int | None = None) -> None:
        if IS_TERMINAL_MODE and self.rich_displayer:
            # Custom display logic
            if msg is None: msg = self.__str__()
            self.rich_container.plain = msg
            self.rich_displayer.refresh()
        else:
            if self.rich_displayer:
                self.rich_displayer.stop()
                self.rich_displayer = None
            super().display(msg, pos)
    
    def __call__(self, arg, *args, **kwds: Any):
        return super().__call__(arg, *args, **kwds) # type: ignore


class GlobalLog:
    LogLevel = Literal["info", "error", "warn", "fatal"]
    
    LOCK: "None | GlobalLog" = None
    Translate = {
        "info": logging.INFO,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "fatal": logging.FATAL,
    }
    
    def __new__(cls, *args, **kwargs) -> "GlobalLog":
        if GlobalLog.LOCK is not None: return GlobalLog.LOCK
        return super().__new__(cls)

    def __init__(self) -> None:
        if GlobalLog.LOCK: return
        GlobalLog.LOCK = self
        
        logging.getLogger("evo").setLevel(logging.CRITICAL)
        logging.getLogger("timm").setLevel(logging.CRITICAL)
        logging.getLogger("numexpr").setLevel(logging.CRITICAL)
        
        logging.basicConfig(
            level="INFO",
            format="PID %(process)d %(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=GlobalConsole)],
        )
        self.__logger = logging.getLogger("AirVIO")

    def write(self, level: LogLevel, msg: Any, marked: bool = False) -> None:
        lg_level = self.Translate[level]
        self.__logger.log(lg_level, msg, stacklevel=2, extra={"markup": marked})
    
    def show_exception(self) -> None:
        GlobalConsole.print_exception()

Logger = GlobalLog()

def save_as_csv(headers: list[str], rows: list[list[float]], filename: str, sort_rows: None | Callable[[list[Any],], Any]=None):
    if sort_rows: rows.sort(key=sort_rows)
    
    with open(filename, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join([(str(r) if r is not None else "") for r in row]) + "\n")
    Logger.write("info", "Save to " + filename)
