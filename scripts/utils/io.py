
"""
I/O utility functions for reading/writing tables, saving figures, 
and configuring logging.

Functions
---------
- read_table      : Load CSV/Parquet into DataFrame.
- setup_logging   : Configure root logger for consistent console output.
- save_figure     : Save Matplotlib figures with timestamp/transparent options.
- get_suffix      : Return suffix string based on mean_subtraction flag.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging, sys


def read_table(path, *, engine=None):
    """
    Read a table from Parquet (.parquet/.pq/.pqt) or CSV (.csv/.txt).

    Parameters
    ----------
    path : Path or str
        Path to file (absolute or relative).
    engine : str, optional
        Parquet engine (default "pyarrow").

    Returns
    -------
    pd.DataFrame
        Loaded table.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    if suffix in (".parquet", ".pq", ".pqt"):
        return pd.read_parquet(p, engine=engine or "pyarrow")
    if suffix in (".csv", ".txt"):
        return pd.read_csv(p)

    raise ValueError(f"Unsupported file extension for {p.name}")

# def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 500, transparent: bool = False) -> None:
#     """
#     Save a Matplotlib figure to the given path.
#     - fig: Matplotlib Figure object
#     - path: Path to file (absolute or relative)
#     - dpi: resolution (default 500)
#     - transparent: whether background is transparent
#     """
#     p = Path(path)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(p, dpi=dpi, transparent=transparent, bbox_inches="tight")
#     print(f"[Saved figure] {p.resolve()}")

def setup_logging(level=logging.INFO):
    """
    Configure root logger for console output.

    - Adds a StreamHandler to stdout.
    - Applies uniform time-stamped formatter.
    - Suppresses noisy logs from matplotlib, seaborn, fontTools.

    Notes
    -----
    - Idempotent: running multiple times won't duplicate handlers.
    - Call this once at the top of a script (before other loggers).
    
    Parameters
    ----------
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger: 
        Configured root logger.
    """
    root = logging.getLogger()
    
    if getattr(root, "_configured_by_setup_logging", False):
        return root

    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("seaborn").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING) 

    root._configured_by_setup_logging = True
    return root


def save_figure(fig, path, *, dpi=500,
                transparent=False, add_timestamp=False):
    """
    Save a Matplotlib figure to path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    path : Path or str
        Destination file path.
    dpi : int, optional
        Resolution (default 500).
    transparent : bool, optional
        Save with transparent background (default False).
    add_timestamp : bool, optional
        Append timestamp to filename to avoid overwriting (default False).

    Notes
    -----
    - Creates parent directory if it does not exist.
    - Logs the save location if logging is configured; else prints.
    """
    p = Path(path)
    log = logging.getLogger(__name__)
    setup_logging()  

    if add_timestamp:
        stem, suffix = p.stem, p.suffix or ".png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = p.with_name(f"{stem}_{timestamp}{suffix}")

    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, transparent=transparent, bbox_inches="tight")

    # Prefer logging if configured, fallback to print
    msg = f"[Saved figure] {p.resolve()}"
    if log.handlers:
        log.info(msg)
    else:
        print(msg)


def get_suffix(mean_subtraction):
    """
    Return suffix string for filenames.

    Parameters
    ----------
    mean_subtraction : bool

    Returns
    -------
    str
        'meansub' if True, else ''.
    """
    return 'meansub' if mean_subtraction else ''
