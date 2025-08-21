

# scripts/utils/io.py
from pathlib import Path
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def read_table(path: Path, *, engine: str | None = None) -> pd.DataFrame:
    """
    Read a table from Parquet or CSV.
    - path: absolute or relative Path (use config.DATAPATH / "file.parquet" in callers)
    - engine: optional pandas engine (e.g., "pyarrow")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(p, engine=engine or "pyarrow")
    if suffix in (".csv", ".txt"):
        return pd.read_csv(p)

    raise ValueError(f"Unsupported file extension for {p.name}")


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 500, transparent: bool = False) -> None:
    """
    Save a Matplotlib figure to the given path.
    - fig: Matplotlib Figure object
    - path: Path to file (absolute or relative)
    - dpi: resolution (default 500)
    - transparent: whether background is transparent
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, transparent=transparent, bbox_inches="tight")
    print(f"[Saved figure] {p.resolve()}")

def get_suffix(mean_subtraction):
    return 'meansub' if mean_subtraction else ''