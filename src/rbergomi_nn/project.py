"""Locations of the project's artefacts and shared plotting helpers for the scripts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the scripts only write files
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "surfaces.npz"
MODEL_PATH = ROOT / "models" / "vol_surface_mlp.pt"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def save_figure(figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"saved {path.relative_to(ROOT)}")
    return path


def save_report(content: dict, name: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(content, indent=2))
    print(f"saved {path.relative_to(ROOT)}")
    return path
