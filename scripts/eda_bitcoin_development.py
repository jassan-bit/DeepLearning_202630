"""Audit only 2023–2024 minute prices from the referenced ML artifact.

Run with Python, numpy and matplotlib:
python scripts/eda_bitcoin_development.py /path/to/BTCUSDT.npy
The source SHA-256 identifies bytes, not independently verified exchange data.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    digest = hashlib.file_digest(args.source.open("rb"), "sha256").hexdigest()
    expected = "a18c60c8622442e7cb11d80f03d25600840548773fc136fc1d772ce27bd0a32d"
    if digest != expected:
        raise ValueError("Source differs from the referenced ML manifest")
    array = np.load(args.source, mmap_mode="r", allow_pickle=False)
    n = (365 + 366) * 1440
    if array.ndim != 1 or len(array) < n:
        raise ValueError("Unexpected minute grid")
    prices = np.asarray(array[:n])  # Do not inspect 2025 test values.
    valid = np.isfinite(prices) & (prices > 0)
    returns = np.full(n - 1, np.nan)
    pairs = valid[1:] & valid[:-1]
    returns[pairs] = np.log(prices[1:][pairs] / prices[:-1][pairs])
    usable = returns[np.isfinite(returns)]
    std = usable.std()
    centered = (usable - usable.mean()) / std
    report = {
        "source_sha256": digest,
        "period": "2023-01-01T00:00Z to 2025-01-01T00:00Z exclusive",
        "frequency": "1 minute", "symbol": "BTCUSDT",
        "grid_rows": n, "valid_prices": int(valid.sum()),
        "nonfinite_prices": int((~np.isfinite(prices)).sum()),
        "nonpositive_prices": int((np.isfinite(prices) & (prices <= 0)).sum()),
        "valid_adjacent_returns": len(usable),
        "return_mean": float(usable.mean()), "return_std_ddof0": float(std),
        "return_skew_moment": float(np.mean(centered**3)),
        "return_excess_kurtosis_moment": float(np.mean(centered**4) - 3),
        "return_quantiles": dict(zip(["0.01", "0.5", "0.99"], map(float, np.quantile(usable, [.01, .5, .99])))),
        "test_values_inspected": False,
        "target_eda_executed": False,
    }
    root = Path(__file__).resolve().parents[1]
    out = root / "results/eda_development"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    days = prices.reshape(-1, 1440)
    axes[0].plot(np.arange(len(days)), days[:, -1], linewidth=.8)
    axes[0].set(xlabel="Días desde 2023-01-01 UTC", ylabel="Último cierre de cada día (USDT)")
    axes[1].hist(usable, bins=150, log=True)
    axes[1].set(xlabel="Retorno logarítmico de un minuto", ylabel="Frecuencia (escala log)")
    fig.suptitle("BTCUSDT: desarrollo 2023–2024; test excluido")
    fig.tight_layout()
    image = root / "book/_static/eda_development.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image, dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
