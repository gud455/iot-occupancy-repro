"""Deterministic, offline reanalysis of UCI Occupancy Detection.

This is a new small baseline experiment, not a reproduction of every model in
Candanedo & Feldheim (2016). No test labels are used to choose thresholds,
preprocessing, priors, features, or hyperparameters. Requires only NumPy.
"""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
FILES = ["datatraining.txt", "datatest.txt", "datatest2.txt"]


def load(name):
    # Files include a row-name column that has no corresponding header field.
    with (ROOT / "data" / name).open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == ["date", *FEATURES, "Occupancy"], header
    assert all(len(row) == 8 for row in rows)
    dates = [row[1] for row in rows]
    x = np.array([[float(v) for v in row[2:7]] for row in rows])
    y = np.array([int(row[7]) for row in rows])
    assert np.isfinite(x).all()
    assert set(np.unique(y)) == {0, 1}
    assert dates == sorted(dates)
    return x, y, dates


def fit_light_rule(x, y):
    """Minimize training 0/1 loss for 'occupied iff Light > threshold'."""
    light = x[:, 2]
    distinct = np.unique(light)
    candidates = np.r_[-np.inf, (distinct[:-1] + distinct[1:]) / 2, np.inf]
    # Ties resolved using the first (lowest) threshold, deterministically.
    losses = np.array([np.count_nonzero((light > t) != y) for t in candidates])
    t = candidates[np.argmin(losses)]
    return float(t), int(losses.min())


def fit_lda(x, y, indices):
    """Two-class LDA with a pooled covariance and training-only scaling.

    A fixed 1e-6 diagonal term in standardized coordinates stabilizes the
    derived-humidity-ratio collinearity. This value is set in advance.
    Store equivalent raw-feature coefficients for transparent inference.
    """
    z = x[:, indices]
    center = z.mean(axis=0)
    scale = z.std(axis=0, ddof=0)
    assert (scale > 0).all()
    z = (z - center) / scale
    mu0, mu1 = z[y == 0].mean(axis=0), z[y == 1].mean(axis=0)
    residual = z - np.where(y[:, None] == 1, mu1, mu0)
    covariance = residual.T @ residual / (len(y) - 2)
    covariance += 1e-6 * np.eye(len(indices))
    w = np.linalg.solve(covariance, mu1 - mu0)
    b = -0.5 * (mu1 + mu0) @ w + np.log(y.mean() / (1 - y.mean()))
    # Check the raw-feature representation equals the standardized one.
    raw_w = w / scale
    raw_b = b - center @ raw_w
    assert np.allclose(z @ w + b, x[:, indices] @ raw_w + raw_b)
    return {"indices": indices, "weights": raw_w.tolist(), "bias": float(raw_b),
            "ridge": 1e-6, "n_coefficients": len(indices) + 1}


def predict_lda(model, x, dtype=np.float64):
    return ((x[:, model["indices"]].astype(dtype) @
             np.array(model["weights"], dtype=dtype) +
             dtype(model["bias"])) > 0).astype(int)


def metrics(y, pred):
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    n = len(y)
    assert tn + fp + fn + tp == n
    return {"n": n, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": (tp + tn) / n,
            "positive_F1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "occupied_false_negative_rate": fn / (tp + fn),
            "empty_false_positive_rate": fp / (tn + fp)}


def consecutive_counts(active, adjacent):
    counts = np.zeros(len(active), dtype=int)
    for i, is_active in enumerate(active):
        if is_active:
            counts[i] = 1 + (counts[i - 1] if adjacent[i] else 0)
    return counts


def vacancy_replay(y, pred, dates):
    """Replay a fixed confirmation rule on the original, open-loop readings."""
    times = [datetime.fromisoformat(value) for value in dates]
    adjacent = [False] + [
        45 <= (end - start).total_seconds() <= 75
        for start, end in zip(times, times[1:])
    ]
    vacancy_counts = consecutive_counts(pred == 0, adjacent)
    false_vacancy_counts = consecutive_counts((y == 1) & (pred == 0), adjacent)
    end = int(false_vacancy_counts.argmax())
    longest = int(false_vacancy_counts[end])
    return {
        "longest_raw_missed_run": longest,
        "longest_raw_missed_run_time": [dates[end - longest + 1], dates[end]] if longest else [],
        "confirmation": {
            str(required): {
                "occupied_rows_confirmed_vacant": int(np.sum((y == 1) & (vacancy_counts >= required))),
                "vacant_rows_confirmed_vacant": int(np.sum((y == 0) & (vacancy_counts >= required))),
            }
            for required in (1, 3)
        },
    }


def main():
    data = {f: load(f) for f in FILES}
    assert [len(data[f][1]) for f in FILES] == [8143, 2665, 9752]
    train_x, train_y, train_dates = data[FILES[0]]
    # Original test 1 predates training; test 2 follows training.
    assert data[FILES[1]][2][-1] < train_dates[0]
    assert train_dates[-1] < data[FILES[2]][2][0]
    date_sets = [set(data[f][2]) for f in FILES]
    assert not any(date_sets[i] & date_sets[j] for i in range(3) for j in range(i))
    threshold, errors = fit_light_rule(train_x, train_y)
    lda5 = fit_lda(train_x, train_y, [0, 1, 2, 3, 4])
    lda4 = fit_lda(train_x, train_y, [0, 1, 3, 4])
    report = {
        "protocol": "Original file split, no shuffle, no test tuning; date/id excluded",
        "features": FEATURES, "rule": {"Light_threshold_lux": threshold,
                                         "training_errors": errors},
        "models": {"LDA-5": lda5, "LDA-no-light": lda4},
        "data": {}, "results": {},
        "vacancy_replay": {
            "protocol": "One vs three consecutive vacant predictions; adjacent gap 45 to 75 seconds; reset at each file, gap or occupied prediction",
            "results": {},
        },
    }
    for f, (x, y, dates) in data.items():
        report["data"][f] = {
            "rows": len(y), "start": dates[0], "end": dates[-1],
            "positive_count": int(y.sum()), "positive_fraction": float(y.mean()),
            "sha256": hashlib.sha256((ROOT / "data" / f).read_bytes()).hexdigest(),
        }
    rows = []
    names = [("Majority", "恒判无人"), ("Light-rule", "照度阈值"),
             ("LDA-5", "五特征 LDA"), ("LDA-no-light", "去照度 LDA")]
    for f in FILES[1:]:
        x, y, dates = data[f]
        predictions = {"Majority": np.zeros(len(y), dtype=int),
                       "Light-rule": (x[:, 2] > threshold).astype(int),
                       "LDA-5": predict_lda(lda5, x),
                       "LDA-no-light": predict_lda(lda4, x)}
        report["results"][f] = {name: metrics(y, pred) for name, pred in predictions.items()}
        report["vacancy_replay"]["results"][f] = {
            name: vacancy_replay(y, pred, dates) for name, pred in predictions.items()
            if name != "Majority"
        }
        for name in ["LDA-5", "LDA-no-light"]:
            model = report["models"][name]
            report["results"][f][name]["float32_prediction_changes"] = int(
                np.sum(predict_lda(model, x, np.float32) != predictions[name]))
    for name, label in names:
        cells = [label]
        for f in FILES[1:]:
            m = report["results"][f][name]
            cells.extend([f'{100*m["accuracy"]:.2f}', f'{m["positive_F1"]:.3f}',
                          f'{100*m["occupied_false_negative_rate"]:.2f}'])
        rows.append(" & ".join(cells) + r" \\")
    (ROOT / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "results-table.tex").write_text(
        "\\newcommand{\\ResultRows}{%\n" + "\n".join(rows) + "\n}\n")
    replay_rows = []
    for name, label in names[1:]:
        result = report["vacancy_replay"]["results"][FILES[2]][name]
        cells = [label, str(result["longest_raw_missed_run"])]
        for field in ("occupied_rows_confirmed_vacant", "vacant_rows_confirmed_vacant"):
            before = result["confirmation"]["1"][field]
            after = result["confirmation"]["3"][field]
            cells.append(f"${before}\\to{after}$")
        replay_rows.append(" & ".join(cells) + r" \\")
    (ROOT / "replay-table.tex").write_text(
        "\\newcommand{\\ReplayRows}{%\n" + "\n".join(replay_rows) + "\n}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
