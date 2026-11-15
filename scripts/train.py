"""Train the SNN from a labeled CSV of gesture-level feature rows.

Expected columns:
tap_interval_ms,mean_speed_px_s,mean_abs_accel_px_s2,path_deviation_px,
pressure_mean,label

Labels must be integer class IDs 0..2. Only use consented, de-identified
research data collected under an approved protocol.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from app.features import FEATURE_NAMES, normalize_feature_vector
from app.snn_model import TouchSNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out", default="artifacts/touch_snn.pt")
    args = parser.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=FEATURE_NAMES + ["label"])
    if len(df) < 30:
        raise SystemExit("Need at least 30 labeled gestures for a basic training run; more is strongly recommended.")
    labels = df["label"].astype(int).to_numpy()
    if not set(np.unique(labels)).issubset({0, 1, 2}):
        raise SystemExit("Labels must be integer class IDs 0, 1, or 2.")

    x = np.stack([normalize_feature_vector(row.to_dict()) for _, row in df.iterrows()])
    # Repeat gesture-level features across time for an initial rate-coded baseline.
    steps = 20
    thresholds = np.linspace(0.15, 0.85, steps, dtype=np.float32)
    seq = (x[:, None, :] >= thresholds[None, :, None]).astype(np.float32)

    # Split by participant/session before training for real studies. This simple
    # random split is only a smoke test and can overestimate generalization.
    rng = np.random.default_rng(7)
    order = rng.permutation(len(x))
    cut = max(1, int(0.8 * len(order)))
    train_idx, val_idx = order[:cut], order[cut:]
    X = torch.tensor(seq, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    model = TouchSNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        spikes, mem = model(X[train_idx])
        # Membrane potential is used as a readout for this baseline.
        logits = mem
        loss = loss_fn(logits, y[train_idx])
        loss.backward()
        optimizer.step()
        print(f"epoch={epoch+1} loss={loss.item():.4f}")

    model.eval()
    if len(val_idx):
        with torch.no_grad():
            _, val_mem = model(X[val_idx])
            acc = (val_mem.argmax(1) == y[val_idx]).float().mean().item()
        print(f"random_split_validation_accuracy={acc:.3f} (not a participant-independent estimate)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": 5,
        "hidden_size": 32,
        "classes": 3,
        "feature_names": FEATURE_NAMES,
    }, out)
    print(f"Saved checkpoint: {out}")

if __name__ == "__main__":
    main()
