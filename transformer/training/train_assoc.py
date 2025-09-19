"""Train the transformer-based association head."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from ..models.association import _AssociationBackbone


@dataclass
class Sample:
    track_features: np.ndarray
    det_features: np.ndarray
    labels: np.ndarray


class AssociationDataset(Dataset[Sample]):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        manifest = self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"missing manifest: {manifest}")
        with manifest.open("r", encoding="utf-8") as fp:
            entries = [json.loads(line) for line in fp if line.strip()]
        if not entries:
            raise RuntimeError("manifest is empty; nothing to train on")
        self.paths: List[Path] = [self.root / entry["npz"] for entry in entries if "npz" in entry]
        if not self.paths:
            raise RuntimeError("no samples listed in manifest")
        self.track_dim = 0
        self.det_dim = 0
        self.track_max = 0
        self.det_max = 0
        for path in self.paths:
            data = np.load(path)
            tf = np.asarray(data["track_features"], dtype=np.float32)
            df = np.asarray(data["det_features"], dtype=np.float32)
            te = np.asarray(data["track_embeddings"], dtype=np.float32)
            de = np.asarray(data["det_embeddings"], dtype=np.float32)
            if te.size:
                tf = np.concatenate([tf, te], axis=1)
            if de.size:
                df = np.concatenate([df, de], axis=1)
            self.track_dim = max(self.track_dim, tf.shape[1] if tf.size else 0)
            self.det_dim = max(self.det_dim, df.shape[1] if df.size else 0)
            self.track_max = max(self.track_max, tf.shape[0])
            self.det_max = max(self.det_max, df.shape[0])
        if self.track_dim == 0 or self.det_dim == 0:
            raise RuntimeError("dataset has zero-dimensional features; check logging configuration")

    def __len__(self) -> int:
        return len(self.paths)

    def _pad(self, arr: np.ndarray, target_dim: int) -> np.ndarray:
        if arr.ndim != 2:
            return np.zeros((0, target_dim), dtype=np.float32)
        if arr.shape[1] == target_dim:
            return arr.astype(np.float32)
        out = np.zeros((arr.shape[0], target_dim), dtype=np.float32)
        take = min(arr.shape[1], target_dim)
        out[:, :take] = arr[:, :take]
        return out

    def __getitem__(self, idx: int) -> Sample:
        data = np.load(self.paths[idx])
        tf = np.asarray(data["track_features"], dtype=np.float32)
        df = np.asarray(data["det_features"], dtype=np.float32)
        te = np.asarray(data["track_embeddings"], dtype=np.float32)
        de = np.asarray(data["det_embeddings"], dtype=np.float32)
        if te.size:
            tf = np.concatenate([tf, te], axis=1)
        if de.size:
            df = np.concatenate([df, de], axis=1)
        tf = self._pad(tf, self.track_dim)
        df = self._pad(df, self.det_dim)
        assigned = np.asarray(data["assigned_track_ids"], dtype=np.int64)
        track_ids = np.asarray(data["track_ids"], dtype=np.int64)
        label = np.zeros((tf.shape[0], df.shape[0]), dtype=np.float32)
        id_map = {int(tid): i for i, tid in enumerate(track_ids.tolist())}
        for det_idx, tid in enumerate(assigned.tolist()):
            track_idx = id_map.get(int(tid))
            if track_idx is not None:
                label[track_idx, det_idx] = 1.0
        return Sample(tf, df, label)


def build_loaders(dataset: AssociationDataset, val_split: float, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - val_split))) if len(indices) > 1 else len(indices)
    train_indices = indices[:split]
    val_indices = indices[split:] if split < len(indices) else []
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices) if val_indices else Subset(dataset, train_indices[:1])

    def collate(batch: Sequence[Sample]) -> Sample:
        return batch[0]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def inspect_dataset(dataset: AssociationDataset) -> None:
    track_counts = []
    det_counts = []
    for path in dataset.paths:
        data = np.load(path)
        track_counts.append(int(data["track_features"].shape[0]))
        det_counts.append(int(data["det_features"].shape[0]))
    print(f"samples: {len(dataset)}")
    print(f"track dims: {dataset.track_dim}  det dims: {dataset.det_dim}")
    print(f"tracks per sample: mean={np.mean(track_counts):.2f} max={max(track_counts)}")
    print(f"dets per sample:   mean={np.mean(det_counts):.2f} max={max(det_counts)}")


def train(args: argparse.Namespace) -> None:
    dataset = AssociationDataset(Path(args.data))
    if args.inspect:
        inspect_dataset(dataset)
        return
    if args.batch_size != 1:
        raise ValueError("Only batch_size=1 is currently supported")

    train_loader, val_loader = build_loaders(dataset, args.val_split, args.batch_size, args.seed)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _AssociationBackbone(dataset.track_dim, dataset.det_dim, embed_dim=args.hidden_dim, nheads=args.heads, nlayers=args.layers)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            tracks = torch.from_numpy(batch.track_features).to(device)
            dets = torch.from_numpy(batch.det_features).to(device)
            labels = torch.from_numpy(batch.labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tracks.unsqueeze(0), dets.unsqueeze(0)).squeeze(0)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tracks = torch.from_numpy(batch.track_features).to(device)
                dets = torch.from_numpy(batch.det_features).to(device)
                labels = torch.from_numpy(batch.labels).to(device)
                logits = model(tracks.unsqueeze(0), dets.unsqueeze(0)).squeeze(0)
                loss = criterion(logits, labels)
                val_loss += float(loss.item())
                preds = (logits.sigmoid() >= 0.5).float()
                if labels.numel() > 0:
                    val_acc += float((preds == labels).float().mean().item())
        val_loss /= max(1, len(val_loader))
        val_acc /= max(1, len(val_loader))
        print(f"epoch {epoch:02d} | train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    if best_state is None:
        best_state = model.state_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "meta": {
                "track_dim": dataset.track_dim,
                "det_dim": dataset.det_dim,
                "hidden_dim": args.hidden_dim,
                "heads": args.heads,
                "layers": args.layers,
            },
        },
        output,
    )
    print(f"saved checkpoint to {output}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True, help="Directory containing manifest.jsonl and .npz samples")
    parser.add_argument("--output", type=str, required=True, help="Path to save the trained checkpoint (.pt)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples per optimizer step")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect", action="store_true", help="Print dataset statistics and exit")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
