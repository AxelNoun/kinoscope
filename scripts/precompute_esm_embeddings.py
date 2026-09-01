#!/usr/bin/env python3
"""Pré-calcul et mise en cache des représentations ESM-2 des poches KLIFS.

Exécuté UNE FOIS. Sortie : un LMDB {accession UniProt -> float16 [85, 1280]}.
~500 kinases x 85 x 1280 x 2 o ~= 110 Mo, entièrement chargeable en RAM à
l'inférence. C'est ce qui rend l'entraînement ~10^3 fois moins coûteux qu'un
forward pass du PLM à chaque batch.

Usage:
    python scripts/precompute_esm_embeddings.py \
        --targets data/processed/targets.parquet \
        --out data/processed/pockets.lmdb \
        --layer 33 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
POCKET_LEN = 85
PLM_DIM = 1280


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=Path, default=Path("data/processed/targets.parquet"))
    p.add_argument("--out", type=Path, default=Path("data/processed/pockets.lmdb"))
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--layer", type=int, default=33, help="Couche de sortie. 33 = dernière ; 30-32 corrèlent mieux avec la structure.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.targets)
    required = {"uniprot", "seq_domain", "klifs_indices"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"colonnes manquantes dans {args.targets}: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmModel.from_pretrained(args.model, output_hidden_states=True).to(args.device).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(args.out), map_size=8 * 1024**3)

    stats: list[dict[str, float]] = []
    rows = df.to_dict("records")
    for start in tqdm(range(0, len(rows), args.batch_size), desc="ESM-2"):
        chunk = rows[start : start + args.batch_size]
        enc = tokenizer([r["seq_domain"] for r in chunk], return_tensors="pt",
                        padding=True, truncation=True, max_length=args.max_length).to(args.device)
        hidden = model(**enc).hidden_states[args.layer]  # [B, L+2, 1280]

        with env.begin(write=True) as txn:
            for i, row in enumerate(chunk):
                # +1 : le token <cls> décale l'indexation des résidus de 1.
                idx = np.asarray(json.loads(row["klifs_indices"]) if isinstance(row["klifs_indices"], str) else row["klifs_indices"])
                mask = idx >= 0                       # -1 = gap dans l'alignement KLIFS
                emb = np.zeros((POCKET_LEN, PLM_DIM), dtype=np.float16)
                valid = torch.as_tensor(idx[mask] + 1, device=hidden.device)
                emb[mask] = hidden[i, valid].to(torch.float16).cpu().numpy()

                txn.put(row["uniprot"].encode(), pickle.dumps({"emb": emb, "mask": mask}))
                stats.append({"uniprot": row["uniprot"], "n_valid": int(mask.sum()),
                              "norm_mean": float(np.linalg.norm(emb[mask], axis=-1).mean())})

    env.close()
    s = pd.DataFrame(stats)
    # Garde-fous d'intégrité : un embedding nul ou explosif signale un mauvais
    # alignement KLIFS ou une troncature de séquence silencieuse.
    assert not s["norm_mean"].isna().any(), "NaN dans les embeddings"
    assert (s["n_valid"] >= 50).all(), "cible avec < 50 positions KLIFS alignées"
    s.to_csv(args.out.with_suffix(".stats.csv"), index=False)
    print(f"{len(s)} cibles écrites -> {args.out}")
    print(s[["n_valid", "norm_mean"]].describe())


if __name__ == "__main__":
    main()
