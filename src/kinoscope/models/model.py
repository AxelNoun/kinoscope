"""
kinoscope/models/model.py
=========================

Architecture hybride pour la prédiction d'affinité cible-médicament (DTI)
sur le kinome humain.

    Ligand   : graphe moléculaire 2D  -> GINE (Graph Isomorphism Network w/ edge features)
    Protéine : poche KLIFS 85 résidus -> embeddings ESM-2 pré-calculés + adaptateur Transformer
    Fusion   : Bilinear Attention Network (BAN) -> carte d'interaction atome-résidu explicite
    Têtes    : régression hétéroscédastique (mu, log sigma^2) + classification actif/inactif

Le modèle expose la carte d'interaction brute afin (i) de la superviser faiblement
contre les contacts cristallographiques (PDBbind / BioLiP, seuil 4.5 A heavy-atom)
et (ii) de produire une interprétabilité vérifiable au moment de l'inférence.

Convention de shapes
--------------------
    B  : taille du batch
    Na : nombre max d'atomes lourds dans le batch (après to_dense_batch)
    Nr : nombre de positions de poche (85 pour l'alignement KLIFS)
    H  : nombre de têtes bilinéaires
    K  : rang de la projection bilinéaire (low-rank)
    D  : dimension latente partagée

Auteur : <you>
Licence : Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

__all__ = [
    "ModelConfig",
    "AtomEncoder",
    "BondEncoder",
    "LigandGraphEncoder",
    "PocketPLMEncoder",
    "BilinearAttentionFusion",
    "HybridDTIModel",
    "DTILoss",
]

# --------------------------------------------------------------------------- #
# Cardinalités du featurizer (src/kinoscope/data/featurizer.py).
# Chaque descripteur est encodé en entier ; l'index 0 est réservé à <unk>.
# --------------------------------------------------------------------------- #
ATOM_FEATURE_DIMS: tuple[int, ...] = (
    120,  # numéro atomique (Z)
    8,  # degré (nb de voisins lourds)
    6,  # charge formelle décalée [-2..+3]
    6,  # hybridation {SP, SP2, SP3, SP3D, SP3D2, other}
    2,  # aromaticité
    6,  # nb d'hydrogènes implicites + explicites
    4,  # tag de chiralité CIP
    2,  # appartenance à un cycle
    6,  # taille du plus petit cycle (SSSR)
)
BOND_FEATURE_DIMS: tuple[int, ...] = (
    5,  # type de liaison {simple, double, triple, aromatique, other}
    2,  # conjugaison
    2,  # appartenance à un cycle
    6,  # stéréochimie
)


@dataclass
class ModelConfig:
    """Configuration sérialisable (mappée 1:1 sur configs/model/*.yaml via Hydra)."""

    # --- dimensions partagées -------------------------------------------------
    hidden_dim: int = 256

    # --- encodeur ligand ------------------------------------------------------
    gnn_layers: int = 5
    gnn_dropout: float = 0.1
    gnn_jk: bool = True  # Jumping Knowledge : concat des représentations par couche

    # --- encodeur protéine ----------------------------------------------------
    plm_dim: int = 1280  # esm2_t33_650M_UR50D
    pocket_len: int = 85  # alignement KLIFS
    adapter_layers: int = 2
    adapter_heads: int = 8
    adapter_dropout: float = 0.1

    # --- fusion bilinéaire ----------------------------------------------------
    ban_heads: int = 4
    ban_rank: int = 96
    ban_dropout: float = 0.2

    # --- covariable de dosage -------------------------------------------------
    # Les endpoints ChEMBL sont hétérogènes (Kd / Ki / IC50 / EC50). Plutôt qu'une
    # conversion Cheng-Prusoff hasardeuse, on conditionne le modèle sur le type
    # d'essai et on ne prédit sur le jeu de test que la distribution ciblée.
    n_assay_types: int = 5
    assay_emb_dim: int = 16

    # --- têtes ----------------------------------------------------------------
    head_hidden: int = 512
    head_dropout: float = 0.3
    heteroscedastic: bool = True
    logvar_clamp: tuple[float, float] = (-6.0, 3.0)

    # --- pondération des pertes ----------------------------------------------
    w_affinity: float = 1.0
    w_activity: float = 0.2
    w_contact: float = 0.1
    activity_threshold: float = 6.5  # pActivity ; ~316 nM

    atom_feature_dims: tuple[int, ...] = field(default_factory=lambda: ATOM_FEATURE_DIMS)
    bond_feature_dims: tuple[int, ...] = field(default_factory=lambda: BOND_FEATURE_DIMS)


# --------------------------------------------------------------------------- #
# Encodeurs de features catégorielles
# --------------------------------------------------------------------------- #
class _CategoricalEncoder(nn.Module):
    """Somme d'embeddings sur des descripteurs catégoriels entiers."""

    def __init__(self, dims: tuple[int, ...], hidden_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=d, embedding_dim=hidden_dim) for d in dims]
        )
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x: Tensor) -> Tensor:
        # x : [N, n_features] entiers
        out: Tensor = self.embeddings[0](x[:, 0])
        for i, emb in enumerate(self.embeddings[1:], start=1):
            out = out + emb(x[:, i])
        return out


class AtomEncoder(_CategoricalEncoder):
    """Descripteurs atomiques RDKit -> R^D."""


class BondEncoder(_CategoricalEncoder):
    """Descripteurs de liaison RDKit -> R^D (requis par GINEConv)."""


# --------------------------------------------------------------------------- #
# 1. Encodeur ligand : GINE
# --------------------------------------------------------------------------- #
class LigandGraphEncoder(nn.Module):
    """
    Pile GINEConv résiduelle sur le graphe moléculaire 2D.

    GINE est retenu plutôt qu'un GAT/GCN parce que son pouvoir discriminant est
    borné par le test de Weisfeiler-Lehman 1-WL (Xu et al., 2019) et qu'il
    consomme nativement les features de liaison — non négociable en chimie
    médicinale, où le type de liaison porte l'essentiel de la sémantique.

    Retourne des embeddings *par atome* (et non un simple pooling global) : la
    couche de fusion en a besoin pour construire la carte d'interaction.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        self.atom_encoder = AtomEncoder(cfg.atom_feature_dims, d)
        self.bond_encoder = BondEncoder(cfg.bond_feature_dims, d)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.gnn_layers):
            mlp = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.BatchNorm1d(2 * d),
                nn.ReLU(inplace=True),
                nn.Linear(2 * d, d),
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=True, edge_dim=d))
            self.norms.append(nn.BatchNorm1d(d))

        self.dropout = nn.Dropout(cfg.gnn_dropout)
        out_dim = d * cfg.gnn_layers if cfg.gnn_jk else d
        self.out_proj = nn.Sequential(nn.Linear(out_dim, d), nn.LayerNorm(d))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        node_repr : [B, Na, D]
        node_mask : [B, Na] booléen (True = atome réel)
        """
        h = self.atom_encoder(x)
        e = self.bond_encoder(edge_attr)

        layer_outputs: list[Tensor] = []
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h_new = conv(h, edge_index, e)
            h_new = norm(h_new)
            h_new = self.dropout(F.relu(h_new))
            h = h + h_new  # connexion résiduelle : stabilise > 4 couches
            layer_outputs.append(h)

        h = torch.cat(layer_outputs, dim=-1) if self.cfg.gnn_jk else h
        h = self.out_proj(h)
        return cast(tuple[Tensor, Tensor], to_dense_batch(h, batch))


# --------------------------------------------------------------------------- #
# 2. Encodeur protéine : adaptateur sur ESM-2 gelé
# --------------------------------------------------------------------------- #
class PocketPLMEncoder(nn.Module):
    """
    Consomme des représentations ESM-2 *pré-calculées* et mises en cache
    (`scripts/precompute_esm_embeddings.py` -> LMDB, fp16).

    Deux décisions de conception :

    1. On n'encode pas la séquence entière de la kinase (300-2500 aa) mais les
       85 positions de la poche définies par l'alignement structural KLIFS. Le
       ratio signal/bruit explose, la longueur devient fixe, et les positions
       acquièrent une identité pharmacophorique stable entre toutes les kinases
       (hinge = 46-48, gatekeeper = 45, motif DFG = 81-83, hélice alphaC = 17-30).
       C'est ce qui autorise un embedding positionnel *appris* — impossible sur
       une séquence full-length.

    2. ESM-2 650M reste gelé par défaut ; seul l'adaptateur (~3M paramètres) est
       entraîné. Avec ~10^5 paires étiquetées, un fine-tuning complet du PLM
       sur-apprend systématiquement. Le fine-tuning LoRA des 4 derniers blocs
       est disponible en phase 2 (cf. `configs/model/gine_esm2_lora.yaml`).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.hidden_dim

        self.input_proj = nn.Sequential(
            nn.LayerNorm(cfg.plm_dim),
            nn.Linear(cfg.plm_dim, d),
            nn.GELU(),
            nn.Dropout(cfg.adapter_dropout),
        )
        # Embedding positionnel indexé sur la position KLIFS (identité fonctionnelle).
        self.pos_embedding = nn.Parameter(torch.zeros(1, cfg.pocket_len, d))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.adapter_heads,
            dim_feedforward=4 * d,
            dropout=cfg.adapter_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN : entraînement stable sans warmup long
        )
        self.adapter = nn.TransformerEncoder(encoder_layer, num_layers=cfg.adapter_layers)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, plm_emb: Tensor, residue_mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        plm_emb      : [B, Nr, plm_dim] représentations ESM-2 par résidu
        residue_mask : [B, Nr] booléen (True = position occupée ; les gaps de
                       l'alignement KLIFS sont masqués)

        Returns
        -------
        [B, Nr, D]
        """
        h = self.input_proj(plm_emb) + self.pos_embedding[:, : plm_emb.size(1)]
        h = self.adapter(h, src_key_padding_mask=~residue_mask)
        return cast(Tensor, self.out_norm(h))


# --------------------------------------------------------------------------- #
# 3. Fusion : Bilinear Attention Network
# --------------------------------------------------------------------------- #
class BilinearAttentionFusion(nn.Module):
    """
    Attention bilinéaire low-rank entre atomes et résidus (Kim et al., 2018 ;
    adaptée au DTI par Bai et al., DrugBAN, Nat. Mach. Intell. 2023).

        A[b,h,i,j] = softmax_{i,j} ( <U_h · l_i , V_h · p_j> / sqrt(K) )
        f[b,h,:]   = sum_{i,j} A[b,h,i,j] * (U_h · l_i) ⊙ (V_h · p_j)

    L'intérêt par rapport à une simple concaténation de vecteurs poolés : le
    modèle est contraint de raisonner sur des *paires* atome-résidu. La matrice A
    est directement comparable à une carte de contacts cristallographique, ce qui
    fournit à la fois un signal de supervision auxiliaire et une interprétabilité
    falsifiable — et non un heatmap décoratif.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.h = cfg.ban_heads
        self.k = cfg.ban_rank
        d = cfg.hidden_dim

        self.ligand_proj = nn.Linear(d, self.h * self.k, bias=False)
        self.protein_proj = nn.Linear(d, self.h * self.k, bias=False)
        self.dropout = nn.Dropout(cfg.ban_dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(self.h * self.k, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.scale = self.k**-0.5

    def forward(
        self,
        ligand_repr: Tensor,  # [B, Na, D]
        protein_repr: Tensor,  # [B, Nr, D]
        ligand_mask: Tensor,  # [B, Na]
        protein_mask: Tensor,  # [B, Nr]
    ) -> tuple[Tensor, Tensor]:
        b, na, _ = ligand_repr.shape
        nr = protein_repr.size(1)

        ql = self.ligand_proj(ligand_repr).view(b, na, self.h, self.k).permute(0, 2, 1, 3)
        qp = self.protein_proj(protein_repr).view(b, nr, self.h, self.k).permute(0, 2, 1, 3)

        # Logits d'interaction : [B, H, Na, Nr]
        logits = torch.einsum("bhik,bhjk->bhij", ql, qp) * self.scale

        pair_mask = ligand_mask[:, None, :, None] & protein_mask[:, None, None, :]
        logits = logits.masked_fill(~pair_mask, float("-inf"))

        # Softmax *joint* sur (Na x Nr) : les paires se concurrencent entre elles,
        # contrairement à un softmax par ligne qui forcerait chaque atome à
        # contacter un résidu.
        attn = torch.softmax(logits.flatten(start_dim=2), dim=-1).view_as(logits)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # Pooling bilinéaire : sum_{i,j} A_ij * (ql_i ⊙ qp_j)
        pooled = torch.einsum("bhij,bhik,bhjk->bhk", attn, ql, qp)
        pooled = pooled.reshape(b, self.h * self.k)

        interaction_logits = logits.masked_fill(~pair_mask, 0.0)
        return self.out_proj(pooled), interaction_logits


# --------------------------------------------------------------------------- #
# 4. Modèle complet
# --------------------------------------------------------------------------- #
class HybridDTIModel(nn.Module):
    """GNN(ligand) x ESM-2(poche) -> pAffinity hétéroscédastique + logit d'activité."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        d = self.cfg.hidden_dim

        self.ligand_encoder = LigandGraphEncoder(self.cfg)
        self.protein_encoder = PocketPLMEncoder(self.cfg)
        self.fusion = BilinearAttentionFusion(self.cfg)
        self.assay_embedding = nn.Embedding(self.cfg.n_assay_types, self.cfg.assay_emb_dim)

        head_in = d + self.cfg.assay_emb_dim
        self.trunk = nn.Sequential(
            nn.Linear(head_in, self.cfg.head_hidden),
            nn.LayerNorm(self.cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.head_dropout),
            nn.Linear(self.cfg.head_hidden, self.cfg.head_hidden // 2),
            nn.GELU(),
            nn.Dropout(self.cfg.head_dropout),
        )
        n_out = 2 if self.cfg.heteroscedastic else 1
        self.affinity_head = nn.Linear(self.cfg.head_hidden // 2, n_out)
        self.activity_head = nn.Linear(self.cfg.head_hidden // 2, 1)

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """
        Clés attendues dans `batch`
        ---------------------------
        x            : [Nnodes, n_atom_feat]  (long)
        edge_index   : [2, Nedges]            (long)
        edge_attr    : [Nedges, n_bond_feat]  (long)
        batch_index  : [Nnodes]               (long)
        plm_emb      : [B, Nr, plm_dim]       (float)
        residue_mask : [B, Nr]                (bool)
        assay_type   : [B]                    (long)
        """
        ligand_repr, ligand_mask = self.ligand_encoder(
            batch["x"], batch["edge_index"], batch["edge_attr"], batch["batch_index"]
        )
        protein_repr = self.protein_encoder(batch["plm_emb"], batch["residue_mask"])

        joint, interaction_logits = self.fusion(
            ligand_repr, protein_repr, ligand_mask, batch["residue_mask"]
        )

        assay = self.assay_embedding(batch["assay_type"])
        z = self.trunk(torch.cat([joint, assay], dim=-1))

        affinity = self.affinity_head(z)
        if self.cfg.heteroscedastic:
            mu, logvar = affinity[:, 0], affinity[:, 1]
            logvar = logvar.clamp(*self.cfg.logvar_clamp)
        else:
            mu, logvar = affinity[:, 0], torch.zeros_like(affinity[:, 0])

        return {
            "affinity_mu": mu,  # [B]  pAffinity prédite
            "affinity_logvar": logvar,  # [B]  incertitude aléatorique
            "activity_logit": self.activity_head(z).squeeze(-1),  # [B]
            "interaction_logits": interaction_logits,  # [B, H, Na, Nr]
            "ligand_mask": ligand_mask,  # [B, Na]
        }

    @torch.inference_mode()
    def predict(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Inférence : renvoie mu et l'écart-type aléatorique (en unités log)."""
        self.eval()
        out = self.forward(batch)
        return {
            "p_affinity": out["affinity_mu"],
            "sigma_aleatoric": torch.exp(0.5 * out["affinity_logvar"]),
            "p_active": torch.sigmoid(out["activity_logit"]),
            "contact_map": out["interaction_logits"].mean(dim=1),  # [B, Na, Nr]
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


# --------------------------------------------------------------------------- #
# 5. Objectif multi-tâche
# --------------------------------------------------------------------------- #
class DTILoss(nn.Module):
    """
    L = w_a * GaussianNLL(pAffinity)
      + w_c * BCE(actif/inactif)
      + w_x * BCE(carte d'interaction | contacts cristallographiques)

    Le terme de contact n'est appliqué que sur le sous-ensemble de paires
    disposant d'une structure co-cristallisée (PDBbind / BioLiP), signalé par
    `contact_available`. Il agit comme un régulariseur structural : il force
    l'attention à se concentrer sur la poche ATP réelle au lieu de corréler avec
    des artefacts de composition atomique.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        mu, logvar = outputs["affinity_mu"], outputs["affinity_logvar"]
        y = targets["p_affinity"]

        # Negative log-likelihood gaussienne (Nix & Weigend, 1994).
        nll = 0.5 * (logvar + (y - mu).pow(2) / logvar.exp())
        loss_affinity = nll.mean()

        y_act = (y >= self.cfg.activity_threshold).float()
        loss_activity = F.binary_cross_entropy_with_logits(outputs["activity_logit"], y_act)

        loss_contact = mu.new_zeros(())
        if "contact_map" in targets and targets.get("contact_available") is not None:
            avail = targets["contact_available"].bool()
            if avail.any():
                pred = outputs["interaction_logits"][avail].mean(dim=1)  # [b', Na, Nr]
                true = targets["contact_map"][avail].float()  # [b', Na, Nr]
                valid = targets["contact_mask"][avail].bool()  # [b', Na, Nr]
                if valid.any():
                    bce = F.binary_cross_entropy_with_logits(pred, true, reduction="none")
                    loss_contact = (bce * valid).sum() / valid.sum().clamp(min=1)

        total = (
            self.cfg.w_affinity * loss_affinity
            + self.cfg.w_activity * loss_activity
            + self.cfg.w_contact * loss_contact
        )
        return {
            "loss": total,
            "loss_affinity": loss_affinity.detach(),
            "loss_activity": loss_activity.detach(),
            "loss_contact": loss_contact.detach(),
        }


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = ModelConfig()
    model = HybridDTIModel(cfg)
    criterion = DTILoss(cfg)

    B, Nr = 4, cfg.pocket_len
    n_atoms_per_mol = [22, 31, 18, 27]
    n_nodes = sum(n_atoms_per_mol)

    x = torch.stack([torch.randint(0, d, (n_nodes,)) for d in cfg.atom_feature_dims], dim=-1)
    batch_index = torch.repeat_interleave(torch.arange(B), torch.tensor(n_atoms_per_mol))
    n_edges = 2 * n_nodes
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    # On garde les arêtes intra-molécule (le batching PyG est bloc-diagonal).
    keep = batch_index[edge_index[0]] == batch_index[edge_index[1]]
    edge_index = edge_index[:, keep]
    edge_attr = torch.stack(
        [torch.randint(0, d, (edge_index.size(1),)) for d in cfg.bond_feature_dims], dim=-1
    )

    batch = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch_index": batch_index,
        "plm_emb": torch.randn(B, Nr, cfg.plm_dim),
        "residue_mask": torch.ones(B, Nr, dtype=torch.bool),
        "assay_type": torch.randint(0, cfg.n_assay_types, (B,)),
    }

    out = model(batch)
    loss = criterion(out, {"p_affinity": torch.rand(B) * 5 + 4})

    print(f"paramètres entraînables : {model.num_parameters():,}")
    for k, v in out.items():
        print(f"  {k:22s} {tuple(v.shape)}")
    print(f"loss = {loss['loss'].item():.4f}")

    loss["loss"].backward()
    print("backward OK")
