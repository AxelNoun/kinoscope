---
project: kinoscope
plan_version: 1.0.0
last_updated: 2026-09-01
current_phase: 0
current_step: "0.4"
blocked_steps: []
completed_steps: 3
total_steps: 58
---

# MASTER_PLAN — KinoScope

> Modèle hybride GNN + ESM-2 pour la prédiction d'affinité kinase–inhibiteur,
> sous protocole d'évaluation sans fuite de données.

---

## 1. Directives pour l'agent (méta-instructions)

**Ce fichier est la source de vérité unique du projet. Lis-le intégralement au début de
chaque session, avant toute autre action, avant même de lister les fichiers du dépôt.**
Localise ensuite la **première** case non cochée `[ ]` en parcourant les phases dans
l'ordre, du haut vers le bas — jamais dans un autre ordre, jamais en sautant une étape
sous prétexte qu'elle semble triviale ou déjà faite. Exécute cette étape et elle seule.
Une étape est *terminée* uniquement lorsque son **Test de sortie** passe et que
`make check` (ruff + mypy + pytest, couverture ≥ 80 %) sort en code 0. Alors, et alors
seulement : coche la case en `[x]`, incrémente `completed_steps` et mets à jour
`current_phase` / `current_step` dans l'en-tête YAML, ajoute une ligne au **Journal de
bord** (§7), et produis un commit unique au format
`feat(<scope>): <description> [PLAN-<id>]`. Un commit par étape, jamais plus.

**Règles d'exécution non négociables :**

1. **Test d'abord.** Écris le test de sortie *avant* le code de production. Vérifie qu'il
   échoue pour la bonne raison, puis implémente jusqu'à ce qu'il passe.
2. **Ne modifie jamais la structure de la feuille de route** (ajout, suppression ou
   réordonnancement d'étapes) sans validation humaine explicite. Si une étape te paraît
   mal spécifiée, marque-la `[?]`, écris ta question dans le Journal, et **arrête-toi**.
3. **Protocole d'échec.** Après 3 tentatives infructueuses sur une même étape, marque-la
   `[!]`, ajoute son identifiant à `blocked_steps`, consigne le diagnostic et l'erreur
   exacte dans le Journal, et arrête-toi. Ne contourne pas, ne désactive pas le test, ne
   passe pas à l'étape suivante.
4. **Interdiction d'inventer.** Aucun chiffre de performance, aucun identifiant UniProt,
   aucun endpoint d'API, aucune signature de fonction de bibliothèque tierce ne doit être
   écrit de mémoire. Vérifie contre la documentation installée
   (`python -c "import X; help(X.f)"`) ou contre une réponse d'API réelle. Si une source
   externe est indisponible, marque l'étape `[!]` — n'écris jamais de données factices en
   la présentant comme réelle. Les fixtures de test synthétiques sont autorisées et
   doivent être nommées `toy_`, `synthetic_` ou `fake_`.
5. **Aucune donnée dans git.** Tout ce qui vit sous `data/` ou `artifacts/` est suivi par
   DVC. Le hook `check-added-large-files` (500 ko) est un garde-fou, pas une autorisation.
6. **Versions épinglées.** N'ajoute jamais une dépendance sans version exacte dans
   `pyproject.toml`, et justifie l'ajout dans le message de commit.
7. **Budget de contexte.** N'ouvre pas le dépôt entier. La carte des fichiers pertinents
   par phase est donnée en §6 : lis uniquement ceux-là, plus ce fichier.
8. **Reproductibilité.** Toute exécution stochastique passe par
   `kinoscope.utils.seed.set_global_seed(cfg.seed)`. Un résultat non reproductible est un
   bug bloquant, pas une fatalité.
9. **Honnêteté des résultats.** Si une métrique s'effondre sous split froid, tu la
   rapportes telle quelle. La dégradation attendue entre `random` et `cold_target` **est**
   le résultat scientifique du projet. Toute tentative de la masquer invalide le dépôt.

**Légende des cases :** `[ ]` à faire · `[x]` terminé et testé · `[!]` bloqué (voir Journal)
· `[?]` spécification à clarifier · `[~]` en cours (une seule à la fois).

---

## 2. Résumé et objectifs scientifiques

### 2.1 Pitch

Les modèles DTI publiés rapportent des corrélations élevées sur Davis et KIBA sous
partition aléatoire. Cette partition place le même ligand des deux côtés de la frontière
train/test : le modèle mémorise un **prior de composé** au lieu d'apprendre une fonction
d'interaction. Le chiffre publié ne prédit donc rien sur un chémotype ou une kinase
inédits, seul régime qui compte en découverte de médicament.

KinoScope livre deux artefacts couplés :

1. **Un modèle hybride** : encodeur de ligand GINE sur graphe moléculaire 2D × encodeur de
   cible ESM-2 restreint aux 85 résidus de la poche ATP définis par l'alignement structural
   KLIFS, fusionnés par un Bilinear Attention Network produisant une carte d'interaction
   atome–résidu explicite. Sortie hétéroscédastique `(μ, log σ²)` enveloppée par une
   prédiction conforme calibrée.
2. **Un protocole d'évaluation sans fuite** : quatre régimes de partition (aléatoire,
   cold-drug par scaffold/Butina, cold-target par groupe de Manning, temporel), audit de
   fuite automatisé en intégration continue, et intervalles de confiance bootstrap sur
   toutes les métriques rapportées.

### 2.2 Hypothèses testables

- **H1** — La performance sous `cold_target` est significativement inférieure à la
  performance sous `random`, à modèle et hyperparamètres identiques.
- **H2** — Les représentations ESM-2 apportent un gain mesurable sur `cold_target` par
  rapport à un encodage one-hot de la séquence, et ce gain est plus faible sous `random`
  (où la mémorisation de la cible suffit).
- **H3** — Restreindre l'entrée protéique aux 85 positions KLIFS surpasse l'utilisation de
  la séquence entière du domaine kinase, à budget de paramètres égal.
- **H4** — La carte d'attention bilinéaire, supervisée faiblement par les contacts
  cristallographiques, recouvre les contacts réels significativement mieux que le hasard
  sur des complexes PDBbind exclus de l'entraînement.

Chaque hypothèse donne lieu à une ablation, un tableau et un IC bootstrap. Une hypothèse
réfutée est rapportée comme telle.

### 2.3 Objectif académique

Démontrer une maîtrise de bout en bout : curation chimio-biologique rigoureuse,
architecture de deep learning géométrique et de PLM, conception d'un protocole
d'évaluation résistant à la fuite, quantification d'incertitude à garantie formelle,
interprétabilité falsifiable contre la vérité structurale, et industrialisation MLOps.
Le livrable n'est pas un score, c'est une **méthodologie reproductible**.

---

## 3. Ressources et prérequis

### 3.1 Bases de données

| Source | Contenu utilisé | Accès | Licence |
|---|---|---|---|
| **ChEMBL 35+** | `activities` × `assays` × `target_dictionary` × `component_sequences` × `docs` | Dump PostgreSQL (recommandé) ou `chembl_webresource_client` | CC BY-SA 3.0 |
| **BindingDB** | `BindingDB_All.tsv` (Ki, Kd, IC50) | Téléchargement direct | CC BY 3.0 |
| **Davis** | 442 kinases × 68 inhibiteurs, matrice **complète** (vrais négatifs) | `tdc.multi_pred.DTI(name='DAVIS')` | CC BY 4.0 |
| **KIBA** | 229 kinases × 2 111 composés | `tdc.multi_pred.DTI(name='KIBA')` | CC BY 4.0 |
| **KLIFS** | Alignement structural 85 positions, annotations gatekeeper / DFG / αC | API REST `klifs.net/api_v2` | Académique |
| **UniProt** | Séquences canoniques, bornes de domaine Pfam | REST `rest.uniprot.org` | CC BY 4.0 |
| **KinHub / arbre de Manning** | Groupes (TK, TKL, STE, CK1, AGC, CAMK, CMGC) et familles | TSV `kinhub.org` | Libre |
| **PDBbind v2020+ / BioLiP2** | Complexes co-cristallisés avec affinité mesurée | Enregistrement académique | Académique |
| **RCSB PDB** | Structures kinase–ligand (mmCIF) | API REST | CC0 |

### 3.2 Bibliothèques

**Cœur** : `torch==2.4.1`, `torch-geometric==2.6.1`, `lightning==2.4.0`,
`transformers==4.44.2` (ESM-2 `facebook/esm2_t33_650M_UR50D`), `rdkit==2024.3.5`.
**Données** : `polars`, `pandas`, `pyarrow`, `lmdb`, `biopython`, `requests`.
**Science** : `numpy<2.0`, `scipy`, `scikit-learn`.
**Config & tracking** : `hydra-core`, `omegaconf`, `mlflow`, `dvc[s3]`.
**Serving** : `fastapi`, `uvicorn`, `pydantic>=2`, `redis`, `onnxruntime`.
**Qualité** : `pytest`, `pytest-cov`, `pytest-xdist`, `hypothesis`, `ruff`, `mypy`,
`pre-commit`, `nbstripout`, `detect-secrets`.

### 3.3 Matériel

Entraînement : 1 GPU ≥ 24 Go VRAM (RTX 4090 / A100 40 Go), < 24 h par run complet grâce au
cache des embeddings ESM-2. Pré-calcul ESM-2 : ~30 min sur GPU pour l'ensemble du kinome.
Inférence : CPU suffisant (le PLM n'est jamais réexécuté).

---

## 4. Configuration et structure du projet

### 4.1 Initialisation

```bash
# Environnement (uv, plus rapide et déterministe que Poetry ou conda pour ce stack)
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[serve,dev]"

# Hooks : ruff, mypy, nbstripout, detect-secrets, check-added-large-files
pre-commit install && pre-commit install --hook-type commit-msg

# Versionnage des données
dvc init && dvc remote add -d storage s3://<bucket>/kinoscope

# Vérification de l'installation
make check
```

RDKit et PyTorch Geometric ne sont pas installables de manière fiable par Poetry seul.
`uv` avec les roues PyPI officielles (`torch-geometric` en pur Python depuis la 2.5)
fonctionne sans index externe. Conda reste une alternative acceptable ; documente le choix
retenu dans `docs/index.md` et n'en maintiens qu'un.

### 4.2 Arborescence cible

```
kinoscope/
├── .github/workflows/{ci,docker,docs,benchmark}.yml
├── configs/{config.yaml,data/,model/,split/,trainer/,experiment/}
├── data/{raw,interim,processed,external}/          # DVC, jamais git
├── docker/{Dockerfile,docker-compose.yml}
├── docs/{index,architecture,data_card,model_card,benchmark,leakage_analysis,api}.md
├── notebooks/01_eda … 05_error_analysis.ipynb
├── scripts/{download_sources,build_dataset,precompute_esm_embeddings,train,evaluate,
│            calibrate_conformal,export_onnx,profile_kinome}.py
├── src/kinoscope/
│   ├── data/{sources/,standardize,featurizer,splits,contacts,dataset,datamodule}.py
│   ├── models/{model,lit_module,losses,layers/}.py
│   ├── eval/{metrics,conformal,applicability,interpretability,leaderboard}.py
│   ├── serving/{api,schemas,predictor,cache}.py
│   └── utils/{seed,logging,registry,chem}.py
├── tests/{unit,integration,fixtures}/
├── {pyproject.toml,Makefile,dvc.yaml,params.yaml,.pre-commit-config.yaml}
└── {README,MASTER_PLAN,MODEL_CARD,CONTRIBUTING,LICENSE,CITATION.cff}
```

---

## 5. Feuille de route itérative

---

### PHASE 0 — Fondations du dépôt et de l'environnement

*Objectif : rendre le dépôt vérifiable avant d'écrire la moindre ligne de science.*

- [x] **0.1 — Squelette du dépôt et packaging**
  - **Action** : créer l'arborescence de §4.2, écrire `pyproject.toml` (layout `src/`,
    build hatchling, dépendances épinglées), `.gitignore`, `LICENSE` (Apache-2.0),
    `CITATION.cff`.
  - **Validation** : `pip install -e .` réussit ; `python -c "import kinoscope"` fonctionne
    depuis un répertoire arbitraire.
  - **Test de sortie** : `pytest tests/unit/test_imports.py` — importe chaque sous-module
    de `kinoscope` et assert qu'aucun `ImportError` ne remonte.

- [x] **0.2 — Qualité statique**
  - **Action** : configurer `ruff` (E, F, W, I, N, UP, B, C4, SIM, ARG, PTH, RUF, NPY) et
    `mypy --strict` dans `pyproject.toml` ; installer `.pre-commit-config.yaml`.
  - **Validation** : zéro violation sur le code existant ; les `ignore_missing_imports`
    sont limités aux bibliothèques sans stubs (RDKit, PyG, LMDB) et justifiés en commentaire.
  - **Test de sortie** : `make lint && make typecheck` sortent en code 0.

- [x] **0.3 — Déterminisme global**
  - **Action** : `src/kinoscope/utils/seed.py::set_global_seed(seed, deterministic=True)`
    couvrant `random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED` et
    `torch.use_deterministic_algorithms`.
  - **Validation** : deux exécutions consécutives d'un mini-forward produisent des tenseurs
    bit-à-bit identiques sur CPU.
  - **Test de sortie** : `pytest tests/unit/test_seed.py::test_reproducible_forward`.

- [ ] **0.4 — Intégration continue**
  - **Action** : `.github/workflows/ci.yml` avec jobs `lint`, `typecheck`, `test`
    (roues CPU `torch`), `leakage-gate` ; cache pip ; `concurrency` avec annulation.
  - **Validation** : la CI passe sur une PR de test et échoue volontairement quand on
    introduit une violation ruff (vérifier les deux sens).
  - **Test de sortie** : run GitHub Actions vert sur la branche courante, capture d'écran
    ou lien consigné dans le Journal.

- [ ] **0.5 — Squelette Docker**
  - **Action** : `docker/Dockerfile` multi-stage (`builder` / `serve` / `train`), utilisateur
    non-root UID 10001, `HEALTHCHECK` ; `docker-compose.yml` avec `api`, `redis`, `mlflow`.
  - **Validation** : image `serve` < 3 Go ; `docker run` répond sur `/healthz` en mode
    dégradé (modèle absent → HTTP 503, pas de crash).
  - **Test de sortie** : `pytest tests/integration/test_docker_build.py -m slow` (build +
    `curl /healthz` attendu en 503).

- [ ] **0.6 — Versionnage des données**
  - **Action** : `dvc init`, remote configuré, `dvc.yaml` avec les stages `ingest`,
    `standardize`, `map_pocket`, `embed`, `featurize`, `split`, `train`.
  - **Validation** : `dvc dag` produit un graphe acyclique complet ; aucun fichier de
    `data/` n'apparaît dans `git status`.
  - **Test de sortie** : `pytest tests/integration/test_dvc_pipeline.py::test_dag_is_acyclic_and_complete`.

---

### PHASE 1 — Ingestion et curation chimio-biologique

*Objectif : un jeu de données propre, tracé et auditable. C'est ici que se gagne ou se perd
la validité de tout le reste.*

- [ ] **1.1 — Extraction ChEMBL**
  - **Action** : `src/kinoscope/data/sources/chembl.py` exécutant la requête de référence
    (`assay_type='B'`, `confidence_score >= 8`, `standard_relation='='`,
    `pchembl_value IS NOT NULL`, `data_validity_comment IS NULL`,
    `potential_duplicate = 0`, `target_type='SINGLE PROTEIN'`, `organism='Homo sapiens'`),
    avec conservation de `doc_year`.
  - **Validation scientifique** : justifier chaque filtre en docstring. Vérifier
    l'**absence totale** de relations censurées (`>`, `<`) : une seule suffit à biaiser la
    régression vers le bas. Rapporter le nombre de lignes rejetées par filtre.
  - **Test de sortie** : `pytest tests/unit/test_chembl_extract.py` sur un dump-fixture de
    1 000 lignes — assert schéma exact, `standard_relation == '='` partout, aucun
    `pchembl_value` nul, `doc_year` dans `[1990, année_courante]`.

- [ ] **1.2 — Restriction au kinome**
  - **Action** : joindre KinHub / arbre de Manning ; conserver les accessions UniProt
    annotées comme protéines kinases, avec `manning_group` et `family`.
  - **Validation** : la couverture des sept groupes canoniques (TK, TKL, STE, CK1, AGC,
    CAMK, CMGC) est effective ; produire l'histogramme du nombre de mesures par kinase et
    documenter la queue longue.
  - **Test de sortie** : `pytest tests/unit/test_kinome_mapping.py` — assert que
    `P00519` (ABL1) est mappé sur `TK` et `Q05655` (PRKCD) sur `AGC`.

- [ ] **1.3 — Standardisation RDKit**
  - **Action** : `standardize.py` — `LargestFragmentChooser` → `Normalizer` → `Uncharger`
    → `TautomerEnumerator.Canonicalize` → filtres (éléments autorisés, 6 ≤ atomes lourds
    ≤ 80, MW ≤ 900). Journaliser chaque motif de rejet.
  - **Validation scientifique** : l'ordre des opérations est critique — le choix de
    fragment doit précéder la neutralisation, sinon un contre-ion est neutralisé puis
    conservé. Vérifier l'**idempotence** : `standardize(standardize(x)) == standardize(x)`.
  - **Test de sortie** : `pytest tests/unit/test_standardize.py` — le mésylate d'imatinib
    donne le même InChIKey que l'imatinib base ; `[Na+].[Cl-]` est rejeté ; idempotence
    vérifiée par `hypothesis` sur 200 SMILES tirés du jeu réel.

- [ ] **1.4 — Fusion multi-sources et déduplication**
  - **Action** : union ChEMBL ∪ BindingDB ∪ Davis ∪ KIBA sur la clé
    `(inchikey, uniprot, assay_type)` ; agrégation par **médiane** du pChEMBL ; rejet des
    paires dont l'écart-type inter-mesures dépasse 1.0 unité log.
  - **Validation scientifique** : sur l'intersection ChEMBL ∩ BindingDB, calculer la
    corrélation de Pearson des valeurs. Un `r < 0.8` signale un bug d'harmonisation
    d'unités, pas du bruit expérimental — investiguer avant de continuer. Rapporter la
    distribution des écarts-types intra-paire dans `docs/data_card.md`.
  - **Test de sortie** : `pytest tests/unit/test_aggregation.py` — assert unicité de la clé
    composite, assert que la médiane est bien utilisée (et non la moyenne, sensible aux
    valeurs aberrantes), assert le rejet effectif d'une paire fabriquée à σ = 1.5.

- [ ] **1.5 — Traitement des endpoints hétérogènes**
  - **Action** : conserver `assay_type` comme covariable catégorielle. **Interdiction
    d'appliquer une conversion Cheng-Prusoff** IC50 → Ki.
  - **Validation scientifique** : documenter dans `docs/data_card.md` que Cheng-Prusoff
    exige la connaissance de `[S]` et de `Km`, absentes de ChEMBL, et que l'appliquer à
    l'aveugle introduit un décalage systématique. Tracer les distributions de pIC50 et de
    pKi côte à côte et rapporter le décalage médian observé.
  - **Test de sortie** : `pytest tests/unit/test_assay_types.py` — assert que `assay_type`
    survit jusqu'au tenseur de batch et que sa cardinalité correspond à
    `ModelConfig.n_assay_types`.

- [ ] **1.6 — Analyse exploratoire**
  - **Action** : `notebooks/01_eda_chembl_kinome.ipynb` — distribution de `p_affinity`,
    mesures par kinase et par groupe de Manning, distribution des poids moléculaires et du
    cLogP, couverture temporelle, matrice de complétude cible × composé.
  - **Validation scientifique** : identifier explicitement le **biais de publication**
    (sur-représentation des actifs dans ChEMBL) et documenter comment Davis, matrice
    complète, sert de contrepoids avec de vrais négatifs.
  - **Test de sortie** : `pytest --nbmake notebooks/01_eda_chembl_kinome.ipynb` — le
    notebook s'exécute de bout en bout sans erreur (sorties dépouillées par `nbstripout`).

- [ ] **1.7 — Carte des données**
  - **Action** : `docs/data_card.md` — provenance, licences, comptes après chaque filtre,
    biais connus, dates de téléchargement, versions exactes des sources.
  - **Validation** : les totaux du data card correspondent exactement aux artefacts DVC.
  - **Test de sortie** : `pytest tests/unit/test_data_card.py` — parse les tableaux du
    markdown et assert la cohérence avec `reports/dataset_stats.json`.

---

### PHASE 2 — Représentation de la cible (KLIFS + ESM-2)

- [ ] **2.1 — Récupération des séquences et bornes de domaine**
  - **Action** : `sources/uniprot.py` — séquence canonique, bornes du domaine kinase
    (Pfam PF00069 / PF07714), avec cache disque et retry exponentiel.
  - **Validation** : longueur du domaine extrait cohérente avec la littérature
    (≈ 250–300 résidus pour un domaine kinase catalytique) ; signaler tout écart.
  - **Test de sortie** : `pytest tests/unit/test_uniprot.py -m network` — la séquence
    récupérée pour `P00519` commence bien par la séquence canonique attendue et sa longueur
    correspond à l'entrée UniProt.

- [ ] **2.2 — Mapping de la poche KLIFS**
  - **Action** : `sources/klifs.py` — pour chaque UniProt, récupérer l'alignement 85
    positions et produire `klifs_indices` (index dans la séquence du domaine, `-1` pour un
    gap).
  - **Validation scientifique** : croiser l'annotation *gatekeeper* renvoyée par KLIFS avec
    la position 45 de l'alignement sur un échantillon de 20 kinases. Ancrages canoniques :
    le gatekeeper d'ABL1 est **T315** (mutation de résistance T315I à l'imatinib) et celui
    d'EGFR est **T790** (T790M). Un mapping qui ne retrouve pas ces deux résidus est faux.
  - **Test de sortie** : `pytest tests/unit/test_klifs_mapping.py` — assert
    `len(klifs_indices) == 85`, assert que le résidu à la position gatekeeper vaut `T` pour
    ABL1 et EGFR, assert que le nombre de positions alignées est ≥ 50 pour toute cible
    retenue.

- [ ] **2.3 — Pré-calcul des embeddings ESM-2**
  - **Action** : `scripts/precompute_esm_embeddings.py` — ESM-2 650M, extraction de la
    couche cible, gather sur les indices KLIFS (avec décalage `+1` pour le token `<cls>`),
    écriture LMDB en `float16`.
  - **Validation scientifique** : vérifier l'absence de NaN et de norme nulle. Le décalage
    du token de début est l'erreur classique de ce pipeline : la vérifier explicitement en
    comparant l'embedding gathered à un forward pass direct sur la sous-séquence.
  - **Test de sortie** : `pytest tests/unit/test_esm_cache.py` — assert shape `(85, 1280)`,
    dtype `float16`, absence de NaN, et **assert du décalage** : l'embedding de la position
    KLIFS *k* est identique à `hidden_states[layer][0, klifs_indices[k] + 1]`.

- [ ] **2.4 — Validation biologique des embeddings**
  - **Action** : calculer la matrice de similarité cosinus entre poches moyennes, et
    projeter en UMAP coloré par groupe de Manning.
  - **Validation scientifique** : la similarité intra-groupe doit dépasser la similarité
    inter-groupe. Tester par permutation (10 000 permutations, `p < 0.01`). Si le signal
    phylogénétique est absent, le mapping KLIFS ou la couche extraite est en cause.
  - **Test de sortie** : `pytest tests/unit/test_embedding_biology.py::test_intra_group_similarity_exceeds_inter_group`.

- [ ] **2.5 — Sélection de la couche ESM-2**
  - **Action** : comparer les couches 26, 30 et 33 sur un sondage linéaire (`ridge`)
    prédisant le groupe de Manning et le pAffinity sur un sous-ensemble de validation.
  - **Validation scientifique** : les couches intermédiaires des PLM encodent souvent
    davantage d'information structurale que la dernière, plus spécialisée sur l'objectif de
    masquage. Rapporter le tableau comparatif et **justifier** la couche retenue par la
    mesure, pas par convention.
  - **Test de sortie** : `pytest tests/unit/test_layer_selection.py` — assert que le
    rapport `reports/layer_probe.json` existe, contient les trois couches, et que la couche
    configurée dans `configs/model/gine_esm2.yaml` est celle de score maximal.

---

### PHASE 3 — Featurisation ligand et couche Dataset

- [ ] **3.1 — Featuriseur moléculaire**
  - **Action** : `featurizer.py` — 9 descripteurs atomiques, 4 descripteurs de liaison,
    graphe bidirectionnel, sortie `torch_geometric.data.Data`.
  - **Validation** : cardinalités strictement synchronisées avec `ATOM_FEATURE_DIMS` /
    `BOND_FEATURE_DIMS` du modèle. Un débordement d'index produit un `IndexError`
    d'embedding — au mieux ; au pire, un modèle entraîné sur des features tronquées.
  - **Test de sortie** : `pytest tests/unit/test_featurizer.py` (fourni) — topologie,
    bornes d'index, déterminisme, et **discrimination des énantiomères**.

- [ ] **3.2 — Cache LMDB des graphes**
  - **Action** : `dataset.py` — écriture LMDB `{inchikey → Data sérialisé}`, lecture
    memory-mapped compatible `num_workers > 0`.
  - **Validation** : LMDB doit être ouvert **après** le fork des workers, jamais avant :
    un handle partagé entre processus corrompt silencieusement les lectures. Implémenter
    l'ouverture paresseuse par worker.
  - **Test de sortie** : `pytest tests/unit/test_dataset.py::test_multiprocess_read_consistency`
    — un `DataLoader(num_workers=4)` renvoie exactement les mêmes tenseurs qu'un
    `num_workers=0`.

- [ ] **3.3 — Collation et DataModule**
  - **Action** : `datamodule.py` (Lightning) — collation du batch mêlant graphes PyG et
    tenseurs denses `plm_emb` / `residue_mask` / `assay_type` / `p_affinity`.
  - **Validation** : `batch_index` doit rester bloc-diagonal ; aucune arête ne doit
    traverser deux molécules. C'est l'erreur de collation la plus destructrice et la plus
    silencieuse du domaine.
  - **Test de sortie** : `pytest tests/unit/test_collate.py::test_no_cross_molecule_edges`
    — pour chaque arête, assert `batch_index[src] == batch_index[dst]`.

- [ ] **3.4 — Extraction des cartes de contacts**
  - **Action** : `contacts.py` — pour les paires disposant d'un complexe PDBbind/BioLiP,
    calculer la matrice binaire atome-ligand × position-KLIFS au seuil 4.5 Å heavy-atom.
  - **Validation scientifique** : le seuil 4.5 Å est le standard pour les contacts non
    covalents ; documenter ce choix et fournir une sensibilité à 4.0 et 5.0 Å. Vérifier
    l'appariement des indices atomiques entre le SMILES standardisé et le ligand du PDB
    (l'ordre des atomes diffère — un appariement par sous-structure est obligatoire).
  - **Test de sortie** : `pytest tests/unit/test_contacts.py -m network` — sur un complexe
    kinase–inhibiteur connu, assert que la région charnière (positions KLIFS 46–48) est
    contactée, et que la densité de contacts est comprise entre 2 % et 25 %.

---

### PHASE 4 — Splits et audit de fuite de données

*Le cœur scientifique du projet. Si une seule assertion de cette phase tombe, tous les
chiffres du README sont invalides.*

- [ ] **4.1 — Implémentation des six stratégies**
  - **Action** : `splits.py` — `random`, `scaffold` (Bemis-Murcko), `butina` (ECFP4,
    distance 0.6), `cold_target` (leave-group-out Manning), `cold_pair`, `temporal`.
  - **Validation** : la répartition des groupes utilise un remplissage glouton du plus gros
    au plus petit ; un simple mélange laisse un cluster massif déséquilibrer le test set.
  - **Test de sortie** : `pytest tests/unit/test_splits.py` (fourni) — exhaustivité,
    disjonction, zéro recouvrement de scaffold, zéro recouvrement de cible, flèche du temps,
    reproductibilité par graine.

- [ ] **4.2 — Audit de fuite quantitatif**
  - **Action** : `audit_leakage()` — InChIKeys partagés, scaffolds partagés, UniProt
    partagés, similarité Tanimoto au plus proche voisin train pour chaque composé de test,
    fraction du test au-dessus de 0.7.
  - **Validation scientifique** : la similarité maximale est **échantillonnée** (le calcul
    exact est O(n·m)) : c'est une borne inférieure honnête, et le rapport doit le dire.
    Un test set dont > 20 % des composés ont un voisin train à Tanimoto > 0.7 n'est pas un
    test cold-drug, quelle que soit la disjonction des scaffolds.
  - **Test de sortie** : `pytest tests/unit/test_splits.py::test_random_split_leaks_and_scaffold_split_does_not`
    — assert positivement la fuite du split aléatoire et son absence sous scaffold.

- [ ] **4.3 — Identité de séquence entre cibles train et test**
  - **Action** : alignement global par paires (Needleman-Wunsch, Biopython) entre les
    domaines kinase du train et du test ; rapporter l'identité maximale par cible de test.
  - **Validation scientifique** : deux paralogues à 95 % d'identité constituent une fuite
    de cible même sous `cold_target` par UniProt. Signaler et documenter toute cible de
    test dont l'identité maximale au train dépasse 90 %.
  - **Test de sortie** : `pytest tests/unit/test_target_identity.py` — assert que
    `reports/target_identity.json` est produit et qu'aucune cible de test sous
    `cold_target(by='manning_group')` ne dépasse le seuil configuré.

- [ ] **4.4 — Garde-fou de fuite en CI**
  - **Action** : job `leakage-gate` dans `ci.yml`, exécutant l'audit et comparant aux
    seuils de `configs/split/*.yaml`.
  - **Validation** : introduire volontairement une fuite dans une branche jetable et
    vérifier que la CI **échoue**. Un garde-fou non testé en échec est décoratif.
  - **Test de sortie** : run CI rouge sur la branche de fuite délibérée, run vert sur
    `main` ; les deux liens consignés dans le Journal.

- [ ] **4.5 — Rapport de fuite**
  - **Action** : `docs/leakage_analysis.md` généré automatiquement — tableau comparatif des
    quatre régimes, distributions de similarité au plus proche voisin, tailles de partition.
  - **Validation scientifique** : ce document est l'argument central du dépôt. Il doit
    permettre à un lecteur de reproduire l'audit sans exécuter le code.
  - **Test de sortie** : `pytest tests/unit/test_reports.py::test_leakage_report_is_complete`
    — assert la présence des quatre régimes et des six champs du `LeakageReport`.

---

### PHASE 5 — Modèle et entraînement

- [ ] **5.1 — Contrat d'architecture**
  - **Action** : finaliser `models/model.py` (`LigandGraphEncoder`, `PocketPLMEncoder`,
    `BilinearAttentionFusion`, `HybridDTIModel`, `DTILoss`).
  - **Validation** : shapes, bornage de `log σ²`, **absence de branche morte** — tout
    paramètre `requires_grad` doit recevoir un gradient non nul.
  - **Test de sortie** : `pytest tests/unit/test_model_shapes.py` (fourni) — shapes, clamp,
    gradient sur tous les paramètres, robustesse au masque partiel.

- [ ] **5.2 — Capacité d'apprentissage**
  - **Action** : test de sur-apprentissage sur un batch unique de 4 exemples.
  - **Validation scientifique** : un modèle incapable de sur-apprendre 4 exemples a un bug
    d'architecture, pas un problème de données. Ce test précède **obligatoirement** tout
    entraînement réel.
  - **Test de sortie** : `pytest tests/unit/test_model_shapes.py::test_can_overfit_single_batch -m slow`
    — assert `MSE < 0.05` après 400 pas.

- [ ] **5.3 — Boucle Lightning**
  - **Action** : `lit_module.py` — AdamW sans weight decay sur les normalisations et les
    biais, `OneCycleLR`, warmup MSE avant bascule sur la log-vraisemblance gaussienne,
    AMP bf16, EMA, gradient clipping à 1.0.
  - **Validation scientifique** : optimiser la NLL gaussienne dès le pas 0 permet au modèle
    de réduire la perte en gonflant σ sans jamais améliorer μ. Le warmup MSE est la
    correction standard ; le documenter en commentaire.
  - **Test de sortie** : `pytest tests/integration/test_train_smoke.py` — 2 époques sur la
    fixture jouet, assert que `train/loss` décroît strictement et qu'aucun NaN n'apparaît.

- [ ] **5.4 — Lignes de base**
  - **Action** : implémenter trois baselines évaluées sur les mêmes splits — (a) Morgan
    ECFP4 + XGBoost, (b) descripteurs RDKit + Random Forest, (c) CNN 1D type DeepDTA sur
    SMILES et séquence.
  - **Validation scientifique** : sans baseline sur le **même** split, un chiffre de deep
    learning n'a aucune signification. La baseline (a) est souvent redoutable sous split
    aléatoire et s'effondre sous cold-target : c'est précisément le contraste à montrer.
  - **Test de sortie** : `pytest tests/integration/test_baselines.py` — assert que chaque
    baseline produit un `metrics.json` valide sur les quatre régimes.

- [ ] **5.5 — Entraînement complet et suivi**
  - **Action** : `scripts/train.py` piloté par Hydra, logging MLflow (hyperparamètres,
    métriques, checkpoints, hash git, hash DVC des données).
  - **Validation** : un run doit être rejouable à l'identique à partir de son seul run_id
    MLflow. Vérifier en relançant et en comparant les métriques finales.
  - **Test de sortie** : `pytest tests/integration/test_reproducibility.py -m slow` — deux
    runs à graine identique produisent une RMSE de validation identique à 1e-4 près.

- [ ] **5.6 — Suite d'ablations (H2, H3, H4)**
  - **Action** : `configs/experiment/ablation_suite.yaml` — `no_plm` (one-hot séquence),
    `no_ban` (concaténation), `full_sequence` (ESM-2 séquence entière), `no_contact_loss`,
    `homoscedastic`.
  - **Validation scientifique** : chaque ablation est évaluée sur les quatre régimes, avec
    IC bootstrap. Une différence dont les intervalles se recouvrent n'est **pas** un
    résultat, et doit être rapportée comme non concluante.
  - **Test de sortie** : `pytest tests/integration/test_ablation_matrix.py` — assert que la
    matrice ablation × split est complète et que chaque cellule porte un IC.

---

### PHASE 6 — Évaluation, calibration et interprétabilité

- [ ] **6.1 — Suite de métriques**
  - **Action** : `eval/metrics.py` — RMSE, MAE, Pearson, Spearman, Concordance Index,
    r²ₘ de Roy, AUPR, AUROC, EF@1 %.
  - **Validation scientifique** : le r²ₘ détecte un biais systématique qu'un simple r²
    masque. L'EF@1 % mesure la capacité de triage réelle, plus pertinente qu'une RMSE pour
    un usage de screening.
  - **Test de sortie** : `pytest tests/unit/test_metrics.py` — sur des vecteurs analytiques
    (`y == yhat` → CI = 1.0 ; `yhat` inversé → CI = 0.0 ; corrélation parfaite → r = 1.0).

- [ ] **6.2 — Intervalles bootstrap**
  - **Action** : `bootstrap_ci()` — 1 000 rééchantillonnages percentile sur toutes les
    métriques.
  - **Validation scientifique** : aucun chiffre du README ne peut être publié sans son IC
    95 %. Une amélioration dont l'IC recouvre celui de la baseline n'est pas une amélioration.
  - **Test de sortie** : `pytest tests/unit/test_metrics.py::test_bootstrap_ci_brackets_point_estimate`
    — assert que l'estimation ponctuelle tombe dans son propre intervalle.

- [ ] **6.3 — Calibration conforme**
  - **Action** : `eval/conformal.py` — split conformal normalisé (score `|y-μ|/σ`),
    quantile à correction d'échantillon fini `⌈(n+1)(1-α)⌉/n`, variante Mondrian par groupe
    de Manning.
  - **Validation scientifique** : sur un split échangeable, le PICP à α = 0.1 doit tomber
    dans `[0.88, 0.92]`. Sous `cold_target`, l'échangeabilité est **violée par
    construction** : mesurer et rapporter l'écart de couverture, ne pas le corriger
    silencieusement. C'est un résultat, pas un défaut.
  - **Test de sortie** : `pytest tests/unit/test_conformal.py::test_marginal_coverage_on_exchangeable_data`
    — sur données synthétiques échangeables, assert `0.88 <= PICP <= 0.92` sur 20 graines.

- [ ] **6.4 — Domaine d'applicabilité**
  - **Action** : `eval/applicability.py` — similarité Tanimoto au plus proche voisin train,
    seuil 0.35, drapeau propagé jusqu'à la réponse d'API.
  - **Validation scientifique** : tracer la RMSE en fonction du décile de similarité au
    plus proche voisin. La courbe doit être monotone décroissante en performance ; si elle
    ne l'est pas, le domaine d'applicabilité ne mesure rien d'utile.
  - **Test de sortie** : `pytest tests/unit/test_applicability.py::test_rmse_degrades_outside_domain`
    — assert que la RMSE du décile le moins similaire dépasse celle du décile le plus similaire.

- [ ] **6.5 — Interprétabilité falsifiable (H4)**
  - **Action** : `eval/interpretability.py` — pour chaque complexe PDBbind exclu de
    l'entraînement, calculer l'AUROC de la carte d'attention contre la carte de contacts
    réelle.
  - **Validation scientifique** : comparer à deux témoins — attention aléatoire et
    attention uniforme. Un AUROC de 0.55 n'est pas de l'interprétabilité. Rapporter la
    distribution complète, pas seulement la moyenne, et exhiber les cas d'échec.
  - **Test de sortie** : `pytest tests/integration/test_interpretability.py` — assert que
    l'AUROC médian dépasse significativement 0.5 (test de Wilcoxon signé, `p < 0.05`) sur
    l'ensemble des complexes de test.

- [ ] **6.6 — Analyse d'erreur**
  - **Action** : `notebooks/05_error_analysis_by_kinase_group.ipynb` — résidus par groupe
    de Manning, par plage d'affinité, par type d'essai, par similarité au plus proche voisin.
  - **Validation scientifique** : identifier les modes d'échec systématiques (régression
    vers la moyenne aux extrémités de la plage d'affinité, sous-performance sur les groupes
    minoritaires) et les inscrire au model card.
  - **Test de sortie** : `pytest --nbmake notebooks/05_error_analysis_by_kinase_group.ipynb`.

- [ ] **6.7 — Leaderboard automatisé**
  - **Action** : `eval/leaderboard.py` + `.github/workflows/benchmark.yml` — régénération
    nocturne du tableau du README avec IC bootstrap.
  - **Validation** : les valeurs du README sont produites par le code, jamais saisies à la
    main. Aucun `TBD` ne doit subsister à la clôture de cette phase.
  - **Test de sortie** : `pytest tests/unit/test_leaderboard.py::test_readme_table_matches_metrics_json`.

---

### PHASE 7 — Serving, MLOps et publication

- [ ] **7.1 — Contrats d'API**
  - **Action** : `serving/schemas.py` — Pydantic v2, validation RDKit du SMILES,
    validation du format d'accession UniProt par expression régulière.
  - **Validation** : un SMILES invalide doit produire un HTTP 422 avec un message
    actionnable, jamais un HTTP 500 ni une prédiction silencieuse.
  - **Test de sortie** : `pytest tests/unit/test_schemas.py` — assert 422 sur SMILES
    invalide, sur accession malformée, et sur un batch de 1 001 éléments.

- [ ] **7.2 — Prédicteur et cache**
  - **Action** : `serving/predictor.py` — chargement unique au démarrage, cache mémoire des
    poches (~110 Mo pour le kinome complet), calibrateur conforme et domaine
    d'applicabilité chargés depuis les artefacts.
  - **Validation critique** : la standardisation appliquée à l'inférence doit être
    **bit-à-bit identique** à celle de l'entraînement. Un écart ici est la première cause
    de dégradation silencieuse en production pour un modèle QSAR.
  - **Test de sortie** : `pytest tests/integration/test_train_serve_parity.py` — assert que
    `predictor.predict_one(x)` reproduit la sortie du `lit_module` sur les mêmes entrées, à
    1e-5 près.

- [ ] **7.3 — Endpoints FastAPI**
  - **Action** : `/predict`, `/predict/batch`, `/profile/kinome`, `/healthz`, `/metrics`
    (Prometheus).
  - **Validation** : `/profile/kinome` renvoie le classement et l'entropie de Shannon du
    profil comme score de promiscuité. `/healthz` distingue « modèle chargé » de « service
    démarré ».
  - **Test de sortie** : `pytest tests/integration/test_api.py` — `TestClient` couvrant les
    chemins nominaux et les quatre cas d'erreur (SMILES invalide, cible inconnue, batch
    trop grand, modèle non chargé).

- [ ] **7.4 — Export ONNX et parité numérique**
  - **Action** : `scripts/export_onnx.py` — export du modèle avec axes dynamiques,
    exécution via `onnxruntime`.
  - **Validation scientifique** : la parité PyTorch ↔ ONNX doit être vérifiée sur 100
    entrées aléatoires. Un écart supérieur à 1e-4 signale un opérateur mal exporté
    (typiquement le `masked_fill` avec `-inf`, qui produit des NaN après export).
  - **Test de sortie** : `pytest tests/integration/test_onnx_parity.py` — assert
    `max|Δ| < 1e-4` sur 100 entrées.

- [ ] **7.5 — Charge et latence**
  - **Action** : scénario `locust` ou `k6` — 50 utilisateurs concurrents, 5 minutes.
  - **Validation** : cibler p95 < 200 ms sur `/predict` en CPU (poches en cache, PLM jamais
    réexécuté). Rapporter p50, p95, p99 et le taux d'erreur.
  - **Test de sortie** : `pytest tests/integration/test_load.py -m slow` — assert
    `p95 < 200 ms` et taux d'erreur nul.

- [ ] **7.6 — Chaîne Docker et publication d'image**
  - **Action** : `.github/workflows/docker.yml` — build multi-arch, scan Trivy, push GHCR,
    tag sémantique.
  - **Validation** : image `serve` < 3 Go, exécution non-root, zéro vulnérabilité HIGH ou
    CRITICAL non justifiée.
  - **Test de sortie** : workflow vert avec rapport Trivy attaché à l'artefact de build.

- [ ] **7.7 — Documentation**
  - **Action** : `mkdocs-material` — `architecture.md` (avec les compromis explicites),
    `data_card.md`, `model_card.md`, `benchmark.md`, `leakage_analysis.md`, `api.md`
    (OpenAPI). Déploiement `gh-pages`.
  - **Validation** : un lecteur externe doit pouvoir reproduire les résultats à partir de
    la seule documentation. Faire relire par un tiers, ou à défaut suivre soi-même les
    instructions depuis un clone vierge.
  - **Test de sortie** : `mkdocs build --strict` (aucun lien mort) + workflow `docs.yml` vert.

- [ ] **7.8 — Publication et archivage**
  - **Action** : `CONTRIBUTING.md`, `CITATION.cff`, tag `v0.1.0`, release GitHub, dépôt
    Zenodo pour l'obtention d'un DOI.
  - **Validation** : le README ne contient plus aucun `TBD` ; chaque chiffre est
    accompagné de son IC et reproductible par `make eval`.
  - **Test de sortie** : `pytest tests/unit/test_readme_integrity.py` — assert l'absence de
    `TBD` et de `<user>` dans `README.md`, et la validité du `CITATION.cff`.

---

## 6. Carte des fichiers par phase

*À consulter pour limiter le contexte chargé. N'ouvre pas les fichiers hors de la phase courante.*

| Phase | Fichiers à lire | Fichiers à écrire |
|---|---|---|
| 0 | `pyproject.toml`, `Makefile` | `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `docker/*`, `dvc.yaml`, `src/kinoscope/utils/seed.py` |
| 1 | `MASTER_PLAN.md` §3 | `src/kinoscope/data/{sources/*,standardize}.py`, `scripts/download_sources.py`, `docs/data_card.md` |
| 2 | `src/kinoscope/models/model.py` (constantes) | `src/kinoscope/data/sources/{uniprot,klifs}.py`, `scripts/precompute_esm_embeddings.py` |
| 3 | `src/kinoscope/models/model.py` (`*_FEATURE_DIMS`) | `src/kinoscope/data/{featurizer,dataset,datamodule,contacts}.py` |
| 4 | `src/kinoscope/data/splits.py` | `src/kinoscope/data/splits.py`, `configs/split/*.yaml`, `docs/leakage_analysis.md` |
| 5 | `src/kinoscope/models/model.py`, `configs/model/*` | `src/kinoscope/models/lit_module.py`, `scripts/train.py`, `configs/experiment/*` |
| 6 | `src/kinoscope/eval/*` | `src/kinoscope/eval/{metrics,conformal,applicability,interpretability,leaderboard}.py` |
| 7 | `src/kinoscope/serving/*` | `src/kinoscope/serving/{api,schemas,predictor,cache}.py`, `docs/*`, workflows |

---

## 7. Journal de bord

*Ajoute une ligne par étape terminée, bloquée ou clarifiée. Ne réécris jamais l'historique.*

| Date | Étape | Statut | Commit | Note |
|---|---|---|---|---|
| 2026-09-01 | — | init | — | Plan initialisé. Prochaine étape : **0.1**. |
| 2026-09-01 | 0.1 | terminé | `f352878` | Arborescence §4.2 complétée (`data/interim/`, `.gitkeep`, `tests/integration/__init__.py`) ; `LICENSE` Apache-2.0 (texte canonique récupéré sur apache.org, non retranscrit de mémoire) ; `CITATION.cff` ; auteur `<you>` retiré de `pyproject.toml`. Environnement : `uv venv --python 3.11` + `uv pip install -e ".[serve,dev]"` — torch 2.4.1+cpu, PyG 2.6.1, lightning 2.4.0, rdkit OK. Test de sortie `tests/unit/test_imports.py` : **3 passed** (parcours `pkgutil` de tous les sous-modules, sous-paquets attendus présents, import depuis un répertoire arbitraire). Falsification vérifiée : `import kinoscope` échoue en `ModuleNotFoundError` sous un interpréteur sans installation. **Réserve explicite : `make check` n'est pas vert**, sur des items hors périmètre de 0.1 — ruff 21 violations + 12 fichiers non formatés et mypy 6 erreurs (stubs pandas/scipy/sklearn/joblib absents de `[[tool.mypy.overrides]]`, plus une erreur de syntaxe dans les stubs livrés par `rdkit==2024.3.5`), tous du ressort de **0.2** ; `test_splits.py` 4 échecs (SMILES `c1cnc2ccccc2n1C` de la fixture jouet non parsable par RDKit) et couverture 60,65 % < 80 %, du ressort de **4.1**. Case cochée sur arbitrage humain explicite, la réserve étant consignée ici plutôt que masquée. |
| 2026-09-01 | 0.1 | complément | *(non commité)* | Identité auteur renseignée sur communication de l'auteur : `LICENSE` (ligne de copyright), `CITATION.cff` (`family-names`/`given-names`/`alias`, `repository-code`), `pyproject.toml` (`authors`), `README.md` (badges CI, URL de clone, entrée BibTeX). Plus aucun `<user>` ni `<you>` dans le dépôt. `affiliation`, `orcid` et le courriel restent vides : non communiqués, donc non inventés. |
| 2026-09-01 | 0.2 | terminé | `701dcf8` | **Ruff : 22 → 0.** 11 corrigées automatiquement (RUF100 sur des `noqa` visant des règles non sélectionnées — ANN, PLC ; UP037 ; SIM910). B905 corrigée à la main en `zip(..., strict=True)` et non `strict=False` : les deux `ModuleList` sont construites en parallèle, une divergence de longueur amputerait silencieusement une couche du GNN — on préfère l'erreur au calcul faux. `N812` ajoutée aux `ignore` avec justification (`torch.nn.functional as F`, `lightning as L` sont les alias conventionnels) ; `ARG002` en per-file-ignore sur `lit_module.py` (`lr`/`weight_decay`/`warmup_pct`/`mse_warmup_steps` transitent par `self.hparams`, `batch_idx` est imposé par l'API Lightning — ruff ne voit ni l'un ni l'autre). `ruff format` appliqué à 14 fichiers. **Mypy `--strict` : 50 → 0** sur 18 fichiers. Deux blocages levés d'abord : (a) `rdkit==2024.3.5` livre un paquet `rdkit-stubs` qui ne se parse pas (`rdmolfiles.pyi:272`), `ignore_missing_imports` n'y suffit pas, seul `follow_imports = "skip"` + `follow_imports_for_stubs = true` l'écarte ; (b) `kinoscope.serving.predictor` n'existe qu'à l'étape 7.2 — `# type: ignore[import-not-found]` posé sur place, que `warn_unused_ignores` signalera comme inutile dès la création du module. `pandas-stubs==2.2.2.240807` et `scipy-stubs==1.14.1.5` ajoutés aux dépendances `dev` plutôt que d'élargir `ignore_missing_imports`, conformément à la validation de l'étape ; seuls `sklearn` et `joblib` l'ont rejoint, faute de stubs publiés. `src/kinoscope/py.typed` créé. Hooks pre-commit installés (`pre-commit` + `commit-msg`) : les 12 passent. **Deux défauts réels corrigés au passage.** Le hook `mirrors-mypy` tournait dans un venv isolé privé du stack et rapportait 36 imports introuvables au lieu des vraies erreurs : remplacé par un hook `local` en `language: system`, qui rend le même verdict que `make typecheck`. `configs/config.yaml` ne se parsait pas — en style flow, les accolades de `${now:%Y-%m-%d}` sont lues comme une imbrication de mapping ; OmegaConf lui-même échouait (vérifié avant/après), donc `scripts/train.py` n'aurait pas démarré. Passé en style bloc. **`make` n'est pas installé sur cette machine** : le test de sortie a été exécuté en lançant les commandes exactes des cibles `lint` et `typecheck` du Makefile — `ruff check src tests scripts && ruff format --check src tests scripts`, puis `mypy src` — toutes deux en code 0. Suite de tests inchangée : 21 passés / 4 échoués (`test_splits.py`, ressort de 4.1), aucune régression introduite. |
| 2026-09-01 | 0.3 | terminé | *(commit suivant)* | `src/kinoscope/utils/seed.py::set_global_seed(seed, deterministic=True)` sème `random`, le générateur global hérité de numpy, `torch` et `torch.cuda`, exporte `PYTHONHASHSEED`, et pose `use_deterministic_algorithms` + `cudnn.deterministic` / `cudnn.benchmark`. Mode déterministe **strict** (pas de `warn_only`) : une opération sans noyau déterministe doit lever, sinon on rend indétectable exactement ce qu'on cherche à détecter. `CUBLAS_WORKSPACE_CONFIG=:4096:8` posé en `setdefault` avant activation, sans quoi cuBLAS lève au premier produit matriciel sur GPU. Deux limites documentées dans le module plutôt que masquées : (a) `PYTHONHASHSEED` est lu au démarrage de l'interpréteur — l'exporter depuis un processus déjà lancé ne sert qu'aux processus fils, et le test le dit explicitement ; (b) `np.random.seed` ne touche que le générateur global hérité, les `np.random.default_rng(seed)` locaux de `splits.py` et `metrics.py` en restant indépendants par construction. Trois `noqa: NPY002` justifiés sur place : semer le générateur hérité est l'objet même de la fonction et l'API `Generator` ne le permet pas ; la règle reste active partout ailleurs. Test écrit avant le code et vu échouer sur `ModuleNotFoundError: kinoscope.utils.seed`. `tests/unit/test_seed.py` : **5 passés** — forward bit-à-bit identique (`torch.equal`, pas de tolérance), graines distinctes donnant des tirages distincts (garde-fou contre une fonction qui ne sèmerait rien), flux python/numpy, export de `PYTHONHASHSEED`, bascule du drapeau déterministe. Une fixture autouse restaure l'état global de torch après chaque test : ces réglages sont propres au processus et contamineraient le reste de la session. Vérifié : `ruff` 0, `mypy --strict` 0 sur 19 fichiers, suite complète 26 passés / 4 échoués — les 4 mêmes qu'avant (`test_splits.py`, ressort de 4.1), aucune régression. |

---

## 8. Définition de « terminé » (checklist globale)

Le projet est publiable lorsque **toutes** ces conditions sont vraies :

- [ ] `make check` sort en 0 sur `main`, couverture ≥ 80 %.
- [ ] Les quatre régimes de split sont évalués, avec IC bootstrap, et le tableau du README
      est généré par le code.
- [ ] L'audit de fuite passe en CI et a été vérifié en échec sur une branche de contrôle.
- [ ] Les cinq ablations sont exécutées sur les quatre régimes.
- [ ] Le PICP conforme est mesuré et rapporté sur chaque régime, y compris là où
      l'échangeabilité est violée.
- [ ] L'AUROC attention ↔ contacts cristallographiques est rapporté avec ses témoins.
- [ ] L'image Docker `serve` démarre et répond, parité ONNX vérifiée.
- [ ] `docs/` déployé, `MODEL_CARD.md` et `docs/data_card.md` complets, limites déclarées.
- [ ] Tag `v0.1.0`, release GitHub, DOI Zenodo.
- [ ] Zéro `TBD`, zéro `<user>`, zéro chiffre non reproductible dans le dépôt.
