# Cartographie reproductible des failles géologiques

Ce dépôt rassemble le notebook de recherche sur les données GeoDAWN et un
socle PyTorch pour préparer des patches spatiaux centrés sur les failles.
L'objectif est de conserver les paramètres, les dépendances, les splits et les
tests nécessaires pour refaire les expériences.

## Organisation

- `notebook/gems_challenge.ipynb` : analyse exploratoire, expériences de
  référence et résultats du projet.
- `configs/fault_dataset.yaml` : paramètres Hydra du Dataset, du seed et du
  tracking MLflow local.
- `dataops/pretreatment.py` : dilatation des labels, création des blocs et
  répartition spatiale en folds.
- `dataops/dataset_torch.py` : création du Dataset de patches et des
  DataLoaders PyTorch.
- `src/` : chargement raster, métrique DTI, visualisations et seeds.
- `src/data.py` et `src/metrics.py` : chargement raster et métrique DTI utilisés
  par le notebook.
- `tests/` : tests unitaires du Dataset, des splits et des seeds.
- `data/raw/` : données sources GeoDAWN, à fournir localement.
- `data/processed/` : caches et résultats dérivés; ils ne sont pas nécessaires
  au dépôt et ne doivent pas être versionnés.
- `submissions/` : livrables et figures de soumission.

## Refaire l'environnement et les vérifications

Installer Python 3.14 et [uv](https://docs.astral.sh/uv/), puis depuis la
racine du dépôt :

```powershell
uv sync --locked
uv run python -m unittest discover -s tests -v
```

Placer les rasters `gems-geodawn-numerical-features.tif` et
`existing_faults.tif` dans `data/raw/`, puis ouvrir
`notebook/gems_challenge.ipynb` dans VS Code et sélectionner l'interpréteur
Python du projet (`.venv`).

Les étapes coûteuses du notebook réutilisent des caches dans `data/processed/`.
Pour recalculer une étape après avoir changé ses entrées ou sa méthode, retirer
le cache correspondant avant de relancer le notebook.

Le seed `seed` de la configuration Hydra pilote l'affectation des folds du
Dataset; les DataLoaders acceptent le même seed pour répéter l'ordre des
échantillons d'entraînement. `seed_everything` synchronise les générateurs
Python, NumPy et PyTorch. La reproductibilité numérique exacte peut néanmoins
varier selon le matériel et les versions des bibliothèques, en particulier sur
GPU.

**Périmètre actuel :** le dépôt fournit la préparation du Dataset de patches,
les splits spatiaux et les visualisations PyTorch, mais pas encore de boucle
d'entraînement ou d'évaluation CNN. Le notebook existant documente les
expériences de modèles classiques; le socle CNN doit être ajouté avant de
prétendre reproduire un entraînement CNN complet.

MLflow est configuré pour un suivi local dans `.mlflow.db`; cette base et les
artefacts générés restent locaux.

---

## Dataset PyTorch de patches spatiaux

`dataops.dataset_torch` fournit un `Dataset` d'un patch par groupe de
faille connecté, ainsi que des splits spatiaux train/validation/test. Le
notebook utilise Hydra via `configs/fault_dataset.yaml`; `KERNEL_SIZE` règle
la dilatation des failles et `seed` contrôle le clustering et les mélanges des
DataLoaders. Les arguments fixes du Dataset, dont `_target_`, sont dans la
section `dataset`. MLflow est réservé au futur entraînement: la
création/visualisation du Dataset ne démarre aucun run et n'est pas enregistrée.

```python
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from dataops.dataset_torch import SpatialFoldSplit, make_fault_dataloaders
from src.reproducibility import seed_everything
from src.torch_viz import plot_fault_overview, plot_fault_sample
import matplotlib.pyplot as plt

with initialize_config_dir(
    config_dir=str((Path.cwd() / "configs").resolve()), version_base=None
):
    cfg = compose(config_name="fault_dataset")
KERNEL_SIZE = int(cfg.KERNEL_SIZE)
seed_everything(int(cfg.seed))

split = SpatialFoldSplit(
    train_folds=tuple(cfg.folds.train),
    validation_fold=int(cfg.folds.validation),
    test_fold=int(cfg.folds.test),
)
dataset = instantiate(
    cfg.dataset,
    features=X,
    raw_labels=Y_RAW,
    transform=tuple(src_x.transform),
)
Y_blocks = dataset.Y_blocks
fold_map_consensus = dataset.fold_map_consensus
loaders = make_fault_dataloaders(dataset, split, batch_size=16, seed=int(cfg.seed))

sample_fig, _ = plot_fault_sample(dataset, index=0, band_names=BAND_NAMES)
overview_fig, _ = plot_fault_overview(dataset, index=0, stride=4)
plt.show()
```

Chaque élément contient `features` `(19, patch_size, patch_size)`, `target`
`(1, patch_size, patch_size)` pour le seul groupe courant et
`supervision_mask` de même forme. La cible positive est la faille originale
fine (`Y_RAW == 1`), pas son buffer. La supervision couvre uniquement la zone
du groupe courant dans `Y_buff`: les positifs originaux et les zéros connus
dans ce buffer; aucun pixel n'est ajouté par une seconde dilatation. Les
pixels inconnus et les autres groupes/folds sont ignorés par la loss. Les
19 bandes sont standardisées séparément avec la moyenne et l'écart-type
calculés uniquement sur les pixels finis appartenant aux folds listés dans
`folds.train` dans `fold_map_consensus` (la carte ne couvre que les blocs
dilatés). Le même scaler est appliqué aux patches train, validation et test.
Les valeurs non finies après standardisation sont remplacées par `0.0` dans
`features` et signalées dans `feature_validity`; les bords raster sont remplis
par réplication et signalés comme invalides.

La section `folds` de la configuration fixe les folds d'entraînement, de
validation et de test. Pour une autre rotation de validation croisée, mettre à
jour ces IDs ensemble. Les DataLoaders filtrent les groupes par fold et
mélangent uniquement l'entraînement.

Les groupes sont créés par composantes connexes à 8 voisins sur `Y_buff`.
Pour chaque groupe, le clustering utilise le centroïde géographique, la moyenne
des 19 covariables et la moyenne de `Y_RAW`. Ces trois descripteurs sont
imputés par médiane et standardisés, puis clusterisés chacun par K-Means
(`n_init=20`, seed 0). Leurs co-associations alimentent un `SpectralClustering`
à 4 folds (`affinity="precomputed"`, `assign_labels="cluster_qr"`), comme dans
le notebook.

`plot_fault_sample` affiche toutes les bandes avec des contrastes par quantiles,
puis le masque de cible, les zéros fiables, le masque supervisé et la fraction
de covariables valides. `plot_fault_overview` situe le groupe sélectionné sur
la carte des folds; `stride` contrôle la décimation de l'affichage.

La section `mlflow` du YAML réserve le tracking local sur SQLite (`.mlflow.db`)
à l'étape d'entraînement. Les runs et métriques MLflow seront créés autour de
l'entraînement, pas lors de la préparation des données.