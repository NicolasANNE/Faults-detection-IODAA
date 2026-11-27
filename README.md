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
- `configs/model/` : hyperparamètres des architectures Hydra.
- `dataops/pretreatment.py` : dilatation des labels, création des blocs et
  répartition spatiale en folds.
- `dataops/dataset_torch.py` : création du Dataset de patches et des
  DataLoaders PyTorch.
- `models/` : architectures PyTorch, dont `Unet_base`.
- `src/` : entraînement, inférence raster, métrique DTI, visualisations et seeds.
- `src/data.py` et `src/metrics.py` : chargement raster et métrique DTI utilisés
  par le notebook.
- `tests/` : tests unitaires du Dataset, des splits, des modèles et de l'inférence.
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

**Périmètre actuel :** le dépôt fournit la préparation reproductible du Dataset,
les splits spatiaux, l'entraînement CNN configurable, la sélection par DTI et
l'inférence raster géoréférencée. Le notebook existant documente aussi les
expériences de modèles classiques.

MLflow est configuré pour un suivi local dans `mlflow.db`; cette base et les
artefacts générés restent locaux.

---

## Dataset PyTorch de patches spatiaux

`dataops.dataset_torch` fournit un `Dataset` d'un patch par groupe de
faille connecté, ainsi que des splits spatiaux train/validation/test. Le
notebook utilise Hydra via `configs/fault_dataset.yaml`; `KERNEL_SIZE` règle
la dilatation des failles et `seed` contrôle le clustering et les mélanges des
DataLoaders. Les arguments fixes du Dataset, dont `_target_`, sont dans la
section `dataset`. La création/visualisation du Dataset ne démarre aucun run
MLflow; le suivi des expériences est effectué par `src.trainer`.

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

La section `mlflow` du YAML réserve le tracking local sur SQLite (`mlflow.db`)
à l'étape d'entraînement. Les runs et métriques MLflow seront créés autour de
l'entraînement, pas lors de la préparation des données. Depuis la racine du
workspace, `mlflow ui` ouvre directement ce même store.

## Entraîner un modèle de segmentation

`src.trainer` lance une expérience configurée dans `configs/train.yaml`. Les
architectures sont choisies par le groupe Hydra `configs/model/`; le modèle
par défaut `Unet_base` reçoit `(B, 19, patch_size, patch_size)` et retourne des
logits de forme `(B, 1, patch_size, patch_size)`. Les masques de validité ne
sont pas des entrées du réseau.

La loss d'entraînement par défaut est l'hybride
`0.7 * BCE masquée + 0.3 * (1 - sDTI)` (`MaskedBCEWithDTILoss` dans
`src/metrics.py`). La BCE ne considère que les pixels de `supervision_mask`;
les poids `training.criterion.bce_weight` et `training.criterion.dti_weight`
sont surchargeables via Hydra; si `dti_weight` n'est pas précisé, il vaut
`1 - bce_weight`, ce qui permet de balayer le poids BCE seul. Pour utiliser
uniquement la BCE, mettre le poids BCE à `1`. Le DTI global exact de validation/test
reste calculé sur la carte reconstruite et les pixels supervisés; il sélectionne
le meilleur checkpoint et mesure le test.

Le checkpoint retenu maximise le DTI sur la carte de validation reconstruite
depuis les origines de patches. Le test n'est évalué qu'une fois, après
restauration du meilleur checkpoint. Les paramètres et checkpoints sont suivis
dans MLflow local; les poids sont également enregistrés sous
`checkpoints/<mlflow-run-id>/`.

### DTI sur l'étendue USGS

En plus des métriques historiques calculées sur les pixels supervisés des
patches, les runs calculent deux DTI sur les probabilités prédites sur toute la
carte (`predict_full_map`):

- `evaluation/spatial_holdout_dti` est le score principal. La région est une
  tuile géographique déterministe, définie par
  `evaluation.spatial_holdout.grid_rows`, `grid_columns` et `region_index`
  (indexation ligne par ligne depuis le coin supérieur gauche). Les groupes
  dont les labels ou les patches chevauchent cette tuile sont retirés des
  ensembles train et validation; le scaler n'utilise pas non plus les pixels
  de la tuile. Le DTI compte tous les pixels valides de la tuile, pas seulement
  les patches autour des failles.
- `evaluation/usgs_full_map_dti` mesure l'accord avec toute la carte
  `existing_faults.tif`. C'est un diagnostic descriptif, pas une estimation
  indépendante de généralisation, car des labels de cette carte sont utilisés
  pendant l'entraînement.

Les deux scores utilisent les pixels où le template de soumission est valide
et `existing_faults.tif` est fini. Les zéros valides sont comptés comme absence
de faille cartographiée par l'USGS; ils ne garantissent pas une absence
géologique réelle. Les valeurs NoData sont ignorées. Les comptes de pixels
évalués sont également journalisés dans MLflow. Le holdout par défaut est la
tuile supérieure gauche d'une grille 2×2; changer `region_index` pour évaluer
une autre tuile. Si elle ne contient aucune faille USGS, le run s'arrête avec
une erreur explicite plutôt que de produire un score trompeur.
L'exclusion peut aussi laisser trop peu de patches train ou validation; dans
ce cas, choisir une autre tuile (`evaluation.spatial_holdout.region_index`) ou
une grille plus fine (`grid_rows` / `grid_columns`).

Ces métriques ne reproduisent pas le DTI du leaderboard : celui-ci utilise des
labels experts privés de failles nouvelles, tandis que ces scores mesurent
l'accord avec l'inventaire USGS actuellement disponible.

## Recherche d'hyperparamètres avec Optuna

Lancer la recherche depuis la racine du dépôt :

```powershell
uv run python -m src.optuna_tuning
```

Le study Optuna persistant (`optuna_studies.db`) maximise `validation/dti` sur
le split courant, avec 50 essais au maximum. Il explore les paramètres de
`Unet_base` (`base_channels`, `dropout`, `activation`) et d'entraînement
(`learning rate`, `weight_decay`, `batch_size`, `bce_weight`). `dti_weight`
reste dérivé de `1 - bce_weight`. Chaque essai s'arrête après 10 époques sans
amélioration du DTI de validation, avec un plafond de sécurité de 300 époques.
Le checkpoint au meilleur DTI validation est conservé.

Les essais n'évaluent pas le fold test ni le DTI holdout. Après sélection, le
meilleur réglage est réentraîné depuis zéro sur les folds train et validation,
pendant le nombre d'époques du meilleur checkpoint Optuna. Le fold test est
alors évalué une seule fois; le DTI du holdout spatial est calculé ensuite,
comme résultat final. Le scaler du refit utilise train+validation et exclut
toujours la région holdout. Les scores de validation ont servi à optimiser
plusieurs essais et ne sont donc pas une estimation indépendante. Aucun
réglage supplémentaire ne doit être fait à partir des scores test ou holdout.

Limite du découpage actuel : l'affectation consensus des folds est construite
sur les groupes de toute la carte et utilise aussi leur moyenne de labels
bruts. Les groupes qui touchent le holdout et leurs patches sont exclus du
train/validation, mais les labels de cette zone participent encore à la
construction de la carte des folds. Le DTI holdout n'est jamais utilisé par
Optuna; ce protocole n'est toutefois pas entièrement aveugle aux labels du
holdout lors de la seule construction des partitions.

Les métriques par époque des essais sont suivies dans MLflow; l'étude est
stockée séparément de MLflow pour permettre de reprendre les essais. Une fois
le protocole entré en évaluation finale, une nouvelle exécution avec le même
nom d'étude est refusée pour éviter de répéter les scores test/holdout.

```powershell
uv run python -m src.trainer `
  model.hyperparameters.dropout=0.15 `
  model.hyperparameters.base_channels=32 `
  training.epochs=50 training.batch_size=16 `
  training.optimizer.lr=0.0005
```

Chaque fichier `configs/model/<nom>.yaml` expose les hyperparamètres sous
`model.hyperparameters` et pointe vers une classe telle que
`models.Unet_base`. Les overrides `model=<nom>`,
`model.hyperparameters.*`, `data.*`, `dataset.*`, `training.*`, `folds.*` et
`evaluation.*`, `mlflow.*` permettent de comparer les expériences. Chaque run
conserve sa configuration effective comme artefact `config.yaml`. Le nom de
l'expérience reste stable; chaque run reçoit un nom aléatoire lisible du type
`monkey-banana-6432823`.

Les entrées natives MLflow utilisent le nom de dataset
`Fault_KS<KERNEL_SIZE>_Patched` et le nom de modèle issu de la classe (par
exemple `Unet_base`). Le dataset référence les rasters source et sa
configuration sans téléverser une deuxième copie des tableaux. Le modèle
enregistré correspond au meilleur checkpoint de validation.

L'artefact `figures/training_curves.png` compare les losses train/validation
et le DTI de validation par epoch. D'autres figures utiles à ajouter ensuite
seraient une superposition prédiction/vérité terrain sur le test ou une carte
des folds spatiaux.

Après l'entraînement, une heatmap de probabilité de la carte complète est
générée par défaut et journalisée comme artefact MLflow sous `figures/`.
Désactiver avec `inference.heatmap_enabled=false`. Le GeoTIFF de soumission est
également généré par défaut et journalisé sous `predictions/`. Il reprend la
grille du raster source et le masque NoData du template
`data/raw/example_submission.tif` : EPSG:32611, résolution 100 m, une bande
`float32`, probabilités dans `[0, 1]` et NoData `NaN`. Les fichiers locaux sont
isolés par run sous `predictions/<run-id>/`. L'export utilise des fenêtres
`patch_size`, un pas `patch_size // 2` et une pondération gaussienne centrée.