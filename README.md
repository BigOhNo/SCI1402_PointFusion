# SCI1402_PointFusion
Projet SCI1402 – Prédiction du point de fusion de molécules organiques. Thermophysical Property: Melting Point https://www.kaggle.com/competitions/melting-point/overview

Le projet s'attaque à un défi de longue date en chimie : la prédiction du point de fusion de molécules
organiques à l'aide de l'apprentissage automatique (machine learning).

Ce projet s’inscrit dans le domaine de la **chimie computationnelle** et de la **modélisation prédictive**.
Il s’agit d’un problème de **régression supervisée** : prédire le point de fusion (en kelvins) de molécules
organiques à partir de descripteurs structuraux dérivés de leur formule SMILES (Simplified Molecular Input
Line Entry System).

Les enjeux sont à la fois **scientifiques** et **industriels** :
- Le point de fusion est une propriété physique critique pour la conception de médicaments, le développement
  de matériaux et la sécurité des procédés chimiques.
- Les mesures expérimentales sont coûteuses, longues et parfois impossibles à réaliser (ex. : composés
  instables).
- Un modèle prédictif fiable permettrait d’accélérer la découverte de nouvelles molécules tout en réduisant
  les coûts et les risques expérimentaux.

## Analyse du contexte.

Actuellement, la mesure expérimentale du point de fusion des molécules est souvent coûteuse, lente à réaliser
ou tout simplement impossible à obtenir pour certaines substances. Ce projet s'inscrit donc dans un contexte
où il existe un besoin pressant pour des méthodes alternatives, plus rapides et économiques.

L'approche proposée ici est de tirer parti de l'apprentissage automatique pour créer des modèles prédictifs.
Ces modèles utiliseront des "group contribution features", qui sont des décomptes de sous-groupes représentant
les groupes fonctionnels au sein de chaque molécule, pour prédire leur comportement thermique. L'objectif est
de voir si des modèles informatiques peuvent généraliser leurs prédictions à travers diverses familles de
produits chimiques et ainsi repousser les limites de la prédiction de propriétés à partir de données.

Le jeu de données provient de la compétition Kaggle **"Melting Point Prediction Challenge"**. Il contient :
- **3328 composés organiques** au total,
- **2662 échantillons d’entraînement** avec la variable cible `Tm` (point de fusion en K),
- **666 échantillons de test** sans la cible,
- Des **descripteurs de type "group contribution"**, représentant le nombre d’occurrences de sous-groupes
  fonctionnels dans chaque molécule,
- Une colonne `SMILES` encodant la structure moléculaire sous forme textuelle standardisée.

Ce type de représentation moléculaire est couramment utilisé en chimiométrie pour relier structure et
propriétés physico-chimiques.

## Définition des objectifs et hypothèses.

### Objectifs
1. **Développer un modèle de régression** capable de prédire précisément le point de fusion à partir des
   descripteurs fournis.
2. **Comparer plusieurs approches** (régression linéaire, forêts aléatoires, XGBoost, réseaux de neurones)
   en termes de performance (MAE) et de robustesse.
3. **Évaluer l’apport potentiel** de l’encodage SMILES (ex. : via des empreintes moléculaires) en complément
   des descripteurs existants.

### Hypothèses

Hypothèse statistique
- Les descripteurs "group contribution" contiennent suffisamment d’information pour capturer les tendances structure-propriété.

H₀ (hypothèse nulle) : Les descripteurs fournis (« Group 1…N » et éventuellement SMILES) n’ont pas de relation prédictive significative avec le point de fusion (Tm). Autrement dit, aucun modèle entraîné sur ces variables ne peut prédire Tm mieux qu’un prédicteur naïf (ex. : prédiction de la moyenne ou de la médiane de Tm sur l’ensemble d’entraînement ou le hasard).

Hₐ (hypothèse alternative) : Les descripteurs fournis contiennent suffisamment  d’information pour permettre à un modèle d’apprentissage automatique de prédire le point de fusion (Tm) avec une erreur absolue moyenne (MAE) significativement inférieure à celle d’un prédicteur naïf.

## Description sommaire de la méthodologie

1. **Exploration et visualisation des données** : analyse descriptive, détection d’anomalies, distribution de
 `Tm`.
2. **Prétraitement** : gestion des valeurs manquantes (si présentes), normalisation/standardisation si
 nécessaire.
3. **Ingénierie de features** (optionnelle) : extraction de descripteurs supplémentaires à partir de `SMILES`
 (ex. : poids moléculaire, polarité, etc.).
4. **Entraînement de modèles** :
   - Modèles de base : régression linéaire, k-plus proches voisins.
   - Modèles avancés : Random Forest, XGBoost.

Le code sera implémenté en **Python**, en utilisant les bibliothèques `pandas`, `scikit-learn`, `xgboost`, et
 éventuellement `rdkit` pour l’analyse chimique. L’accent sera mis sur la **reproductibilité**, la **clarté du
 code** et la **documentation** des étapes. 

