# information personnelle
# IKRAM AZIZI CAC1 
<img src="th (3).jpg" style="height:250px;margin-right:432px"/>


# Dataset Heart Disease (donné le 30/06/1988)

## Contexte général
Ce dataset regroupe des données médicales de patients examinés pour la détection de maladies cardiaques coronariennes. Il est constitué de quatre bases de données provenant de différentes régions :
- Cleveland, USA
- Hongrie
- Suisse
- VA Long Beach, USA

La base Cleveland est la plus utilisée en apprentissage automatique pour la prédiction de la maladie.

## Population concernée
- Patients examinés principalement dans les années 1980 (1981-1987 selon la base)
- Patients suspects de maladie coronarienne, sans antécédents cardiaques graves connus au départ
- Total env. 1000 patients combinés dans les 4 bases

## Données et caractéristiques principales
- 76 attributs initiaux, dont un sous-ensemble de 14, utilisés pour les études
- Variables démographiques, cliniques et de tests cardiaques (ex : âge, sexe, douleur thoracique, pression artérielle, cholestérol, fréquence cardiaque maximale)
- Variable cible : présence de maladie cardiaque codée de 0 (absent) à 4 (présence avec différenciation)
- Données anonymisées (suppression des noms et numéros de sécurité sociale)

## Variables utilisées (extrait)
| Variable           | Description                                   | Type          |
|--------------------|-----------------------------------------------|---------------|
| age                | Âge en années                                 | Entier        |
| sex                | Sexe (1 = homme, 0 = femme)                   | Catégoriel    |
| cp                 | Type de douleur thoracique                     | Catégoriel    |
| trestbps           | Pression artérielle au repos (mm Hg)          | Entier        |
| chol               | Cholestérol sérique (mg/dl)                    | Entier        |
| fbs                | Glycémie à jeun > 120 mg/dl (1 = vrai, 0 = faux) | Catégoriel |
| restecg            | Résultat ECG au repos                          | Catégoriel    |
| thalach            | Fréquence cardiaque maximale atteinte         | Entier        |
| exang              | Angine induite par l'exercice (1 = oui, 0 = non) | Catégoriel  |
| oldpeak            | Dépression ST induite par exercice             | Réel          |
| slope              | Pente segment ST à l'effort                     | Catégoriel    |
| ca                 | Nombre de vaisseaux majeurs atteints           | Entier        |
| thal               | Type de défaut thallium                         | Catégoriel    |
| num (cible)        | Diagnostic de maladie cardiaque                  | Entier (0-4)  |

## Objectif scientifique
- Utilisation par des chercheurs pour développer des modèles de classification machine learning 
- Prédiction binaire courante : présence (num>0) ou absence (num=0) de maladie coronarienne
- Publication de l’algorithme probabiliste international par Detrano et al. (1989)

## Sources historiques
- Données recueillies Mai 1981 à Septembre 1987 selon les bases
- Publication clé : "International application of a new probability algorithm for the diagnosis of coronary artery disease", 1989

---

## Références
- Dataset original et descriptions : UCI Machine Learning Repository
- Études pousées sur Cleveland dataset en ML
- Publication Detrano et al., American Journal of Cardiology, 1989
