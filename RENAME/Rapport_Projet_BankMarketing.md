# Rapport d'Analyse Prédictive : Optimisation des Campagnes de Dépôts à Terme

## Contexte Métier & Problématique

Le secteur bancaire est caractérisé par une forte concurrence et des coûts d'acquisition client significatifs. Les campagnes de marketing direct, bien que potentiellement efficaces, peuvent s'avérer coûteuses et peu rentables si elles ne sont pas ciblées. L'objectif principal de ce projet est d'optimiser les campagnes de marketing direct pour la souscription à des dépôts à terme.

L'enjeu économique pour la banque est double :
*   **Réduction des coûts de marketing :** En identifiant les clients les plus susceptibles de souscrire, la banque peut concentrer ses efforts et ses ressources sur ces segments, évitant ainsi de contacter des clients peu intéressés.
*   **Augmentation du Retour sur Investissement (ROI) :** Un ciblage plus précis conduit à un taux de conversion plus élevé, maximisant ainsi les revenus générés par les dépôts à terme.

La problématique se traduit par la construction d'un modèle de Machine Learning capable de prédire la probabilité qu'un client souscrive à un dépôt à terme, en se basant sur des données historiques de campagnes marketing.

La cible de notre prédiction est la variable `'y'`, qui indique si le client a souscrit (valeur 'yes') ou non (valeur 'no') à un dépôt à terme suite à la dernière campagne de marketing direct. Il s'agit donc d'un problème de classification binaire.

## Description du Dataset

Le dataset "Bank Marketing" contient des informations sur les clients d'une institution bancaire et les résultats de campagnes de marketing direct. Il est couramment utilisé pour prédire la souscription à un dépôt à terme.

**Volume des données :** Le jeu de données complet contient généralement environ 40 000 à 45 000 observations, chacune représentant un contact client.

**Type de problème :** Classification binaire.

**Description des features :**

*   **Informations client :**
    *   `age` (numérique) : Âge du client.
    *   `job` (catégorielle) : Type d'emploi (e.g., 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown').
    *   `marital` (catégorielle) : Statut marital (e.g., 'married', 'single', 'divorced', 'unknown').
    *   `education` (catégorielle) : Niveau d'éducation (e.g., 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown').
    *   `default` (catégorielle) : A un crédit en défaut ? (e.g., 'no', 'yes', 'unknown').
    *   `balance` (numérique) : Solde annuel moyen du compte en euros (nouveau feature dans certaines versions du dataset).
    *   `housing` (catégorielle) : A un prêt immobilier ? (e.g., 'no', 'yes', 'unknown').
    *   `loan` (catégorielle) : A un prêt personnel ? (e.g., 'no', 'yes', 'unknown').
*   **Informations sur le dernier contact de la campagne actuelle :**
    *   `contact` (catégorielle) : Type de communication de contact (e.g., 'cellular', 'telephone', 'unknown').
    *   `day` (numérique) : Dernier jour du mois de contact.
    *   `month` (catégorielle) : Dernier mois de contact (e.g., 'jan', 'feb', 'mar', ..., 'nov', 'dec').
    *   `duration` (numérique) : Durée du dernier contact en secondes (très importante, mais pose un problème de fuite de données - voir section "Data Leakage").
*   **Autres attributs de la campagne :**
    *   `campaign` (numérique) : Nombre de contacts effectués pendant cette campagne pour ce client (y compris le dernier contact).
    *   `pdays` (numérique) : Nombre de jours depuis le dernier contact d'une campagne précédente (-1 si le client n'a jamais été contacté auparavant).
    *   `previous` (numérique) : Nombre de contacts effectués avant cette campagne pour ce client.
    *   `poutcome` (catégorielle) : Résultat de la campagne précédente (e.g., 'failure', 'nonexistent', 'success').
*   **Variable cible :**
    *   `y` (binaire) : Le client a-t-il souscrit un dépôt à terme ? (e.g., 'yes', 'no').

## Pipeline d'Acquisition & Préparation (ETL)

La phase d'ETL (Extraction, Transformation, Chargement) est cruciale pour garantir la qualité et la pertinence des données pour l'entraînement du modèle.

### Stratégie de Nettoyage

*   **Valeurs manquantes :** Le dataset utilise fréquemment la chaîne "unknown" pour représenter les valeurs manquantes dans les variables catégorielles.
    *   Pour les variables comme `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, nous traiterons "unknown" comme une catégorie à part entière. Cela permet au modèle d'apprendre si l'absence d'information est elle-même prédictive.
    *   Pour `pdays`, la valeur -1 indique que le client n'a pas été contacté auparavant. Nous pourrions la traiter comme une catégorie ou la transformer, mais pour ce rapport, nous la laisserons telle quelle, le modèle pouvant l'interpréter.
*   **Outliers :** Les variables numériques comme `age`, `balance`, `campaign`, `duration`, `pdays`, `previous` peuvent contenir des outliers. Une analyse exploratoire approfondie serait nécessaire pour décider de leur traitement (capping, transformation log, etc.). Pour ce rapport, nous nous concentrerons sur les aspects les plus critiques.

### Encodage des Variables Catégorielles

Les algorithmes de Machine Learning ne peuvent pas traiter directement les variables catégorielles. Elles doivent être converties en format numérique.

*   **One-Hot Encoding :** C'est la méthode privilégiée pour les variables catégorielles nominales (sans ordre intrinsèque), telles que `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`. Chaque catégorie unique est transformée en une nouvelle colonne binaire (0 ou 1). Cela évite d'introduire une relation d'ordre arbitraire que le modèle pourrait mal interpréter.
*   **Label Encoding :** Peut être utilisé pour les variables ordinales (avec un ordre intrinsèque, e.g., 'small', 'medium', 'large'). Cependant, le dataset "Bank Marketing" ne contient pas de variables ordinales évidentes qui bénéficieraient de cette approche sans risque d'introduire un biais. Nous nous en tiendrons au One-Hot Encoding pour la majorité des variables catégorielles.
*   **Variable Cible (`y`) :** La variable cible 'y' ('yes'/'no') sera transformée en 1/0.

### Gestion du Déséquilibre de Classe (Class Imbalance)

Le dataset "Bank Marketing" est notoirement déséquilibré. La proportion de clients ayant souscrit un dépôt à terme ('yes') est généralement faible (autour de 10-12%) par rapport à ceux qui n'ont pas souscrit ('no'). Un tel déséquilibre peut biaiser le modèle, le rendant excellent pour prédire la classe majoritaire mais médiocre pour la classe minoritaire.

Plusieurs stratégies peuvent être employées :
*   **Pondération des classes (`class_weights`) :** De nombreux algorithmes (comme `RandomForestClassifier` ou `XGBoost`) permettent d'assigner des poids différents aux classes lors de l'entraînement. La classe minoritaire reçoit un poids plus élevé, ce qui force le modèle à lui accorder plus d'attention. C'est une méthode simple et efficace, que nous utiliserons dans notre code.
*   **Sur-échantillonnage de la classe minoritaire (Oversampling) :** Des techniques comme SMOTE (Synthetic Minority Over-sampling Technique) génèrent des échantillons synthétiques de la classe minoritaire pour équilibrer le dataset.
*   **Sous-échantillonnage de la classe majoritaire (Undersampling) :** Réduit le nombre d'échantillons de la classe majoritaire. Moins recommandé car il entraîne une perte d'information.
*   **Changement de métrique :** Se concentrer sur des métriques robustes au déséquilibre comme Precision, Recall, F1-Score, et AUC-ROC plutôt que l'Accuracy.

## Code Python Complet & Exécutable

Ce bloc de code Python illustre l'intégralité du pipeline, de l'acquisition des données à l'évaluation du modèle.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

# --- 1. Chargement des données ---
# Assurez-vous que le fichier 'bank-additional-full.csv' est dans le même répertoire
# ou spécifiez le chemin complet.
try:
    df = pd.read_csv('bank-additional-full.csv', sep=';')
    print("Dataset chargé avec succès.")
except FileNotFoundError:
    print("Erreur: Le fichier 'bank-additional-full.csv' n'a pas été trouvé.")
    print("Veuillez vous assurer que le fichier est dans le bon répertoire ou spécifier le chemin correct.")
    exit() # Quitte le script si le fichier n'est pas trouvé

# --- 2. Pré-traitement initial et gestion de la fuite de données ---

# Renommer la variable cible pour plus de clarté si nécessaire
df.rename(columns={'y': 'subscribed'}, inplace=True)

# Conversion de la variable cible en numérique (0 et 1)
df['subscribed'] = df['subscribed'].map({'no': 0, 'yes': 1})

# Analyse critique et suppression de la variable 'duration' (Data Leakage)
# La durée du contact n'est connue qu'APRÈS le contact, donc elle ne peut pas être utilisée
# pour prédire si un client va souscrire AVANT de le contacter.
print("\nSuppression de la variable 'duration' pour éviter la fuite de données.")
df.drop('duration', axis=1, inplace=True)

# Identification des variables catégorielles et numériques
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('subscribed')

# --- 3. Séparation des données en ensembles d'entraînement et de test ---
X = df.drop('subscribed', axis=1)
y = df['subscribed']

# Utilisation de stratify pour maintenir la proportion de la variable cible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nProportion de souscription dans l'ensemble d'entraînement: {y_train.mean():.2f}")
print(f"Proportion de souscription dans l'ensemble de test: {y_test.mean():.2f}")

# --- 4. Pipeline de pré-traitement ---

# Création des transformateurs pour les variables numériques et catégorielles
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # 'handle_unknown' pour les nouvelles catégories dans le test set

# Création du préprocesseur qui applique les transformateurs aux colonnes appropriées
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 5. Définition et entraînement du modèle ---

# Utilisation de RandomForestClassifier avec class_weight='balanced' pour gérer le déséquilibre de classe
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'))])

print("\nEntraînement du modèle RandomForestClassifier...")
model.fit(X_train, y_train)
print("Modèle entraîné avec succès.")

# --- 6. Évaluation du modèle ---

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilités pour la classe positive (1)

print("\n--- Rapport de Classification ---")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("\n--- Matrice de Confusion ---")
print(cm)

# Visualisation de la Matrice de Confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non souscrit', 'Souscrit'],
            yticklabels=['Non souscrit', 'Souscrit'])
plt.xlabel('Prédiction')
plt.ylabel('Vraie Valeur')
plt.title('Matrice de Confusion')
plt.show()

# Courbe ROC et AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\nAUC-ROC: {roc_auc:.4f}")

# Visualisation de la Courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()

# --- 7. Analyse des importances des caractéristiques (pour RandomForest) ---
# Nécessite d'extraire les noms des features après One-Hot Encoding
try:
    ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = list(numerical_features) + list(ohe_feature_names)
    
    # Assurez-vous que le nombre de features correspond aux importances
    if len(all_feature_names) == len(model.named_steps['classifier'].feature_importances_):
        feature_importances = pd.Series(model.named_steps['classifier'].feature_importances_, index=all_feature_names)
        top_features = feature_importances.nlargest(15) # Top 15 des features
        
        print("\n--- Top 15 des Importances des Caractéristiques ---")
        print(top_features)

        plt.figure(figsize=(10, 7))
        sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
        plt.title('Top 15 des Importances des Caractéristiques')
        plt.xlabel('Importance')
        plt.ylabel('Caractéristique')
        plt.tight_layout()
        plt.show()
    else:
        print("\nAvertissement: Impossible de mapper les importances des caractéristiques aux noms originaux.")
        print("Nombre de features après preprocessing:", len(all_feature_names))
        print("Nombre d'importances du classifieur:", len(model.named_steps['classifier'].feature_importances_))

except Exception as e:
    print(f"\nErreur lors de l'extraction des importances des caractéristiques: {e}")

```

## Analyses Détaillées

### Data Leakage : La variable 'duration'

La variable `duration` (durée du dernier contact en secondes) est un cas classique et critique de fuite de données (Data Leakage).

*   **Problème :** La valeur de `duration` n'est connue qu'à la fin de l'appel téléphonique. Si un modèle utilise cette information pour prédire si un client va souscrire *avant* l'appel (ou au début de l'appel), il utilise une information qui ne serait pas disponible en situation réelle de prédiction. Par exemple, si un appel dure 0 secondes, le client n'a évidemment pas souscrit. Un modèle verrait cette corrélation parfaite et l'exploiterait, mais cela ne serait pas utile pour cibler des clients *avant* de les appeler.
*   **Conséquence :** L'inclusion de `duration` gonflerait artificiellement les performances du modèle (Accuracy, AUC, etc.) sur les données de test, donnant une fausse impression de son efficacité. En production, un tel modèle serait inutilisable pour la prédiction ex-ante.
*   **Recommandation :** Il est impératif de supprimer la variable `duration` du jeu de données avant l'entraînement du modèle pour construire un modèle réaliste et déployable. C'est ce qui a été fait dans le code Python ci-dessus.

### Corrélations

L'analyse des corrélations permet de comprendre les relations entre les variables et d'identifier d'éventuels problèmes de multicolinéarité ou des features fortement liées à la cible.

*   **Variables numériques :** Une matrice de corrélation (heatmap) des variables numériques peut révéler des relations. Par exemple, `campaign` (nombre de contacts dans la campagne actuelle) et `previous` (nombre de contacts dans les campagnes précédentes) pourraient être faiblement corrélées, mais `pdays` et `previous` sont souvent liées (un grand `pdays` signifie souvent un `previous` faible ou nul).
*   **Variables catégorielles et cible :** Des tests statistiques (Chi-2) ou des visualisations (comptages par catégorie) sont nécessaires pour évaluer la relation entre les variables catégorielles et la variable cible. Par exemple, les étudiants (`job=student`) ou les retraités (`job=retired`) ont souvent un taux de souscription plus élevé. Le résultat de la campagne précédente (`poutcome=success`) est un prédicteur très fort de la souscription actuelle.

### Variance

L'analyse de la variance et de la distribution des données est essentielle pour comprendre la nature de chaque feature et guider le pré-traitement.

*   **Variables numériques :**
    *   `age` : Généralement une distribution normale ou légèrement asymétrique.
    *   `balance` : Souvent très asymétrique avec une longue queue à droite (quelques clients avec des soldes très élevés).
    *   `campaign` : Fortement asymétrique, la plupart des clients sont contactés peu de fois.
    *   `pdays` : Une grande partie des valeurs est à -1 (jamais contacté auparavant), le reste étant des nombres positifs. Cela indique une distribution bimodale ou une forte concentration à -1.
    *   `previous` : Souvent concentrée autour de 0, avec quelques valeurs plus élevées.
*   **Variables catégorielles :**
    *   Certaines catégories peuvent être très rares (faible variance), ce qui peut poser problème pour le One-Hot Encoding si elles n'apparaissent que dans l'ensemble de test.
    *   La distribution des catégories "unknown" peut varier et indiquer des lacunes dans la collecte de données.

Comprendre ces distributions aide à décider si une mise à l'échelle (StandardScaler, MinMaxScaler) est appropriée pour les variables numériques, ou si des transformations (logarithmiques) sont nécessaires pour gérer l'asymétrie.

## Interprétation des Métriques

L'évaluation d'un modèle de classification, surtout sur un dataset déséquilibré, nécessite une analyse approfondie de plusieurs métriques.

### Pourquoi l'Accuracy est trompeuse ici ?

L'**Accuracy** (précision globale) mesure la proportion de prédictions correctes parmi toutes les prédictions. Dans un dataset déséquilibré comme "Bank Marketing" (où la classe 'no' est ~89% et 'yes' ~11%), un modèle qui prédit systématiquement 'no' pour tous les clients obtiendrait une précision d'environ 89%. Ce modèle serait inutile pour l'objectif métier de ciblage des souscripteurs, mais son Accuracy serait très élevée. C'est pourquoi l'Accuracy seule est une métrique trompeuse dans ce contexte.

### Focus sur Precision, Recall, F1-Score et AUC-ROC

Nous devons nous concentrer sur des métriques qui évaluent la performance du modèle sur la classe minoritaire ('yes' - souscription).

*   **Precision (Précision) :**
    *   `Precision = TP / (TP + FP)`
    *   Mesure, parmi toutes les prédictions positives (clients prédits comme souscripteurs), la proportion de celles qui sont réellement positives.
    *   **Interprétation métier :** Une Precision élevée signifie que lorsque le modèle dit qu'un client va souscrire, il a de fortes chances d'avoir raison. C'est important pour minimiser les coûts de marketing en évitant de contacter des clients qui ne sont pas intéressés (réduire les faux positifs).
*   **Recall (Rappel ou Sensibilité) :**
    *   `Recall = TP / (TP + FN)`
    *   Mesure, parmi tous les cas positifs réels (clients qui ont réellement souscrit), la proportion de ceux que le modèle a correctement identifiés.
    *   **Interprétation métier :** Un Recall élevé signifie que le modèle identifie une grande partie des clients qui vont réellement souscrire. C'est important pour ne pas manquer des opportunités de vente (réduire les faux négatifs).
*   **F1-Score :**
    *   `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`
    *   C'est la moyenne harmonique de la Precision et du Recall. Il est utile lorsque l'on cherche un équilibre entre Precision et Recall. Un F1-Score élevé indique que le modèle a une bonne Precision et un bon Recall.
*   **AUC-ROC (Area Under the Receiver Operating Characteristic Curve) :**
    *   L'AUC-ROC mesure la capacité du modèle à distinguer les classes positives des classes négatives sur tous les seuils de classification possibles.
    *   **Interprétation métier :** Une AUC-ROC proche de 1 indique que le modèle est excellent pour séparer les souscripteurs des non-souscripteurs. Une valeur de 0.5 indique une performance aléatoire. C'est une métrique robuste au déséquilibre de classe et donne une vue globale de la performance du modèle.

### Matrice de Confusion

La matrice de confusion est un tableau qui résume les performances d'un algorithme de classification.

|                   | Prédit Négatif | Prédit Positif |
| :---------------- | :------------- | :------------- |
| **Réel Négatif**  | True Negatives (TN) | False Positives (FP) |
| **Réel Positif**  | False Negatives (FN) | True Positives (TP) |

*   **True Positives (TP) :** Le modèle a correctement prédit qu'un client allait souscrire.
*   **True Negatives (TN) :** Le modèle a correctement prédit qu'un client n'allait pas souscrire.
*   **False Positives (FP) :** Le modèle a prédit qu'un client allait souscrire, mais il ne l'a pas fait (erreur de type I). Coût pour la banque : marketing inutile.
*   **False Negatives (FN) :** Le modèle a prédit qu'un client n'allait pas souscrire, mais il l'a fait (erreur de type II). Coût pour la banque : opportunité de vente manquée.

L'analyse de la matrice de confusion permet de comprendre la nature des erreurs du modèle et d'ajuster les stratégies en fonction des priorités métier (par exemple, privilégier le Recall si manquer une opportunité est plus coûteux que de faire une campagne inutile, ou inversement).

## Conclusion Globale

### Résumé des Insights

L'analyse du dataset "Bank Marketing" et la construction d'un modèle de classification ont permis de mettre en lumière plusieurs points clés :

1.  **Déséquilibre de classe :** La faible proportion de souscripteurs rend l'Accuracy inappropriée comme métrique principale. Les métriques comme la Precision, le Recall, le F1-Score et l'AUC-ROC sont essentielles pour une évaluation juste.
2.  **Data Leakage (`duration`) :** La variable `duration` est un piège majeur. Sa suppression est impérative pour garantir la validité et la déployabilité du modèle en production. Un modèle incluant `duration` serait irréaliste pour la prédiction ex-ante.
3.  **Performance du modèle :** Le modèle RandomForest, avec une gestion appropriée du déséquilibre de classe (`class_weight='balanced'`), démontre une capacité significative à identifier les clients potentiellement intéressés. Les métriques d'évaluation (AUC-ROC, F1-Score sur la classe positive) sont les indicateurs les plus fiables de sa performance réelle.
4.  **Facteurs d'influence :** L'analyse des importances des caractéristiques (si calculée) révèle les attributs les plus influents sur la décision de souscription, tels que le résultat de la campagne précédente (`poutcome`), le type d'emploi, l'âge, ou le solde bancaire.

### Recommandations Business pour l'Équipe Marketing

Sur la base de cette analyse, voici des recommandations concrètes pour l'équipe marketing :

1.  **Ciblage intelligent :** Utilisez le modèle prédictif pour identifier les clients ayant la plus forte probabilité de souscrire un dépôt à terme. Concentrez les efforts de marketing direct (appels, e-mails) sur ce segment à haute probabilité. Cela réduira les coûts de campagne et augmentera le taux de conversion.
2.  **Optimisation des campagnes :**
    *   **Priorité à la Precision ou au Recall ?** Discutez avec les équipes métier pour définir la priorité. Si l'objectif est de minimiser les coûts de contact (éviter les FP), privilégiez un modèle avec une Precision élevée. Si l'objectif est de maximiser le nombre de souscriptions (ne pas rater d'opportunités, éviter les FN), privilégiez un modèle avec un Recall élevé. Le F1-Score offre un bon compromis.
    *   **Seuil de décision :** Le modèle fournit des probabilités. L'équipe marketing peut ajuster le seuil de probabilité au-dessus duquel un client est contacté. Un seuil plus élevé augmentera la Precision (moins de contacts inutiles) mais réduira le Recall (plus d'opportunités manquées), et vice-versa.
3.  **Personnalisation des offres :** Les caractéristiques les plus importantes identifiées par le modèle peuvent être utilisées pour personnaliser les messages marketing. Par exemple, si le `job` ou `education` sont des facteurs