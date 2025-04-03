
# MedicalAnalysisApp

**MedicalAnalysisApp** est une application interactive développée avec Streamlit pour analyser et visualiser des données médicales de différents types. Elle offre des outils d'exploration, de traitement, et de prédiction adaptés aux données tabulaires, textuelles, d'images, et de séries temporelles biomédicales. Conçue pour les professionnels de santé, chercheurs, et data scientists, cette application vise à faciliter l'analyse des données médicales dans un environnement convivial.

---

## Étape 1 : Analyse exploratoire et visualisation

- **Objectif** : Explorer et visualiser les données tabulaires médicales.
- **Fonctionnalités** :
  - Statistiques descriptives (moyenne, écart-type, etc.).
  - Histogrammes et matrices de corrélation.
  - Visualisations interactives avec Plotly (diagrammes en barres, boîtes à moustaches, etc.).
  - Détection des valeurs manquantes et gestion des données aberrantes.

---

## Étape 2 : Analyse d’images médicales

- **Objectif** : Traiter et analyser des images médicales (DICOM, PNG, JPEG).
- **Fonctionnalités** :
  - Chargement et visualisation des images médicales.
  - Ajustement de la luminosité et du contraste.
  - Détection de contours et segmentation basique avec OpenCV.
  - Exportation des images traitées.

---

## Étape 3 : Analyse de texte médical et NLP

- **Objectif** : Analyser des rapports médicaux ou textes cliniques avec des techniques de traitement du langage naturel (NLP).
- **Fonctionnalités** :
  - Prétraitement du texte (minuscules, suppression des mots vides, lemmatisation).
  - Analyse de la fréquence des mots et nuage de mots.
  - Reconnaissance des entités nommées (NER) avec spaCy.
  - Analyse de sentiment basique.
  - Exportation des textes traités ou des fréquences en TXT/CSV.

---

## Étape 4 : Analyse de séries temporelles biomédicales

- **Objectif** : Explorer et modéliser des données temporelles médicales (ex. rythme cardiaque, pression artérielle).
- **Fonctionnalités** :
  - Visualisation des séries temporelles avec Plotly.
  - Prétraitement (interpolation, lissage, suppression des outliers).
  - Décomposition en tendance, saisonnalité et résidu avec `statsmodels`.
  - Détection d’anomalies basée sur le Z-score.
  - Statistiques descriptives et autocorrélation.
  - Exportation des données prétraitées en CSV.

---

## Étape 5 : Modèles de prédiction médicale

- **Objectif** : Construire et évaluer des modèles prédictifs pour des données médicales tabulaires.
- **Fonctionnalités** :
  - Prétraitement des données (normalisation avec `StandardScaler`).
  - Entraînement de modèles : Régression Logistique, SVM, Random Forest.
  - Évaluation avec précision, rapport de classification, et matrice de confusion.
  - Visualisation de l’importance des caractéristiques (pour Random Forest).
  - Exportation des prédictions en CSV.

---

## Prérequis

- Python 3.8 ou supérieur
- Dépendances listées dans `requirements.txt` :

```
streamlit
pandas
numpy
plotly
scikit-learn>=1.0.0
pillow
opencv-python
pydicom
matplotlib
spacy
nltk>=3.8.1
wordcloud
statsmodels
```

---

## Installation

1. Clonez le dépôt :

```bash
git clone https://github.com/has1elb/MedicalAnalysisApp.git
cd MedicalAnalysisApp
```

2. Créez un environnement virtuel (optionnel mais recommandé) :

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installez les dépendances :

```bash
pip install -r requirements.txt
```

4. Téléchargez le modèle spaCy pour l’analyse NLP :

```bash
python -m spacy download en_core_web_sm
```

---

## Utilisation

Lancez l’application :

```bash
streamlit run main.py
```

Accédez à l’interface via votre navigateur à l’adresse :

```
Local URL: http://localhost:8501
```

---

## Structure du Projet

- **main.py** : Point d’entrée de l’application, gère le menu et les redirections.
- **config.py** : Configuration de la page Streamlit (titre, layout, etc.).
- **exploratory_analysis.py** : Module pour l’analyse exploratoire.
- **image_analysis.py** : Module pour l’analyse d’images.
- **analyse_texte_nlp.py** : Module pour l’analyse NLP.
- **analyse_serie_temporelle.py** : Module pour les séries temporelles.
- **analyse_predictive.py** : Module pour les modèles prédictifs.
- **requirements.txt** : Liste des dépendances.

---

## Contributions

Les contributions sont bienvenues !

1. Forkez le dépôt.
2. Créez une branche pour vos modifications :

```bash
git checkout -b ma-nouvelle-fonctionnalite
```

3. Soumettez une pull request avec une description claire des changements.

---

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

---

## Contact

Pour toute question ou suggestion, contactez votre-email@example.com.
