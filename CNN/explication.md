Voici un **document explicatif** du notebook `preprocessing.ipynb`, destiné à décrire son fonctionnement, ses objectifs et ses différentes parties.

---

# 📘 Document d’explication — Notebook de Prétraitement d’Images Médicales

## 🧠 Objectif général

Ce notebook a pour but de :
- Charger des images médicales (formats standards comme PNG, JPG, mais surtout **DICOM**),
- Appliquer des traitements de prétraitement personnalisés (normalisation, réduction de bruit, amélioration de contraste, redimensionnement),
- Fournir une **interface interactive** via `ipywidgets` pour permettre à l'utilisateur de tester facilement différentes configurations de traitement d'image.

---

## 🧰 Bibliothèques utilisées

- `pydicom` : Lecture d’images médicales au format DICOM.
- `numpy`, `opencv-python` : Manipulations et traitements d’images.
- `scikit-image` : Redimensionnement et transformations.
- `matplotlib`, `Pillow` : Affichage et chargement d’images.
- `ipywidgets` : Interface utilisateur interactive.
- `streamlit` (optionnel) : Interface graphique alternative.
- `scikit-learn` : Non utilisé ici, probablement prévu pour des traitements futurs.

---

## 📦 Partie 0 – Installation des bibliothèques

Une cellule `pip install` est incluse pour installer toutes les dépendances nécessaires à l'exécution du notebook.

---

## 📂 Partie 1 – Chargement et gestion des images

### 🔹 `load_image(filepath)`
Permet de charger une image à partir d’un fichier :
- Si c’est un fichier `.dcm`, il utilise `pydicom` et applique la **LUT** (Lookup Table) pour visualiser correctement les intensités.
- Si c’est une image classique (jpg, png), elle est chargée avec `Pillow` et convertie en `numpy array`.

### 🔹 `get_input_image()`
Détermine l'image d'entrée selon que l'utilisateur :
- A téléchargé un fichier via un widget (`FileUpload`),
- A coché l’option d'utiliser une image d’exemple.

Cette fonction gère le traitement des fichiers temporaires et retourne l’image + les métadonnées + le nom du fichier.

---

## 🧪 Partie 2 – Fonctions de prétraitement

### 🔸 `normalize_image(image)`
Normalise les valeurs de l'image entre 0 et 1 (si float) ou entre 0 et 255 (si uint8).

### 🔸 `denoise_image(image, kernel_size)`
Applique un **filtre médian** pour éliminer le bruit. Vérifie que la taille du noyau est impaire.

### 🔸 `enhance_contrast(image)`
Convertit l'image en niveaux de gris si nécessaire, puis applique une **égalisation d’histogramme** pour renforcer le contraste.

### 🔸 `resize_image(image, target_size)`
Redimensionne l'image avec **anti-aliasing**. Si l'image a plusieurs canaux, chaque canal est redimensionné séparément.

### 🔸 `save_processed_image(image)`
Sauvegarde l’image sous forme de fichier `.npy`.

---

## ⚙️ Partie 3 – Fonction principale `process_image()`

Cette fonction applique les traitements choisis :
- Elle extrait une **slice centrale** si l’image est volumique (3D),
- Applique les traitements sélectionnés : normalisation, débruitage, amélioration de contraste, redimensionnement.

Les conditions sont contrôlées par des **checkboxes** interactives.

---

## 🖱️ Partie 4 – Interface Interactive

### Composants :
- **Widgets** : boutons, sliders, file upload, checkboxes pour choisir les traitements.
- `process_button` : lance le traitement.
- `on_process_clicked()` : callback qui s’exécute au clic, affiche l’image d’origine et l’image prétraitée.

Les widgets sont affichés avec `display(...)`, et le traitement est lancé avec `process_button.on_click(...)`.

---

## 🔄 Fonctionnement général

1. L’utilisateur charge une image (upload ou exemple généré).
2. Il choisit les options de traitement avec les widgets.
3. Il clique sur **Lancer le prétraitement**.
4. L’image originale et l’image transformée sont affichées côte à côte.
5. L’image traitée peut être sauvegardée localement.

---

Souhaitez-vous que je transforme ce résumé en un fichier PDF, Markdown ou autre format ?