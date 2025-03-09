import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import numpy as np

class ExploratoryAnalysis:
    def __init__(self, data):
        self.data = data
    
    def descriptive_stats(self):
        """Calcule les statistiques descriptives."""
        return self.data.describe()
    
    def plot_histogram(self, column, output_path="histogram.png"):
        """Génère un histogramme pour une colonne donnée."""
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(8, 6))
        self.data[column].hist(bins=20)
        plt.title(f"Histogramme de {column}")
        plt.xlabel(column)
        plt.ylabel("Fréquence")
        plt.savefig(output_path)
        plt.close()
    
    def plot_correlation_matrix(self, output_path="correlation.png"):
        """Génère une matrice de corrélation."""
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice de corrélation")
        plt.savefig(output_path)
        plt.close()
    
    def apply_pca(self, n_components=2):
        """Applique une PCA et retourne les données réduites."""
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self.data.select_dtypes(include=[np.number]))
        return pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])