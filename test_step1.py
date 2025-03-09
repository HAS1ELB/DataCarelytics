from src.data_processing.tabular_processor import TabularProcessor
from src.analysis.exploratory import ExploratoryAnalysis

try:
    # Charger et prétraiter les données
    processor = TabularProcessor("data/tabular/biomarkers.csv")
    data = processor.clean_data()
    data = processor.normalize()

    # Analyse exploratoire
    explorer = ExploratoryAnalysis(data)
    print("Statistiques descriptives :")
    print(explorer.descriptive_stats())
    explorer.plot_histogram("glucose", "output/histogram_glucose.png")
    print("Histogramme sauvegardé dans output/histogram_glucose.png")
    explorer.plot_correlation_matrix("output/correlation_matrix.png")
    print("Matrice de corrélation sauvegardée dans output/correlation_matrix.png")
    pca_data = explorer.apply_pca()
    print("Résultats PCA :")
    print(pca_data)

except FileNotFoundError as e:
    print(f"Erreur : Fichier non trouvé - {e}")
except Exception as e:
    print(f"Une erreur est survenue : {e}")