import pandas as pd
import numpy as np

class TabularProcessor:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
    
    def clean_data(self):
        """Supprime les lignes avec des valeurs manquantes."""
        self.data.dropna(inplace=True)
        return self.data
    
    def normalize(self):
        """Normalise les colonnes numériques."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = (self.data[numeric_cols] - self.data[numeric_cols].mean()) / self.data[numeric_cols].std()
        return self.data
