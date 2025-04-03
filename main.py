import streamlit as st
from config import setup_page_config
from exploratory_analysis import exploratory_analysis
from image_analysis import medical_image_analysis
from analyse_texte_nlp import medical_text_nlp_analysis
from analyse_serie_temporelle import biomedical_time_series_analysis
from analyse_predictive import medical_predictive_models

def main():
    setup_page_config()
    st.sidebar.title("🧪 Menu")
    analysis_type = st.sidebar.radio(
        "Choisir un type d’analyse",
        [
            "Analyse exploratoire et visualisation",
            "Modèles de prédiction médicale",
            "Analyse d’images médicales",
            "Analyse de texte médical et NLP",
            "Analyse de séries temporelles biomédicales"
        ]
    )

    if analysis_type == "Analyse exploratoire et visualisation":
        exploratory_analysis()
    elif analysis_type == "Modèles de prédiction médicale":
        medical_predictive_models()
    elif analysis_type == "Analyse d’images médicales":
        medical_image_analysis()
    elif analysis_type == "Analyse de texte médical et NLP":
        medical_text_nlp_analysis()
    elif analysis_type == "Analyse de séries temporelles biomédicales":
        biomedical_time_series_analysis()

if __name__ == "__main__":
    main()