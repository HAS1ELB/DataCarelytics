import streamlit as st
from config import setup_page_config
from exploratory_analysis import exploratory_analysis
from image_analysis import medical_image_analysis

def main():
    # Apply page configuration from config.py
    setup_page_config()

    # Sidebar navigation
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

    # Route to the appropriate analysis function
    if analysis_type == "Analyse exploratoire et visualisation":
        exploratory_analysis()
    elif analysis_type == "Modèles de prédiction médicale":
        st.title("🩺 Modèles de Prédiction Médicale")
        st.write("Section en développement : Régression logistique, SVM, Random Forest, etc.")
    elif analysis_type == "Analyse d’images médicales":
        medical_image_analysis()
    elif analysis_type == "Analyse de texte médical et NLP":
        st.title("📄 Analyse de Texte Médical et NLP")
        st.write("Section en développement : BERT, NER, Chatbots, etc.")
    elif analysis_type == "Analyse de séries temporelles biomédicales":
        st.title("📈 Analyse de Séries Temporelles Biomédicales")
        st.write("Section en développement : LSTM, DTW, Fourier, etc.")

if __name__ == "__main__":
    main()