import streamlit as st
import pandas as pd
import numpy as np
import spacy
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import plotly.express as px
from io import StringIO
import re
import datetime

# Download required NLTK data (run once during app initialization)
nltk.download('punkt', quiet=True)      # Original punkt data (backward compatibility)
nltk.download('punkt_tab', quiet=True)  # New punkt_tab data for updated NLTK versions
nltk.download('stopwords', quiet=True)

# Load English language model for spaCy (assuming 'en_core_web_sm' is installed)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Modèle spaCy 'en_core_web_sm' non installé. Exécutez 'python -m spacy download en_core_web_sm' et redémarrez l'application.")
    st.stop()

def medical_text_nlp_analysis():
    st.title("📄 Analyse de Texte Médical et NLP")
    st.markdown("**Analysez des textes médicaux avec des outils de traitement du langage naturel.**", unsafe_allow_html=True)

    # Session state initialization
    if 'text_data' not in st.session_state:
        st.session_state.text_data = None
        st.session_state.processed_text = None

    # Container for text input
    with st.container():
        st.markdown("### 📝 Chargement des Données Textuelles")
        data_source = st.radio("Choisir la source des données", ["Télécharger un fichier", "Entrer du texte manuellement", "Utiliser un exemple"])

        if data_source == "Télécharger un fichier":
            uploaded_file = st.file_uploader("Choisissez un fichier texte ou CSV", type=["txt", "csv"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            st.session_state.text_data = " ".join(df['text'].astype(str))
                        else:
                            st.error("Le fichier CSV doit contenir une colonne nommée 'text'.")
                            return
                    else:
                        st.session_state.text_data = uploaded_file.read().decode("utf-8")
                    st.success("Texte chargé avec succès !")
                except Exception as e:
                    st.error(f"Erreur lors du chargement du fichier : {e}")
                    return

        elif data_source == "Entrer du texte manuellement":
            st.session_state.text_data = st.text_area("Entrez le texte médical ici", height=200)

        elif data_source == "Utiliser un exemple":
            example_text = """
            Patient presents with fever, cough, and shortness of breath. Suspected diagnosis: pneumonia. 
            Past medical history includes hypertension and diabetes mellitus. Current medications: 
            Lisinopril 10 mg daily, Metformin 500 mg twice daily. Lab results indicate elevated 
            white blood cell count and C-reactive protein. Chest X-ray shows bilateral infiltrates.
            """
            st.session_state.text_data = example_text
            st.info("Texte d'exemple chargé : rapport médical fictif.")

        if st.session_state.text_data:
            st.markdown("### 📄 Texte Chargé")
            st.write(st.session_state.text_data[:500] + "..." if len(st.session_state.text_data) > 500 else st.session_state.text_data)

    # Check if text data is available
    if st.session_state.text_data:
        text = st.session_state.text_data

        # Preprocessing options
        with st.container():
            st.markdown("### 🛠️ Prétraitement du Texte")
            preprocess_options = st.multiselect(
                "Options de prétraitement",
                ["Minuscules", "Supprimer la ponctuation", "Supprimer les mots vides", "Lemmatisation"]
            )

            processed_text = text
            if preprocess_options:
                if "Minuscules" in preprocess_options:
                    processed_text = processed_text.lower()
                if "Supprimer la ponctuation" in preprocess_options:
                    processed_text = re.sub(r'[^\w\s]', '', processed_text)
                if "Supprimer les mots vides" in preprocess_options:
                    stop_words = set(stopwords.words('english'))
                    words = word_tokenize(processed_text)
                    processed_text = " ".join([word for word in words if word not in stop_words])
                if "Lemmatisation" in preprocess_options:
                    doc = nlp(processed_text)
                    processed_text = " ".join([token.lemma_ for token in doc])

                st.session_state.processed_text = processed_text
                st.markdown("### 🔄 Texte Prétraité")
                st.write(processed_text[:500] + "..." if len(processed_text) > 500 else processed_text)

        # Advanced NLP analysis
        with st.expander("🔍 Analyse Avancée", expanded=False):
            analysis_type = st.selectbox(
                "Type d'analyse",
                ["Fréquence des mots", "Entités nommées (NER)", "Analyse de sentiment (basique)", "Nuage de mots"]
            )

            if analysis_type == "Fréquence des mots":
                words = word_tokenize(processed_text if st.session_state.processed_text else text)
                word_freq = Counter(words)
                top_n = st.slider("Nombre de mots à afficher", 5, 50, 10)
                common_words = word_freq.most_common(top_n)
                df_freq = pd.DataFrame(common_words, columns=["Mot", "Fréquence"])
                
                st.write("**Fréquence des mots :**")
                st.table(df_freq)

                fig = px.bar(df_freq, x="Mot", y="Fréquence", title="Fréquence des mots les plus courants",
                             template="plotly_white")
                st.plotly_chart(fig)

            elif analysis_type == "Entités nommées (NER)":
                doc = nlp(processed_text if st.session_state.processed_text else text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                
                if entities:
                    df_entities = pd.DataFrame(entities, columns=["Entité", "Type"])
                    st.write("**Entités détectées :**")
                    st.table(df_entities)

                    entity_counts = Counter([ent[1] for ent in entities])
                    df_entity_counts = pd.DataFrame(list(entity_counts.items()), columns=["Type", "Nombre"])
                    fig = px.pie(df_entity_counts, names="Type", values="Nombre", 
                                 title="Répartition des types d'entités", template="plotly_white")
                    st.plotly_chart(fig)
                else:
                    st.warning("Aucune entité nommée détectée.")

            elif analysis_type == "Analyse de sentiment (basique)":
                st.warning("Cette analyse de sentiment est basique et utilise une heuristique simple.")
                doc = nlp(processed_text if st.session_state.processed_text else text)
                positive_words = set(["good", "normal", "stable", "improved"])
                negative_words = set(["fever", "pain", "severe", "abnormal"])
                
                words = [token.text.lower() for token in doc]
                pos_count = sum(1 for word in words if word in positive_words)
                neg_count = sum(1 for word in words if word in negative_words)
                total = pos_count + neg_count or 1  # Avoid division by zero
                
                sentiment_score = (pos_count - neg_count) / total
                sentiment_label = "Positif" if sentiment_score > 0 else "Négatif" if sentiment_score < 0 else "Neutre"
                
                st.write(f"**Score de sentiment :** {sentiment_score:.2f} ({sentiment_label})")
                st.write(f"Mots positifs détectés : {pos_count}")
                st.write(f"Mots négatifs détectés : {neg_count}")

            elif analysis_type == "Nuage de mots":
                from wordcloud import WordCloud
                wordcloud = WordCloud(width=800, height=400, background_color="white", 
                                      max_words=100, contour_width=1, contour_color='steelblue')
                wordcloud.generate(processed_text if st.session_state.processed_text else text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

        # Export results
        with st.container():
            st.markdown("### 💾 Exporter les Résultats")
            export_format = st.selectbox("Format d'exportation", ["TXT", "CSV"])
            if st.button("Exporter"):
                if export_format == "TXT":
                    content = processed_text if st.session_state.processed_text else text
                    buf = StringIO()
                    buf.write(content)
                    buf.seek(0)
                    st.download_button(
                        label="Télécharger le texte",
                        data=buf.getvalue(),
                        file_name=f"processed_text_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                elif export_format == "CSV":
                    words = word_tokenize(processed_text if st.session_state.processed_text else text)
                    word_freq = Counter(words)
                    df_freq = pd.DataFrame(word_freq.items(), columns=["Mot", "Fréquence"])
                    buf = StringIO()
                    df_freq.to_csv(buf, index=False)
                    buf.seek(0)
                    st.download_button(
                        label="Télécharger les fréquences",
                        data=buf.getvalue(),
                        file_name=f"word_freq_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )