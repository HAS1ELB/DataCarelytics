import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(page_title="Analyse Médicale", page_icon="🩺", layout="wide")

# CSS personnalisé pour styliser les boutons et l'interface
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Fonction pour charger les données avec mise en cache
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Fonction pour l'analyse exploratoire
def exploratory_analysis():
    st.title("🩺 Analyse Exploratoire et Visualisation")
    st.markdown("**Explorez vos données médicales avec des outils interactifs.**", unsafe_allow_html=True)

    # Initialisation de l'état de session
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.numeric_cols = None
        st.session_state.categorical_cols = None

    # Conteneur pour le chargement des données
    with st.container():
        st.markdown("### 📂 Chargement des Données")
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                st.session_state.df = load_data(uploaded_file)
                st.session_state.numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
                st.session_state.categorical_cols = st.session_state.df.select_dtypes(exclude=[np.number]).columns
                st.success("Fichier chargé avec succès !")
                st.write("**Aperçu des données :**", st.session_state.df.head())
            except Exception as e:
                st.error(f"Erreur : {e}")
                return

    # Vérifier si des données sont chargées
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        numeric_cols = st.session_state.numeric_cols
        categorical_cols = st.session_state.categorical_cols

        # Conteneur pour la description des données
        with st.container():
            st.markdown("### 📊 Description des Données")
            st.write(f"**Nombre total de lignes** : {df.shape[0]}")
            st.write(f"**Nombre total de colonnes** : {df.shape[1]}")
            st.write(f"**Colonnes numériques** : {len(numeric_cols)} ({', '.join(numeric_cols)})")
            st.write(f"**Colonnes catégoriques** : {len(categorical_cols)} ({', '.join(categorical_cols)})")
            st.write("**Types de données** :", df.dtypes)
            st.divider()

        # Conteneur pour les modifications des données
        with st.container():
            st.markdown("### 🛠️ Modifier les Données")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Supprimer les doublons"):
                    df.drop_duplicates(inplace=True)
                    st.session_state.df = df
                    st.success(f"Doublons supprimés. Lignes restantes : {len(df)}")
                    st.write("Nouveau aperçu :", df.head())
            with col2:
                if st.button("✂️ Supprimer les valeurs nulles"):
                    df.dropna(inplace=True)
                    st.session_state.df = df
                    st.success(f"Valeurs nulles supprimées. Lignes restantes : {len(df)}")
                    st.write("Nouveau aperçu :", df.head())
            with col3:
                column_to_drop = st.selectbox("Choisir une colonne", df.columns, key="drop_column")
                if st.button("❌ Supprimer la colonne"):
                    df.drop(columns=[column_to_drop], inplace=True)
                    st.session_state.df = df
                    st.session_state.numeric_cols = df.select_dtypes(include=[np.number]).columns
                    st.session_state.categorical_cols = df.select_dtypes(exclude=[np.number]).columns
                    st.success(f"Colonne '{column_to_drop}' supprimée.")
                    st.write("Nouveau aperçu :", df.head())

            if st.button("🔄 Réinitialiser les données"):
                st.session_state.df = None
                st.session_state.numeric_cols = None
                st.session_state.categorical_cols = None
                st.success("Données réinitialisées.")
                return
            st.divider()

        # Conteneur pour les analyses
        with st.expander("🔍 Analyses Avancées", expanded=False):
            if st.checkbox("Afficher les valeurs manquantes"):
                st.write("**Valeurs manquantes par colonne :**", df.isnull().sum())

            if st.checkbox("Normaliser les données numériques"):
                if len(numeric_cols) > 0:
                    numeric_df = df[numeric_cols]
                    scaler = StandardScaler()
                    normalized_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_cols)
                    st.session_state.df[numeric_cols] = normalized_df
                    st.write("**Données normalisées :**", st.session_state.df.head())
                else:
                    st.warning("Aucune colonne numérique disponible.")

            if st.checkbox("Statistiques descriptives (numériques)"):
                if len(numeric_cols) > 0:
                    st.write("**Statistiques descriptives :**", df[numeric_cols].describe())
                else:
                    st.warning("Aucune colonne numérique disponible.")

            if st.checkbox("Analyser les données catégoriques"):
                if len(categorical_cols) > 0:
                    st.subheader("Analyse des colonnes catégoriques")
                    cat_column = st.selectbox("Choisir une colonne", categorical_cols)
                    value_counts = df[cat_column].value_counts()
                    st.write(f"**Fréquence dans {cat_column} :**", value_counts)
                    fig_bar = px.bar(x=value_counts.index, y=value_counts.values, 
                                     labels={"x": cat_column, "y": "Nombre"}, title=f"Fréquence de {cat_column}",
                                     template="plotly_white")
                    st.plotly_chart(fig_bar)
                    fig_pie = px.pie(names=value_counts.index, values=value_counts.values, 
                                     title=f"Répartition de {cat_column}", template="plotly_white")
                    st.plotly_chart(fig_pie)
                else:
                    st.warning("Aucune colonne catégorique disponible.")

            if st.checkbox("Relations catégoriques-numériques"):
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    cat_col = st.selectbox("Colonne catégorique", categorical_cols, key="cat_rel")
                    num_col = st.selectbox("Colonne numérique", numeric_cols, key="num_rel")
                    fig = px.box(df, x=cat_col, y=num_col, title=f"{num_col} par {cat_col}", template="plotly_white")
                    st.plotly_chart(fig)
                else:
                    st.warning("Pas assez de colonnes.")

            if st.checkbox("Histogrammes (numériques)"):
                if len(numeric_cols) > 0:
                    columns = st.multiselect("Colonnes pour l'histogramme", numeric_cols)
                    if columns:
                        sample_df = df[columns].sample(min(1000, len(df)))
                        fig = px.histogram(sample_df, x=columns, marginal="rug", title="Histogramme", 
                                           template="plotly_white")
                        st.plotly_chart(fig)
                else:
                    st.warning("Aucune colonne numérique disponible.")

            if st.checkbox("Boxplots (numériques)"):
                if len(numeric_cols) > 0:
                    column = st.selectbox("Colonne pour le boxplot", numeric_cols)
                    sample_df = df[[column]].sample(min(1000, len(df)))
                    fig = px.box(sample_df, y=column, title=f"Boxplot de {column}", template="plotly_white")
                    st.plotly_chart(fig)
                else:
                    st.warning("Aucune colonne numérique disponible.")

            if st.checkbox("Matrice de corrélation (numériques)"):
                if len(numeric_cols) > 0:
                    sample_df = df[numeric_cols].sample(min(1000, len(df)))
                    fig = px.imshow(sample_df.corr(), text_auto=True, color_continuous_scale="RdBu_r",
                                    title="Matrice de corrélation", template="plotly_white")
                    st.plotly_chart(fig)
                else:
                    st.warning("Aucune colonne numérique disponible.")

            if st.checkbox("PCA (avec encodage optionnel)"):
                if len(numeric_cols) > 0:
                    df_pca = df[numeric_cols].dropna()
                    if st.checkbox("Inclure colonnes catégoriques"):
                        if len(categorical_cols) > 0:
                            df_encoded = pd.get_dummies(df[categorical_cols])
                            df_pca = pd.concat([df_pca, df_encoded], axis=1)
                            st.write("**Données encodées :**", df_pca.head())
                    if len(df_pca.columns) > 1:
                        n_components = st.slider("Composantes PCA", 2, min(5, len(df_pca.columns)), 2)
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(df_pca.sample(min(1000, len(df_pca))))
                        st.write("**Explained variance ratio :**", pca.explained_variance_ratio_)
                        fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], title="PCA (2 premières composantes)",
                                         template="plotly_white")
                        st.plotly_chart(fig)
                    else:
                        st.warning("Pas assez de colonnes pour PCA.")
                else:
                    st.warning("Aucune colonne numérique disponible.")

# Menu principal
def main():
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
        st.title("🩺 Modèles de Prédiction Médicale")
        st.write("Section en développement : Régression logistique, SVM, Random Forest, etc.")
    elif analysis_type == "Analyse d’images médicales":
        st.title("📸 Analyse d’Images Médicales")
        st.write("Section en développement : CNN, U-Net, GANs, etc.")
    elif analysis_type == "Analyse de texte médical et NLP":
        st.title("📄 Analyse de Texte Médical et NLP")
        st.write("Section en développement : BERT, NER, Chatbots, etc.")
    elif analysis_type == "Analyse de séries temporelles biomédicales":
        st.title("📈 Analyse de Séries Temporelles Biomédicales")
        st.write("Section en développement : LSTM, DTW, Fourier, etc.")

if __name__ == "__main__":
    main()