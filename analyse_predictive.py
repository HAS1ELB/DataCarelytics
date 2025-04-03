import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import datetime

def medical_predictive_models():
    st.title("🩺 Modèles de Prédiction Médicale")
    st.markdown("**Construisez et évaluez des modèles prédictifs pour des données médicales.**", unsafe_allow_html=True)

    # Session state initialization
    if 'predictive_data' not in st.session_state:
        st.session_state.predictive_data = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'target' not in st.session_state:
        st.session_state.target = None
    if 'y_pred' not in st.session_state:
        st.session_state.y_pred = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None

    # Container for data input
    with st.container():
        st.markdown("### 📂 Chargement des Données")
        data_source = st.radio("Choisir la source des données", ["Télécharger un fichier CSV", "Utiliser un exemple"])

        if data_source == "Télécharger un fichier CSV":
            uploaded_file = st.file_uploader("Choisissez un fichier CSV (avec caractéristiques et cible)", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.predictive_data = df
                    st.session_state.features = None
                    st.session_state.target = None
                    st.session_state.y_pred = None
                    st.session_state.X_test = None
                    st.session_state.y_test = None
                    st.success("Données chargées avec succès !")
                    st.write("**Aperçu des données :**", df.head())
                except Exception as e:
                    st.error(f"Erreur lors du chargement du fichier : {e}")
                    return

        elif data_source == "Utiliser un exemple":
            np.random.seed(42)
            n_samples = 500
            data = {
                "Age": np.random.randint(20, 80, n_samples),
                "BMI": np.random.normal(27, 5, n_samples),
                "BloodPressure": np.random.normal(120, 20, n_samples),
                "Glucose": np.random.normal(100, 30, n_samples),
                "Diabetes": np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
            }
            df = pd.DataFrame(data)
            st.session_state.predictive_data = df
            st.session_state.features = ["Age", "BMI", "BloodPressure", "Glucose"]
            st.session_state.target = "Diabetes"
            st.session_state.y_pred = None
            st.session_state.X_test = None
            st.session_state.y_test = None
            st.info("Données d'exemple chargées : prédiction du diabète (0 = non, 1 = oui).")
            st.write("**Aperçu des données :**", df.head())

    if st.session_state.predictive_data is not None:
        df = st.session_state.predictive_data.copy()

        with st.container():
            st.markdown("### ⚙️ Configuration des Données")
            st.write("Sélectionnez les colonnes pour l'analyse :")
            feature_cols = st.multiselect("Caractéristiques (X)", df.columns, 
                                         default=st.session_state.features if st.session_state.features else [])
            target_col = st.selectbox("Cible (y)", df.columns, 
                                     index=df.columns.get_loc(st.session_state.target) if st.session_state.target in df.columns else 0)

            if st.button("Confirmer la sélection"):
                if not feature_cols or not target_col:
                    st.error("Veuillez sélectionner au moins une caractéristique et une cible.")
                    return
                try:
                    X = df[feature_cols].dropna()
                    y = df[target_col].dropna()
                    if len(X) != len(y):
                        st.error("Les caractéristiques et la cible doivent avoir le même nombre de lignes après suppression des valeurs manquantes.")
                        return
                    st.session_state.predictive_data = df
                    st.session_state.features = feature_cols
                    st.session_state.target = target_col
                    st.session_state.y_pred = None
                    st.session_state.X_test = None
                    st.session_state.y_test = None
                    st.success("Colonnes configurées avec succès !")
                except Exception as e:
                    st.error(f"Erreur lors de la configuration : {e}")
                    return

        if st.session_state.features and st.session_state.target:
            X = df[st.session_state.features]
            y = df[st.session_state.target]

            with st.container():
                st.markdown("### 🛠️ Prétraitement des Données")
                preprocess = st.checkbox("Normaliser les caractéristiques (StandardScaler)")
                test_size = st.slider("Taille de l'ensemble de test (%)", 10, 50, 20) / 100

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                if preprocess:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    st.write("Caractéristiques normalisées avec StandardScaler.")

            with st.container():
                st.markdown("### 🤖 Entraînement du Modèle")
                model_type = st.selectbox("Choisir un modèle", ["Régression Logistique", "SVM", "Random Forest"])
                
                if model_type == "Régression Logistique":
                    model = LogisticRegression(max_iter=1000)
                elif model_type == "SVM":
                    kernel = st.selectbox("Noyau SVM", ["linear", "rbf"], index=1)
                    model = SVC(kernel=kernel)
                elif model_type == "Random Forest":
                    n_estimators = st.slider("Nombre d'arbres", 10, 200, 100)
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

                if st.button("Entraîner le modèle"):
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.session_state.y_pred = y_pred
                        st.session_state.X_test = pd.DataFrame(X_test, columns=st.session_state.features, index=X_test.index if hasattr(X_test, 'index') else None)
                        st.session_state.y_test = y_test

                        st.write("Debug: Premières prédictions stockées:", st.session_state.y_pred[:5])
                        accuracy = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, output_dict=True)
                        cm = confusion_matrix(y_test, y_pred)

                        st.write(f"**Précision du modèle :** {accuracy:.2f}")
                        st.write("**Rapport de classification :**")
                        st.table(pd.DataFrame(report).T)

                        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", 
                                          title="Matrice de Confusion", 
                                          labels=dict(x="Prédit", y="Réel", color="Nombre"))
                        st.plotly_chart(fig_cm)

                        if model_type == "Random Forest":
                            importance = model.feature_importances_
                            df_importance = pd.DataFrame({"Caractéristique": st.session_state.features, "Importance": importance})
                            fig_importance = px.bar(df_importance, x="Caractéristique", y="Importance", 
                                                   title="Importance des Caractéristiques", template="plotly_white")
                            st.plotly_chart(fig_importance)

                    except Exception as e:
                        st.error(f"Erreur lors de l'entraînement : {e}")

            with st.container():
                st.markdown("### 💾 Exporter les Résultats")
                if st.button("Exporter les prédictions en CSV"):
                    if st.session_state.y_pred is None or st.session_state.X_test is None or st.session_state.y_test is None:
                        # Auto-train if predictions are not available
                        try:
                            model.fit(X_train, y_train)
                            st.session_state.y_pred = model.predict(X_test)
                            st.session_state.X_test = pd.DataFrame(X_test, columns=st.session_state.features, index=X_test.index if hasattr(X_test, 'index') else None)
                            st.session_state.y_test = y_test
                            st.info("Modèle entraîné automatiquement pour l'exportation.")
                        except Exception as e:
                            st.error(f"Erreur lors de l'entraînement automatique : {e}")
                            return

                    try:
                        y_pred_df = pd.DataFrame({"Prédiction": st.session_state.y_pred}, index=st.session_state.X_test.index)
                        result_df = pd.concat([st.session_state.X_test, st.session_state.y_test.rename("Réel"), y_pred_df], axis=1)
                        buf = StringIO()
                        result_df.to_csv(buf, index=False)
                        buf.seek(0)
                        st.download_button(
                            label="Télécharger les prédictions",
                            data=buf.getvalue(),
                            file_name=f"predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Erreur lors de l'exportation : {e}")

# No main() function here as this is a module to be imported