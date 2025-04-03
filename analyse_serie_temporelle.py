import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from io import StringIO
import datetime

def biomedical_time_series_analysis():
    st.title("📈 Analyse de Séries Temporelles Biomédicales")
    st.markdown("**Analysez des données temporelles biomédicales avec des outils interactifs.**", unsafe_allow_html=True)

    # Session state initialization
    if 'ts_data' not in st.session_state:
        st.session_state.ts_data = None
    if 'selected_time_col' not in st.session_state:
        st.session_state.selected_time_col = None
    if 'selected_value_col' not in st.session_state:
        st.session_state.selected_value_col = None

    # Container for data input
    with st.container():
        st.markdown("### 📂 Chargement des Données Temporelles")
        data_source = st.radio("Choisir la source des données", ["Télécharger un fichier CSV", "Utiliser un exemple"])

        if data_source == "Télécharger un fichier CSV":
            uploaded_file = st.file_uploader("Choisissez un fichier CSV (avec colonne de temps et valeur)", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.ts_data = df
                    st.session_state.selected_time_col = None  # Reset column selections
                    st.session_state.selected_value_col = None
                    st.success("Données chargées avec succès !")
                    st.write("**Aperçu des données :**", df.head())
                except Exception as e:
                    st.error(f"Erreur lors du chargement du fichier : {e}")
                    return

        elif data_source == "Utiliser un exemple":
            # Simulate a biomedical time series (e.g., heart rate over time)
            dates = pd.date_range(start="2023-01-01", end="2023-01-07", freq="h")
            np.random.seed(42)
            heart_rate = np.random.normal(70, 10, len(dates)) + np.sin(np.linspace(0, 10, len(dates))) * 5
            df = pd.DataFrame({"Time": dates, "HeartRate": heart_rate})
            st.session_state.ts_data = df
            st.session_state.selected_time_col = "Time"  # Preselect for example
            st.session_state.selected_value_col = "HeartRate"
            st.info("Données d'exemple chargées : rythme cardiaque simulé sur une semaine.")
            st.write("**Aperçu des données :**", df.head())

    # Check if data is available
    if st.session_state.ts_data is not None:
        df = st.session_state.ts_data.copy()

        # Select time and value columns
        with st.container():
            st.markdown("### ⚙️ Configuration des Données")
            st.write("Sélectionnez les colonnes pour l'analyse :")
            time_col = st.selectbox("Colonne de temps", df.columns, 
                                   index=df.columns.get_loc(st.session_state.selected_time_col) if st.session_state.selected_time_col in df.columns else 0, 
                                   key="time_col")
            value_col = st.selectbox("Colonne de valeur", df.columns, 
                                    index=df.columns.get_loc(st.session_state.selected_value_col) if st.session_state.selected_value_col in df.columns else 0, 
                                    key="value_col")

            # Update session state only if columns change
            if st.button("Confirmer la sélection"):
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                    df = df.sort_values(time_col)
                    st.session_state.ts_data = df
                    st.session_state.selected_time_col = time_col
                    st.session_state.selected_value_col = value_col
                    st.success("Colonnes configurées avec succès !")
                except Exception as e:
                    st.error(f"Erreur lors de la conversion de la colonne de temps : {e}")
                    return

        # Use selected columns if confirmed, otherwise skip further processing
        if st.session_state.selected_time_col and st.session_state.selected_value_col:
            time_col = st.session_state.selected_time_col
            value_col = st.session_state.selected_value_col

            # Basic visualization
            with st.container():
                st.markdown("### 📊 Visualisation de Base")
                fig = px.line(df, x=time_col, y=value_col, title=f"{value_col} au fil du temps", 
                              template="plotly_white")
                fig.update_layout(xaxis_title="Temps", yaxis_title=value_col)
                st.plotly_chart(fig)

            # Preprocessing options
            with st.container():
                st.markdown("### 🛠️ Prétraitement des Données")
                preprocess_options = st.multiselect(
                    "Options de prétraitement",
                    ["Interpolation des valeurs manquantes", "Lissage (moyenne mobile)", "Supprimer les valeurs aberrantes"]
                )

                processed_df = df.copy()
                if preprocess_options:
                    if "Interpolation des valeurs manquantes" in preprocess_options:
                        processed_df[value_col] = processed_df[value_col].interpolate(method='linear')
                        st.write("Valeurs manquantes interpolées.")

                    if "Lissage (moyenne mobile)" in preprocess_options:
                        window_size = st.slider("Taille de la fenêtre de lissage", 3, 30, 7)
                        processed_df[f"{value_col}_smoothed"] = processed_df[value_col].rolling(window=window_size, center=True).mean()
                        st.write(f"Lissage appliqué avec une fenêtre de {window_size}.")

                    if "Supprimer les valeurs aberrantes" in preprocess_options:
                        z_score_threshold = st.slider("Seuil de Z-score", 1.0, 5.0, 3.0)
                        mean_val = processed_df[value_col].mean()
                        std_val = processed_df[value_col].std()
                        processed_df = processed_df[
                            (processed_df[value_col] >= mean_val - z_score_threshold * std_val) &
                            (processed_df[value_col] <= mean_val + z_score_threshold * std_val)
                        ]
                        st.write(f"Valeurs aberrantes supprimées (Z-score > {z_score_threshold}).")

                    st.session_state.ts_data = processed_df
                    st.markdown("### 🔄 Données Prétraitées")
                    fig_processed = px.line(processed_df, x=time_col, y=[col for col in processed_df.columns if col != time_col],
                                            title="Données après prétraitement", template="plotly_white")
                    fig_processed.update_layout(xaxis_title="Temps", yaxis_title="Valeur")
                    st.plotly_chart(fig_processed)

            # Advanced time series analysis
            with st.expander("🔍 Analyse Avancée", expanded=False):
                analysis_type = st.selectbox(
                    "Type d'analyse",
                    ["Décomposition (tendance, saisonnalité)", "Détection d'anomalies", "Statistiques de base"]
                )

                if analysis_type == "Décomposition (tendance, saisonnalité)":
                    period = st.slider("Période estimée (en unités de temps)", 1, 100, 24)
                    if len(processed_df) > 2 * period:
                        result = seasonal_decompose(processed_df[value_col].dropna(), model='additive', period=period)
                        
                        # Plot components
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=processed_df[time_col], y=result.observed, mode='lines', name='Observé'))
                        fig.add_trace(go.Scatter(x=processed_df[time_col], y=result.trend, mode='lines', name='Tendance'))
                        fig.add_trace(go.Scatter(x=processed_df[time_col], y=result.seasonal, mode='lines', name='Saisonnalité'))
                        fig.add_trace(go.Scatter(x=processed_df[time_col], y=result.resid, mode='lines', name='Résidu'))
                        fig.update_layout(title="Décomposition de la série temporelle", 
                                          xaxis_title="Temps", yaxis_title=value_col, template="plotly_white")
                        st.plotly_chart(fig)
                    else:
                        st.warning("Les données sont trop courtes pour une décomposition avec cette période.")

                elif analysis_type == "Détection d'anomalies":
                    z_score_threshold = st.slider("Seuil de Z-score pour anomalies", 1.0, 5.0, 3.0)
                    rolling_mean = processed_df[value_col].rolling(window=5, center=True).mean()
                    rolling_std = processed_df[value_col].rolling(window=5, center=True).std()
                    anomalies = processed_df[
                        (processed_df[value_col] > rolling_mean + z_score_threshold * rolling_std) |
                        (processed_df[value_col] < rolling_mean - z_score_threshold * rolling_std)
                    ]
                    
                    fig = px.line(processed_df, x=time_col, y=value_col, title="Détection d'anomalies", 
                                  template="plotly_white")
                    fig.add_scatter(x=anomalies[time_col], y=anomalies[value_col], mode='markers', 
                                    name='Anomalies', marker=dict(color='red', size=10))
                    fig.update_layout(xaxis_title="Temps", yaxis_title=value_col)
                    st.plotly_chart(fig)
                    st.write(f"**Nombre d'anomalies détectées :** {len(anomalies)}")

                elif analysis_type == "Statistiques de base":
                    stats = processed_df[value_col].describe()
                    st.write("**Statistiques descriptives :**")
                    st.table(stats)

                    # Autocorrelation plot
                    autocorr = [processed_df[value_col].autocorr(lag=i) for i in range(1, 11)]
                    fig = px.bar(x=list(range(1, 11)), y=autocorr, title="Autocorrélation (lags 1-10)", 
                                 template="plotly_white")
                    fig.update_layout(xaxis_title="Lag", yaxis_title="Autocorrélation")
                    st.plotly_chart(fig)

            # Export results
            with st.container():
                st.markdown("### 💾 Exporter les Résultats")
                if st.button("Exporter les données prétraitées en CSV"):
                    buf = StringIO()
                    processed_df.to_csv(buf, index=False)
                    buf.seek(0)
                    st.download_button(
                        label="Télécharger le CSV",
                        data=buf.getvalue(),
                        file_name=f"processed_timeseries_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )