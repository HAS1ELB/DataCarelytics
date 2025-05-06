import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, roc_curve, auc, precision_recall_curve, 
                            mean_squared_error, r2_score, mean_absolute_error)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabular_data.utils import (create_confusion_matrix_plot, create_roc_curve_plot, 
                  create_precision_recall_curve_plot, create_learning_curve_plot, 
                  display_metrics_dashboard)

def handle_machine_learning():
    st.markdown("<div class='category-box'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Advanced Machine Learning</h3>", unsafe_allow_html=True)
    
    # Vérifier si le prétraitement est terminé
    if 'preprocessing_complete' not in st.session_state or not st.session_state['preprocessing_complete']:
        st.error("❌ Preprocessing is not complete. Please finalize preprocessing in the 'Advanced Data Preprocessing' section.")
        return
    
    # Vérifier si le DataFrame prétraité existe
    if 'df' not in st.session_state:
        st.error("❌ No dataset found. Please upload and preprocess a dataset in the 'Advanced Data Preprocessing' section.")
        return
    
    df = st.session_state['df']
    
    # Vérifier la présence de la colonne cible
    if 'target' not in df.columns:
        st.error("❌ No 'target' column found in the dataset. Please select a target column in the 'Advanced Data Preprocessing' section.")
        return
    
    # Vérifier qu'il y a au moins une feature
    if df.shape[1] <= 1:
        st.error("❌ The dataset has too few columns. Ensure you have at least one feature in addition to the target column.")
        return
    
    # Vérifier l'absence de valeurs manquantes
    if df.isnull().sum().sum() > 0:
        st.error("❌ The dataset contains missing values. Please handle them in the 'Advanced Data Preprocessing' section.")
        return
    
    # Determine problem type based on target variable
    target_unique = df['target'].nunique()
    is_classification = target_unique <= 10  # Assume classification if <= 10 unique values
    
    problem_type = st.radio(
        "Problem Type:",
        ["Classification", "Regression"],
        index=0 if is_classification else 1,
        horizontal=True
    )
    
    ml_tabs = st.tabs(["🔄 Model Setup", "🔬 Training & Evaluation", "📊 Interpretability", "🧪 Cross-Validation", "📈 Model Comparison"])
    
    with ml_tabs[0]:
        st.subheader("Model Configuration")
        
        # Select features
        all_columns = df.columns.tolist()
        feature_cols = [col for col in all_columns if col != 'target']
        
        selected_features = st.multiselect(
            "Select features for modeling",
            feature_cols,
            default=feature_cols
        )
        
        if not selected_features:
            st.warning("⚠️ Please select at least one feature for modeling.")
            return
        
        # Model selection (allow multiple models)
        if problem_type == "Classification":
            model_options = [
                "Logistic Regression",
                "Random Forest",
                "Support Vector Machine",
                "K-Nearest Neighbors",
                "Gradient Boosting",
                "AdaBoost",
                "Neural Network (MLP)",
                "XGBoost",
                "LightGBM",
                "CatBoost"
            ]
        else:
            model_options = [
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "Random Forest",
                "Support Vector Machine",
                "K-Nearest Neighbors",
                "Gradient Boosting",
                "AdaBoost",
                "Neural Network (MLP)",
                "XGBoost",
                "LightGBM",
                "CatBoost"
            ]
        
        selected_models = st.multiselect(
            "Select Models to Train",
            model_options,
            default=["Random Forest"]  # Default to Random Forest for simplicity
        )
        
        if not selected_models:
            st.warning("⚠️ Please select at least one model to train.")
            return
        
        # Train-test split
        test_size = st.slider(
            "Test set size (proportion)",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )
        
        # Store configuration in session state
        st.session_state['ml_config'] = {
            'problem_type': problem_type,
            'selected_features': selected_features,
            'selected_models': selected_models,
            'test_size': test_size
        }
    
    with ml_tabs[1]:
        st.subheader("Model Training & Evaluation")
        
        if 'ml_config' not in st.session_state:
            st.info("Please configure the model in the 'Model Setup' tab first.")
            return
        
        if st.button("Train Selected Models"):
            with st.spinner("Training models..."):
                try:
                    # Prepare data
                    X = df[selected_features]
                    y = df['target']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Dictionary to store results
                    results = {}
                    
                    # Train each selected model
                    for selected_model in selected_models:
                        # Initialize model
                        if problem_type == "Classification":
                            if selected_model == "Logistic Regression":
                                model = LogisticRegression(random_state=42)
                            elif selected_model == "Random Forest":
                                model = RandomForestClassifier(n_estimators=100, random_state=42)
                            elif selected_model == "Support Vector Machine":
                                model = SVC(probability=True, random_state=42)
                            elif selected_model == "K-Nearest Neighbors":
                                model = KNeighborsClassifier()
                            elif selected_model == "Gradient Boosting":
                                model = GradientBoostingClassifier(random_state=42)
                            elif selected_model == "AdaBoost":
                                model = AdaBoostClassifier(random_state=42)
                            elif selected_model == "Neural Network (MLP)":
                                model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
                            elif selected_model == "XGBoost":
                                model = xgb.XGBClassifier(random_state=42)
                            elif selected_model == "LightGBM":
                                model = lgb.LGBMClassifier(random_state=42)
                            elif selected_model == "CatBoost":
                                model = CatBoostClassifier(verbose=0, random_state=42)
                        else:  # Regression
                            if selected_model == "Linear Regression":
                                model = LinearRegression()
                            elif selected_model == "Ridge Regression":
                                model = Ridge(random_state=42)
                            elif selected_model == "Lasso Regression":
                                model = Lasso(random_state=42)
                            elif selected_model == "Random Forest":
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                            elif selected_model == "Support Vector Machine":
                                model = SVR()
                            elif selected_model == "K-Nearest Neighbors":
                                model = KNeighborsRegressor()
                            elif selected_model == "Gradient Boosting":
                                model = GradientBoostingRegressor(random_state=42)
                            elif selected_model == "AdaBoost":
                                model = AdaBoostRegressor(random_state=42)
                            elif selected_model == "Neural Network (MLP)":
                                model = MLPRegressor(hidden_layer_sizes=(100,), random_state=42)
                            elif selected_model == "XGBoost":
                                model = xgb.XGBRegressor(random_state=42)
                            elif selected_model == "LightGBM":
                                model = lgb.LGBMRegressor(random_state=42)
                            elif selected_model == "CatBoost":
                                model = CatBoostRegressor(verbose=0, random_state=42)
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        if problem_type == "Classification":
                            y_pred = model.predict(X_test)
                            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                        else:
                            y_pred = model.predict(X_test)
                            y_pred_proba = None
                        
                        # Calculate metrics
                        if problem_type == "Classification":
                            metrics = {
                                "Accuracy": accuracy_score(y_test, y_pred),
                                "Precision": precision_score(y_test, y_pred, average='weighted'),
                                "Recall": recall_score(y_test, y_pred, average='weighted'),
                                "F1-Score": f1_score(y_test, y_pred, average='weighted')
                            }
                        else:
                            metrics = {
                                "MSE": mean_squared_error(y_test, y_pred),
                                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                                "MAE": mean_absolute_error(y_test, y_pred),
                                "R² Score": r2_score(y_test, y_pred)
                            }
                        
                        # Store results
                        results[selected_model] = {
                            'model': model,
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba,
                            'metrics': metrics,
                            'X_train': X_train,
                            'X_test': X_test,
                            'y_test': y_test
                        }
                    
                    # Display results for each model
                    for model_name, result in results.items():
                        st.markdown(f"### Results for {model_name}")
                        
                        # Display metrics
                        st.write("Model Performance Metrics:")
                        display_metrics_dashboard({
                            k: f"{v:.4f}" for k, v in result['metrics'].items()
                        })
                        
                        # Visualizations
                        if problem_type == "Classification":
                            # Confusion Matrix
                            cm = confusion_matrix(result['y_test'], result['y_pred'])
                            fig_cm = create_confusion_matrix_plot(cm)
                            st.plotly_chart(fig_cm, use_container_width=True)
                            
                            # ROC Curve (if binary classification)
                            if result['y_pred_proba'] is not None and len(np.unique(result['y_test'])) == 2:
                                fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                                roc_auc = auc(fpr, tpr)
                                fig_roc = create_roc_curve_plot(fpr, tpr, roc_auc)
                                st.plotly_chart(fig_roc, use_container_width=True)
                            
                            # Precision-Recall Curve (if binary classification)
                            if result['y_pred_proba'] is not None and len(np.unique(result['y_test'])) == 2:
                                precision, recall, _ = precision_recall_curve(result['y_test'], result['y_pred_proba'])
                                avg_precision = auc(recall, precision)
                                fig_pr = create_precision_recall_curve_plot(precision, recall, avg_precision)
                                st.plotly_chart(fig_pr, use_container_width=True)
                        
                        else:  # Regression
                            # Predicted vs Actual Plot
                            fig = px.scatter(
                                x=result['y_test'], y=result['y_pred'],
                                labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                title=f'Predicted vs Actual Values ({model_name})'
                            )
                            fig.add_scatter(x=[result['y_test'].min(), result['y_test'].max()], 
                                          y=[result['y_test'].min(), result['y_test'].max()], 
                                          mode='lines', 
                                          name='Ideal')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Store results in session state for comparison
                    st.session_state['ml_results'] = results
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    
                    st.success("✅ All selected models trained and evaluated successfully!")
                
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
    
    with ml_tabs[2]:
        st.subheader("Model Interpretability")
        
        if 'ml_results' not in st.session_state:
            st.info("Please train models in the 'Training & Evaluation' tab first.")
            return
        
        # Select model for SHAP analysis
        model_name = st.selectbox("Select Model for SHAP Analysis", list(st.session_state['ml_results'].keys()))
        model = st.session_state['ml_results'][model_name]['model']
        X_train = st.session_state['ml_results'][model_name]['X_train']
        
        if st.button("Generate SHAP Feature Importance"):
            with st.spinner("Calculating SHAP values..."):
                try:
                    # Use a smaller sample for SHAP to avoid long computation
                    X_sample = X_train.sample(min(100, len(X_train)), random_state=42)
                    
                    # Initialize SHAP explainer
                    if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor, 
                                      lgb.LGBMClassifier, lgb.LGBMRegressor,
                                      CatBoostClassifier, CatBoostRegressor,
                                      GradientBoostingClassifier, GradientBoostingRegressor)):
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.KernelExplainer(model.predict, X_sample)
                    
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(X_sample)
                    
                    # For classification, use absolute SHAP values for multi-class
                    if problem_type == "Classification" and len(np.unique(df['target'])) > 2:
                        shap_values = np.abs(shap_values).mean(axis=0)
                    elif problem_type == "Classification":
                        shap_values = shap_values[1]  # Use positive class for binary classification
                    
                    # Create feature importance plot
                    importance_df = pd.DataFrame({
                        'Feature': X_train.columns,
                        'SHAP Importance': np.abs(shap_values).mean(axis=0)
                    }).sort_values('SHAP Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x='SHAP Importance',
                        y='Feature',
                        orientation='h',
                        title=f'SHAP Feature Importance ({model_name})'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display SHAP summary
                    st.write("SHAP Summary Plot:")
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
                    st.pyplot(fig)
                    plt.close(fig)  # Close the figure to prevent memory leaks
                
                except Exception as e:
                    st.error(f"Error calculating SHAP values: {str(e)}")
    
    with ml_tabs[3]:
        st.subheader("Cross-Validation & Learning Curves")
        
        if 'ml_config' not in st.session_state:
            st.info("Please configure the model in the 'Model Setup' tab first.")
            return
        
        # Select model for cross-validation
        selected_model = st.selectbox("Select Model for Cross-Validation", st.session_state['ml_config']['selected_models'])
        
        # Cross-validation
        cv_folds = st.slider("Number of CV folds", min_value=3, max_value=10, value=5)
        
        if st.button("Perform Cross-Validation"):
            with st.spinner("Performing cross-validation..."):
                try:
                    X = df[selected_features]
                    y = df['target']
                    
                    # Initialize model
                    if problem_type == "Classification":
                        if selected_model == "Logistic Regression":
                            model = LogisticRegression(random_state=42)
                        elif selected_model == "Random Forest":
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                        elif selected_model == "Support Vector Machine":
                            model = SVC(probability=True, random_state=42)
                        elif selected_model == "K-Nearest Neighbors":
                            model = KNeighborsClassifier()
                        elif selected_model == "Gradient Boosting":
                            model = GradientBoostingClassifier(random_state=42)
                        elif selected_model == "AdaBoost":
                            model = AdaBoostClassifier(random_state=42)
                        elif selected_model == "Neural Network (MLP)":
                            model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
                        elif selected_model == "XGBoost":
                            model = xgb.XGBClassifier(random_state=42)
                        elif selected_model == "LightGBM":
                            model = lgb.LGBMClassifier(random_state=42)
                        elif selected_model == "CatBoost":
                            model = CatBoostClassifier(verbose=0, random_state=42)
                    else:  # Regression
                        if selected_model == "Linear Regression":
                            model = LinearRegression()
                        elif selected_model == "Ridge Regression":
                            model = Ridge(random_state=42)
                        elif selected_model == "Lasso Regression":
                            model = Lasso(random_state=42)
                        elif selected_model == "Random Forest":
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif selected_model == "Support Vector Machine":
                            model = SVR()
                        elif selected_model == "K-Nearest Neighbors":
                            model = KNeighborsRegressor()
                        elif selected_model == "Gradient Boosting":
                            model = GradientBoostingRegressor(random_state=42)
                        elif selected_model == "AdaBoost":
                            model = AdaBoostRegressor(random_state=42)
                        elif selected_model == "Neural Network (MLP)":
                            model = MLPRegressor(hidden_layer_sizes=(100,), random_state=42)
                        elif selected_model == "XGBoost":
                            model = xgb.XGBRegressor(random_state=42)
                        elif selected_model == "LightGBM":
                            model = lgb.LGBMRegressor(random_state=42)
                        elif selected_model == "CatBoost":
                            model = CatBoostRegressor(verbose=0, random_state=42)
                    
                    # Perform cross-validation
                    scoring = 'accuracy' if problem_type == "Classification" else 'r2'
                    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
                    
                    # Display CV results
                    st.write(f"Cross-Validation {scoring.capitalize()} Scores for {selected_model}:")
                    cv_results = pd.DataFrame({
                        'Fold': [f"Fold {i+1}" for i in range(cv_folds)],
                        'Score': cv_scores
                    })
                    st.dataframe(cv_results, use_container_width=True)
                    
                    metrics = {
                        "Mean Score": f"{cv_scores.mean():.4f}",
                        "Std Dev": f"{cv_scores.std():.4f}",
                        "Min Score": f"{cv_scores.min():.4f}",
                        "Max Score": f"{cv_scores.max():.4f}"
                    }
                    display_metrics_dashboard(metrics)
                    
                    # Plot CV scores
                    fig = px.bar(
                        cv_results,
                        x='Fold',
                        y='Score',
                        title=f'{scoring.capitalize()} Scores Across Folds ({selected_model})'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error during cross-validation: {str(e)}")
        
        # Learning Curves
        if st.button("Generate Learning Curves"):
            with st.spinner("Generating learning curves..."):
                try:
                    X = df[selected_features]
                    y = df['target']
                    
                    # Initialize model
                    if problem_type == "Classification":
                        if selected_model == "Logistic Regression":
                            model = LogisticRegression(random_state=42)
                        elif selected_model == "Random Forest":
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                        elif selected_model == "Support Vector Machine":
                            model = SVC(probability=True, random_state=42)
                        elif selected_model == "K-Nearest Neighbors":
                            model = KNeighborsClassifier()
                        elif selected_model == "Gradient Boosting":
                            model = GradientBoostingClassifier(random_state=42)
                        elif selected_model == "AdaBoost":
                            model = AdaBoostClassifier(random_state=42)
                        elif selected_model == "Neural Network (MLP)":
                            model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
                        elif selected_model == "XGBoost":
                            model = xgb.XGBClassifier(random_state=42)
                        elif selected_model == "LightGBM":
                            model = lgb.LGBMClassifier(random_state=42)
                        elif selected_model == "CatBoost":
                            model = CatBoostClassifier(verbose=0, random_state=42)
                    else:  # Regression
                        if selected_model == "Linear Regression":
                            model = LinearRegression()
                        elif selected_model == "Ridge Regression":
                            model = Ridge(random_state=42)
                        elif selected_model == "Lasso Regression":
                            model = Lasso(random_state=42)
                        elif selected_model == "Random Forest":
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif selected_model == "Support Vector Machine":
                            model = SVR()
                        elif selected_model == "K-Nearest Neighbors":
                            model = KNeighborsRegressor()
                        elif selected_model == "Gradient Boosting":
                            model = GradientBoostingRegressor(random_state=42)
                        elif selected_model == "AdaBoost":
                            model = AdaBoostRegressor(random_state=42)
                        elif selected_model == "Neural Network (MLP)":
                            model = MLPRegressor(hidden_layer_sizes=(100,), random_state=42)
                        elif selected_model == "XGBoost":
                            model = xgb.XGBRegressor(random_state=42)
                        elif selected_model == "LightGBM":
                            model = lgb.LGBMRegressor(random_state=42)
                        elif selected_model == "CatBoost":
                            model = CatBoostRegressor(verbose=0, random_state=42)
                    
                    # Calculate learning curves
                    train_sizes, train_scores, test_scores = learning_curve(
                        model, X, y, cv=5, 
                        scoring='accuracy' if problem_type == "Classification" else 'r2',
                        train_sizes=np.linspace(0.1, 1.0, 10)
                    )
                    
                    # Plot learning curves
                    fig = create_learning_curve_plot(train_sizes, train_scores, test_scores)
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error generating learning curves: {str(e)}")
    
    with ml_tabs[4]:
        st.subheader("Model Comparison")
        
        if 'ml_results' not in st.session_state:
            st.info("Please train models in the 'Training & Evaluation' tab first.")
            return
        
        results = st.session_state['ml_results']
        
        # Create comparison table
        metrics_df = pd.DataFrame({
            model_name: result['metrics'] for model_name, result in results.items()
        }).T
        
        st.write("Performance Metrics Comparison:")
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
        
        # Plot comparison bar chart
        if problem_type == "Classification":
            fig = go.Figure()
            for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                fig.add_trace(go.Bar(
                    x=metrics_df.index,
                    y=metrics_df[metric],
                    name=metric
                ))
            fig.update_layout(
                title="Model Performance Comparison (Classification Metrics)",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curves comparison (if binary classification)
            if len(np.unique(df['target'])) == 2:
                fig = go.Figure()
                for model_name, result in results.items():
                    if result['y_pred_proba'] is not None:
                        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                        roc_auc = auc(fpr, tpr)
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            name=f"{model_name} (AUC = {roc_auc:.2f})",
                            mode='lines'
                        ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    line=dict(dash='dash'), name='Random Guess'
                ))
                fig.update_layout(
                    title="ROC Curves Comparison",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:  # Regression
            fig = go.Figure()
            for metric in ["MSE", "RMSE", "MAE", "R² Score"]:
                fig.add_trace(go.Bar(
                    x=metrics_df.index,
                    y=metrics_df[metric],
                    name=metric
                ))
            fig.update_layout(
                title="Model Performance Comparison (Regression Metrics)",
                xaxis_title="Model",
                yaxis_title="Metric Value",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional visualization: Predicted vs Actual for all models
            fig = go.Figure()
            for model_name, result in results.items():
                fig.add_trace(go.Scatter(
                    x=result['y_test'],
                    y=result['y_pred'],
                    mode='markers',
                    name=model_name,
                    opacity=0.6
                ))
            fig.add_trace(go.Scatter(
                x=[df['target'].min(), df['target'].max()],
                y=[df['target'].min(), df['target'].max()],
                mode='lines',
                name='Ideal',
                line=dict(dash='dash')
            ))
            fig.update_layout(
                title="Predicted vs Actual Values (All Models)",
                xaxis_title="Actual Values",
                yaxis_title="Predicted Values",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)