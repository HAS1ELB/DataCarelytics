import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def create_feature_importance_plot(importance, features, plot_type='bar'):
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    if plot_type == 'bar':
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale='blues')
    else:  # plot_type == 'scatter'
        fig = px.scatter(importance_df, x=range(len(importance_df)), y='Importance',
                        hover_name='Feature',
                        title='Feature Importance',
                        labels={'x': 'Feature Rank', 'y': 'Importance'},
                        color='Importance',
                        color_continuous_scale='blues')
    
    fig.update_layout(height=500, width=700)
    return fig

def create_confusion_matrix_plot(cm, classes=None):
    if classes is None:
        classes = [str(i) for i in range(len(cm))]
    
    fig = px.imshow(cm, 
                   x=classes, 
                   y=classes,
                   labels=dict(x="Predicted", y="True", color="Count"),
                   text_auto=True,
                   color_continuous_scale='blues')
    
    fig.update_layout(title='Confusion Matrix',
                     width=600, height=500)
    return fig

def create_roc_curve_plot(fpr, tpr, auc_score):
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc_score:.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(yaxis_scaleanchor='x')
    return fig

def create_precision_recall_curve_plot(precision, recall, avg_precision):
    fig = px.area(
        x=recall, y=precision,
        title=f'Precision-Recall Curve (AP={avg_precision:.4f})',
        labels=dict(x='Recall', y='Precision'),
        width=700, height=500
    )
    fig.update_yaxes(range=[0, 1.05])
    fig.update_xaxes(range=[0, 1])
    return fig

def create_learning_curve_plot(train_sizes, train_scores, test_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_scores_mean,
        mode='lines+markers',
        name='Training score',
        line=dict(color='blue'),
        error_y=dict(
            type='data',
            array=train_scores_std,
            visible=True,
            color='blue',
            thickness=1.5,
            width=3
        )
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_scores_mean,
        mode='lines+markers',
        name='Cross-validation score',
        line=dict(color='red'),
        error_y=dict(
            type='data',
            array=test_scores_std,
            visible=True,
            color='red',
            thickness=1.5,
            width=3
        )
    ))
    
    fig.update_layout(
        title='Learning Curves',
        xaxis_title='Training Examples',
        yaxis_title='Score',
        width=700, height=500
    )
    
    return fig

def create_image_preview(img, title="Image Preview", colormap=None):
    fig = px.imshow(img, title=title, color_continuous_scale=colormap)
    fig.update_layout(coloraxis_showscale=False)
    return fig

def display_metrics_dashboard(metrics_dict):
    cols = st.columns(len(metrics_dict))
    for i, (metric_name, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{value}</div>
                <div class='metric-label'>{metric_name}</div>
            </div>
            """, unsafe_allow_html=True)

def display_home_page():
    st.write("")  # Add some spacing
    
    # Create a modern 3-column layout with feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card animate-fade-in delay-1'>
            <div class='feature-icon'>📊</div>
            <div class='feature-title'>Tabular Data Analysis</div>
            <p>Advanced analysis of medical datasets with state-of-the-art machine learning algorithms,
            feature selection, and model interpretability tools.</p>
            <ul>
                <li>Ensemble methods (XGBoost, LightGBM, CatBoost)</li>
                <li>Automated feature engineering</li>
                <li>Explainable AI with SHAP values</li>
                <li>Hyperparameter optimization</li>
                <li>Cross-validation strategies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card animate-fade-in delay-2'>
            <div class='feature-icon'>🔬</div>
            <div class='feature-title'>Image Processing</div>
            <p>Comprehensive medical image analysis with enhanced visualization, segmentation,
            and feature extraction techniques.</p>
            <ul>
                <li>Advanced image enhancement methods</li>
                <li>Multi-algorithm segmentation approaches</li>
                <li>Deep learning-based classification</li>
                <li>Texture and feature analysis</li>
                <li>DICOM file support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card animate-fade-in delay-3'>
            <div class='feature-icon'>📝</div>
            <div class='feature-title'>Medical Text Analysis</div>
            <p>Extract valuable insights from clinical notes and medical literature using
            advanced NLP techniques.</p>
            <ul>
                <li>Advanced text preprocessing</li>
                <li>Medical named entity recognition</li>
                <li>Topic modeling with LDA</li>
                <li>Transformer-based models</li>
                <li>Sentiment and emotion analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")  # Add some spacing
    
    # New section: deep learning
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-card animate-fade-in delay-4'>
            <div class='feature-icon'>🧠</div>
            <div class='feature-title'>Deep Learning</div>
            <p>Apply state-of-the-art deep learning models to medical data for superior prediction accuracy.</p>
            <ul>
                <li>Transfer learning with pre-trained models</li>
                <li>Custom neural network architectures</li>
                <li>Convolutional networks for imaging</li>
                <li>Recurrent networks for sequence data</li>
                <li>Transformers for medical text</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card animate-fade-in delay-5'>
            <div class='feature-icon'>📈</div>
            <div class='feature-title'>Interactive Visualization</div>
            <p>Explore your data with dynamic, interactive visualizations for better insights.</p>
            <ul>
                <li>Plotly-powered interactive charts</li>
                <li>3D visualization capabilities</li>
                <li>Custom dashboards</li>
                <li>Time-series analysis tools</li>
                <li>Comparative visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")  # Add some spacing
    
    # Getting started section
    st.markdown("<h2 class='sub-header'>Getting Started</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='category-box animate-fade-in'>
        <p>Follow these steps to begin your advanced medical data analysis:</p>
        <ol>
            <li><strong>Select a module</strong> from the sidebar navigation</li>
            <li><strong>Upload your data</strong> in the supported formats</li>
            <li><strong>Choose analysis techniques</strong> from the comprehensive options</li>
            <li><strong>Review the results</strong> using our advanced visualization tools</li>
            <li><strong>Export your findings</strong> for use in reports or publications</li>
        </ol>
        <p>Each module offers best-in-class algorithms and visualization capabilities, specifically optimized for medical data analysis.</p>
        <p>All processing is done locally, ensuring your sensitive medical data remains secure and private.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example showcase
    st.markdown("<h2 class='sub-header'>Example Applications</h2>", unsafe_allow_html=True)
    
    examples = st.columns(2)
    
    with examples[0]:
        st.markdown("""
        <div class='category-box'>
            <h3>Clinical Prediction Models</h3>
            <p>Develop models to predict patient outcomes, disease progression, treatment response, 
            and readmission risk using structured clinical data.</p>
            <p><strong>Techniques:</strong> Ensemble learning, survival analysis, time-series forecasting</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='category-box'>
            <h3>Medical Image Classification</h3>
            <p>Automatically classify medical images for disease detection, severity assessment, 
            and anatomical structure identification.</p>
            <p><strong>Techniques:</strong> Convolutional neural networks, transfer learning, gradient-weighted class activation mapping</p>
        </div>
        """, unsafe_allow_html=True)
    
    with examples[1]:
        st.markdown("""
        <div class='category-box'>
            <h3>Clinical Text Mining</h3>
            <p>Extract structured information from unstructured clinical notes, discharge summaries, 
            and medical literature.</p>
            <p><strong>Techniques:</strong> Transformer models, named entity recognition, relation extraction</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='category-box'>
            <h3>Multimodal Analysis</h3>
            <p>Combine data from multiple sources (imaging, genomics, clinical variables) for comprehensive 
            patient analysis and personalized medicine.</p>
            <p><strong>Techniques:</strong> Feature fusion, multi-view learning, ensemble strategies</p>
        </div>
        """, unsafe_allow_html=True)