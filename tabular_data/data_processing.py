import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, RobustScaler, LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tabular_data.utils import create_feature_importance_plot, display_metrics_dashboard

def handle_tabular_data_analysis():
    st.markdown("<h2 class='sub-header'>Advanced Tabular Data Analysis</h2>", unsafe_allow_html=True)
    
    # Initialiser la liste des étapes de prétraitement
    if 'preprocessing_steps' not in st.session_state:
        st.session_state['preprocessing_steps'] = []
    if 'preprocessing_complete' not in st.session_state:
        st.session_state['preprocessing_complete'] = False
    
    # File upload section
    st.markdown("<div class='category-box'>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📤 Upload Data", "🧪 Sample datasets"])
    
    with tab1:
        st.subheader("Upload Your Dataset")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file:
            try:
                with st.spinner("Loading and processing your data..."):
                    # Read the data
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state['df'] = df
                    st.session_state['preprocessing_steps'].append(f"Uploaded dataset: {uploaded_file.name} ({df.shape[0]} rows × {df.shape[1]} columns)")
                    st.session_state['preprocessing_complete'] = False
                    st.success(f"✅ File uploaded successfully! Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with tab2:
        st.subheader("Use a Sample Dataset")
        sample_dataset = st.selectbox(
            "Select a sample medical dataset",
            ["Diabetes", "Heart Disease", "Breast Cancer", "Thyroid Disease"]
        )
        
        if st.button("Load Sample Dataset"):
            with st.spinner("Loading sample dataset..."):
                if sample_dataset == "Diabetes":
                    try:
                        from sklearn.datasets import load_diabetes
                        data = load_diabetes(as_frame=True)
                        df = data.data
                        df['Outcome'] = data.target  # Renamed to avoid 'target'
                        st.session_state['preprocessing_steps'].append(f"Loaded sample dataset: Diabetes ({df.shape[0]} rows × {df.shape[1]} columns)")
                        st.success("✅ Diabetes dataset loaded successfully!")
                    except:
                        df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
                        st.session_state['preprocessing_steps'].append(f"Loaded sample dataset: Diabetes from CSV ({df.shape[0]} rows × {df.shape[1]} columns)")
                        st.success("✅ Diabetes dataset loaded successfully!")
                
                elif sample_dataset == "Heart Disease":
                    try:
                        from sklearn.datasets import fetch_ucirepo
                        heart = fetch_ucirepo(id=45)
                        df = heart.data.features
                        df['Condition'] = heart.data.targets  # Renamed to avoid 'target'
                        st.session_state['preprocessing_steps'].append(f"Loaded sample dataset: Heart Disease ({df.shape[0]} rows × {df.shape[1]} columns)")
                        st.success("✅ Heart Disease dataset loaded successfully!")
                    except:
                        df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/heart.csv")
                        st.session_state['preprocessing_steps'].append(f"Loaded sample dataset: Heart Disease from CSV ({df.shape[0]} rows × {df.shape[1]} columns)")
                        st.success("✅ Heart Disease dataset loaded successfully!")
                
                elif sample_dataset == "Breast Cancer":
                    try:
                        from sklearn.datasets import load_breast_cancer
                        data = load_breast_cancer(as_frame=True)
                        df = data.data
                        df['Diagnosis'] = data.target  # Renamed to avoid 'target'
                        st.session_state['preprocessing_steps'].append(f"Loaded sample dataset: Breast Cancer ({df.shape[0]} rows × {df.shape[1]} columns)")
                        st.success("✅ Breast Cancer dataset loaded successfully!")
                    except:
                        st.error("Could not load breast cancer dataset.")
                        df = None
                
                elif sample_dataset == "Thyroid Disease":
                    try:
                        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/thyroid0387.data", header=None)
                        st.session_state['preprocessing_steps'].append(f"Loaded sample dataset: Thyroid Disease ({df.shape[0]} rows × {df.shape[1]} columns)")
                        st.success("✅ Thyroid Disease dataset loaded successfully!")
                    except:
                        st.error("Could not load thyroid disease dataset.")
                        df = None
                
                if df is not None:
                    st.session_state['df'] = df
                    st.session_state['preprocessing_complete'] = False
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Main analysis section - only show if data is loaded
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Data Overview
        st.markdown("<div class='category-box'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Data Overview</h3>", unsafe_allow_html=True)
        
        overview_tabs = st.tabs(["📋 Preview", "📊 Summary", "📈 Distributions", "🔄 Correlations"])
        
        with overview_tabs[0]:
            st.dataframe(df.head(10), use_container_width=True)
            st.text(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        with overview_tabs[1]:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("Statistical Summary")
                st.dataframe(df.describe(), use_container_width=True)
            
            with col2:
                st.write("Data Types")
                dtypes_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Missing': df.isnull().sum(),
                    'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(dtypes_df, use_container_width=True)
                
                # Missing values summary
                if df.isnull().sum().sum() > 0:
                    missing_cols = df.columns[df.isnull().any()].tolist()
                    st.warning(f"⚠️ Found {len(missing_cols)} columns with missing values.")
                else:
                    st.success("✅ No missing values detected in the dataset.")
        
        with overview_tabs[2]:
            # Select column to visualize
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column to visualize", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram with KDE
                    fig = px.histogram(df, x=selected_col, histnorm='probability density',
                                      title=f"Distribution of {selected_col}",
                                      marginal="box")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Basic stats
                    stats = {
                        'Mean': f"{df[selected_col].mean():.4f}",
                        'Median': f"{df[selected_col].median():.4f}",
                        'Std Dev': f"{df[selected_col].std():.4f}",
                        'Min': f"{df[selected_col].min():.4f}",
                        'Max': f"{df[selected_col].max():.4f}"
                    }
                    
                    display_metrics_dashboard(stats)
            else:
                st.info("No numeric columns available for distribution visualization.")
        
        with overview_tabs[3]:
            if len(numeric_cols) > 1:
                corr_method = st.radio("Correlation Method", ["Pearson", "Spearman"], horizontal=True)
                
                # Calculate correlation matrix
                corr_matrix = df[numeric_cols].corr(method=corr_method.lower())
                
                # Plot interactive heatmap
                fig = px.imshow(corr_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               color_continuous_scale='RdBu_r',
                               title=f"{corr_method} Correlation Matrix")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top correlations
                st.subheader("Top Correlations")
                
                # Get the upper triangle of the correlation matrix
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                corr_df = corr_matrix.mask(mask).stack().reset_index()
                corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
                corr_df = corr_df.sort_values('Correlation', ascending=False)
                
                # Display top positive and negative correlations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Strongest Positive Correlations")
                    st.dataframe(corr_df.head(5), use_container_width=True)
                
                with col2:
                    st.write("Strongest Negative Correlations")
                    st.dataframe(corr_df.sort_values('Correlation').head(5), use_container_width=True)
                
                # Scatter plot for selected variables
                st.subheader("Relationship Explorer")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    x_var = st.selectbox("X Variable", numeric_cols)
                
                with col2:
                    y_var = st.selectbox("Y Variable", [col for col in numeric_cols if col != x_var])
                
                fig = px.scatter(df, x=x_var, y=y_var,
                               title=f"Relationship between {x_var} and {y_var}",
                               trendline="ols")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns to calculate correlations.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data Preprocessing
        st.markdown("<div class='category-box'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Advanced Data Preprocessing</h3>", unsafe_allow_html=True)
        
        preprocessing_tabs = st.tabs(["🧹 Cleaning", "📊 Feature Engineering", "🔄 Transformation", "🎯 Feature Selection", "✅ Finalize Preprocessing"])
        
        with preprocessing_tabs[0]:
            st.write("Handle Missing Values")
            
            # Get columns with missing values
            missing_cols = df.columns[df.isnull().any()].tolist()
            
            if missing_cols:
                st.write(f"Found {len(missing_cols)} columns with missing values:")
                missing_df = pd.DataFrame({
                    'Column': missing_cols,
                    'Missing Count': [df[col].isnull().sum() for col in missing_cols],
                    'Missing Percentage': [f"{df[col].isnull().sum() / len(df) * 100:.2f}%" for col in missing_cols]
                })
                st.dataframe(missing_df, use_container_width=True)
                
                # Methods for handling missing values
                missing_method = st.radio(
                    "Choose missing value handling method:",
                    ["Drop rows", "Fill with mean/median/mode", "KNN Imputation", "Forward/Backward Fill"],
                    horizontal=True
                )
                
                if st.button("Apply Missing Value Treatment"):
                    with st.spinner("Processing..."):
                        if missing_method == "Drop rows":
                            df_cleaned = df.dropna()
                            st.session_state['preprocessing_steps'].append(f"Dropped rows with missing values (reduced from {len(df)} to {len(df_cleaned)} rows)")
                            st.success(f"✅ Dropped rows with missing values. Rows reduced from {len(df)} to {len(df_cleaned)}")
                        
                        elif missing_method == "Fill with mean/median/mode":
                            df_cleaned = df.copy()
                            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                            
                            # For numeric columns, fill with median
                            for col in numeric_cols:
                                if df_cleaned[col].isnull().any():
                                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                            
                            # For categorical columns, fill with mode
                            for col in categorical_cols:
                                if df_cleaned[col].isnull().any():
                                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                            
                            st.session_state['preprocessing_steps'].append("Filled missing values with mean/median/mode")
                            st.success("✅ Filled missing values with mean/median/mode")
                        
                        elif missing_method == "KNN Imputation":
                            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            if numeric_cols:
                                df_cleaned = df.copy()
                                numeric_missing = [col for col in numeric_cols if df[col].isnull().any()]
                                
                                if numeric_missing:
                                    imputer = KNNImputer(n_neighbors=5)
                                    df_cleaned[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                                    st.session_state['preprocessing_steps'].append(f"Applied KNN imputation to numeric columns: {', '.join(numeric_missing)}")
                                    st.success("✅ Applied KNN imputation to numeric columns with missing values")
                                else:
                                    st.info("No missing values in numeric columns.")
                                
                                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                                for col in categorical_cols:
                                    if df_cleaned[col].isnull().any():
                                        df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                            else:
                                st.error("KNN imputation requires numeric features.")
                                df_cleaned = df.copy()
                        
                        elif missing_method == "Forward/Backward Fill":
                            df_cleaned = df.copy()
                            df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
                            st.session_state['preprocessing_steps'].append("Applied forward/backward fill for missing values")
                            st.success("✅ Applied forward/backward fill to handle missing values")
                        
                        # Update the dataframe in session state
                        st.session_state['df'] = df_cleaned
                        
                        # Show preview of cleaned data
                        st.write("Preview of processed data:")
                        st.dataframe(df_cleaned.head(), use_container_width=True)
            else:
                st.success("✅ No missing values found in the dataset!")
            
            # Handle duplicate rows
            st.write("Handle Duplicate Rows")
            
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                st.warning(f"⚠️ Found {duplicate_count} duplicate rows in the dataset.")
                if st.button("Remove Duplicate Rows"):
                    df_no_dupes = df.drop_duplicates()
                    st.session_state['preprocessing_steps'].append(f"Removed {duplicate_count} duplicate rows (reduced from {len(df)} to {len(df_no_dupes)} rows)")
                    st.success(f"✅ Removed {duplicate_count} duplicate rows. Rows reduced from {len(df)} to {len(df_no_dupes)}")
                    st.session_state['df'] = df_no_dupes
            else:
                st.success("✅ No duplicate rows found in the dataset!")
            
            # Remove irrelevant columns
            st.write("Remove Irrelevant Columns (e.g., ID)")
            all_columns = df.columns.tolist()
            columns_to_remove = st.multiselect(
                "Select columns to remove (e.g., ID, PatientID)",
                all_columns,
                key="remove_columns"
            )
            
            if columns_to_remove and st.button("Remove Selected Columns"):
                with st.spinner("Removing selected columns..."):
                    df_cleaned = df.drop(columns=columns_to_remove)
                    st.session_state['preprocessing_steps'].append(f"Removed columns: {', '.join(columns_to_remove)}")
                    st.success(f"✅ Removed {len(columns_to_remove)} columns: {', '.join(columns_to_remove)}")
                    st.session_state['df'] = df_cleaned
                    st.write("Preview of data after removing columns:")
                    st.dataframe(df_cleaned.head(), use_container_width=True)
            
            # Handle outliers
            st.write("Outlier Detection and Handling")
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column for outlier analysis", numeric_cols, key="outlier_col")
                
                # Calculate IQR boundaries
                Q1 = df[selected_col].quantile(0.25)
                Q3 = df[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)][selected_col]
                outlier_count = len(outliers)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.box(df, y=selected_col, title=f"Boxplot of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Outliers Detected", outlier_count)
                    st.metric("Outlier Percentage", f"{outlier_count / len(df) * 100:.2f}%")
                    st.write(f"Lower bound: {lower_bound:.2f}")
                    st.write(f"Upper bound: {upper_bound:.2f}")
                
                if outlier_count > 0:
                    outlier_method = st.radio(
                        "Choose outlier handling method:",
                        ["Keep outliers", "Remove outliers", "Cap outliers", "Winsorize"],
                        horizontal=True
                    )
                    
                    if st.button("Apply Outlier Treatment"):
                        with st.spinner("Processing..."):
                            df_processed = df.copy()
                            
                            if outlier_method == "Remove outliers":
                                df_processed = df_processed[(df_processed[selected_col] >= lower_bound) & 
                                                         (df_processed[selected_col] <= upper_bound)]
                                st.session_state['preprocessing_steps'].append(f"Removed {outlier_count} outliers from {selected_col} (reduced from {len(df)} to {len(df_processed)} rows)")
                                st.success(f"✅ Removed {outlier_count} outliers from {selected_col}. Rows reduced from {len(df)} to {len(df_processed)}")
                            
                            elif outlier_method == "Cap outliers":
                                df_processed.loc[df_processed[selected_col] < lower_bound, selected_col] = lower_bound
                                df_processed.loc[df_processed[selected_col] > upper_bound, selected_col] = upper_bound
                                st.session_state['preprocessing_steps'].append(f"Capped {outlier_count} outliers in {selected_col} to IQR boundaries")
                                st.success(f"✅ Capped {outlier_count} outliers in {selected_col} to IQR boundaries")
                            
                            elif outlier_method == "Winsorize":
                                df_processed[selected_col] = df_processed[selected_col].clip(
                                    lower=df_processed[selected_col].quantile(0.05),
                                    upper=df_processed[selected_col].quantile(0.95)
                                )
                                st.session_state['preprocessing_steps'].append(f"Winsorized {selected_col} at 5th and 95th percentiles")
                                st.success(f"✅ Winsorized {selected_col} (capped at 5th and 95th percentiles)")
                            
                            else:
                                st.info("Keeping outliers in the dataset.")
                            
                            # Update the dataframe in session state
                            st.session_state['df'] = df_processed
                            
                            # Show histograms before and after
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.histogram(df, x=selected_col, title="Before Outlier Treatment")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = px.histogram(df_processed, x=selected_col, title="After Outlier Treatment")
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns available for outlier analysis.")
        
        with preprocessing_tabs[1]:
            st.write("Feature Engineering")
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            feature_eng_method = st.radio(
                "Choose feature engineering method:",
                ["Polynomial Features", "Interaction Terms", "Binning", "Log Transform", "Custom Formula"],
                horizontal=True
            )
            
            if feature_eng_method == "Polynomial Features":
                if numeric_cols:
                    selected_cols = st.multiselect("Select columns for polynomial features", numeric_cols)
                    
                    if selected_cols:
                        degree = st.slider("Polynomial Degree", min_value=2, max_value=5, value=2)
                        
                        if st.button("Generate Polynomial Features"):
                            with st.spinner("Generating polynomial features..."):
                                df_engineered = df.copy()
                                poly = PolynomialFeatures(degree=degree, include_bias=False)
                                poly_features = poly.fit_transform(df_engineered[selected_cols])
                                
                                feature_names = poly.get_feature_names_out(selected_cols)
                                
                                for i, name in enumerate(feature_names):
                                    if name not in selected_cols:
                                        df_engineered[name] = poly_features[:, i]
                                
                                st.session_state['preprocessing_steps'].append(f"Generated {len(feature_names) - len(selected_cols)} polynomial features from {', '.join(selected_cols)} (degree={degree})")
                                st.success(f"✅ Generated {len(feature_names) - len(selected_cols)} new polynomial features")
                                
                                st.session_state['df'] = df_engineered
                                
                                st.write("Preview with new features:")
                                st.dataframe(df_engineered.head(), use_container_width=True)
                    else:
                        st.info("Please select at least one numeric column.")
                else:
                    st.info("No numeric columns available for polynomial features.")
            
            elif feature_eng_method == "Interaction Terms":
                if len(numeric_cols) >= 2:
                    col1 = st.selectbox("Select first column", numeric_cols, key="inter_col1")
                    col2 = st.selectbox("Select second column", [c for c in numeric_cols if c != col1], key="inter_col2")
                    
                    if st.button("Create Interaction Feature"):
                        with st.spinner("Creating interaction feature..."):
                            df_engineered = df.copy()
                            interaction_name = f"{col1}_{col2}_interaction"
                            df_engineered[interaction_name] = df_engineered[col1] * df_engineered[col2]
                            
                            st.session_state['preprocessing_steps'].append(f"Created interaction feature '{interaction_name}' from {col1} and {col2}")
                            st.success(f"✅ Created interaction feature '{interaction_name}'")
                            
                            st.session_state['df'] = df_engineered
                            
                            st.write("Preview with interaction feature:")
                            st.dataframe(df_engineered.head(), use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns to create interaction terms.")
            
            elif feature_eng_method == "Binning":
                if numeric_cols:
                    selected_col = st.selectbox("Select column for binning", numeric_cols, key="bin_col")
                    num_bins = st.slider("Number of bins", min_value=2, max_value=10, value=4)
                    
                    fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    binning_method = st.radio(
                        "Binning method:", 
                        ["Equal-width", "Equal-frequency (quantiles)", "Custom breaks"],
                        horizontal=True
                    )
                    
                    if binning_method == "Custom breaks":
                        min_val, max_val = float(df[selected_col].min()), float(df[selected_col].max())
                        st.write(f"Column range: {min_val} to {max_val}")
                        
                        breaks_input = st.text_input(
                            "Enter custom break points (comma-separated values)", 
                            value=", ".join([str(round(min_val + i*(max_val-min_val)/(num_bins), 2)) for i in range(1, num_bins)])
                        )
                        
                        try:
                            custom_breaks = [float(x.strip()) for x in breaks_input.split(",") if x.strip()]
                            breaks = [min_val] + custom_breaks + [max_val]
                            valid_breaks = True
                        except:
                            st.error("Invalid break points. Please enter numeric values separated by commas.")
                            valid_breaks = False
                    else:
                        valid_breaks = True
                    
                    if st.button("Create Binned Feature") and valid_breaks:
                        with st.spinner("Creating binned feature..."):
                            df_engineered = df.copy()
                            
                            if binning_method == "Equal-width":
                                bins = np.linspace(df[selected_col].min(), df[selected_col].max(), num_bins + 1)
                            elif binning_method == "Equal-frequency (quantiles)":
                                bins = np.percentile(df[selected_col], np.linspace(0, 100, num_bins + 1))
                            else:
                                bins = breaks
                            
                            bin_labels = [f"Bin {i+1}" for i in range(len(bins)-1)]
                            
                            binned_col_name = f"{selected_col}_binned"
                            df_engineered[binned_col_name] = pd.cut(
                                df_engineered[selected_col], 
                                bins=bins, 
                                labels=bin_labels, 
                                include_lowest=True
                            )
                            
                            st.session_state['preprocessing_steps'].append(f"Created binned feature '{binned_col_name}' with {len(bin_labels)} bins")
                            st.success(f"✅ Created binned feature '{binned_col_name}' with {len(bin_labels)} bins")
                            
                            bin_counts = df_engineered[binned_col_name].value_counts().sort_index()
                            fig = px.bar(
                                x=bin_counts.index, 
                                y=bin_counts.values,
                                labels={'x': 'Bin', 'y': 'Count'},
                                title=f"Distribution of {binned_col_name}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.session_state['df'] = df_engineered
                            
                            st.write("Preview with binned feature:")
                            st.dataframe(df_engineered.head(), use_container_width=True)
                else:
                    st.info("No numeric columns available for binning.")
            
            elif feature_eng_method == "Log Transform":
                if numeric_cols:
                    selected_cols = st.multiselect("Select columns for log transformation", numeric_cols)
                    
                    if selected_cols:
                        handle_zeros = st.checkbox("Add small constant to handle zeros/negative values", value=True)
                        
                        if st.button("Apply Log Transformation"):
                            with st.spinner("Applying log transformation..."):
                                df_engineered = df.copy()
                                
                                for col in selected_cols:
                                    min_val = df_engineered[col].min()
                                    if min_val <= 0 and handle_zeros:
                                        shift = abs(min_val) + 1
                                        df_engineered[f"{col}_log"] = np.log(df_engineered[col] + shift)
                                        st.session_state['preprocessing_steps'].append(f"Applied log transform to {col} with shift {shift} to handle non-positive values")
                                        st.info(f"Added {shift} to {col} before log transform to handle non-positive values")
                                    elif min_val <= 0 and not handle_zeros:
                                        st.warning(f"Skipped {col} because it contains non-positive values and constant addition was not selected")
                                    else:
                                        df_engineered[f"{col}_log"] = np.log(df_engineered[col])
                                        st.session_state['preprocessing_steps'].append(f"Applied log transform to {col}")
                                
                                st.success(f"✅ Applied log transformation to {len(selected_cols)} columns")
                                
                                if selected_cols:
                                    col = selected_cols[0]
                                    log_col = f"{col}_log"
                                    fig = make_subplots(
                                        rows=1, cols=2,
                                        subplot_titles=(f"Original: {col}", f"Log-transformed: {log_col}")
                                    )
                                    fig.add_trace(
                                        go.Histogram(x=df_engineered[col], name="Original"),
                                        row=1, col=1
                                    )
                                    fig.add_trace(
                                        go.Histogram(x=df_engineered[log_col], name="Log-transformed"),
                                        row=1, col=2
                                    )
                                    fig.update_layout(title="Comparison of Original vs. Log-transformed Distribution")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                st.session_state['df'] = df_engineered
                                
                                st.write("Preview with log-transformed features:")
                                st.dataframe(df_engineered.head(), use_container_width=True)
                    else:
                        st.info("Please select at least one numeric column for log transformation.")
                else:
                    st.info("No numeric columns available for log transformation.")
            
            elif feature_eng_method == "Custom Formula":
                st.write("Create a custom feature using a mathematical formula")
                
                if numeric_cols:
                    selected_vars = st.multiselect("Select variables to use in your formula", numeric_cols)
                    
                    if selected_vars:
                        var_info = pd.DataFrame({
                            'Variable': selected_vars,
                            'Min': [df[var].min() for var in selected_vars],
                            'Max': [df[var].max() for var in selected_vars],
                            'Mean': [df[var].mean() for var in selected_vars]
                        })
                        st.dataframe(var_info, use_container_width=True)
                        
                        st.markdown("""
                        **Formula syntax examples:**
                        - `A + B` (addition)
                        - `A - B` (subtraction)
                        - `A * B` (multiplication)
                        - `A / B` (division)
                        - `A ** 2` (exponentiation)
                        - `np.sqrt(A)` (square root)
                        - `np.log(A)` (natural logarithm)
                        - `np.sin(A)` (sine function)
                        """)
                        
                        formula = st.text_input("Enter your formula using the selected variables", 
                                              placeholder="e.g., A * np.sqrt(B) + C")
                        
                        new_feature_name = st.text_input("Name for the new feature", 
                                                       placeholder="e.g., custom_feature_1")
                        
                        if st.button("Create Custom Feature"):
                            if formula and new_feature_name:
                                try:
                                    with st.spinner("Creating custom feature..."):
                                        df_engineered = df.copy()
                                        var_dict = {var: df_engineered[var] for var in selected_vars}
                                        var_dict['np'] = np
                                        formula_to_eval = formula
                                        for var in selected_vars:
                                            formula_to_eval = formula_to_eval.replace(var, f"var_dict['{var}']")
                                        df_engineered[new_feature_name] = eval(formula_to_eval)
                                        st.session_state['preprocessing_steps'].append(f"Created custom feature '{new_feature_name}' using formula: {formula}")
                                        st.success(f"✅ Created custom feature '{new_feature_name}' using formula: {formula}")
                                        fig = px.histogram(
                                            df_engineered, 
                                            x=new_feature_name,
                                            title=f"Distribution of {new_feature_name}"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        st.session_state['df'] = df_engineered
                                        st.write("Preview with custom feature:")
                                        st.dataframe(df_engineered.head(), use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating custom feature: {str(e)}")
                            else:
                                st.warning("Please enter both a formula and a name for the new feature.")
                    else:
                        st.info("Please select at least one variable to use in your formula.")
                else:
                    st.info("No numeric columns available for creating custom features.")
        
        with preprocessing_tabs[2]:
            st.write("Data Transformation")
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            transformation_tabs = st.tabs(["Scaling", "Encoding", "Dimensionality Reduction"])
            
            with transformation_tabs[0]:
                st.subheader("Feature Scaling")
                
                if numeric_cols:
                    scaling_method = st.radio(
                        "Choose scaling method:",
                        ["StandardScaler (μ=0, σ=1)", "MinMaxScaler (0-1)", "RobustScaler (median, IQR)"],
                        horizontal=True
                    )
                    
                    selected_cols = st.multiselect("Select columns to scale", numeric_cols, default=numeric_cols)
                    
                    if selected_cols and st.button("Apply Scaling"):
                        with st.spinner("Scaling features..."):
                            df_scaled = df.copy()
                            
                            if scaling_method == "StandardScaler (μ=0, σ=1)":
                                scaler = StandardScaler()
                            elif scaling_method == "MinMaxScaler (0-1)":
                                scaler = MinMaxScaler()
                            else:
                                scaler = RobustScaler()
                            
                            df_scaled[selected_cols] = scaler.fit_transform(df_scaled[selected_cols])
                            
                            st.session_state['preprocessing_steps'].append(f"Applied {scaling_method} to columns: {', '.join(selected_cols)}")
                            st.success(f"✅ Applied {scaling_method} to {len(selected_cols)} columns")
                            
                            if len(selected_cols) > 0:
                                col = selected_cols[0]
                                fig = make_subplots(
                                    rows=1, cols=2,
                                    subplot_titles=(f"Before scaling: {col}", f"After scaling: {col}")
                                )
                                fig.add_trace(
                                    go.Histogram(x=df[col], name="Original"),
                                    row=1, col=1
                                )
                                fig.add_trace(
                                    go.Histogram(x=df_scaled[col], name="Scaled"),
                                    row=1, col=2
                                )
                                fig.update_layout(title=f"Effect of {scaling_method} on {col}")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.session_state['df'] = df_scaled
                            
                            st.write("Preview with scaled features:")
                            st.dataframe(df_scaled.head(), use_container_width=True)
                else:
                    st.info("No numeric columns available for scaling.")
            
            with transformation_tabs[1]:
                st.subheader("Categorical Encoding")
                
                if categorical_cols:
                    encoding_method = st.radio(
                        "Choose encoding method:",
                        ["One-Hot Encoding", "Label Encoding", "Frequency Encoding"],
                        horizontal=True
                    )
                    
                    selected_cols = st.multiselect("Select categorical columns to encode", categorical_cols, default=categorical_cols)
                    
                    if selected_cols and st.button("Apply Encoding"):
                        with st.spinner("Encoding categorical variables..."):
                            df_encoded = df.copy()
                            
                            if encoding_method == "One-Hot Encoding":
                                df_encoded = pd.get_dummies(df_encoded, columns=selected_cols, drop_first=True)
                                st.session_state['preprocessing_steps'].append(f"Applied One-Hot Encoding to columns: {', '.join(selected_cols)}")
                                st.success(f"✅ Applied One-Hot Encoding to {len(selected_cols)} columns")
                            
                            elif encoding_method == "Label Encoding":
                                for col in selected_cols:
                                    le = LabelEncoder()
                                    df_encoded[f"{col}_encoded"] = le.fit_transform(df_encoded[col].astype(str))
                                st.session_state['preprocessing_steps'].append(f"Applied Label Encoding to columns: {', '.join(selected_cols)}")
                                st.success(f"✅ Applied Label Encoding to {len(selected_cols)} columns")
                            
                            elif encoding_method == "Frequency Encoding":
                                for col in selected_cols:
                                    frequency = df_encoded[col].value_counts(normalize=True)
                                    df_encoded[f"{col}_freq_encoded"] = df_encoded[col].map(frequency)
                                st.session_state['preprocessing_steps'].append(f"Applied Frequency Encoding to columns: {', '.join(selected_cols)}")
                                st.success(f"✅ Applied Frequency Encoding to {len(selected_cols)} columns")
                            
                            st.session_state['df'] = df_encoded
                            
                            st.write("Preview with encoded features:")
                            st.dataframe(df_encoded.head(), use_container_width=True)
                else:
                    st.info("No categorical columns available for encoding.")
            
            with transformation_tabs[2]:
                st.subheader("Dimensionality Reduction")
                
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                if len(numeric_cols) > 2:
                    dr_method = st.radio(
                        "Choose dimensionality reduction method:",
                        ["PCA", "t-SNE", "UMAP"],
                        horizontal=True
                    )
                    
                    selected_cols = st.multiselect("Select columns for dimensionality reduction", numeric_cols, default=numeric_cols)
                    
                    if len(selected_cols) >= 2:
                        n_components = st.slider("Number of components", min_value=2, max_value=min(len(selected_cols), 10), value=2)
                        
                        if st.button("Apply Dimensionality Reduction"):
                            with st.spinner(f"Applying {dr_method}..."):
                                from sklearn.decomposition import PCA
                                from sklearn.preprocessing import StandardScaler
                                
                                df_dr = df.copy()
                                scaler = StandardScaler()
                                scaled_data = scaler.fit_transform(df_dr[selected_cols])
                                
                                try:
                                    if dr_method == "PCA":
                                        reducer = PCA(n_components=n_components)
                                        reduced_data = reducer.fit_transform(scaled_data)
                                        for i in range(n_components):
                                            df_dr[f"PC{i+1}"] = reduced_data[:, i]
                                        st.session_state['preprocessing_steps'].append(f"Applied PCA with {n_components} components to columns: {', '.join(selected_cols)}")
                                        st.success(f"✅ Applied PCA, explained variance: {sum(reducer.explained_variance_ratio_):.2%}")
                                        fig = px.bar(
                                            x=[f"PC{i+1}" for i in range(n_components)],
                                            y=reducer.explained_variance_ratio_,
                                            labels={"x": "Principal Component", "y": "Explained Variance Ratio"},
                                            title="Explained Variance by Component"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    elif dr_method == "t-SNE":
                                        from sklearn.manifold import TSNE
                                        reducer = TSNE(n_components=n_components, random_state=42)
                                        reduced_data = reducer.fit_transform(scaled_data)
                                        for i in range(n_components):
                                            df_dr[f"tSNE{i+1}"] = reduced_data[:, i]
                                        st.session_state['preprocessing_steps'].append(f"Applied t-SNE with {n_components} components to columns: {', '.join(selected_cols)}")
                                        st.success(f"✅ Applied t-SNE with {n_components} components")
                                    
                                    elif dr_method == "UMAP":
                                        try:
                                            import umap
                                            reducer = umap.UMAP(n_components=n_components, random_state=42)
                                            reduced_data = reducer.fit_transform(scaled_data)
                                            for i in range(n_components):
                                                df_dr[f"UMAP{i+1}"] = reduced_data[:, i]
                                            st.session_state['preprocessing_steps'].append(f"Applied UMAP with {n_components} components to columns: {', '.join(selected_cols)}")
                                            st.success(f"✅ Applied UMAP with {n_components} components")
                                        except:
                                            st.error("UMAP is not installed or unavailable. Please install it with 'pip install umap-learn'.")
                                    
                                    if n_components >= 2:
                                        if dr_method == "PCA":
                                            x_col, y_col = "PC1", "PC2"
                                        elif dr_method == "t-SNE":
                                            x_col, y_col = "tSNE1", "tSNE2"
                                        else:
                                            x_col, y_col = "UMAP1", "UMAP2"
                                        
                                        fig = px.scatter(
                                            df_dr, x=x_col, y=y_col,
                                            title=f"{dr_method}: 2D Visualization"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.session_state['df'] = df_dr
                                    
                                    st.write("Preview with reduced dimensions:")
                                    st.dataframe(df_dr.head(), use_container_width=True)
                                
                                except Exception as e:
                                    st.error(f"Error applying dimensionality reduction: {str(e)}")
                    else:
                        st.info("Please select at least 2 columns for dimensionality reduction.")
                else:
                    st.info("Need at least 3 numeric columns for meaningful dimensionality reduction.")
        
        with preprocessing_tabs[3]:
            st.subheader("Feature Selection")
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_cols) > 1:
                feature_sel_method = st.radio(
                    "Choose feature selection method:",
                    ["Statistical Tests (ANOVA F-value)", "Feature Importance from Tree Models", "Recursive Feature Elimination (RFE)", "L1-based Selection"],
                    horizontal=True
                )
                
                selected_cols = st.multiselect("Select features to consider", numeric_cols, default=numeric_cols)
                
                if len(selected_cols) > 0:
                    k_features = st.slider("Number of features to select", min_value=1, max_value=len(selected_cols), value=min(5, len(selected_cols)))
                    
                    st.warning("⚠️ Feature selection requires a target column. You will select the target column in the 'Finalize Preprocessing' tab. Results here are preliminary and assume a temporary target.")
                    
                    temp_target = st.selectbox("Select a temporary target column for feature selection", df.columns)
                    
                    if st.button("Perform Feature Selection"):
                        with st.spinner(f"Performing feature selection using {feature_sel_method}..."):
                            try:
                                X = df[selected_cols]
                                y = df[temp_target]
                                
                                if feature_sel_method == "Statistical Tests (ANOVA F-value)":
                                    selector = SelectKBest(f_classif, k=k_features)
                                    selector.fit(X, y)
                                    selected_features = [selected_cols[i] for i in selector.get_support(indices=True)]
                                    scores = selector.scores_
                                    feature_scores = pd.DataFrame({
                                        'Feature': selected_cols,
                                        'Score': scores
                                    }).sort_values('Score', ascending=False)
                                    st.session_state['preprocessing_steps'].append(f"Selected {k_features} features using ANOVA F-value (with temporary target {temp_target}): {', '.join(selected_features)}")
                                    st.success(f"✅ Selected top {k_features} features based on ANOVA F-value")
                                    st.write("Feature Scores:")
                                    st.dataframe(feature_scores, use_container_width=True)
                                    fig = create_feature_importance_plot(feature_scores['Score'], feature_scores['Feature'])
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif feature_sel_method == "Feature Importance from Tree Models":
                                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                                    model.fit(X, y)
                                    importances = model.feature_importances_
                                    feature_importances = pd.DataFrame({
                                        'Feature': selected_cols,
                                        'Importance': importances
                                    }).sort_values('Importance', ascending=False)
                                    selected_features = feature_importances.head(k_features)['Feature'].tolist()
                                    st.session_state['preprocessing_steps'].append(f"Selected {k_features} features using Random Forest importance (with temporary target {temp_target}): {', '.join(selected_features)}")
                                    st.success(f"✅ Selected top {k_features} features based on Random Forest importance")
                                    st.write("Feature Importances:")
                                    st.dataframe(feature_importances, use_container_width=True)
                                    fig = create_feature_importance_plot(feature_importances['Importance'], feature_importances['Feature'])
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif feature_sel_method == "Recursive Feature Elimination (RFE)":
                                    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                                    selector = RFE(estimator, n_features_to_select=k_features, step=1)
                                    selector = selector.fit(X, y)
                                    selected_features = [selected_cols[i] for i in range(len(selected_cols)) if selector.support_[i]]
                                    feature_ranks = pd.DataFrame({
                                        'Feature': selected_cols,
                                        'Rank': selector.ranking_
                                    }).sort_values('Rank')
                                    st.session_state['preprocessing_steps'].append(f"Selected {k_features} features using RFE (with temporary target {temp_target}): {', '.join(selected_features)}")
                                    st.success(f"✅ Selected {k_features} features using Recursive Feature Elimination")
                                    st.write("Feature Rankings (1 = selected):")
                                    st.dataframe(feature_ranks, use_container_width=True)
                                    fig = px.bar(
                                        feature_ranks.head(k_features),
                                        x='Rank', y='Feature',
                                        orientation='h',
                                        title='Top Features by RFE'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif feature_sel_method == "L1-based Selection":
                                    selector = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=42)
                                    selector.fit(X, y)
                                    coefficients = selector.coef_[0]
                                    feature_coefs = pd.DataFrame({
                                        'Feature': selected_cols,
                                        'Coefficient': np.abs(coefficients)
                                    }).sort_values('Coefficient', ascending=False)
                                    selected_features = feature_coefs.head(k_features)['Feature'].tolist()
                                    st.session_state['preprocessing_steps'].append(f"Selected {k_features} features using L1-based selection (with temporary target {temp_target}): {', '.join(selected_features)}")
                                    st.success(f"✅ Selected top {k_features} features based on L1 regularization")
                                    st.write("Feature Coefficients (absolute value):")
                                    st.dataframe(feature_coefs, use_container_width=True)
                                    fig = create_feature_importance_plot(feature_coefs['Coefficient'], feature_coefs['Feature'])
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                df_selected = df[selected_features]
                                st.write("Dataset with selected features (preliminary):")
                                st.dataframe(df_selected.head(), use_container_width=True)
                                
                                if st.checkbox("Update dataset with selected features only"):
                                    st.session_state['df'] = df_selected
                                    st.session_state['preprocessing_steps'].append(f"Updated dataset to include only selected features (with temporary target {temp_target}): {', '.join(selected_features)}")
                                    st.success("✅ Dataset updated with selected features")
                            
                            except Exception as e:
                                st.error(f"Error during feature selection: {str(e)}")
                else:
                    st.info("Please select at least one feature for feature selection.")
            else:
                st.info("Need at least 2 numeric features for feature selection.")
        
        with preprocessing_tabs[4]:
            st.subheader("Finalize Preprocessing")
            
            st.write("### Review Preprocessing Steps")
            if st.session_state['preprocessing_steps']:
                st.write("The following preprocessing steps have been applied to the dataset:")
                for i, step in enumerate(st.session_state['preprocessing_steps'], 1):
                    st.write(f"{i}. {step}")
            else:
                st.warning("⚠️ No preprocessing steps have been applied yet.")
            
            st.write("### Current Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            st.write("### Dataset Summary")
            dtypes_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Missing': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(dtypes_df, use_container_width=True)
            
            if df.isnull().sum().sum() > 0:
                st.warning("⚠️ The dataset still contains missing values. Consider handling them in the 'Cleaning' tab.")
            
            if df.shape[1] < 2:
                st.error("❌ The dataset has too few columns. Ensure you have at least two columns (one for target and one for features).")
            
            st.write("### Select Target Column")
            all_columns = df.columns.tolist()
            target_col = st.selectbox(
                "Choose the target column for machine learning",
                options=['None'] + all_columns,
                index=0
            )
            
            if target_col == 'None':
                st.warning("⚠️ Please select a target column to proceed with machine learning.")
            
            if st.button("Confirm Preprocessing and Proceed to Machine Learning"):
                if target_col == 'None':
                    st.error("❌ Please select a target column before confirming preprocessing.")
                elif df.isnull().sum().sum() > 0:
                    st.error("❌ Cannot confirm preprocessing. The dataset contains missing values.")
                elif df.shape[1] < 2:
                    st.error("❌ Cannot confirm preprocessing. The dataset must have at least two columns.")
                else:
                    # Rename selected column to 'target'
                    df_final = df.copy()
                    if target_col != 'target':
                        df_final = df_final.rename(columns={target_col: 'target'})
                        st.session_state['preprocessing_steps'].append(f"Renamed column '{target_col}' to 'target'")
                    
                    st.session_state['df'] = df_final
                    st.session_state['preprocessing_complete'] = True
                    st.success("✅ Preprocessing confirmed! Target column set to 'target'. You can now proceed to the Machine Learning section.")
        
        st.markdown("</div>", unsafe_allow_html=True)