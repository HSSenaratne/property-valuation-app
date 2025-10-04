import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Try to import your model with error handling
try:
    from property_model import PropertyValuationModel, create_sample_dataset
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    st.error(f"Model module import failed: {e}")

# Page configuration
st.set_page_config(
    page_title="Property Valuation App",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #bee5eb;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

class PropertyValuationApp:
    def __init__(self):
        if MODEL_AVAILABLE:
            self.model = PropertyValuationModel()
        else:
            self.model = None
        self.df = None
        
    def run(self):
        # Header
        st.markdown('<h1 class="main-header">🏠 Property Valuation Data Mining App</h1>', 
                   unsafe_allow_html=True)
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        
        # Sidebar
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.selectbox("Choose Section", 
                                       ["Home", "Data Overview", "EDA", "Model Training", 
                                        "Predictions", "Results"])
        
        if app_mode == "Home":
            self.show_home()
        elif app_mode == "Data Overview":
            self.show_data_overview()
        elif app_mode == "EDA":
            self.show_eda()
        elif app_mode == "Model Training":
            self.show_model_training()
        elif app_mode == "Predictions":
            self.show_predictions()
        elif app_mode == "Results":
            self.show_results()
    
    def show_home(self):
        st.markdown("""
        ## Welcome to the Property Valuation Data Mining Application
        
        This application uses machine learning to predict property values based on various features.
        
        ### 🚀 Quick Start:
        1. **Upload your dataset** below (CSV format) or use our sample data
        2. **Explore the data** in the Data Overview section
        3. **Train machine learning models** 
        4. **Make predictions** and analyze results
        
        ### 📊 Supported Data Format:
        Your CSV should include columns like: `price`, `bed`, `bath`, `sqft`, `city`, `state`, etc.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Option 1: Upload Your Dataset")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
            
            if uploaded_file is not None:
                try:
                    self.df = pd.read_csv(uploaded_file)
                    st.session_state.current_data = self.df
                    st.session_state.data_loaded = True
                    
                    st.markdown(f'<div class="success-box">✅ Dataset loaded successfully!<br>Shape: {self.df.shape}</div>', 
                               unsafe_allow_html=True)
                    
                    st.subheader("Dataset Preview")
                    st.dataframe(self.df.head(10))
                    
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        with col2:
            st.subheader("Option 2: Use Sample Data")
            st.write("Don't have a dataset? Use our sample property data to test the application.")
            
            if st.button("Generate Sample Data", key="sample_data"):
                try:
                    if MODEL_AVAILABLE:
                        self.df = create_sample_dataset()
                        st.session_state.current_data = self.df
                        st.session_state.data_loaded = True
                        
                        st.markdown(f'<div class="success-box">✅ Sample dataset created!<br>Shape: {self.df.shape}</div>', 
                                   unsafe_allow_html=True)
                        
                        st.subheader("Sample Data Preview")
                        st.dataframe(self.df.head(10))
                    else:
                        st.error("Model module not available. Cannot generate sample data.")
                except Exception as e:
                    st.error(f"Error creating sample data: {e}")
    
    def show_data_overview(self):
        st.header("📊 Data Overview")
        
        if not st.session_state.data_loaded or st.session_state.current_data is None:
            st.markdown('<div class="warning-box">⚠️ Please upload a dataset or generate sample data in the Home section first.</div>', 
                       unsafe_allow_html=True)
            return
        
        self.df = st.session_state.current_data
        
        # Basic Information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", f"{self.df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", f"{self.df.shape[1]}")
        with col3:
            st.metric("Memory Usage", f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data Preview
        st.subheader("Data Preview")
        st.dataframe(self.df.head())
        
        # Data Types and Missing Values
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types")
            dtype_info = self.df.dtypes.reset_index()
            dtype_info.columns = ['Column', 'Data Type']
            st.dataframe(dtype_info)
        
        with col2:
            st.subheader("Missing Values")
            missing_data = self.df.isnull().sum().reset_index()
            missing_data.columns = ['Column', 'Missing Values']
            missing_data['Percentage'] = (missing_data['Missing Values'] / len(self.df)) * 100
            st.dataframe(missing_data)
        
        # Numeric Statistics
        if not self.df.select_dtypes(include=[np.number]).empty:
            st.subheader("Numeric Columns Statistics")
            st.dataframe(self.df.describe())
    
    def show_eda(self):
        st.header("📈 Exploratory Data Analysis")
        
        if not st.session_state.data_loaded or st.session_state.current_data is None:
            st.markdown('<div class="warning-box">⚠️ Please upload a dataset or generate sample data in the Home section first.</div>', 
                       unsafe_allow_html=True)
            return
        
        self.df = st.session_state.current_data
        
        # NO SAMPLING - Using full dataset for visualizations
        st.info(f"📊 Showing visualizations for all {len(self.df):,} records")
        
        # Price distribution
        if 'price' in self.df.columns:
            st.subheader("Price Distribution")
            fig = px.histogram(self.df, x='price', nbins=50, 
                             title='Distribution of Property Prices',
                             labels={'price': 'Price ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap for numeric columns
        st.subheader("Correlation Heatmap")
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Limit to reasonable number of columns to prevent performance issues
        if len(numeric_df.columns) > 10:
            numeric_df = numeric_df.iloc[:, :10]
            st.info("Showing correlation for first 10 numeric columns")
        
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(corr_matrix, 
                          title='Correlation Matrix',
                          color_continuous_scale='RdBu_r',
                          aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature relationships
        st.subheader("Feature Relationships")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X-axis", options=numeric_cols, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis", options=numeric_cols, 
                                    index=min(1, len(numeric_cols)-1))
            
            if x_axis != y_axis:
                # Use sampling only for scatter plot to prevent browser crashes
                if len(self.df) > 10000:
                    scatter_sample = self.df.sample(n=10000, random_state=42)
                    st.info("Using 10,000 sampled records for scatter plot to ensure smooth performance")
                else:
                    scatter_sample = self.df
                    
                fig = px.scatter(scatter_sample, x=x_axis, y=y_axis, 
                               title=f'{y_axis} vs {x_axis}')
                st.plotly_chart(fig, use_container_width=True)

    def show_model_training(self):
        st.header("🤖 Model Training")
        
        if not st.session_state.data_loaded:
            st.markdown('<div class="warning-box">⚠️ Please upload a dataset or generate sample data in the Home section first.</div>', 
                       unsafe_allow_html=True)
            return
        
        if not MODEL_AVAILABLE:
            st.error("❌ Model module not available. Please check property_model.py")
            return
        
        self.df = st.session_state.current_data
        
        # Check if dataset has required columns
        if 'price' not in self.df.columns:
            st.error("❌ Dataset must contain a 'price' column for training.")
            return
        
        st.markdown("""
        <div class="info-box">
        💡 <b>Training Information:</b><br>
        - We'll train machine learning models to predict property prices<br>
        - The data will be automatically cleaned and preprocessed<br>
        - Training may take a few minutes depending on dataset size
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        st.subheader("Model Selection")
        available_models = ["Random Forest", "LightGBM", "XGBoost", "Linear Regression"]
        
        selected_models = st.multiselect(
            "Select models to train:",
            options=available_models,
            default=["Random Forest", "LightGBM"]
        )
        
        if not selected_models:
            st.error("❌ Please select at least one model to train.")
            return
        
        # Training configuration
        st.subheader("Training Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            training_mode = st.selectbox(
                "Training Mode",
                ["Fast Training", "Balanced", "Comprehensive"],
                help="Fast: Quick results, Balanced: Good accuracy, Comprehensive: Best accuracy"
            )
        
        with col2:
            validation_method = st.selectbox(
                "Validation Method",
                ["Random Split", "Geographic Split"],
                help="Geographic split tests generalization across locations"
            )
        
        # Feature selection
        st.subheader("Feature Selection")
        available_features = [col for col in self.df.columns if col != 'price']
        
        if available_features:
            selected_features = st.multiselect(
                "Select features to use for training:",
                options=available_features,
                default=available_features[:min(8, len(available_features))]
            )
            
            if not selected_features:
                st.error("❌ Please select at least one feature for training.")
                return
        else:
            st.error("❌ No features available for training (only 'price' column found).")
            return
        
        if st.button("🚀 Train Machine Learning Models", key="train_models"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Initialize model
                    self.model = PropertyValuationModel()
                    
                    # Load and prepare data
                    st.info("📊 Preparing data...")
                    success = self.model.load_dataframe(self.df, selected_features)
                    
                    if success:
                        st.info("🤖 Training models...")
                        training_success = self.model.train_models_optimized(
                            models_to_train=selected_models,
                            training_mode=training_mode,
                            validation_method=validation_method
                        )
                        
                        if training_success:
                            st.info("📈 Evaluating models...")
                            regression_results = self.model.evaluate_models()
                            
                            st.info("💰 Training price band classifier...")
                            classification_results = self.model.train_price_band_classifier()
                            
                            if regression_results:
                                st.session_state.models_trained = True
                                st.session_state.regression_results = regression_results
                                st.session_state.classification_results = classification_results
                                st.session_state.trained_model = self.model
                                st.session_state.training_features = selected_features
                                st.session_state.selected_models = selected_models
                                
                                st.markdown('<div class="success-box">✅ Models trained successfully!</div>', 
                                           unsafe_allow_html=True)
                                
                                # Display results
                                self.display_model_results(regression_results, classification_results)
                            else:
                                st.error("❌ Model evaluation failed.")
                        else:
                            st.error("❌ Model training failed.")
                    else:
                        st.error("❌ Data preparation failed. Please check your dataset has numeric features like 'bed', 'bath', 'sqft'.")
                        
                except Exception as e:
                    st.error(f"❌ Error training models: {str(e)}")
                    st.info("💡 Tip: Make sure your dataset has a 'price' column and numeric features like 'bed', 'bath', or 'sqft'.")
    
    def display_model_results(self, regression_results, classification_results):
        if not regression_results:
            return
        
        st.subheader("📊 Model Performance Comparison")
        
        # Regression Results
        st.write("#### Regression Models (Price Prediction)")
        for model_name, metrics in regression_results.items():
            if model_name in st.session_state.selected_models:
                st.write(f"**{model_name}**")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("MAE", f"${metrics['MAE']:,.0f}")
                with col2:
                    st.metric("RMSE", f"${metrics['RMSE']:,.0f}")
                with col3:
                    st.metric("R² Score", f"{metrics['R2']:.3f}")
                with col4:
                    st.metric("Within 10%", f"{metrics.get('Within_10_Percent', 0):.1f}%")
                with col5:
                    st.metric("Within 20%", f"{metrics['Within_20_Percent']:.1f}%")
        
        # Classification Results
        if classification_results:
            st.write("#### Classification Models (Price Bands)")
            for model_name, metrics in classification_results.items():
                st.write(f"**{model_name}**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                with col2:
                    st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                with col4:
                    st.metric("F1-Score", f"{metrics.get('f1', 0):.3f}")

    def show_predictions(self):
        st.header("🔮 Price Prediction")
        
        if not st.session_state.models_trained:
            st.markdown("""
            <div class="warning-box">
            ⚠️ <b>Models not trained yet!</b><br>
            Please go to the <b>Model Training</b> section and train the models first.
            </div>
            """, unsafe_allow_html=True)
            return
        
        st.subheader("Predict Property Value")
        st.write("Enter property details below to get a price prediction:")
        
        # Get the features used during training
        if hasattr(st.session_state, 'training_features'):
            feature_columns = st.session_state.training_features
        else:
            # Fallback to available features
            feature_columns = [col for col in st.session_state.current_data.columns if col != 'price']
        
        # Simple prediction form
        col1, col2 = st.columns(2)
        
        with col1:
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
            bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
            sqft = st.number_input("Square Feet", min_value=500, max_value=10000, value=2000)
        
        with col2:
            year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)
            lot_size = st.number_input("Lot Size (sq ft)", min_value=1000, max_value=100000, value=10000)
            city = st.text_input("City", value="New York")
            state = st.text_input("State", value="NY")
        
        # Additional features
        st.subheader("Additional Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            property_type = st.selectbox("Property Type", 
                                       ["Single Family", "Condo", "Townhouse", "Multi-Family"])
            stories = st.number_input("Stories", min_value=1, max_value=5, value=2)
        
        with col2:
            garage = st.number_input("Garage Spaces", min_value=0, max_value=5, value=2)
            pool = st.selectbox("Has Pool", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        with col3:
            condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
        
        if st.button("Predict Price", key="predict_price"):
            try:
                model = st.session_state.trained_model
                
                # Prepare features
                features = {
                    'bed': bedrooms,
                    'bath': bathrooms,
                    'sqft': sqft,
                    'year_built': year_built,
                    'lot_size': lot_size,
                    'city': city,
                    'state': state,
                    'property_type': property_type,
                    'stories': stories,
                    'garage': garage,
                    'pool': pool,
                    'condition': condition
                }
                
                # Filter to only include features that were used in training
                prediction_features = {}
                for feature, value in features.items():
                    if feature in feature_columns:
                        prediction_features[feature] = value
                
                # Make prediction
                price_prediction = model.predict_price(prediction_features)
                price_band = model.predict_price_band(prediction_features)
                
                if price_prediction is not None and price_prediction > 0:
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>🎯 Prediction Result</h3>
                    <p style="font-size: 2.5rem; color: #28a745; font-weight: bold; text-align: center;">
                        Estimated Value: <b>${price_prediction:,.0f}</b>
                    </p>
                    <p style="font-size: 1.5rem; color: #17a2b8; text-align: center;">
                        Price Band: <b>{price_band}</b>
                    </p>
                    <p><strong>Input Features:</strong></p>
                    <ul>
                        <li>Bedrooms: {bedrooms}</li>
                        <li>Bathrooms: {bathrooms}</li>
                        <li>Square Feet: {sqft}</li>
                        <li>Year Built: {year_built}</li>
                        <li>Lot Size: {lot_size} sq ft</li>
                        <li>Location: {city}, {state}</li>
                        <li>Property Type: {property_type}</li>
                        <li>Stories: {stories}</li>
                        <li>Garage Spaces: {garage}</li>
                        <li>Pool: {'Yes' if pool == 1 else 'No'}</li>
                        <li>Condition: {condition}/5</li>
                    </ul>
                    <p><em>Based on trained machine learning models ({', '.join(st.session_state.selected_models)})</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                        
                else:
                    st.error("❌ Prediction failed. Please check your input values and try again.")
                    
            except Exception as e:
                st.error(f"❌ Error in prediction: {e}")
                st.info("💡 Using fallback calculation with standard market rates.")

    def show_results(self):
        st.header("📋 Results & Analysis")
        
        if not st.session_state.models_trained:
            st.markdown("""
            <div class="warning-box">
            ⚠️ <b>No results available yet!</b><br>
            Please train models in the <b>Model Training</b> section first.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Get results from session state
        regression_results = st.session_state.regression_results
        classification_results = st.session_state.classification_results
        model = st.session_state.trained_model
        
        if not regression_results:
            st.error("No results available. Please train models again.")
            return
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🔍 Feature Importance", "📈 Model Comparison", "💡 Business Insights"])
        
        with tab1:
            st.subheader("Model Performance Analysis")
            
            # Display performance metrics in a nice table
            performance_data = []
            for model_name, metrics in regression_results.items():
                if model_name in st.session_state.selected_models:
                    performance_data.append({
                        'Model': model_name,
                        'MAE': f"${metrics['MAE']:,.0f}",
                        'RMSE': f"${metrics['RMSE']:,.0f}",
                        'R² Score': f"{metrics['R2']:.4f}",
                        'Within 10%': f"{metrics.get('Within_10_Percent', 0):.1f}%",
                        'Within 20%': f"{metrics['Within_20_Percent']:.1f}%"
                    })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
            
            # Best model identification
            best_model = min(regression_results.items(), key=lambda x: x[1]['RMSE'])
            st.success(f"🏆 **Best Performing Model**: {best_model[0]} (RMSE: ${best_model[1]['RMSE']:,.0f})")
            
            # Performance interpretation
            st.subheader("Performance Interpretation")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Best R² Score", f"{max([m['R2'] for m in regression_results.values()]):.4f}")
                st.metric("Best Within 10%", f"{max([m.get('Within_10_Percent', 0) for m in regression_results.values()]):.1f}%")
            
            with col2:
                st.metric("Lowest MAE", f"${min([m['MAE'] for m in regression_results.values()]):,.0f}")
                st.metric("Lowest RMSE", f"${min([m['RMSE'] for m in regression_results.values()]):,.0f}")
        
        with tab2:
            st.subheader("Feature Importance Analysis")
            
            # Get feature importance from the best model
            best_model_name = min(regression_results.items(), key=lambda x: x[1]['RMSE'])[0]
            importance_df = model.get_feature_importance(best_model_name)
            
            if importance_df is not None:
                # Display feature importance chart
                fig = px.bar(importance_df, 
                            x='importance', 
                            y='feature',
                            orientation='h',
                            title=f'Feature Importance - {best_model_name}',
                            labels={'importance': 'Importance Score', 'feature': 'Feature'},
                            color='importance',
                            color_continuous_scale='viridis')
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display feature importance table
                st.subheader("Feature Importance Rankings")
                st.dataframe(importance_df, use_container_width=True)
                
                # Feature interpretation
                st.subheader("Key Insights")
                top_feature = importance_df.iloc[0]['feature']
                top_importance = importance_df.iloc[0]['importance']
                
                st.info(f"""
                **🔑 Most Important Feature**: `{top_feature}`
                
                - Accounts for **{top_importance:.1%}** of the model's decision-making
                - This feature has the strongest influence on property price predictions
                - Focus on accurate data collection for this feature
                """)
            else:
                st.warning("Feature importance data not available for the selected model.")
        
        with tab3:
            st.subheader("Model Comparison")
            
            # Create comparison charts
            models_to_compare = [m for m in st.session_state.selected_models if m in regression_results]
            
            if len(models_to_compare) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # RMSE comparison
                    rmse_values = [regression_results[model]['RMSE'] for model in models_to_compare]
                    
                    fig_rmse = go.Figure(data=[
                        go.Bar(name='RMSE (Lower is Better)', x=models_to_compare, y=rmse_values)
                    ])
                    fig_rmse.update_layout(title='Model RMSE Comparison')
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                with col2:
                    # R² comparison
                    r2_values = [regression_results[model]['R2'] for model in models_to_compare]
                    
                    fig_r2 = go.Figure(data=[
                        go.Bar(name='R² Score (Higher is Better)', x=models_to_compare, y=r2_values)
                    ])
                    fig_r2.update_layout(title='Model R² Score Comparison')
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                # Accuracy comparison
                accuracy_values = [regression_results[model]['Within_20_Percent'] for model in models_to_compare]
                
                fig_accuracy = go.Figure(data=[
                    go.Bar(name='Within 20% Accuracy', x=models_to_compare, y=accuracy_values)
                ])
                fig_accuracy.update_layout(title='Model Accuracy Comparison (% within 20% of actual price)')
                st.plotly_chart(fig_accuracy, use_container_width=True)
            else:
                st.info("Need at least 2 models for comparison.")
        
        with tab4:
            st.subheader("Business Insights & Recommendations")
            
            # Calculate business metrics
            best_accuracy = max([regression_results[model]['Within_20_Percent'] for model in regression_results.keys()])
            avg_mae = np.mean([regression_results[model]['MAE'] for model in regression_results.keys()])
            best_model_name = min(regression_results.items(), key=lambda x: x[1]['RMSE'])[0]
            
            st.markdown(f"""
            ### 📈 Performance Summary
            
            - **Best Model**: `{best_model_name}`
            - **Prediction Accuracy**: **{best_accuracy:.1f}%** of predictions within 20% of actual prices
            - **Average Error**: **${avg_mae:,.0f}** mean absolute error
            - **Model Reliability**: {'Excellent' if best_accuracy > 85 else 'Good' if best_accuracy > 75 else 'Moderate'}
            
            ### 💼 Business Recommendations
            
            1. **Model Deployment**: Use `{best_model_name}` for production as it shows the best performance
            2. **Confidence Intervals**: Provide price ranges of ±20% for client communications
            3. **Data Quality**: Focus on collecting accurate data for the most important features
            4. **Model Monitoring**: Regularly retrain models with new market data
            5. **Risk Management**: Use the accuracy metrics to set client expectations
            
            ### 🎯 Practical Applications
            
            - **Real Estate Agents**: Quick property valuation for listings
            - **Buyers/Sellers**: Market price estimation for negotiations  
            - **Investors**: Identify undervalued properties
            - **Banks/Lenders**: Mortgage and loan valuation support
            """)

def main():
    app = PropertyValuationApp()
    app.run()

if __name__ == "__main__":
    main()