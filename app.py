import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from property_model import PropertyValuationModel, create_sample_dataset
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

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
        self.model = PropertyValuationModel()
        self.df = None
        self.results = None
        
        # Initialize session state
        if 'df_loaded' not in st.session_state:
            st.session_state.df_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'current_df' not in st.session_state:
            st.session_state.current_df = None
    
    def run(self):
        # Header
        st.markdown('<h1 class="main-header">🏠 Property Valuation Data Mining App</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.selectbox("Choose Section", 
                                       ["Home", "Data Overview", "EDA", "Model Training", 
                                        "Predictions", "Results"])
        
        # Load data first if available
        self.load_data()
        
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
    
    def load_data(self):
        """Load data into session state"""
        if st.session_state.df_loaded and st.session_state.current_df is not None:
            self.df = st.session_state.current_df
    
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
                    st.session_state.current_df = self.df
                    st.session_state.df_loaded = True
                    
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
                with st.spinner("Creating sample dataset..."):
                    self.df = create_sample_dataset()
                    st.session_state.current_df = self.df
                    st.session_state.df_loaded = True
                    
                    st.markdown(f'<div class="success-box">✅ Sample dataset created!<br>Shape: {self.df.shape}</div>', 
                               unsafe_allow_html=True)
                    
                    st.subheader("Sample Data Preview")
                    st.dataframe(self.df.head(10))
    
    def show_data_overview(self):
        st.header("📊 Data Overview")
        
        if not st.session_state.df_loaded:
            st.markdown('<div class="warning-box">⚠️ Please upload a dataset or generate sample data in the Home section first.</div>', 
                       unsafe_allow_html=True)
            return
        
        self.df = st.session_state.current_df
        
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
        
        if not st.session_state.df_loaded:
            st.markdown('<div class="warning-box">⚠️ Please upload a dataset or generate sample data in the Home section first.</div>', 
                       unsafe_allow_html=True)
            return
        
        self.df = st.session_state.current_df
        
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
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            # Limit to first 10 columns for readability
            numeric_df_limited = numeric_df.iloc[:, :10]
            corr_matrix = numeric_df_limited.corr()
            
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
                fig = px.scatter(self.df, x=x_axis, y=y_axis, 
                               title=f'{y_axis} vs {x_axis}',
                               hover_data=self.df.columns.tolist())
                st.plotly_chart(fig, use_container_width=True)
    
    def show_model_training(self):
        st.header("🤖 Model Training")
        
        if not st.session_state.df_loaded:
            st.markdown('<div class="warning-box">⚠️ Please upload a dataset or generate sample data in the Home section first.</div>', 
                       unsafe_allow_html=True)
            return
        
        self.df = st.session_state.current_df
        
        st.markdown("""
        <div class="info-box">
        💡 <b>Training Information:</b><br>
        - We'll train multiple machine learning models to predict property prices<br>
        - The data will be automatically cleaned and preprocessed<br>
        - Training may take a few minutes depending on dataset size
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Train Machine Learning Models", key="train_models"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Initialize and train model
                    self.model = PropertyValuationModel()
                    
                    # Load and preprocess data
                    df_processed, X, y = self.model.load_and_preprocess(df=self.df)
                    
                    if df_processed is not None:
                        # Train models
                        success = self.model.train_models()
                        
                        if success:
                            # Evaluate models
                            self.results = self.model.evaluate_models()
                            st.session_state.models_trained = True
                            st.session_state.model = self.model
                            st.session_state.results = self.results
                            
                            st.markdown('<div class="success-box">✅ Models trained successfully!</div>', 
                                       unsafe_allow_html=True)
                            
                            # Display results
                            self.display_model_results()
                        else:
                            st.error("Model training failed.")
                    else:
                        st.error("Data preprocessing failed. Please check your dataset.")
                        
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
                    st.info("💡 Tip: Make sure your dataset has a 'price' column and some numeric features like 'bed', 'bath', or 'sqft'.")
        
        # Show training status
        if st.session_state.models_trained:
            st.markdown("""
            <div class="success-box">
            ✅ <b>Models are trained and ready for predictions!</b><br>
            You can now use the Predictions section to estimate property values.
            </div>
            """, unsafe_allow_html=True)
    
    def display_model_results(self):
        if 'results' not in st.session_state:
            return
        
        results = st.session_state.results
        
        st.subheader("📊 Model Performance Comparison")
        
        # Create metrics display
        for model_name, metrics in results.items():
            st.write(f"**{model_name}**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAE", f"${metrics['MAE']:,.0f}")
            with col2:
                st.metric("RMSE", f"${metrics['RMSE']:,.0f}")
            with col3:
                st.metric("R² Score", f"{metrics['R2']:.3f}")
            with col4:
                st.metric("Within 20%", f"{metrics['Within_20_Percent']:.1f}%")
        
        # Best model
        if results:
            best_model = min(results.items(), key=lambda x: x[1]['RMSE'])
            st.success(f"**Best Performing Model:** {best_model[0]} (RMSE: ${best_model[1]['RMSE']:,.0f})")
    
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
        
        self.model = st.session_state.model
        self.df = st.session_state.current_df
        
        st.subheader("Predict Property Value")
        st.write("Enter property details below to get a price prediction:")
        
        # Create input form based on available features
        input_features = {}
        
        # Get the feature names from the trained model
        feature_names = self.model.feature_names
        
        # Create inputs for each feature
        cols = st.columns(2)
        col_index = 0
        
        for i, feature in enumerate(feature_names):
            if 'bed' in feature.lower() or feature == 'bed':
                input_features[feature] = cols[col_index].number_input(
                    "Bedrooms", min_value=0, max_value=10, value=3, key=f"bed_{i}"
                )
            elif 'bath' in feature.lower() or feature == 'bath':
                input_features[feature] = cols[col_index].number_input(
                    "Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5, key=f"bath_{i}"
                )
            elif 'sqft' in feature.lower() or feature in ['sqft', 'square_feet']:
                input_features[feature] = cols[col_index].number_input(
                    "Square Feet", min_value=500, max_value=10000, value=2000, key=f"sqft_{i}"
                )
            elif 'age' in feature.lower():
                input_features[feature] = cols[col_index].number_input(
                    "Property Age (years)", min_value=0, max_value=100, value=20, key=f"age_{i}"
                )
            elif 'pool' in feature.lower():
                input_features[feature] = cols[col_index].selectbox(
                    "Has Pool", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key=f"pool_{i}"
                )
            else:
                # For other features, use a default value
                input_features[feature] = cols[col_index].number_input(
                    f"{feature}", value=0.0, key=f"other_{i}"
                )
            
            col_index = (col_index + 1) % 2
        
        if st.button("Predict Price", key="predict_price"):
            try:
                # Prepare input features in correct order
                feature_vector = [input_features[feature] for feature in feature_names]
                
                # Make prediction
                predicted_price = self.model.predict_single_property(feature_vector)
                
                if predicted_price is not None:
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>🎯 Prediction Result</h3>
                    <p style="font-size: 2rem; color: #28a745; font-weight: bold;">
                        Estimated Value: <b>${predicted_price:,.0f}</b>
                    </p>
                    <p><strong>Input Features:</strong></p>
                    <ul>
                        {"".join([f"<li>{feature}: {input_features[feature]}</li>" for feature in feature_names])}
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Prediction failed. Please check the input values.")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
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
        
        self.results = st.session_state.results
        self.model = st.session_state.model
        
        # Feature Importance
        st.subheader("🔍 Feature Importance")
        
        importance_df = self.model.get_feature_importance()
        if importance_df is not None:
            fig = px.bar(importance_df.head(10), 
                        x='importance', 
                        y='feature',
                        orientation='h',
                        title='Top 10 Most Important Features',
                        labels={'importance': 'Importance', 'feature': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("Feature Importance Details:")
            st.dataframe(importance_df)
        else:
            st.info("Feature importance data not available.")
        
        # Model Comparison Chart
        st.subheader("📈 Model Performance Comparison")
        
        if self.results:
            models = list(self.results.keys())
            rmse_values = [self.results[model]['RMSE'] for model in models]
            
            fig = go.Figure(data=[
                go.Bar(name='RMSE (Lower is Better)', x=models, y=rmse_values)
            ])
            fig.update_layout(title='Model RMSE Comparison')
            st.plotly_chart(fig, use_container_width=True)

def main():
    app = PropertyValuationApp()
    app.run()

if __name__ == "__main__":
    main()