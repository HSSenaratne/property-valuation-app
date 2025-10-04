import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')



# Try to import optional packages with fallbacks
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not available, using Random Forest only")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using Random Forest only")

class PropertyValuationModel:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.label_encoders = {}
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.is_trained = False
        
    def load_dataframe(self, df):
        """Load dataframe directly"""
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        self.df = df.copy()
        print(f"DataFrame loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return True
    
    def load_and_preprocess(self, data_path=None, df=None):
        """Load and preprocess the property data"""
        try:
            # Load dataset from dataframe if provided
            if df is not None:
                self.df = df.copy()
            elif data_path:
                if data_path.endswith('.csv'):
                    self.df = pd.read_csv(data_path)
                else:
                    self.df = pd.read_excel(data_path)
            else:
                raise ValueError("Either data_path or df must be provided")
            
            print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Basic cleaning
            self.df = self.clean_data(self.df)
            
            # Feature engineering
            self.df = self.feature_engineering(self.df)
            
            # Prepare features and target
            X, y = self.prepare_features(self.df)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"Training set: {self.X_train.shape[0]} samples")
            print(f"Testing set: {self.X_test.shape[0]} samples")
            print(f"Features used: {self.feature_names}")
            
            return self.df, X, y
            
        except Exception as e:
            print(f"Error in load_and_preprocess: {e}")
            return None, None, None
    
    def clean_data(self, df):
        """Clean the property dataset"""
        # Make a copy
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {initial_rows - df_clean.shape[0]} duplicate rows")
        
        # Handle missing values - only for numeric columns that exist
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                print(f"Filled missing values in {col}")
        
        # Handle categorical missing values
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna('Unknown', inplace=True)
                print(f"Filled missing values in {col}")
        
        # Remove extreme outliers in price if price column exists
        if 'price' in df_clean.columns:
            initial_price_rows = df_clean.shape[0]
            Q1 = df_clean['price'].quantile(0.05)
            Q3 = df_clean['price'].quantile(0.95)
            df_clean = df_clean[(df_clean['price'] >= Q1) & (df_clean['price'] <= Q3)]
            print(f"Removed {initial_price_rows - df_clean.shape[0]} price outliers")
        
        print(f"After cleaning: {df_clean.shape[0]} rows")
        return df_clean
    
    def feature_engineering(self, df):
        """Create new features for better prediction"""
        df_fe = df.copy()
        
        # Create price per square foot if sqft exists
        if all(col in df_fe.columns for col in ['price', 'sqft']):
            df_fe['price_per_sqft'] = df_fe['price'] / df_fe['sqft']
            # Handle infinite values and zeros
            df_fe['price_per_sqft'] = df_fe['price_per_sqft'].replace([np.inf, -np.inf], np.nan)
            df_fe['price_per_sqft'].fillna(df_fe['price_per_sqft'].median(), inplace=True)
            print("Created price_per_sqft feature")
        
        # Create property age if year_built exists
        if 'year_built' in df_fe.columns:
            current_year = pd.Timestamp.now().year
            df_fe['property_age'] = current_year - df_fe['year_built']
            # Handle invalid years
            df_fe['property_age'] = df_fe['property_age'].clip(lower=0, upper=200)
            df_fe['property_age'].fillna(df_fe['property_age'].median(), inplace=True)
            print("Created property_age feature")
        
        # Create bedroom to bathroom ratio
        if all(col in df_fe.columns for col in ['bed', 'bath']):
            df_fe['bed_bath_ratio'] = df_fe['bed'] / np.maximum(df_fe['bath'], 0.5)
            df_fe['bed_bath_ratio'].fillna(1.0, inplace=True)
            print("Created bed_bath_ratio feature")
        
        # Create boolean flags for amenities
        pool_columns = [col for col in df_fe.columns if 'pool' in col.lower()]
        if pool_columns:
            df_fe['has_pool'] = df_fe[pool_columns[0]].apply(
                lambda x: 1 if str(x).lower() in ['yes', 'true', '1', 'y'] else 0
            )
            print("Created has_pool feature")
        else:
            df_fe['has_pool'] = 0
        
        return df_fe
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        # Select relevant features that exist in the dataset
        feature_columns = []
        
        # Common numeric features in real estate datasets
        possible_numeric_features = [
            'bed', 'bath', 'sqft', 'acre_lot', 'year_built', 
            'price_per_sqft', 'property_age', 'bed_bath_ratio', 'has_pool',
            'bathrooms', 'bedrooms', 'square_feet', 'lot_size'
        ]
        
        for feature in possible_numeric_features:
            if feature in df.columns:
                # Handle missing values for this specific feature
                if df[feature].isnull().sum() > 0:
                    df[feature].fillna(df[feature].median(), inplace=True)
                feature_columns.append(feature)
        
        # Categorical features
        possible_categorical_features = ['state', 'city', 'property_type', 'status']
        for feature in possible_categorical_features:
            if feature in df.columns:
                # Use label encoding for categorical variables
                le = LabelEncoder()
                encoded_col = feature + '_encoded'
                df[encoded_col] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
                feature_columns.append(encoded_col)
        
        # If no features found, use all numeric columns except price
        if not feature_columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'price' in numeric_cols:
                numeric_cols.remove('price')
            feature_columns = numeric_cols[:4]  # Use first 4 numeric columns
            print(f"No standard features found, using: {feature_columns}")
        
        # Target variable
        target_col = 'price'
        if 'price' not in df.columns:
            # Look for alternative price columns
            price_columns = [col for col in df.columns if 'price' in col.lower() and col != 'price_per_sqft']
            if price_columns:
                target_col = price_columns[0]
                print(f"Using '{target_col}' as target variable")
            else:
                raise ValueError("No price column found in dataset")
        
        self.feature_names = feature_columns
        X = df[feature_columns]
        y = df[target_col]
        
        print(f"Selected {len(feature_columns)} features: {feature_columns}")
        print(f"Target variable: {target_col}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def train_models(self):
        """Train multiple machine learning models"""
        if self.X_train is None:
            raise ValueError("Data not prepared. Call load_and_preprocess first.")
        
        print("Training machine learning models...")
        
        # Always train Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=50,  # Reduced for faster training
            random_state=42, 
            n_jobs=-1,
            max_depth=10
        )
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        
        # Try LightGBM if available
        if LGBM_AVAILABLE:
            print("Training LightGBM...")
            try:
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=50,
                    random_state=42, 
                    n_jobs=-1,
                    verbose=-1
                )
                lgb_model.fit(self.X_train, self.y_train)
                self.models['LightGBM'] = lgb_model
            except Exception as e:
                print(f"LightGBM training failed: {e}")
        
        # Try XGBoost if available
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            try:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=50,
                    random_state=42, 
                    n_jobs=-1
                )
                xgb_model.fit(self.X_train, self.y_train)
                self.models['XGBoost'] = xgb_model
            except Exception as e:
                print(f"XGBoost training failed: {e}")
        
        # Calculate feature importance
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        self.is_trained = True
        print(f"Successfully trained {len(self.models)} models")
        return True
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        if not self.models:
            raise ValueError("No models trained. Call train_models first.")
        
        results = {}
        
        for name, model in self.models.items():
            try:
                # Predictions
                y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y_test, y_pred)
                
                # Percentage within 20% of actual price
                within_20_percent = np.mean(
                    np.abs((self.y_test - y_pred) / np.maximum(self.y_test, 1)) <= 0.2
                ) * 100
                
                results[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'Within_20_Percent': within_20_percent,
                    'predictions': y_pred
                }
                
                print(f"\n{name} Performance:")
                print(f"MAE: ${mae:,.2f}")
                print(f"RMSE: ${rmse:,.2f}")
                print(f"R² Score: {r2:.4f}")
                print(f"Within 20% of actual price: {within_20_percent:.2f}%")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue
        
        return results
    
    def predict_single_property(self, input_features):
        """Predict price for a single property"""
        if not self.models:
            raise ValueError("No models trained.")
        
        # Use first available model
        best_model = list(self.models.values())[0]
        
        try:
            # Ensure input features match training features
            if len(input_features) != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features, got {len(input_features)}")
            
            prediction = best_model.predict([input_features])[0]
            return prediction
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def get_feature_importance(self, model_name='Random Forest'):
        """Get feature importance for a specific model"""
        if model_name in self.feature_importance:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.feature_importance[model_name]
            }).sort_values('importance', ascending=False)
            return importance_df
        return None
    
    def save_models(self, filepath='property_models.joblib'):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'feature_importance': self.feature_importance,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='property_models.joblib'):
        """Load trained models"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.feature_importance = model_data['feature_importance']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        print(f"Models loaded from {filepath}")

def create_sample_dataset():
    """Create a sample dataset for testing"""
    np.random.seed(42)
    n_samples = 500  # Reduced for faster testing
    
    sample_data = {
        'price': np.random.normal(350000, 150000, n_samples).astype(int),
        'bed': np.random.randint(1, 6, n_samples),
        'bath': np.random.randint(1, 4, n_samples),
        'sqft': np.random.normal(2000, 800, n_samples).astype(int),
        'city': np.random.choice(['Los Angeles', 'San Diego', 'San Francisco', 'Sacramento'], n_samples),
        'state': 'CA',
        'property_type': np.random.choice(['Single Family', 'Condo', 'Townhouse'], n_samples),
        'year_built': np.random.randint(1950, 2020, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    # Remove any negative prices
    df['price'] = df['price'].clip(lower=50000)
    return df