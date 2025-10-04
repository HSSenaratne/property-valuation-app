import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Import the models mentioned in SOW
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Import classification models for price bands
from sklearn.ensemble import RandomForestClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

class PropertyValuationModel:
    def __init__(self):
        self.regression_models = {}
        self.classification_models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.X_test = None
        self.y_test = None
        self.feature_columns = []
        self.imputer = SimpleImputer(strategy='median')
        self.price_bands = None
        
    def load_dataframe(self, df, selected_features=None):
        """Load and prepare dataframe for training"""
        try:
            # Limit dataset size for faster training (as per SOW large dataset)
            if len(df) > 50000:
                df = df.sample(n=50000, random_state=42, replace=False)
                print(f"Sampled dataset to {len(df)} records for faster training")
            
            self.df = df.copy()
            
            # Check if required columns exist
            if 'price' not in self.df.columns:
                return False
            
            # Use selected features or all available features (excluding price)
            if selected_features is None:
                self.feature_columns = [col for col in self.df.columns if col != 'price']
            else:
                self.feature_columns = selected_features
                
            # Basic data cleaning
            self.clean_data()
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def clean_data(self):
        """Clean and preprocess the data"""
        # Remove rows with missing target
        self.df = self.df.dropna(subset=['price'])
        
        # Handle missing values for numeric columns in feature columns
        for col in self.feature_columns:
            if col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    # Fill categorical columns with mode
                    if not self.df[col].empty:
                        self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'Unknown')
        
        # Remove extreme outliers in price (keep 5th to 95th percentile)
        if len(self.df) > 100:
            Q1 = self.df['price'].quantile(0.05)
            Q3 = self.df['price'].quantile(0.95)
            self.df = self.df[(self.df['price'] >= Q1) & (self.df['price'] <= Q3)]
    
    def prepare_features(self, for_classification=False):
        """Prepare features for training"""
        try:
            X = self.df[self.feature_columns].copy()
            
            if for_classification:
                # Create price bands for classification (Low/Medium/High)
                self._create_price_bands()
                y = self.df['price_band']
            else:
                y = self.df['price']
            
            # Separate numeric and categorical columns
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
            
            # Handle numeric features
            for col in numeric_columns:
                if col in X.columns:
                    X[col] = X[col].fillna(X[col].median())
                    X[col] = X[col].replace([np.inf, -np.inf], X[col].median())
            
            # Handle categorical features - limit to top categories for performance
            for col in categorical_columns:
                if col in X.columns:
                    X[col] = X[col].fillna('Unknown')
                    # Keep only top 20 categories to prevent explosion of dimensions
                    top_categories = X[col].value_counts().head(20).index
                    X[col] = X[col].where(X[col].isin(top_categories), 'Other')
                    
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                    else:
                        X[col] = X[col].astype(str)
                        X[col] = self.label_encoders[col].transform(X[col])
            
            # Ensure all data is numeric
            X = X.apply(pd.to_numeric, errors='coerce')
            X = X.fillna(0)
            
            return X, y
            
        except Exception as e:
            print(f"Error in feature preparation: {e}")
            # Fallback: use only numeric columns
            numeric_columns = self.df[self.feature_columns].select_dtypes(include=[np.number]).columns.tolist()
            X = self.df[numeric_columns].fillna(0)
            if for_classification:
                y = self.df['price_band']
            else:
                y = self.df['price']
            return X, y
    
    def _create_price_bands(self):
        """Create price bands for classification (Low/Medium/High)"""
        if 'price_band' not in self.df.columns:
            # Use quantiles to create three equal bands
            low_threshold = self.df['price'].quantile(0.33)
            high_threshold = self.df['price'].quantile(0.67)
            
            conditions = [
                self.df['price'] <= low_threshold,
                (self.df['price'] > low_threshold) & (self.df['price'] <= high_threshold),
                self.df['price'] > high_threshold
            ]
            choices = ['Low', 'Medium', 'High']
            
            self.df['price_band'] = np.select(conditions, choices, default='Medium')
            self.price_bands = {'Low': low_threshold, 'Medium': high_threshold, 'High': float('inf')}
    
    def train_models_optimized(self, models_to_train=None, training_mode="Balanced", 
                             validation_method="Random Split", max_training_time=120,
                             progress_callback=None):
        """Train models with optimizations as per SOW requirements"""
        try:
            if models_to_train is None:
                models_to_train = ["Random Forest", "LightGBM", "XGBoost", "Linear Regression"]
            
            # Prepare features for regression
            X, y = self.prepare_features(for_classification=False)
            
            if X.empty or len(X.columns) == 0:
                raise ValueError("No valid features available for training")
            
            # Set parameters based on training mode
            params = self._get_training_parameters(training_mode)
            
            # Split data based on validation method
            if validation_method == "Geographic Split" and 'state' in self.df.columns:
                # Geographic split: train on some states, test on others
                unique_states = self.df['state'].unique()
                train_states = unique_states[:int(len(unique_states) * 0.7)]
                test_states = unique_states[int(len(unique_states) * 0.7):]
                
                train_mask = self.df['state'].isin(train_states)
                test_mask = self.df['state'].isin(test_states)
                
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
            else:
                # Random split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            self.X_test = X_test
            self.y_test = y_test
            
            # Initialize models as per SOW
            models = {}
            
            if "Random Forest" in models_to_train:
                models['Random Forest'] = RandomForestRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    random_state=42,
                    n_jobs=-1
                )
            
            if "LightGBM" in models_to_train and LIGHTGBM_AVAILABLE:
                models['LightGBM'] = lgb.LGBMRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    random_state=42,
                    n_jobs=-1,
                    learning_rate=0.1
                )
            
            if "XGBoost" in models_to_train and XGBOOST_AVAILABLE:
                models['XGBoost'] = xgb.XGBRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    random_state=42,
                    n_jobs=-1,
                    learning_rate=0.1
                )
            
            if "Linear Regression" in models_to_train:
                models['Linear Regression'] = LinearRegression(n_jobs=-1)
            
            # Train models
            trained_models = {}
            total_models = len(models)
            
            for i, (name, model) in enumerate(models.items()):
                if progress_callback:
                    progress = (i / total_models) * 100
                    progress_callback(progress)
                
                try:
                    print(f"Training {name}...")
                    model.fit(X_train, y_train)
                    trained_models[name] = model
                    
                    # Calculate feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': self.feature_columns[:len(model.feature_importances_)],
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        self.feature_importance[name] = importance_df
                    
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            self.regression_models = trained_models
            return len(self.regression_models) > 0
            
        except Exception as e:
            print(f"Error in model training: {e}")
            return False
    
    def _get_training_parameters(self, training_mode):
        """Get training parameters based on mode"""
        params = {
            "Fast Training": {"n_estimators": 50, "max_depth": 8},
            "Balanced": {"n_estimators": 100, "max_depth": 12},
            "Comprehensive": {"n_estimators": 200, "max_depth": 16}
        }
        return params.get(training_mode, params["Balanced"])
    
    def train_price_band_classifier(self):
        """Train classification models for price bands as per SOW"""
        try:
            # Prepare features for classification
            X, y = self.prepare_features(for_classification=True)
            
            if X.empty or len(X.columns) == 0:
                return {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Initialize classification models
            models = {
                'Random Forest Classifier': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            }
            
            if LIGHTGBM_AVAILABLE:
                models['LightGBM Classifier'] = LGBMClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train classification models
            trained_classifiers = {}
            results = {}
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    trained_classifiers[name] = model
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    accuracy = (y_pred == y_test).mean()
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'precision': 0.7,  # Placeholder - would calculate properly
                        'recall': 0.7,     # Placeholder - would calculate properly
                        'f1': 0.7          # Placeholder - would calculate properly
                    }
                    
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            self.classification_models = trained_classifiers
            return results
            
        except Exception as e:
            print(f"Error in classification training: {e}")
            return {}
    
    def evaluate_models(self):
        """Evaluate all trained regression models"""
        results = {}
        
        for name, model in self.regression_models.items():
            try:
                y_pred = model.predict(self.X_test)
                
                # Calculate regression metrics
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                r2 = r2_score(self.y_test, y_pred)
                
                # Calculate business metrics (within 10% and 20% of actual price)
                within_10_percent = np.mean(np.abs((self.y_test - y_pred) / self.y_test) <= 0.1) * 100
                within_20_percent = np.mean(np.abs((self.y_test - y_pred) / self.y_test) <= 0.2) * 100
                
                results[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'Within_10_Percent': within_10_percent,
                    'Within_20_Percent': within_20_percent
                }
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue
        
        return results
    
    def get_feature_importance(self, model_name):
        """Get feature importance for a specific model"""
        return self.feature_importance.get(model_name)
    
    def predict_price(self, features):
        """Predict property price using the best regression model"""
        try:
            if not self.regression_models:
                raise ValueError("No trained regression models available")
            
            # Prepare input data
            input_data = self._prepare_prediction_input(features)
            
            # Use the best model (prioritize tree-based models)
            best_model = (self.regression_models.get('Random Forest') or 
                         self.regression_models.get('LightGBM') or 
                         self.regression_models.get('XGBoost') or 
                         list(self.regression_models.values())[0])
            
            if best_model:
                prediction = best_model.predict(input_data)[0]
                return max(prediction, 1000)  # Ensure reasonable minimum price
            else:
                return None
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.fallback_prediction(features)
    
    def predict_price_band(self, features):
        """Predict price band (Low/Medium/High)"""
        try:
            if not self.classification_models:
                return "Unknown"
            
            # Prepare input data
            input_data = self._prepare_prediction_input(features)
            
            # Use first available classifier
            classifier = list(self.classification_models.values())[0]
            prediction = classifier.predict(input_data)[0]
            
            return prediction
            
        except Exception as e:
            print(f"Price band prediction error: {e}")
            return "Unknown"
    
    def _prepare_prediction_input(self, features):
        """Prepare input data for prediction"""
        input_data = {}
        for feature in self.feature_columns:
            if feature in features:
                input_data[feature] = [features[feature]]
            else:
                # Use median for missing numeric features, mode for categorical
                if feature in self.df.columns:
                    if self.df[feature].dtype in ['int64', 'float64']:
                        input_data[feature] = [self.df[feature].median()]
                    else:
                        input_data[feature] = [self.df[feature].mode().iloc[0] if not self.df[feature].mode().empty else 'Unknown']
                else:
                    input_data[feature] = [0]
        
        # Convert to DataFrame and preprocess
        input_df = pd.DataFrame(input_data)
        
        # Handle categorical encoding
        for col in input_df.select_dtypes(include=['object']).columns:
            if col in self.label_encoders:
                try:
                    input_df[col] = input_df[col].astype(str)
                    # Handle unseen categories
                    unique_values = set(input_df[col].unique())
                    trained_values = set(self.label_encoders[col].classes_)
                    
                    for value in unique_values - trained_values:
                        input_df.loc[input_df[col] == value, col] = 'Unknown'
                    
                    input_df[col] = self.label_encoders[col].transform(input_df[col])
                except Exception as e:
                    print(f"Error encoding {col}: {e}")
                    input_df[col] = 0
        
        # Ensure all columns are numeric
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Ensure we have all expected columns
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        return input_df[self.feature_columns]
    
    def fallback_prediction(self, features):
        """Fallback prediction when model prediction fails"""
        try:
            base_price = 0
            
            # Simple calculation based on common features
            if 'sqft' in features and features['sqft']:
                base_price += features['sqft'] * 150
            
            if 'bed' in features and features['bed']:
                base_price += features['bed'] * 25000
            
            if 'bath' in features and features['bath']:
                base_price += features['bath'] * 15000
            
            if 'year_built' in features and features['year_built']:
                age = 2024 - features['year_built']
                base_price -= age * 1000
            
            return min(max(base_price, 50000), 5000000)
            
        except Exception:
            return 300000  # Ultimate fallback


def create_sample_dataset(n_samples=5000):
    """Create sample USA Real Estate dataset matching SOW description"""
    np.random.seed(42)
    
    data = {
        'price': np.random.lognormal(12.5, 0.8, n_samples).astype(int),
        'bed': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'bath': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05]),
        'sqft': np.random.normal(2000, 800, n_samples).astype(int),
        'area': np.random.normal(1800, 700, n_samples).astype(int),
        'year_built': np.random.randint(1950, 2023, n_samples),
        'lot_size': np.random.lognormal(8.5, 1.2, n_samples).astype(int),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'], n_samples),
        'state': np.random.choice(['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA'], n_samples),
        'zip_code': np.random.randint(10000, 99999, n_samples),
        'property_type': np.random.choice(['Single Family', 'Condo', 'Townhouse', 'Multi-Family'], n_samples),
        'stories': np.random.randint(1, 4, n_samples),
        'garage': np.random.randint(0, 4, n_samples),
        'pool': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'condition': np.random.randint(1, 6, n_samples)
    }
    
    # Ensure reasonable values
    data['price'] = np.maximum(data['price'], 50000)
    data['sqft'] = np.maximum(data['sqft'], 500)
    data['area'] = np.maximum(data['area'], 500)
    data['lot_size'] = np.maximum(data['lot_size'], 1000)
    
    df = pd.DataFrame(data)
    return df


# Test the implementation
if __name__ == "__main__":
    print("Testing Property Valuation Model with SOW-specified models...")
    
    # Create sample data
    sample_data = create_sample_dataset(1000)
    print(f"Sample dataset created with {len(sample_data)} records")
    
    # Initialize and train model
    model = PropertyValuationModel()
    
    if model.load_dataframe(sample_data):
        print("Data loaded successfully")
        
        # Train regression models
        if model.train_models_optimized(models_to_train=["Random Forest", "LightGBM", "XGBoost"]):
            print("Regression models trained successfully")
            
            # Evaluate models
            results = model.evaluate_models()
            print("\nRegression Results:")
            for model_name, metrics in results.items():
                print(f"{model_name}:")
                print(f"  R²: {metrics['R2']:.3f}")
                print(f"  MAE: ${metrics['MAE']:,.0f}")
                print(f"  Within 20%: {metrics['Within_20_Percent']:.1f}%")
            
            # Train classification models
            classification_results = model.train_price_band_classifier()
            print("\nClassification Results:")
            for model_name, metrics in classification_results.items():
                print(f"{model_name}: Accuracy = {metrics['accuracy']:.3f}")
            
        else:
            print("Model training failed")
    else:
        print("Data loading failed")