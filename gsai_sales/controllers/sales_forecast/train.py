import os
import frappe
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from gsai_sales.controllers.sales_forecast.data import get_forecast_fields

MODEL_PATH = frappe.get_site_path("indexes/sales_forecast/models")

class SalesForecastTrainer:
    def __init__(self, settings_doctype="GSAI-S Forecast Settings"):
        self.settings = frappe.get_single(settings_doctype)
        self.model_directory = MODEL_PATH
        self.ensure_model_directory()
        self.scaler = None
        self.model = None
        self.has_trained_model = False
        self.load_or_initialize_model()
        self.load_features()
        
    def ensure_model_directory(self):
        """Ensure model directory exists"""
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        
    def load_or_initialize_model(self):
        """Load existing model or initialize a new one"""
        model_path = os.path.join(self.model_directory, "forecast_model.pkl")
        scaler_path = os.path.join(self.model_directory, "scaler.pkl")
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)

                self.has_trained_model = True

                print("Loaded existing sales forecast model")
            else:
                self.initialize_new_model()
        except Exception as e:
            # frappe.logger().error(f"Error loading model, initializing new: {str(e)}")
            print(f"Error loading model, initializing new: {str(e)}")
            self.initialize_new_model()
    
    def load_features(self):
        ftypes = {
            "binary": ("Check", ),
            "numeric": ("Currency", "Int", "Long Int", "Float", "Percent", "Check", "Duration", "Phone"),
            "datetime": ("Date", "Datetime", "Time"),
            "periodic": ("Periodic", ),
            "categorical": [],
        }

        self.feature_type = {}
        self.feature_list = []

        features = get_forecast_fields(as_list=False, extended=True)

        for d in features:
            self.feature_list.append(d.get("fieldname"))
            self.feature_type[d.get("fieldname")] = "categorical"

            for ftype, values in ftypes.items():
                if d.get("fieldtype") in values:
                    self.feature_type[d.get("fieldname")] = ftype
                    break
    
    def initialize_new_model(self):
        """Initialize a new model"""
        # XGBoost base regressor
        base_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
        # MultiOutputRegressor for multi-target prediction (qty and value)
        self.model = MultiOutputRegressor(base_model)
        self.scaler = StandardScaler()
        # frappe.logger().info("Initialized new XGBoost sales forecast model")
        print("Initialized new XGBoost sales forecast model")
    
    def get_training_data(self, from_date=None, to_date=None):
        """Fetch training data from forecast indicators"""

        ignore_older_than = frappe.db.get_single_value("GSAI-S Forecast Settings", "ignore_older_than")

        cond = ""

        if from_date:
            cond += f" and creation >= {from_date}"

        if to_date:
            cond += f" and creation <= {to_date}"

        if ignore_older_than:
            cond += f" and datediff(now(), date) <= {ignore_older_than}"

        indicators = frappe.db.sql(
            f"""
                select hash_key, data, target_qty, target_value
                from `tabSales Forecast Indicator`
                where 1 = 1
                {cond}
            """,
            as_dict=True
        )
        
        if not indicators:
            # frappe.logger().warning("No training data found")
            print("No training data found")
            return None, None, None
            
        # Convert features from JSON to dataframe columns
        df_list = []
        for ind in indicators:
            try:
                features = json.loads(ind.data)
                features["hash_key"] = ind.hash_key
                features["target_qty"] = ind.target_qty
                features["target_value"] = ind.target_value
                df_list.append(features)
            except Exception as e:
                print(f"Error processing indicator: {str(e)}")
                continue
            
        if not df_list:
            return None, None, None
            
        df = pd.DataFrame(df_list)
        
        # Separate features and targets
        X = df[self.feature_list]
        y = df[["target_qty", "target_value"]]
        
        # Default weights
        sample_weight = np.ones(len(X))
        
        return X, y, sample_weight

    def preprocess_features(self, X):
        """Preprocess features for model training"""
        X = X.copy()
        
        for col in X.columns:
            feature_type = self.feature_type.get(col, 'categorical')
            
            if feature_type == 'categorical':
                # Categorical: use category codes
                X[col] = X[col].astype('category').cat.codes
            
            elif feature_type == 'numeric':
                # Numeric: ensure numeric type
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            elif feature_type == 'datetime':
                # Datetime: convert to numeric (e.g., timestamp)
                X[col] = pd.to_datetime(X[col], errors='coerce').astype(int) // 10**9
            
            elif feature_type == 'binary':
                # Binary: convert to 0 or 1
                X[col] = X[col].astype(int)

            elif feature_type == 'periodic':
                X[col] = (X[col] - X[col].min()) / (X[col].max() - X[col].min())

        # Handle missing values
        X = X.fillna(0)
        
        return X
    
    def train(self, from_date=None, to_date=None, incremental=True):
        """Train the model with data from indicators"""
        X, y, sample_weight = self.get_training_data(from_date, to_date)
        
        if X is None or y is None:
            frappe.throw("No training data available")
            return False
            
        X = self.preprocess_features(X)
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            if not incremental:
                # Full retraining
                self.initialize_new_model()
                self.model.fit(X_scaled, y)
            else:
                # For incremental learning:
                # If first time, just fit normally
                if not hasattr(self.model, 'estimators_') or not self.model.estimators_:
                    self.model.fit(X_scaled, y)
                else:
                    # For subsequent incremental trainings
                    # We need to create a new model with more trees and train it
                    new_n_estimators = 20  # Add 20 more trees each time
                    
                    # For each target (qty and value)
                    for i, estimator in enumerate(self.model.estimators_):
                        target = y.iloc[:, i]  # Current target (qty or value)
                        
                        # Save current parameters
                        params = estimator.get_params()
                        
                        # Increase the number of trees
                        new_params = {**params}
                        new_params['n_estimators'] = params['n_estimators'] + new_n_estimators
                        
                        # Lower the learning rate for incremental learning
                        new_params['learning_rate'] = params.get('learning_rate', 0.1) * 0.8
                        
                        # Create a new estimator with updated parameters
                        new_estimator = xgb.XGBRegressor(**new_params)
                        
                        # Train on all data (base model + new data)
                        # This is the safer approach for incremental learning
                        new_estimator.fit(X_scaled, target, sample_weight=sample_weight)
                        
                        # Replace the old estimator with the new one
                        self.model.estimators_[i] = new_estimator
            
            # Save the trained model
            self.save_model()
            
            self.has_trained_model = True

            print(f"XGBoost model {'incrementally trained' if incremental else 'fully retrained'} successfully")
            return True
            
        except Exception as e:
            # frappe.logger().error(f"Error training model: {str(e)}")
            frappe.throw(f"Error training model: {str(e)}")
            return False
    
    def save_model(self):
        """Save the trained model to disk"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Save current model
        model_path = os.path.join(self.model_directory, "forecast_model.pkl")
        scaler_path = os.path.join(self.model_directory, "scaler.pkl")
        
        # Also save a timestamped backup
        backup_model_path = os.path.join(self.model_directory, f"forecast_model_{timestamp}.pkl")
        backup_scaler_path = os.path.join(self.model_directory, f"scaler_{timestamp}.pkl")
        
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            # Backup copies
            joblib.dump(self.model, backup_model_path)
            joblib.dump(self.scaler, backup_scaler_path)
            
            # frappe.logger().info(f"Model saved successfully with backup at {timestamp}")
            print(f"Model saved successfully with backup at {timestamp}")
            return True
        except Exception as e:
            # frappe.logger().error(f"Error saving model: {str(e)}")
            print(f"Error saving model: {str(e)}")
            return False

# Utility functions
def train_forecast_model(from_date=None, to_date=None, incremental=True):
    """Utility function to train the forecast model"""
    trainer = SalesForecastTrainer()
    return trainer.train(from_date, to_date, incremental)