# gsai_sales/controllers/sales_forecast/train.py
"""
Sales Forecast Model Training
Handles high variance data with log transformation and stratified weighting
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import frappe
import json
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

MODEL_PATH = frappe.get_site_path("indexes/sales_forecast/models")

class SalesForecastTrainer:
    """Main trainer class for sales forecasting"""

    def __init__(self):
        self.model_directory = MODEL_PATH
        self.ensure_model_directory()

        self.scaler = RobustScaler()
        self.model_qty = None
        self.model_value = None

        self.categorical_cols = []
        self.numeric_cols = []
        self.categorical_encodings = {}
        self.use_log_transform = True

        self.load_models()

    def ensure_model_directory(self):
        """Create model directory if it doesn't exist"""
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)

    def determine_feature_types(self, df):
        """Automatically determine categorical vs numeric features"""
        self.categorical_cols = []
        self.numeric_cols = []

        exclude_cols = ['date', 'sales_order', 'target_qty', 'target_value', 'hash_key', 'source']

        for col in df.columns:
            if col in exclude_cols:
                continue

            dtype = df[col].dtype

            if dtype in ['object', 'string']:
                self.categorical_cols.append(col)
            elif dtype in ['int64', 'float64', 'int32', 'float32']:
                n_unique = df[col].nunique()
                if n_unique < 20 and col not in ['year', 'month', 'day', 'quarter', 'week', 'dayofweek']:
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)

    def create_stratified_weights(self, y_qty, y_value):
        """
        Create sample weights that balance small and large orders
        Ensures model learns patterns from both typical and large orders
        """
        # Create bins for order sizes
        qty_bins = pd.qcut(y_qty, q=5, labels=False, duplicates='drop')
        value_bins = pd.qcut(y_value, q=5, labels=False, duplicates='drop')

        # Count samples in each bin
        qty_counts = qty_bins.value_counts()
        value_counts = value_bins.value_counts()

        # Inverse frequency weighting - rare bins get higher weight
        qty_weights = qty_bins.map(lambda x: 1.0 / qty_counts.get(x, 1))
        value_weights = value_bins.map(lambda x: 1.0 / value_counts.get(x, 1))

        # Average the weights
        stratified_weights = (qty_weights + value_weights) / 2

        # Normalize to mean 1.0
        stratified_weights = stratified_weights / stratified_weights.mean()

        return stratified_weights.values

    def get_training_data(self, from_date=None, to_date=None):
        """Fetch and prepare training data"""
        try:
            conditions = ["source = 'actual'"]
            if from_date:
                conditions.append(f"date >= '{from_date}'")
            if to_date:
                conditions.append(f"date <= '{to_date}'")

            where_clause = " AND ".join(conditions)

            indicators = frappe.db.sql(f"""
                SELECT * FROM `tabSales Forecast Indicator`
                WHERE {where_clause}
                ORDER BY date ASC
            """, as_dict=True)

            if not indicators:
                return None, None, None

            frappe.msgprint(f"Loaded {len(indicators)} training samples")

            # Parse data
            df_list = []
            for ind in indicators:
                try:
                    row_data = json.loads(ind['data']) if ind.get('data') else {}
                    row_data['date'] = ind.get('date')
                    row_data['target_qty'] = float(ind.get('target_qty', 0))
                    row_data['target_value'] = float(ind.get('target_value', 0))
                    row_data['sales_order'] = ind.get('sales_order')
                    df_list.append(row_data)
                except Exception as e:
                    frappe.log_error(f"Error parsing indicator: {str(e)}")
                    continue

            df = pd.DataFrame(df_list)
            df['date'] = pd.to_datetime(df['date'])

            # Show data distribution
            frappe.msgprint(f"""
                Data Summary:
                - Quantity range: {df['target_qty'].min():.0f} to {df['target_qty'].max():.0f}
                - Quantity median: {df['target_qty'].median():.0f}
                - Value range: ₹{df['target_value'].min():,.0f} to ₹{df['target_value'].max():,.0f}
                - Date range: {df['date'].min().date()} to {df['date'].max().date()}
            """)

            # Determine feature types
            if not self.categorical_cols and not self.numeric_cols:
                self.determine_feature_types(df)

            # Create stratified weights to balance order sizes
            stratified_weights = self.create_stratified_weights(
                df['target_qty'],
                df['target_value']
            )

            # Also apply time decay (recent data more important)
            max_date = df['date'].max()
            days_old = (max_date - df['date']).dt.days
            time_weights = np.exp(-days_old / 365)

            # Combine weights
            sample_weights = stratified_weights * time_weights

            # Separate features and targets
            exclude_cols = ['date', 'sales_order', 'target_qty', 'target_value']
            feature_cols = [col for col in df.columns if col not in exclude_cols]

            X = df[feature_cols].copy()
            y = df[['target_qty', 'target_value']].copy()

            # Apply log transformation to handle variance
            if self.use_log_transform:
                y['target_qty'] = np.log1p(y['target_qty'])
                y['target_value'] = np.log1p(y['target_value'])

            return X, y, sample_weights

        except Exception as e:
            frappe.log_error(f"Error getting training data: {str(e)}")
            frappe.throw(f"Error loading training data: {str(e)}")
            return None, None, None

    def preprocess_features(self, X, fit=False):
        """Preprocess features with encoding and scaling"""
        try:
            X = X.copy()

            # Handle numeric columns
            for col in self.numeric_cols:
                if col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

            # Encode categorical features
            if fit:
                for col in self.categorical_cols:
                    if col in X.columns:
                        unique_vals = X[col].fillna('MISSING').unique()
                        self.categorical_encodings[col] = {
                            val: idx for idx, val in enumerate(unique_vals)
                        }

            # Apply categorical encoding
            for col in self.categorical_cols:
                if col in X.columns and col in self.categorical_encodings:
                    X[col] = X[col].fillna('MISSING').map(
                        lambda x: self.categorical_encodings[col].get(x, -1)
                    )

            # Fill remaining NaN values
            X = X.fillna(0)

            # Ensure all values are numeric
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

            # Apply robust scaling
            if fit:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)

            return X_scaled

        except Exception as e:
            frappe.log_error(f"Error preprocessing features: {str(e)}")
            raise

    def train(self, from_date=None, to_date=None, validate=True):
        """
        Train the sales forecast model

        Args:
            from_date: Start date for training data
            to_date: End date for training data
            validate: Whether to use validation split
        """
        try:
            frappe.msgprint("Starting model training...")

            # Load training data
            X, y, sample_weights = self.get_training_data(from_date, to_date)

            if X is None or len(X) < 50:
                frappe.throw("Insufficient training data. Need at least 50 samples.")
                return False

            frappe.msgprint(f"Training on {len(X)} samples with {X.shape[1]} features")

            if validate:
                # Time-based train/validation split (80/20)
                split_idx = int(len(X) * 0.8)

                X_train = X.iloc[:split_idx]
                X_val = X.iloc[split_idx:]
                y_train = y.iloc[:split_idx]
                y_val = y.iloc[split_idx:]
                weights_train = sample_weights[:split_idx]

                # Keep original values for metric calculation
                y_val_original = np.expm1(y_val) if self.use_log_transform else y_val

                # Preprocess features
                X_train_scaled = self.preprocess_features(X_train, fit=True)
                X_val_scaled = self.preprocess_features(X_val, fit=False)

                frappe.msgprint(f"Train: {len(X_train)}, Validation: {len(X_val)}")
            else:
                # Use all data for training
                X_train_scaled = self.preprocess_features(X, fit=True)
                y_train = y
                weights_train = sample_weights

            frappe.msgprint("Initializing XGBoost models...")

            # XGBoost parameters optimized for high variance data
            params = {
                'n_estimators': 400,
                'learning_rate': 0.02,
                'max_depth': 6,
                'min_child_weight': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'gamma': 0.3,
                'reg_alpha': 0.5,
                'reg_lambda': 3.0,
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1
            }

            self.model_qty = xgb.XGBRegressor(**params)
            self.model_value = xgb.XGBRegressor(**params)

            # Train quantity model
            frappe.msgprint("Training quantity prediction model...")
            if validate:
                self.model_qty.fit(
                    X_train_scaled,
                    y_train['target_qty'],
                    sample_weight=weights_train,
                    eval_set=[(X_val_scaled, y_val['target_qty'])],
                    verbose=False
                )
            else:
                self.model_qty.fit(
                    X_train_scaled,
                    y_train['target_qty'],
                    sample_weight=weights_train
                )

            # Train value model
            frappe.msgprint("Training value prediction model...")
            if validate:
                self.model_value.fit(
                    X_train_scaled,
                    y_train['target_value'],
                    sample_weight=weights_train,
                    eval_set=[(X_val_scaled, y_val['target_value'])],
                    verbose=False
                )
            else:
                self.model_value.fit(
                    X_train_scaled,
                    y_train['target_value'],
                    sample_weight=weights_train
                )

            # Calculate validation metrics
            metrics = {}
            if validate:
                frappe.msgprint("Calculating validation metrics...")

                # Make predictions
                y_pred_qty = self.model_qty.predict(X_val_scaled)
                y_pred_value = self.model_value.predict(X_val_scaled)

                # Transform back from log space
                if self.use_log_transform:
                    y_pred_qty = np.expm1(y_pred_qty)
                    y_pred_value = np.expm1(y_pred_value)

                # Calculate overall metrics
                metrics['qty_mae'] = mean_absolute_error(y_val_original['target_qty'], y_pred_qty)
                metrics['qty_rmse'] = np.sqrt(mean_squared_error(y_val_original['target_qty'], y_pred_qty))
                metrics['qty_mape'] = np.mean(
                    np.abs((y_val_original['target_qty'] - y_pred_qty) / (y_val_original['target_qty'] + 1))
                ) * 100

                metrics['value_mae'] = mean_absolute_error(y_val_original['target_value'], y_pred_value)
                metrics['value_rmse'] = np.sqrt(mean_squared_error(y_val_original['target_value'], y_pred_value))
                metrics['value_mape'] = np.mean(
                    np.abs((y_val_original['target_value'] - y_pred_value) / (y_val_original['target_value'] + 1))
                ) * 100

                # Calculate metrics by order size
                val_qty = y_val_original['target_qty'].values
                threshold = np.percentile(val_qty, 75)
                small_orders = val_qty <= threshold
                large_orders = val_qty > threshold

                small_mape = np.mean(
                    np.abs((val_qty[small_orders] - y_pred_qty[small_orders]) / (val_qty[small_orders] + 1))
                ) * 100

                large_mape = np.mean(
                    np.abs((val_qty[large_orders] - y_pred_qty[large_orders]) / (val_qty[large_orders] + 1))
                ) * 100 if large_orders.sum() > 0 else 0

                # Display results
                frappe.msgprint(f"""
                    <h4>✅ Model Training Completed Successfully!</h4>

                    <h5>Overall Performance:</h5>
                    <b>Quantity Predictions:</b><br>
                    • MAE: {metrics['qty_mae']:.2f} units<br>
                    • RMSE: {metrics['qty_rmse']:.2f} units<br>
                    • MAPE: {metrics['qty_mape']:.2f}%<br>
                    <br>
                    <b>Value Predictions:</b><br>
                    • MAE: ₹{metrics['value_mae']:,.2f}<br>
                    • RMSE: ₹{metrics['value_rmse']:,.2f}<br>
                    • MAPE: {metrics['value_mape']:.2f}%<br>

                    <h5>By Order Size:</h5>
                    • Smaller orders (75%): MAPE {small_mape:.1f}%<br>
                    • Larger orders (25%): MAPE {large_mape:.1f}%<br>
                    <br>
                    <i>💡 MAPE < 25% is good, < 15% is excellent</i>
                """, title="Training Results", indicator="green")

                # Save metrics
                self.save_metrics(metrics)

            # Save models
            frappe.msgprint("Saving models...")
            self.save_models()

            frappe.msgprint("✅ Training completed successfully!", indicator="green")
            return True

        except Exception as e:
            frappe.log_error(f"Error during training: {str(e)}", "Sales Forecast Training")
            frappe.throw(f"Training failed: {str(e)}")
            return False

    def save_models(self):
        """Save trained models and configurations"""
        try:
            # Save XGBoost models
            joblib.dump(self.model_qty, os.path.join(self.model_directory, "model_qty.pkl"))
            joblib.dump(self.model_value, os.path.join(self.model_directory, "model_value.pkl"))

            # Save scaler
            joblib.dump(self.scaler, os.path.join(self.model_directory, "scaler.pkl"))

            # Save configuration
            config = {
                'categorical_encodings': self.categorical_encodings,
                'categorical_cols': self.categorical_cols,
                'numeric_cols': self.numeric_cols,
                'use_log_transform': self.use_log_transform,
                'feature_names': list(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else []
            }
            joblib.dump(config, os.path.join(self.model_directory, "feature_config.pkl"))

        except Exception as e:
            frappe.log_error(f"Error saving models: {str(e)}")
            raise

    def load_models(self):
        """Load existing trained models"""
        try:
            model_path = os.path.join(self.model_directory, "model_qty.pkl")
            if os.path.exists(model_path):
                self.model_qty = joblib.load(model_path)
                self.model_value = joblib.load(os.path.join(self.model_directory, "model_value.pkl"))
                self.scaler = joblib.load(os.path.join(self.model_directory, "scaler.pkl"))

                config = joblib.load(os.path.join(self.model_directory, "feature_config.pkl"))
                self.categorical_encodings = config['categorical_encodings']
                self.categorical_cols = config['categorical_cols']
                self.numeric_cols = config['numeric_cols']
                self.use_log_transform = config.get('use_log_transform', True)

                return True
            return False
        except Exception as e:
            frappe.log_error(f"Error loading models: {str(e)}")
            return False

    def save_metrics(self, metrics):
        """Save training metrics for tracking"""
        try:
            # Check if doctype exists
            if not frappe.db.exists("DocType", "Sales Forecast Model Metrics"):
                return

            metrics_doc = frappe.get_doc({
                "doctype": "Sales Forecast Model Metrics",
                "date": datetime.now(),
                "qty_mae": metrics.get('qty_mae'),
                "qty_rmse": metrics.get('qty_rmse'),
                "qty_mape": metrics.get('qty_mape'),
                "value_mae": metrics.get('value_mae'),
                "value_rmse": metrics.get('value_rmse'),
                "value_mape": metrics.get('value_mape'),
            })
            metrics_doc.insert(ignore_permissions=True)
            frappe.db.commit()
        except Exception as e:
            frappe.log_error(f"Error saving metrics: {str(e)}")


# Whitelisted API functions
@frappe.whitelist()
def train_model(from_date=None, to_date=None, validate=True):
    """
    Train the sales forecast model

    Args:
        from_date: Optional start date
        to_date: Optional end date
        validate: Whether to use validation split (default: True)

    Returns:
        True if successful
    """
    # Convert string to boolean if needed
    if isinstance(validate, str):
        validate = validate.lower() in ['true', '1', 'yes']

    trainer = SalesForecastTrainer()
    return trainer.train(from_date=from_date, to_date=to_date, validate=validate)

@frappe.whitelist()
def check_model_status():
    """Check if model exists and get information"""
    MODEL_PATH = frappe.get_site_path("indexes/sales_forecast/models")

    status = {
        "model_exists": False,
        "files": [],
        "path": MODEL_PATH
    }

    if os.path.exists(MODEL_PATH):
        files = os.listdir(MODEL_PATH)
        status["files"] = files
        status["model_exists"] = "model_qty.pkl" in files and "model_value.pkl" in files

    return status
