# gsai_sales/controllers/sales_forecast/predict.py
"""
Sales Forecast Predictions
Make predictions using trained models
"""

import frappe
import json
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

MODEL_PATH = frappe.get_site_path("indexes/sales_forecast/models")

class SalesForecastPredictor:
    """Main predictor class for sales forecasting"""

    def __init__(self):
        self.model_directory = MODEL_PATH
        self.model_qty = None
        self.model_value = None
        self.scaler = None
        self.categorical_encodings = {}
        self.categorical_cols = []
        self.numeric_cols = []
        self.feature_names = []
        self.use_log_transform = True

        self.load_models()

    def load_models(self):
        """Load all trained models and configurations"""
        try:
            # Load XGBoost models
            self.model_qty = joblib.load(os.path.join(self.model_directory, "model_qty.pkl"))
            self.model_value = joblib.load(os.path.join(self.model_directory, "model_value.pkl"))

            # Load scaler
            self.scaler = joblib.load(os.path.join(self.model_directory, "scaler.pkl"))

            # Load configuration
            config = joblib.load(os.path.join(self.model_directory, "feature_config.pkl"))
            self.categorical_encodings = config['categorical_encodings']
            self.categorical_cols = config['categorical_cols']
            self.numeric_cols = config['numeric_cols']
            self.use_log_transform = config.get('use_log_transform', True)
            self.feature_names = config.get('feature_names', [])

            if not self.feature_names and hasattr(self.scaler, 'feature_names_in_'):
                self.feature_names = list(self.scaler.feature_names_in_)

            return True
        except Exception as e:
            frappe.log_error(f"Error loading models: {str(e)}", "Sales Forecast - Load Models")
            return False

    def preprocess_features(self, X):
        """Preprocess features using trained encoder and scaler"""
        try:
            X = X.copy()

            # Handle numeric columns
            for col in self.numeric_cols:
                if col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

            # Encode categorical features
            for col in self.categorical_cols:
                if col in X.columns and col in self.categorical_encodings:
                    X[col] = X[col].fillna('MISSING').map(
                        lambda x: self.categorical_encodings[col].get(x, -1)
                    )

            # Fill NaN values
            X = X.fillna(0)

            # Ensure all values are numeric
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

            # Scale features
            X_scaled = self.scaler.transform(X)

            return X_scaled
        except Exception as e:
            frappe.log_error(f"Error preprocessing: {str(e)}", "Sales Forecast - Preprocess")
            raise

    def predict(self, features):
        """
        Make sales forecast prediction

        Args:
            features: dict of feature values

        Returns:
            dict with qty and value predictions
        """
        if self.model_qty is None or self.model_value is None:
            frappe.throw("Models not loaded. Please train the model first.")
            return None

        try:
            # Get expected features
            if self.feature_names:
                expected_features = self.feature_names
            elif hasattr(self.scaler, 'feature_names_in_'):
                expected_features = list(self.scaler.feature_names_in_)
            else:
                frappe.throw("Cannot determine expected features")
                return None

            # Create DataFrame with all expected features initialized to 0
            feature_df = pd.DataFrame(0, index=[0], columns=expected_features)

            # Fill in provided values
            for key, value in features.items():
                if key in expected_features:
                    feature_df.loc[0, key] = value

            # Preprocess features
            features_scaled = self.preprocess_features(feature_df)

            # Make predictions
            pred_qty = self.model_qty.predict(features_scaled)[0]
            pred_value = self.model_value.predict(features_scaled)[0]

            # Transform back from log space if needed
            if self.use_log_transform:
                pred_qty = np.expm1(pred_qty)
                pred_value = np.expm1(pred_value)

            return {
                "qty": float(max(0, pred_qty)),
                "amount": float(max(0, pred_value)),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            frappe.log_error(f"Prediction error: {str(e)}", "Sales Forecast - Predict")
            frappe.throw(f"Error making prediction: {str(e)}")
            return None

    def batch_predict(self, features_list):
        """
        Make predictions for multiple feature sets

        Args:
            features_list: list of feature dicts

        Returns:
            list of prediction dicts
        """
        predictions = []
        for features in features_list:
            try:
                pred = self.predict(features)
                if pred:
                    predictions.append(pred)
            except Exception as e:
                frappe.log_error(f"Batch prediction error: {str(e)}")
                continue

        return predictions

    def get_required_features(self):
        """Get list of required features for prediction"""
        if self.feature_names:
            return self.feature_names
        elif hasattr(self.scaler, 'feature_names_in_'):
            return list(self.scaler.feature_names_in_)
        else:
            return []


# Whitelisted API functions
@frappe.whitelist()
def predict_sales(features):
    """
    Make a sales forecast prediction

    Args:
        features: dict or JSON string of feature values

    Returns:
        dict with predictions
    """
    if isinstance(features, str):
        features = json.loads(features)

    predictor = SalesForecastPredictor()
    return predictor.predict(features)

@frappe.whitelist()
def batch_predict_sales(features_list):
    """
    Make predictions for multiple scenarios

    Args:
        features_list: list or JSON string of feature dicts

    Returns:
        list of prediction dicts
    """
    if isinstance(features_list, str):
        features_list = json.loads(features_list)

    predictor = SalesForecastPredictor()
    return predictor.batch_predict(features_list)

@frappe.whitelist()
def get_model_info():
    """Get information about the trained model"""
    try:
        predictor = SalesForecastPredictor()

        return {
            "status": "Model loaded successfully",
            "model_exists": True,
            "required_features": predictor.get_required_features(),
            "categorical_features": predictor.categorical_cols,
            "numeric_features": predictor.numeric_cols,
            "total_features": len(predictor.get_required_features()),
            "uses_log_transform": predictor.use_log_transform
        }
    except Exception as e:
        return {
            "status": f"Error: {str(e)}",
            "model_exists": False
        }

@frappe.whitelist()
def evaluate_model():
    """Evaluate model on recent data"""
    try:
        from datetime import timedelta
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Get last 30 days of data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        # Fetch actual data
        indicators = frappe.db.sql(f"""
            SELECT * FROM `tabSales Forecast Indicator`
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            AND source = 'actual'
            ORDER BY date ASC
            LIMIT 100
        """, as_dict=True)

        if not indicators:
            return {"error": "No validation data found in last 30 days"}

        predictor = SalesForecastPredictor()

        actuals_qty = []
        actuals_value = []
        predictions_qty = []
        predictions_value = []

        for ind in indicators:
            try:
                # Parse features
                features = json.loads(ind['data']) if ind.get('data') else {}

                # Make prediction
                pred = predictor.predict(features)

                if pred:
                    actuals_qty.append(float(ind['target_qty']))
                    actuals_value.append(float(ind['target_value']))
                    predictions_qty.append(pred['qty'])
                    predictions_value.append(pred['amount'])
            except:
                continue

        if not actuals_qty:
            return {"error": "Could not generate predictions"}

        # Calculate metrics
        qty_mae = mean_absolute_error(actuals_qty, predictions_qty)
        qty_rmse = np.sqrt(mean_squared_error(actuals_qty, predictions_qty))
        qty_mape = np.mean(np.abs((np.array(actuals_qty) - np.array(predictions_qty)) / (np.array(actuals_qty) + 1))) * 100

        value_mae = mean_absolute_error(actuals_value, predictions_value)
        value_rmse = np.sqrt(mean_squared_error(actuals_value, predictions_value))
        value_mape = np.mean(np.abs((np.array(actuals_value) - np.array(predictions_value)) / (np.array(actuals_value) + 1))) * 100

        return {
            "samples_evaluated": len(actuals_qty),
            "date_range": f"{start_date} to {end_date}",
            "quantity_metrics": {
                "mae": round(qty_mae, 2),
                "rmse": round(qty_rmse, 2),
                "mape": round(qty_mape, 2)
            },
            "value_metrics": {
                "mae": round(value_mae, 2),
                "rmse": round(value_rmse, 2),
                "mape": round(value_mape, 2)
            }
        }

    except Exception as e:
        frappe.log_error(f"Error evaluating model: {str(e)}", "Sales Forecast - Evaluate")
        return {"error": str(e)}

@frappe.whitelist()
def get_feature_importance():
    """Get feature importance from trained models"""
    try:
        # Load models directly
        model_qty = joblib.load(os.path.join(MODEL_PATH, "model_qty.pkl"))
        model_value = joblib.load(os.path.join(MODEL_PATH, "model_value.pkl"))

        # Load feature names
        config = joblib.load(os.path.join(MODEL_PATH, "feature_config.pkl"))
        feature_names = config.get('feature_names', [])

        if not feature_names:
            return {"error": "Feature names not found"}

        # Get importance scores
        qty_importance = model_qty.feature_importances_
        value_importance = model_value.feature_importances_

        # Create list
        importance_list = []
        for i, feature in enumerate(feature_names):
            importance_list.append({
                "feature": feature,
                "qty_importance": round(float(qty_importance[i]), 4),
                "value_importance": round(float(value_importance[i]), 4),
                "avg_importance": round(float((qty_importance[i] + value_importance[i]) / 2), 4)
            })

        # Sort by average importance
        importance_list.sort(key=lambda x: x['avg_importance'], reverse=True)

        return {
            "top_features": importance_list[:20],
            "all_features": importance_list
        }

    except Exception as e:
        frappe.log_error(f"Error getting feature importance: {str(e)}", "Sales Forecast - Feature Importance")
        return {"error": str(e)}

@frappe.whitelist()
def test_prediction():
    """Test prediction with a sample from the database"""
    try:
        # Get a recent sample
        sample = frappe.db.sql("""
            SELECT * FROM `tabSales Forecast Indicator`
            WHERE source = 'actual'
            ORDER BY date DESC
            LIMIT 1
        """, as_dict=True)

        if not sample:
            return {"error": "No sample data found"}

        # Parse features
        features = json.loads(sample[0]['data']) if sample[0].get('data') else {}

        # Make prediction
        prediction = predict_sales(features)

        if not prediction:
            return {"error": "Prediction failed"}

        # Calculate error
        actual_qty = float(sample[0]['target_qty'])
        pred_qty = prediction['qty']
        error_pct = abs(actual_qty - pred_qty) / actual_qty * 100 if actual_qty > 0 else 0

        return {
            "date": str(sample[0].get('date')),
            "item_code": sample[0].get('item_code'),
            "actual_qty": actual_qty,
            "actual_value": float(sample[0]['target_value']),
            "predicted_qty": pred_qty,
            "predicted_value": prediction['amount'],
            "error_percentage": round(error_pct, 2),
            "sample_features": {k: v for k, v in list(features.items())[:5]}  # Show first 5
        }

    except Exception as e:
        frappe.log_error(f"Test prediction error: {str(e)}", "Sales Forecast - Test")
        return {"error": str(e)}
