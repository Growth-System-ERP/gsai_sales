import frappe
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from .train import MODEL_PATH, SalesForecastTrainer

class SalesForecastPredictor(SalesForecastTrainer):
    def __init__(self):
        super(SalesForecastPredictor, self).__init__()

    def predict(self, features):
        """Make predictions for given features"""
        if self.model is None or not self.has_trained_model:
            frappe.throw("Model not loaded")
            return None
                
        try:
            # Get expected feature names from scaler
            expected_features = self.scaler.feature_names_in_
            
            # Create DataFrame with expected column structure
            feature_df = pd.DataFrame(columns=expected_features)
            feature_df.loc[0] = 0  # Add a row with zeros
            
            # Fill in the values we have
            for key, value in features.items():
                # Try to find matching column
                matching_cols = [col for col in expected_features if key in col]
                
                for col in matching_cols:
                    # Determine feature type
                    feature_df.loc[0, col] = self.feature_encode(col, value)
            
            # Handle missing columns by filling with 0
            feature_df = feature_df.fillna(0)
            
            # Ensure all expected features are present
            for col in expected_features:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            
            # Reorder columns to match scaler's expected order
            feature_df = feature_df[expected_features]
            
            # Handle categorical features
            for col in feature_df.select_dtypes(include=['object']).columns:
                feature_df[col] = pd.Categorical(feature_df[col]).codes

            # Scale features
            features_scaled = self.scaler.transform(feature_df)
            
            # Make prediction
            predictions = self.model.predict(features_scaled)
            
            # Return as dictionary
            return {
                "qty": float(predictions[0][0]),
                "amount": float(predictions[0][1]),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            frappe.log_error(f"Prediction error: {str(e)}")
            frappe.throw(f"Error making prediction: {str(e)}")
            return None

    def feature_encode(self, col, value):
        feature_type = self.feature_type.get(col, 'categorical')

        if feature_type == 'categorical':
            # Use consistent categorical encoding
            return self._encode_categorical(value)
        
        elif feature_type == 'numeric':
            # Ensure numeric type
            return pd.to_numeric(value, errors='coerce')
    
        elif feature_type == 'binary':
            # Convert to 0 or 1
            return int(bool(value))
        
        elif feature_type == 'periodic':
            # Normalize periodic features
            return float(value)
        
        elif feature_type == 'datetime':
            # Convert to timestamp
            return pd.to_datetime(value).timestamp() if value else 0
        
        else:
            # Default to original value
            return value

    def _encode_categorical(self, value):
        """
        Encode categorical values consistently
        
        This method ensures:
        1. Consistent encoding across predictions
        2. Numeric representation of categorical features
        """
        if pd.isna(value):
            return 0
        
        # Use a consistent hashing method
        return hash(str(value)) % 1000
