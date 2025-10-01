import frappe
import json
from .predict import SalesForecastPredictor

def inspect_model_features(model, feature_names):
    """
    Inspect feature importances in the XGBoost model
    """
    try:
        # For MultiOutputRegressor, we'll check each estimator
        importances = []
        for i, estimator in enumerate(model.estimators_):
            # XGBoost feature importance
            importance = estimator.feature_importances_
            
            # Create a dictionary of feature importances
            feature_imp = dict(zip(feature_names, importance))
            
            # Sort features by importance
            sorted_features = sorted(
                feature_imp.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            print(f"\nFeature Importance for Target {i}:")
            for feature, imp in sorted_features[:10]:  # Top 10 features
                print(f"{feature}: {imp}")
        
        return sorted_features
    
    except Exception as e:
        frappe.log_error(f"Feature importance error: {str(e)}")
        return None

def analyze_model_details():
    """
    Comprehensive model analysis
    """
    try:
        # Initialize predictor to load the model
        predictor = SalesForecastPredictor()
        
        # Get feature names
        feature_names = predictor.scaler.feature_names_in_
        
        print("\n--- Model Inspection ---")
        
        # Print model type
        print("Model Type:", type(predictor.model))
        
        # Inspect feature importances
        print("\nFeature Importances:")
        inspect_model_features(predictor.model, feature_names)
        
        # Check scaler details
        print("\nScaler Details:")
        print("Feature Names:", feature_names)
        print("Mean values:", predictor.scaler.mean_)
        print("Scale values:", predictor.scaler.scale_)
        
        # Analyze training data
        import pandas as pd
        df = predictor._fetch_training_data()
        
        print("\nTraining Data Summary:")
        print("Total records:", len(df))
        print("\nColumn Statistics:")
        print(df.describe())
        
        # Check for data distribution
        print("\nColumn Value Counts:")
        for col in df.columns[:10]:  # First 10 columns
            print(f"\n{col} unique values:")
            print(df[col].value_counts().head())
    
    except Exception as e:
        frappe.log_error(f"Model analysis error: {str(e)}")
        print(f"Error: {str(e)}")

# Run the analysis
@frappe.whitelist()
def run_model_inspection():
    """
    API endpoint to run model inspection
    """
    analyze_model_details()
    return "Model inspection completed. Check logs for details."