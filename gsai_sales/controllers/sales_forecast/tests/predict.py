import json 
import numpy as np
import pandas as pd

def generate_test_cases():
    """
    Generate a list of test cases for prediction testing
    """
    test_cases = [
        # Full feature set
        {
            "name": "Full Feature Set",
            "features": {
                "year": 2025, 
                "month": 2, 
                "day": 24, 
                "weekday": 0, 
                "quarter": 1,
                "`tabAddress`.country": "India", 
                "`tabAddress`.city": "SATARA", 
                "`tabAddress`.state": "Maharashtra", 
                "`tabSales Order Item`.item_group": "SMD Bulk", 
                "`tabSales Order Item`.is_free_item": 0, 
                "`tabSales Order Item`.uom": "Nos", 
                "`tabSales Order`.selling_price_list": "Standard Selling (Base Price)", 
                "`tabSales Order`.territory": "SATARA", 
                "`tabSales Order`.customer_group": "OUTSTATION"
            }
        },
        # Minimal feature set
        {
            "name": "Minimal Features",
            "features": {
                "year": 2025, 
                "month": 2, 
                "item_group": "SMD Bulk",
                "`tabAddress`.state": "Rajasthan",
            }
        },
        # Different year and month
        {
            "name": "Different Time Period",
            "features": {
                "year": 2024, 
                "month": 6, 
                "`tabSales Order Item`.item_group": "SMD Bulk",
                "`tabAddress`.state": "Rajasthan",
            }
        },
        # Different item group
        {
            "name": "Different Item Group",
            "features": {
                "year": 2025, 
                "month": 2, 
                "`tabSales Order Item`.item_group": "Electronics",
                "`tabAddress`.state": "Rajasthan",
            }
        }
    ]
    
    return test_cases

def run_prediction_tests(predictor):
    """
    Run predictions for each test case
    
    :param predictor: Prediction model
    :return: List of test results
    """
    test_cases = generate_test_cases()
    results = []
    
    for test_case in test_cases:
        try:
            # Make prediction
            prediction = predictor.predict(test_case['features'])
            
            # Prepare result
            result = {
                "test_name": test_case['name'],
                "features": test_case['features'],
                "prediction": prediction
            }
            
            results.append(result)
            
            # Print individual test results
            print(f"\nTest: {test_case['name']}")
            print("Features:", json.dumps(test_case['features'], indent=2))
            print("Prediction:", json.dumps(prediction, indent=2))
        
        except Exception as e:
            print(f"Error in test {test_case['name']}: {str(e)}")
    
    return results

def compare_predictions(results):
    """
    Compare predictions across test cases
    
    :param results: List of prediction results
    """
    print("\n--- Prediction Comparison ---")
    
    # Extract prediction values
    qty_values = [result['prediction']['qty'] for result in results]
    amount_values = [result['prediction']['amount'] for result in results]
    
    print("Quantity Predictions:", qty_values)
    print("Amount Predictions:", amount_values)
    
    # Calculate variations
    qty_variation = max(qty_values) - min(qty_values)
    amount_variation = max(amount_values) - min(amount_values)
    
    print(f"\nQuantity Variation: {qty_variation}")
    print(f"Amount Variation: {amount_variation}")
    
    return {
        "qty_predictions": qty_values,
        "amount_predictions": amount_values,
        "qty_variation": qty_variation,
        "amount_variation": amount_variation
    }

# Usage example
def execute():
    from gsai_sales.controllers.sales_forecast.predict import SalesForecastPredictor
    
    # Initialize predictor
    predictor = SalesForecastPredictor()
    
    # Run prediction tests
    test_results = run_prediction_tests(predictor)
    
    # Compare predictions
    comparison = compare_predictions(test_results)

def compare_predictions_to_synthetic_data(test_cases, predictor):
    """
    Compare predictions against the synthetic data generation logic
    """
    for test_case in test_cases:
        features = test_case['features']
        
        # Recreate synthetic data generation logic
        expected_qty = (
            10 * (features.get('month', 0) / 6) + 
            5 * (features.get('`tabSales Order Item`.item_group', '') == 'Electronics') + 
            3 * (features.get('`tabAddress`.state', '') == 'Maharashtra') + 
            np.random.normal(0, 2)
        )
        
        expected_value = (
            100 * (features.get('month', 0) / 6) + 
            50 * (features.get('`tabSales Order`.customer_group', '') == 'Wholesale') + 
            25 * (features.get('`tabSales Order Item`.item_group', '') == 'Electronics') + 
            np.random.normal(0, 10)
        )
        
        # Make prediction
        prediction = predictor.predict(features)
        
        print(f"\nTest: {test_case['name']}")
        print("Features:", json.dumps(features, indent=2))
        print("Predicted Qty:", prediction['qty'])
        print("Expected Qty (Synthetic Logic):", expected_qty)
        print("Predicted Amount:", prediction['amount'])
        print("Expected Amount (Synthetic Logic):", expected_value)
        
        # Calculate absolute error
        qty_error = abs(prediction['qty'] - expected_qty)
        amount_error = abs(prediction['amount'] - expected_value)
        
        print(f"Qty Absolute Error: {qty_error}")
        print(f"Amount Absolute Error: {amount_error}")

def create_synthetic_training_data(n_samples=1000):
    """
    Create a synthetic dataset for controlled testing
    """
    np.random.seed(42)
    
    # Create synthetic data with controlled variations
    data = {
        "`tabSales Order Item`.item_group": np.random.choice(['Electronics', 'Clothing', 'Groceries'], n_samples),
        "`tabAddress`.state": np.random.choice(['Maharashtra', 'Karnataka', 'Tamil Nadu'], n_samples),
        "`tabAddress`.city": np.random.choice(['City'], n_samples),
        "`tabAddress`.country": np.random.choice(['India'], n_samples),
        "`tabSales Order`.customer_group": np.random.choice(['Retail', 'Wholesale', 'Online'], n_samples),
        "`tabSales Order`.territory": np.random.choice(['Tr3'], n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'day': np.random.randint(1, 29, n_samples),
        'year': np.random.randint(2020, 2025, n_samples),
        'weekday': np.random.randint(0, 7, n_samples),
        'quarter': np.random.randint(1, 5, n_samples),
        '`tabSales Order Item`.uom': "Ns", 
        '`tabSales Order`.selling_price_list': "No", 
        '`tabSales Order Item`.is_free_item': 0
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variables with controlled complexity
    df['target_qty'] = (
        10 * (df['month'] / 6) + 
        5 * (df["`tabSales Order Item`.item_group"] == 'Electronics') + 
        3 * (df["`tabAddress`.state"] == 'Maharashtra') + 
        np.random.normal(0, 2, n_samples)
    )
    
    df['target_value'] = (
        100 * (df['month'] / 6) + 
        50 * (df["`tabSales Order`.customer_group"] == 'Wholesale') + 
        25 * (df["`tabSales Order Item`.item_group"] == 'Electronics') + 
        np.random.normal(0, 10, n_samples)
    )
    
    return df

def run_synthetic_data_comparison():
    # Initialize predictor
    from gsai_sales.controllers.sales_forecast.predict import SalesForecastPredictor
    predictor = SalesForecastPredictor()
    
    # Generate test cases
    test_cases = generate_test_cases()
    
    # Compare predictions
    compare_predictions_to_synthetic_data(test_cases, predictor)
