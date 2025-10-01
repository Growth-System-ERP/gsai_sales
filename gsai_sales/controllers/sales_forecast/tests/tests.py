import frappe
import json
from gsai_sales.controllers.sales_forecast.predict import SalesForecastPredictor

def test_single_prediction():
    """
    Comprehensive test for single prediction
    """
    # Initialize predictor
    predictor = SalesForecastPredictor()
    
    # Test cases with different feature subsets
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
    
    # Run tests
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

def compare_predictions():
    """
    Compare predictions across different feature sets
    """
    # Run single prediction tests
    test_results = test_single_prediction()
    
    # Analyze differences
    print("\n--- Prediction Comparison ---")
    qty_values = [result['prediction']['qty'] for result in test_results]
    amount_values = [result['prediction']['amount'] for result in test_results]
    
    print("Quantity Predictions:", qty_values)
    print("Amount Predictions:", amount_values)
    
    # Check for significant variations
    qty_variation = max(qty_values) - min(qty_values)
    amount_variation = max(amount_values) - min(amount_values)
    
    print(f"\nQuantity Variation: {qty_variation}")
    print(f"Amount Variation: {amount_variation}")
    
    return test_results

# Run the tests
@frappe.whitelist()
def run_forecast_prediction_tests():
    """
    API endpoint to run prediction tests
    """
    try:
        results = compare_predictions()
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        frappe.log_error(f"Prediction test error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }