import frappe
import json
from frappe import _
from gsai_sales.controllers.sales_forecast.predict import SalesForecastPredictor
from gsai_sales.controllers.sales_forecast.data import get_forecast_fields

def execute(filters=None):
    """
    Generate report data for Frappe report
    """

    if not filters:
        frappe.throw(_("Please select filters"))

    columns = get_columns()
    data = generate_forecast_report(filters)

    return columns, data

def get_columns():
    cols = [{**d, "fieldtype": "Data"} for d in get_forecast_fields(as_list=False, extended=True)]

    cols.extend([
        {
            "fieldname": "predicted_qty",
            "label": "Predicted Qty",
            "fieldtype": "Float",
        },
        {
            "fieldname": "predicted_amount",
            "label": "Predicted Amount",
            "fieldtype": "Currency",
        },
    ])

    return cols

def generate_forecast_report(filters=None, top_n=50):
    """
    Generate a comprehensive forecast report with intelligent feature exploration
    
    :param filters: Dictionary of fixed feature values
    :param top_n: Number of top predictions to return
    """
    try:
        # Prepare predictor
        predictor = SalesForecastPredictor()
        
        # Get available forecast fields
        available_fields = get_forecast_fields(extended=True)
        
        # Prepare base features with given filters
        base_features = filters or {}
        
        # Fetch unique values for unspecified fields
        def get_unique_values_for_field(field):
            try:
                # Try to get unique values from Sales Forecast Indicators
                unique_values = frappe.get_all(
                    "Sales Forecast Indicator", 
                    fields=['data'],
                    distinct=True
                )
                
                # Extract values for the specific field
                values = set()
                for ind in unique_values:
                    try:
                        data = json.loads(ind['data'])
                        if field in data:
                            values.add(data[field])
                    except:
                        pass
                
                return list(values)
            except:
                return []
        
        # Identify missing features
        missing_features = [
            field for field in available_fields 
            if field not in base_features
        ]
        
        # Generate prediction scenarios
        forecast_results = []
        
        # First, try to fill in most important features
        priority_order = [
            '`tabSales Order Item`.item_group',
            '`tabAddress`.state',
            '`tabSales Order`.customer_group',
            'month',
            'year'
        ]
        
        # Combine base features with explored features
        for priority_field in priority_order:
            if priority_field in missing_features:
                unique_values = get_unique_values_for_field(priority_field)
                
                for value in unique_values:
                    # Create a copy of base features
                    scenario_features = base_features.copy()
                    scenario_features[priority_field] = value
                    
                    # Try to fill remaining fields
                    for field in missing_features:
                        if field not in scenario_features:
                            field_values = get_unique_values_for_field(field)
                            if field_values:
                                # Take the first value
                                scenario_features[field] = field_values[0]
                    
                    # Make prediction
                    try:
                        prediction = predictor.predict(scenario_features)
                        
                        result_item = {
                            **scenario_features,
                            'predicted_qty': prediction.get('qty', 0),
                            'predicted_amount': prediction.get('amount', 0)
                        }
                        
                        forecast_results.append(result_item)
                    
                    except Exception as e:
                        frappe.log_error(f"Prediction error for {scenario_features}: {str(e)}")
        
        # Sort results by predicted amount
        forecast_results.sort(
            key=lambda x: abs(x.get('predicted_qty', 0)), 
            reverse=True
        )
        
        # Return top N results
        return forecast_results[:top_n]
    
    except Exception as e:
        frappe.log_error(f"Forecast report generation error: {str(e)}")
        frappe.throw(f"Error generating forecast report: {str(e)}")