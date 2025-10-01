import frappe
from .data import INDICATOR_DOCTYPE
from frappe.utils import now_datetime, add_to_date, get_datetime, today, nowdate

def process_new_sales_orders():
    """Process new sales orders since last run"""
    last_processed = frappe.db.get_single_value("GSAI-S Forecast Settings", "last_processed_date")
    current_time = now_datetime()
    
    # If no last processed time, default to 7 days ago
    if not last_processed:
        last_processed = add_to_date(current_time, days=-7)
    else:
        last_processed = get_datetime(last_processed)
    
    # Format dates for SQL
    from_date = last_processed.strftime('%Y-%m-%d %H:%M:%S')
    to_date = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Process orders in this time range
    from gsai_sales.controllers.sales_forecast.data import update_sales_forecast_indicators
    count = update_sales_forecast_indicators(from_date=from_date, to_date=to_date)
    
    # Update the last processed time
    frappe.db.set_value("GSAI-S Forecast Settings", "GSAI-S Forecast Settings", 
                        "last_processed_date", current_time)
    
    return count

def update_forecast_model():
    """Daily update of forecast model with new indicators"""
    # Get indicators from the last day
    yesterday = add_to_date(now_datetime(), days=-1)
    from_date = yesterday.strftime('%Y-%m-%d')
    
    # Perform incremental training
    from gsai_sales.controllers.sales_forecast.train import train_forecast_model
    result = train_forecast_model(from_date=from_date, incremental=True)
    
    return result

def full_retrain_model():
    """Retrain the model from scratch using all indicators"""
    from gsai_sales.controllers.sales_forecast.train import train_forecast_model
    train_forecast_model(incremental=False)

    frappe.db.set_value("GSAI-S Forecast Settings", "GSAI-S Forecast Settings", "last_full_training_date", today())
    return 

def remove_old_indicators():
    """Remove indicators older than the specified days"""
    cutoff_days = frappe.db.get_single_value("GSAI-S Forecast Settings", "ignore_older_than") or 1095

    cutoff_date = add_to_date(nowdate(), days=-cutoff_days)

    count = frappe.db.count(INDICATOR_DOCTYPE, {"creation": ["<", cutoff_date]})
    
    if count > 0:
        frappe.db.sql(f"""
            DELETE FROM `tab{INDICATOR_DOCTYPE}`
            WHERE creation < %s
        """, cutoff_date)
        
        frappe.db.commit()

    return count    