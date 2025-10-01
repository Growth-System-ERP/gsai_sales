# Create a new file: forecast_views.py

import frappe
import json
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from .predict import SalesForecastPredictor

class ForecastDashboard:
    def __init__(self):
        self.predictor = SalesForecastPredictor()
        
    def get_item_forecast_dashboard(self, filters=None, period_count=6):
        """
        Generate a comprehensive forecast dashboard for items
        
        Returns forecasts grouped by:
        - Top items by sales volume
        - Top items by revenue
        - Items by product category
        - Growth trends
        """
        if not filters:
            filters = {}
            
        # Get top items by historical data
        top_items = self._get_top_items(10, filters)
        
        # Generate forecasts for each segment
        forecasts = {
            "top_items_by_volume": self._forecast_for_items(
                top_items["by_volume"], 
                period_count
            ),
            "top_items_by_value": self._forecast_for_items(
                top_items["by_value"], 
                period_count
            ),
            "category_forecasts": self._forecast_by_category(
                filters, 
                period_count
            ),
            "growth_trends": self._calculate_growth_trends(
                top_items["by_volume"] + top_items["by_value"],
                period_count
            )
        }
        
        return forecasts
    
    def _get_top_items(self, limit=10, filters=None):
        """Get top selling items by volume and value"""
        filter_conditions = ""
        values = []
        
        if filters:
            for key, value in filters.items():
                if value:
                    filter_conditions += f" AND `{key}` = %s"
                    values.append(value)
        
        # Top by volume
        volume_query = f"""
            SELECT 
                item_code, 
                SUM(target_qty) as total_qty
            FROM `tabSales Forecast Indicator`
            WHERE 1=1 {filter_conditions}
            GROUP BY item_code
            ORDER BY total_qty DESC
            LIMIT {limit}
        """
        
        # Top by value
        value_query = f"""
            SELECT 
                item_code, 
                SUM(target_value) as total_value
            FROM `tabSales Forecast Indicator`
            WHERE 1=1 {filter_conditions}
            GROUP BY item_code
            ORDER BY total_value DESC
            LIMIT {limit}
        """
        
        top_by_volume = frappe.db.sql(volume_query, values, as_dict=True)
        top_by_value = frappe.db.sql(value_query, values, as_dict=True)
        
        return {
            "by_volume": top_by_volume,
            "by_value": top_by_value
        }
    
    def _forecast_for_items(self, items, period_count=6):
        """Generate forecasts for a list of items"""
        results = []
        
        for item in items:
            item_code = item["item_code"]
            
            # Get item details
            item_doc = frappe.get_doc("Item", item_code)
            
            # Feature set for this item
            features = {
                "item_code": item_code,
                "item_group": item_doc.item_group
            }
            
            # Get time series forecast
            forecast = self.predictor.forecast_timeseries(
                features, 
                periods=period_count, 
                period_type="month"
            )
            
            if forecast:
                # Add item details to the forecast
                for period in forecast:
                    period["item_name"] = item_doc.item_name
                    period["item_group"] = item_doc.item_group
                
                # Calculate trend metrics
                qty_values = [period["qty"] for period in forecast]
                amount_values = [period["amount"] for period in forecast]
                
                trend = {
                    "item_code": item_code,
                    "item_name": item_doc.item_name,
                    "forecast": forecast,
                    "total_forecast_qty": sum(qty_values),
                    "total_forecast_amount": sum(amount_values),
                    "avg_forecast_qty": sum(qty_values) / len(qty_values),
                    "avg_forecast_amount": sum(amount_values) / len(amount_values),
                    "trend_direction": "up" if qty_values[-1] > qty_values[0] else "down"
                }
                
                results.append(trend)
        
        return results
    
    def _forecast_by_category(self, filters=None, period_count=6):
        """Generate forecasts grouped by item category"""
        if not filters:
            filters = {}
            
        # Get top categories
        categories = frappe.db.sql("""
            SELECT 
                item_group,
                COUNT(*) as item_count,
                SUM(target_qty) as total_qty,
                SUM(target_value) as total_value
            FROM `tabSales Forecast Indicator`
            GROUP BY item_group
            ORDER BY total_value DESC
            LIMIT 10
        """, as_dict=True)
        
        results = []
        
        for category in categories:
            # Feature set for this category
            features = {
                "item_group": category["item_group"]
            }
            
            # Add any additional filters
            for key, value in filters.items():
                if value and key != "item_group":
                    features[key] = value
            
            # Get time series forecast
            forecast = self.predictor.forecast_timeseries(
                features, 
                periods=period_count, 
                period_type="month"
            )
            
            if forecast:
                category_result = {
                    "item_group": category["item_group"],
                    "item_count": category["item_count"],
                    "historical_qty": category["total_qty"],
                    "historical_value": category["total_value"],
                    "forecast": forecast,
                    "forecast_total_qty": sum(period["qty"] for period in forecast),
                    "forecast_total_value": sum(period["amount"] for period in forecast)
                }
                
                results.append(category_result)
        
        return results
    
    def _calculate_growth_trends(self, items, period_count=6):
        """Calculate growth trends for forecasted items"""
        growth_items = []
        decline_items = []
        
        for item in items:
            item_code = item["item_code"]
            
            # Get historical data for comparison
            historical = frappe.db.sql("""
                SELECT 
                    SUM(target_qty) as historical_qty,
                    SUM(target_value) as historical_value
                FROM `tabSales Forecast Indicator`
                WHERE item_code = %s
                AND creation > DATE_SUB(NOW(), INTERVAL %s MONTH)
            """, (item_code, period_count), as_dict=True)[0]
            
            # Feature set for this item
            features = {"item_code": item_code}
            
            # Get forecast
            forecast = self.predictor.forecast_timeseries(
                features, 
                periods=period_count, 
                period_type="month"
            )
            
            if forecast and historical:
                # Calculate forecast totals
                forecast_qty = sum(period["qty"] for period in forecast)
                forecast_value = sum(period["amount"] for period in forecast)
                
                # Calculate growth percentages
                if historical["historical_qty"] > 0:
                    qty_growth = (forecast_qty - historical["historical_qty"]) / historical["historical_qty"] * 100
                else:
                    qty_growth = 100  # New item
                    
                if historical["historical_value"] > 0:
                    value_growth = (forecast_value - historical["historical_value"]) / historical["historical_value"] * 100
                else:
                    value_growth = 100  # New item
                
                growth_data = {
                    "item_code": item_code,
                    "historical_qty": historical["historical_qty"],
                    "historical_value": historical["historical_value"],
                    "forecast_qty": forecast_qty,
                    "forecast_value": forecast_value,
                    "qty_growth": qty_growth,
                    "value_growth": value_growth,
                    "forecast": forecast
                }
                
                # Categorize by growth
                if qty_growth > 0 or value_growth > 0:
                    growth_items.append(growth_data)
                else:
                    decline_items.append(growth_data)
        
        # Sort by growth percentage
        growth_items = sorted(growth_items, key=lambda x: x["value_growth"], reverse=True)
        decline_items = sorted(decline_items, key=lambda x: x["value_growth"])
        
        return {
            "growth": growth_items[:5],  # Top 5 growth items
            "decline": decline_items[:5]  # Top 5 declining items
        }

    def get_opportunity_insights(self, filters=None):
        """Find potential sales opportunities based on patterns"""
        if not filters:
            filters = {}
            
        # Find items that are frequently purchased together
        bundle_opportunities = self._find_bundle_opportunities()
        
        # Find seasonal patterns
        seasonal_opportunities = self._find_seasonal_opportunities()
        
        # Find customer segment opportunities
        segment_opportunities = self._find_customer_segment_opportunities(filters)
        
        # Find underperforming territories
        territory_opportunities = self._find_territory_opportunities(filters)
        
        return {
            "bundle_opportunities": bundle_opportunities,
            "seasonal_opportunities": seasonal_opportunities,
            "segment_opportunities": segment_opportunities,
            "territory_opportunities": territory_opportunities
        }
    
    def _find_bundle_opportunities(self):
        """Find items that are frequently purchased together"""
        # This would analyze sales orders to find co-occurring items
        # Simplified example:
        bundle_query = """
            SELECT 
                i1.item_code as item1,
                i2.item_code as item2,
                COUNT(*) as co_occurrence
            FROM `tabSales Order Item` i1
            JOIN `tabSales Order Item` i2 ON i1.parent = i2.parent AND i1.item_code < i2.item_code
            GROUP BY i1.item_code, i2.item_code
            HAVING COUNT(*) > 5
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """
        
        bundles = frappe.db.sql(bundle_query, as_dict=True)
        
        results = []
        for bundle in bundles:
            # Get item details
            item1 = frappe.get_doc("Item", bundle["item1"])
            item2 = frappe.get_doc("Item", bundle["item2"])
            
            # Generate combined forecast
            features = {
                "item_code": [bundle["item1"], bundle["item2"]]
            }
            
            results.append({
                "items": [
                    {"item_code": bundle["item1"], "item_name": item1.item_name},
                    {"item_code": bundle["item2"], "item_name": item2.item_name}
                ],
                "co_occurrence": bundle["co_occurrence"],
                "opportunity_type": "bundle",
                "description": f"These items are frequently purchased together ({bundle['co_occurrence']} times)"
            })
        
        return results
    
    def _find_seasonal_opportunities(self):
        """Find seasonal patterns in sales data"""
        seasonal_query = """
            SELECT 
                item_code,
                MONTH(creation) as month,
                SUM(target_qty) as monthly_qty
            FROM `tabSales Forecast Indicator`
            WHERE creation > DATE_SUB(NOW(), INTERVAL 24 MONTH)
            GROUP BY item_code, MONTH(creation)
            ORDER BY item_code, month
        """
        
        seasonal_data = frappe.db.sql(seasonal_query, as_dict=True)
        
        # Group by item
        items_by_month = {}
        for row in seasonal_data:
            if row["item_code"] not in items_by_month:
                items_by_month[row["item_code"]] = {}
                
            items_by_month[row["item_code"]][row["month"]] = row["monthly_qty"]
        
        results = []
        for item_code, months in items_by_month.items():
            if len(months) >= 6:  # Enough data for seasonal analysis
                # Find peak months
                avg_qty = sum(months.values()) / len(months)
                peak_months = [month for month, qty in months.items() if qty > avg_qty * 1.5]
                
                if peak_months:
                    item = frappe.get_doc("Item", item_code)
                    peak_month_names = [datetime(2000, m, 1).strftime("%B") for m in peak_months]
                    
                    results.append({
                        "item_code": item_code,
                        "item_name": item.item_name,
                        "peak_months": peak_month_names,
                        "opportunity_type": "seasonal",
                        "description": f"Sales peak during {', '.join(peak_month_names)}"
                    })
        
        return results[:10]  # Top 10 seasonal opportunities
    
    def _find_customer_segment_opportunities(self, filters):
        """Find customer segments with growth potential"""
        segment_query = """
            SELECT 
                customer_group,
                SUM(target_value) as total_value,
                COUNT(DISTINCT customer) as customer_count
            FROM `tabSales Forecast Indicator`
            WHERE customer_group IS NOT NULL
            GROUP BY customer_group
            ORDER BY total_value DESC
        """
        
        segments = frappe.db.sql(segment_query, as_dict=True)
        
        results = []
        for segment in segments[:5]:  # Top 5 segments
            # Generate forecast for this segment
            features = {
                "customer_group": segment["customer_group"]
            }
            
            forecast = self.predictor.forecast_timeseries(
                features, 
                periods=6, 
                period_type="month"
            )
            
            if forecast:
                forecast_total = sum(period["amount"] for period in forecast)
                
                results.append({
                    "customer_group": segment["customer_group"],
                    "customer_count": segment["customer_count"],
                    "historical_value": segment["total_value"],
                    "forecast_value": forecast_total,
                    "opportunity_type": "customer_segment",
                    "description": f"Customer group with {segment['customer_count']} customers and forecast value of {forecast_total:.2f}"
                })
        
        return results
    
    def _find_territory_opportunities(self, filters):
        """Find territories with growth potential"""
        territory_query = """
            SELECT 
                territory,
                SUM(target_value) as total_value,
                COUNT(DISTINCT customer) as customer_count,
                COUNT(DISTINCT item_code) as item_count
            FROM `tabSales Forecast Indicator`
            WHERE territory IS NOT NULL
            GROUP BY territory
            ORDER BY total_value DESC
        """
        
        territories = frappe.db.sql(territory_query, as_dict=True)
        
        results = []
        for territory in territories:
            # Generate forecast for this territory
            features = {
                "territory": territory["territory"]
            }
            
            forecast = self.predictor.forecast_timeseries(
                features, 
                periods=6, 
                period_type="month"
            )
            
            if forecast:
                forecast_total = sum(period["amount"] for period in forecast)
                
                # Calculate growth potential
                growth = (forecast_total - territory["total_value"]) / territory["total_value"] * 100 if territory["total_value"] > 0 else 0
                
                results.append({
                    "territory": territory["territory"],
                    "customer_count": territory["customer_count"],
                    "item_count": territory["item_count"],
                    "historical_value": territory["total_value"],
                    "forecast_value": forecast_total,
                    "growth_potential": growth,
                    "opportunity_type": "territory",
                    "description": f"Territory with {territory['customer_count']} customers and {growth:.1f}% growth potential"
                })
        
        # Sort by growth potential
        results = sorted(results, key=lambda x: x["growth_potential"], reverse=True)
        
        return results[:10]  # Top 10 territory opportunities

# API endpoints
@frappe.whitelist()
def get_sales_forecast_dashboard(filters_json=None):
    """Get comprehensive sales forecast dashboard"""
    filters = json.loads(filters_json) if filters_json else {}
    
    dashboard = ForecastDashboard()
    results = dashboard.get_item_forecast_dashboard(filters)
    
    return results

@frappe.whitelist()
def get_sales_opportunities():
    """Get sales opportunity insights"""
    dashboard = ForecastDashboard()
    results = dashboard.get_opportunity_insights()
    
    return results