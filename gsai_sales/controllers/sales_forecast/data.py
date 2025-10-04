# gsai_sales/controllers/sales_forecast/data.py
import frappe
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
import re

INDICATOR_DOCTYPE = "Sales Forecast Indicator"

def validate_sql_field(field):
    """Validate field to prevent SQL injection"""
    pattern = r'^`tab[A-Za-z0-9 ]+`\.[a-z_]+$'
    if not re.match(pattern, field):
        frappe.throw(f"Invalid field format: {field}")
    return field

def get_forecast_fields(filters={}, as_list=True, extended=False):
    """Get forecast fields with validation"""
    try:
        # Get user-configured fields
        fields = frappe.get_all(
            "GSAI-S Forecast Field",
            filters={**filters, "is_feature": 1},
            fields=["fieldname", "fieldtype"]
        )

        # Add time features if extended
        if extended:
            time_fields = [
                {"fieldname": "year", "fieldtype": "Int"},
                {"fieldname": "month", "fieldtype": "Int"},
                {"fieldname": "day", "fieldtype": "Int"},
                {"fieldname": "dayofweek", "fieldtype": "Int"},
                {"fieldname": "quarter", "fieldtype": "Int"},
                {"fieldname": "week", "fieldtype": "Int"},
                {"fieldname": "month_sin", "fieldtype": "Float"},
                {"fieldname": "month_cos", "fieldtype": "Float"},
                {"fieldname": "dayofweek_sin", "fieldtype": "Float"},
                {"fieldname": "dayofweek_cos", "fieldtype": "Float"},
            ]
            fields.extend(time_fields)

        if as_list:
            return [d.get("fieldname") for d in fields]

        return fields
    except Exception as e:
        frappe.log_error(f"Error getting forecast fields: {str(e)}")
        return [] if as_list else []

def fetch_sales_data_raw(since_date=None, till_date=None):
    """
    Fetch raw sales data WITHOUT aggregation
    Keep transaction-level granularity for proper time series
    """
    try:
        # Get user-selected fields
        select_fields = [f for f in get_forecast_fields()
                        if not f.startswith("year")
                        and not f.startswith("month")
                        and not f.startswith("day")
                        and not f.startswith("week")
                        and not f.startswith("quarter")]

        # Build SELECT clause - handle empty fields
        print(select_fields)
        if select_fields:
            # Validate all fields
            validated_fields = []
            for field in select_fields:
                try:
                    validated_field = validate_sql_field(field)
                    validated_fields.append(validated_field)
                except:
                    frappe.log_error(f"Skipping invalid field: {field}")
                    continue

            print(validated_fields)
            if validated_fields:
                select_field_sql = ", " + ", ".join([f"{field} as '{field}'" for field in validated_fields])
            else:
                select_field_sql = ""
        else:
            select_field_sql = ""

        # Get settings
        try:
            ignore_older_than = frappe.db.get_single_value(
                "GSAI-S Forecast Settings", "ignore_older_than"
            ) or 730  # Default 2 years
        except:
            ignore_older_than = 730

        # Build WHERE conditions
        conditions = ["`tabSales Order`.docstatus = 1"]

        if since_date:
            conditions.append(f"`tabSales Order`.transaction_date >= '{since_date}'")
        if till_date:
            conditions.append(f"`tabSales Order`.transaction_date <= '{till_date}'")
        if ignore_older_than:
            conditions.append(
                f"DATEDIFF(NOW(), `tabSales Order`.transaction_date) <= {ignore_older_than}"
            )

        where_clause = " AND ".join(conditions)

        # Execute query - NO AGGREGATION, keep all rows
        query = f"""
            SELECT
                `tabSales Order`.name AS sales_order,
                `tabSales Order`.transaction_date as date,
                `tabSales Order Item`.item_code,
                `tabSales Order Item`.qty as target_qty,
                `tabSales Order Item`.base_amount as target_value,
                `tabItem`.item_group
                {select_field_sql}
            FROM `tabSales Order`
            LEFT JOIN `tabSales Order Item`
                ON `tabSales Order Item`.parent = `tabSales Order`.name
            LEFT JOIN `tabItem`
                ON `tabSales Order Item`.item_code = `tabItem`.name
            LEFT JOIN `tabAddress`
                ON `tabAddress`.name = `tabSales Order`.customer_address
            LEFT JOIN `tabCustomer`
                ON `tabCustomer`.name = `tabSales Order`.customer
            WHERE {where_clause}
            ORDER BY `tabSales Order`.transaction_date ASC
        """

        # Execute and convert to list of dicts
        result = frappe.db.sql(query, as_dict=True)

        return result

    except Exception as e:
        frappe.log_error(f"Error fetching sales data: {str(e)}", "Sales Forecast Data Extraction")
        frappe.throw(f"Error fetching sales data: {str(e)}")
        return []

def engineer_features(df):
    """
    Create engineered features from raw data
    """
    try:
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['week'] = df['date'].dt.isocalendar().week
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

        # Cyclical encoding for periodic features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        return df
    except Exception as e:
        frappe.log_error(f"Error engineering features: {str(e)}")
        return df

def create_lag_features(df, group_cols, target_cols, lags=[1, 7, 30]):
    """
    Create lag features grouped by relevant dimensions
    """
    try:
        df = df.sort_values('date')

        # Filter group_cols to only those that exist
        existing_group_cols = [col for col in group_cols if col in df.columns]

        if not existing_group_cols:
            # No grouping, use overall lags
            for target in target_cols:
                if target in df.columns:
                    for lag in lags:
                        df[f'{target}_lag_{lag}'] = df[target].shift(lag)

                    # Rolling statistics
                    df[f'{target}_rolling_mean_7d'] = df[target].rolling(7, min_periods=1).mean()
                    df[f'{target}_rolling_std_7d'] = df[target].rolling(7, min_periods=1).std()
        else:
            # Group-wise lags
            for target in target_cols:
                if target in df.columns:
                    for lag in lags:
                        df[f'{target}_lag_{lag}'] = df.groupby(existing_group_cols)[target].shift(lag)

                    # Rolling statistics within groups
                    df[f'{target}_rolling_mean_7d'] = df.groupby(existing_group_cols)[target].transform(
                        lambda x: x.rolling(7, min_periods=1).mean()
                    )
                    df[f'{target}_rolling_std_7d'] = df.groupby(existing_group_cols)[target].transform(
                        lambda x: x.rolling(7, min_periods=1).std()
                    )

        return df
    except Exception as e:
        frappe.log_error(f"Error creating lag features: {str(e)}")
        return df

@frappe.whitelist()
def update_sales_forecast_indicators(from_date=None, to_date=None):
    """
    Update indicators with RAW transaction data
    """
    try:
        frappe.msgprint("Starting data extraction...")

        # Fetch raw data
        sales_data = fetch_sales_data_raw(from_date, to_date)

        if not sales_data:
            frappe.msgprint("No sales data found")
            return 0

        frappe.msgprint(f"Extracted {len(sales_data)} raw records. Processing...")

        # Convert to DataFrame - IMPORTANT: Handle the frappe dict properly
        df = pd.DataFrame.from_records(sales_data)

        # Engineer features
        df = engineer_features(df)

        # Get grouping fields (if configured)
        try:
            group_by_fields = get_forecast_fields(filters={"is_group_by_field": 1})
            # Filter to only existing columns
            group_by_fields = [f for f in group_by_fields if f in df.columns]
        except:
            group_by_fields = []

        if group_by_fields:
            # Create lag features within groups
            df = create_lag_features(
                df,
                group_cols=group_by_fields,
                target_cols=['target_qty', 'target_value'],
                lags=[1, 7, 30]
            )
        else:
            # Create lag features without grouping
            df = create_lag_features(
                df,
                group_cols=[],
                target_cols=['target_qty', 'target_value'],
                lags=[1, 7, 30]
            )

        frappe.msgprint("Features engineered. Saving indicators...")

        # Fill NaN values in lag/rolling features
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        df[lag_cols] = df[lag_cols].fillna(0)

        # Save to indicators - KEEP GRANULARITY
        count = 0
        batch_size = 500
        indicators_to_insert = []

        for idx, row in df.iterrows():
            # Create unique key for this specific transaction
            # Include date to maintain time granularity
            key_data = {
                'date': row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else '',
                'item_code': str(row.get('item_code', '')),
                'sales_order': str(row.get('sales_order', ''))
            }

            # Add group_by fields to key
            if group_by_fields:
                for field in group_by_fields:
                    if field in row:
                        key_data[field] = str(row.get(field, ''))

            hash_key = hashlib.md5(
                json.dumps(key_data, sort_keys=True).encode()
            ).hexdigest()

            # Check if exists
            if not frappe.db.exists(INDICATOR_DOCTYPE, {"hash_key": hash_key}):
                # Convert row to dict, handling NaN values
                row_dict = row.to_dict()
                for key, value in row_dict.items():
                    if pd.isna(value):
                        row_dict[key] = None

                # Create indicator doc
                doc_dict = {
                    "doctype": INDICATOR_DOCTYPE,
                    "hash_key": hash_key,
                    "date": row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else None,
                    "item_code": row.get('item_code'),
                    "item_group": row.get('item_group'),
                    "target_qty": float(row.get('target_qty', 0)),
                    "target_value": float(row.get('target_value', 0)),
                    "data": json.dumps(row_dict, default=str),
                    "source": "actual",
                    "sales_order": row.get('sales_order')
                }

                indicators_to_insert.append(doc_dict)
                count += 1

                # Batch insert
                if len(indicators_to_insert) >= batch_size:
                    for doc_dict in indicators_to_insert:
                        doc = frappe.get_doc(doc_dict)
                        doc.insert(ignore_permissions=True)
                    frappe.db.commit()
                    indicators_to_insert = []
                    frappe.msgprint(f"Saved {count} indicators so far...")

        # Insert remaining
        if indicators_to_insert:
            for doc_dict in indicators_to_insert:
                doc = frappe.get_doc(doc_dict)
                doc.insert(ignore_permissions=True)
            frappe.db.commit()

        frappe.msgprint(f"✅ Successfully created {count} new indicators!")
        return count

    except Exception as e:
        frappe.log_error(f"Error in update_sales_forecast_indicators: {str(e)}", "Sales Forecast Update")
        frappe.throw(f"Error updating indicators: {str(e)}")
        return 0


@frappe.whitelist()
def test_data_extraction(limit=10):
    """
    Test function to check data extraction
    """
    try:
        from datetime import datetime, timedelta

        # Test last 30 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        data = fetch_sales_data_raw(start_date, end_date)

        if data:
            sample = data[:limit]
            return {
                "success": True,
                "total_rows": len(data),
                "sample": sample,
                "columns": list(sample[0].keys()) if sample else []
            }
        else:
            return {
                "success": False,
                "message": "No data found"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
