import frappe
import json
from frappe.utils import getdate, add_days
import hashlib

INDICATOR_DOCTYPE = "Sales Forecast Indicator"

def fetch_sales_data(since_date, till_date):
	select_fields = get_forecast_fields()
	select_field_sql = ", ".join([f"{d} as '{d}'" for d in select_fields])
	select_field_hash = get_fieldset_hash(select_fields)

	ignore_older_than = frappe.db.get_single_value("GSAI-S Forecast Settings", "ignore_older_than")

	conditions = ""

	if since_date:
		conditions += f" AND `tabSales Order`.transaction_date >= '{since_date}'"

	if till_date:
		conditions += f" AND `tabSales Order`.transaction_date <= '{till_date}'"

	if ignore_older_than:
		conditions += f" AND DATEDIFF(NOW(), `tabSales Order`.transaction_date) <= {ignore_older_than}"

	return frappe.db.sql(f"""
		SELECT 
			`tabSales Order`.name AS sales_order,
			DATE_FORMAT(`tabSales Order`.transaction_date, '%Y-%m-%d') as date,
			YEAR(`tabSales Order`.transaction_date) as year,
			MONTH(`tabSales Order`.transaction_date) as month,
			DAYOFMONTH(`tabSales Order`.transaction_date) as day,
			WEEKDAY(`tabSales Order`.transaction_date) as weekday,
			QUARTER(`tabSales Order`.transaction_date) as quarter,

			`tabSales Order Item`.qty as target_qty, 
			`tabSales Order Item`.base_amount as target_value,
			`tabSales Order Item`.item_code,

			"{select_field_hash}" as select_field_hash, 
			"actual" as source, 

			`tabItem`.item_group, 

			{select_field_sql}
		FROM `tabSales Order`
		LEFT JOIN `tabSales Order Item` ON `tabSales Order Item`.parent = `tabSales Order`.name
		LEFT JOIN `tabItem` ON `tabSales Order Item`.item_code = `tabItem`.name
		LEFT JOIN `tabAddress` ON `tabAddress`.name = `tabSales Order`.customer_address
		WHERE `tabSales Order`.docstatus = 1 
		{conditions}
		ORDER BY `tabSales Order`.transaction_date ASC
	""", as_dict=True)

def get_forecast_fields(filters={}, as_list=True, extended=False):
	static_fields = [{"fieldname": "year", "fieldtype": "Int"}, {"fieldname": "month", "fieldtype": "Int"}, {"fieldname": "day", "fieldtype": "Int"}, {"fieldname": "weekday", "fieldtype": "Periodic"}, {"fieldname": "quarter", "fieldtype": "Periodic"}]

	fields = frappe.get_all("GSAI-S Forecast Field", filters=filters, fields=["*"])

	if extended:
		fields.extend(static_fields)

	if as_list:
		return [d.get("fieldname") for d in fields]

	return fields

# We can even use this later for grouping, frequency counting, or model feedback loops.
def create_hash_key(row, selected_fields):
	key_string = "|".join(str(row.get(f) or "").lower().strip() for f in selected_fields)

	return hashlib.md5(key_string.encode()).hexdigest()

def get_fieldset_hash(fields):
	field_string = "|".join(sorted(fields))
	return hashlib.md5(field_string.encode()).hexdigest()

@frappe.whitelist()
def update_sales_forecast_indicators(from_date=None, to_date=None):
	sales_data = fetch_sales_data(from_date, to_date)
	selected_fields = get_forecast_fields(extended=True)

	grouped_data = {}

	for row in sales_data:
		key = create_hash_key(row, selected_fields)

		if key not in grouped_data:
			grouped_data[key] = {
				"target_qty": 0,
				"target_value": 0,
				"hash_key": key,
				"data": json.dumps(row),
				**row
			}

		grouped_data[key]["target_qty"] += float(row["target_qty"] or 0)
		grouped_data[key]["target_value"] += float(row["target_value"] or 0)

	for key, data in grouped_data.items():
		if not frappe.db.exists(INDICATOR_DOCTYPE, {"hash_key": key}):
			doc = frappe.get_doc({
				"doctype": INDICATOR_DOCTYPE,
				**data
			})
			doc.save(ignore_permissions=True)
		else:
			frappe.db.set_value(
				INDICATOR_DOCTYPE,
				{"hash_key": key},
				{"target_qty": data["target_qty"], "target_value": data["target_value"]}
			)
