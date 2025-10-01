// Copyright (c) 2025, GWS and contributors
// For license information, please see license.txt

frappe.ui.form.on("GSAI-S Forecast Settings", {
	refresh(frm) {
		frm.trigger("load_forecast_fields");
	},

	validate(frm) {
		frm.doc.fields?.forEach(row => {

			if (!row.fieldname) return;

			row.label = frm.forecast_fields_dict[row.fieldname]?.label;
			row.fieldtype = frm.forecast_fields_dict[row.fieldname]?.fieldtype;
		})
	},

	load_forecast_fields(frm) {
		const process_fields = (dt) => {

			frappe.get_meta(dt).fields.forEach(df => {
				if (frappe.model.no_value_type.includes(df.fieldtype) || frappe.model.table_fields.includes(df.fieldtype)) {
					return
				}

				const label = `${df.label||df.fieldnme} (${dt})`;
				const fieldname = `\`tab${dt}\`.${df.fieldname}`;

				const field = {
					"label": label,
					"value": fieldname,
					"fieldtype": df.fieldtype,
				}

				valid_fields.push(field);

				frm.forecast_fields_dict[fieldname] = field;
			});
		}


		const doctypes = ["Sales Order", "Sales Order Item", "Item", "Customer", "Address"];

		let promises = [];
		let valid_fields = [];
		frm.forecast_fields_dict = {};

		doctypes.forEach((dt) => {
			const p = frappe.model.with_doctype(dt).then(()=>process_fields(dt));
			promises.push(p);
		});

		Promise.all(promises).then(() => {
			frm.fields_dict.fields.grid.update_docfield_property("fieldname", "options", valid_fields);
		})
	}
});


frappe.ui.form.on("GSAI-S Forecast Field", {
	fieldname(frm, cdt, cdn) {
		let row = locals[cdt][cdn];

		row.label = frm.forecast_fields_dict[row.fieldname]?.label||"";
		row.fieldtype = frm.forecast_fields_dict[row.fieldname]?.fieldtype||"";
	}
})