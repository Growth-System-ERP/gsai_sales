// Copyright (c) 2025, GWS and contributors
// For license information, please see license.txt

frappe.query_reports["Sales Forecast"] = {
	"filters": [
		{
			fieldname: "`tabSales Order Item`.item_group",
			label: "Item Group",
			fieldtype: "Link",
			options: "Item Group",
			reqd: 1,
		},
		{
			fieldname: "`tabAddress`.state",
			label: "State",
			fieldtype: "Select",
			options: "\nAndaman and Nicobar Islands\nAndhra Pradesh\nArunachal Pradesh\nAssam\nBihar\nChandigarh\nChhattisgarh\nDadra and Nagar Haveli and Daman and Diu\nDelhi\nGoa\nGujarat\nHaryana\nHimachal Pradesh\nJammu and Kashmir\nJharkhand\nKarnataka\nKerala\nLadakh\nLakshadweep Islands\nMadhya Pradesh\nMaharashtra\nManipur\nMeghalaya\nMizoram\nNagaland\nOdisha\nOther Territory\nPondicherry\nPunjab\nRajasthan\nSikkim\nTamil Nadu\nTelangana\nTripura\nUttar Pradesh\nUttarakhand\nWest Bengal",
			reqd: 1,
		}
	]
};
