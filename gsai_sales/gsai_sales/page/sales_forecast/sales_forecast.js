// forecast_dashboard.js

frappe.pages['sales-forecast'].on_page_show = function (wrapper) {
    var page = frappe.ui.make_app_page({
        parent: wrapper,
        title: 'Sales Forecast Dashboard',
        single_column: true
    });
    
    // Add filters
    page.add_field({
        fieldtype: 'Link',
        label: 'Item Group',
        fieldname: 'item_group',
        options: 'Item Group',
        change: function() {
            load_dashboard();
        }
    });
    
    page.add_field({
        fieldtype: 'Link',
        label: 'Territory',
        fieldname: 'territory',
        options: 'Territory',
        change: function() {
            load_dashboard();
        }
    });
    
    page.add_field({
        fieldtype: 'Link',
        label: 'Customer Group',
        fieldname: 'customer_group',
        options: 'Customer Group',
        change: function() {
            load_dashboard();
        }
    });
    
    // Add dashboard sections
    var top_items_section = $('<div class="forecast-section"><h4>Top Items Forecast</h4><div id="top-items-chart"></div></div>').appendTo(page.main);
    var category_section = $('<div class="forecast-section"><h4>Category Forecast</h4><div id="category-chart"></div></div>').appendTo(page.main);
    var growth_section = $('<div class="forecast-section"><h4>Growth Trends</h4><div id="growth-chart"></div></div>').appendTo(page.main);
    var opportunity_section = $('<div class="forecast-section"><h4>Sales Opportunities</h4><div id="opportunities-list"></div></div>').appendTo(page.main);
    
    function load_dashboard() {
        let filters = {
            item_group: page.fields_dict.item_group.get_value(),
            territory: page.fields_dict.territory.get_value(),
            customer_group: page.fields_dict.customer_group.get_value()
        };
        
        frappe.call({
            method: "gsai_sales.controllers.sales_forecast.forecast_views.get_sales_forecast_dashboard",
            args: {
                filters_json: JSON.stringify(filters)
            },
            callback: function(r) {
                if (r.message) {
                    render_dashboard(r.message);
                }
            }
        });
        
        // frappe.call({
        //     method: "gsai_sales.controllers.sales_forecast.forecast_views.get_sales_opportunities",
        //     callback: function(r) {
        //         if (r.message) {
        //             render_opportunities(r.message);
        //         }
        //     }
        // });
    }
    
    function render_dashboard(data) {
        // Render top items chart
        render_items_chart(data.top_items_by_value, 'top-items-chart', 'Top Items by Value');
        
        // Render category chart
        render_category_chart(data.category_forecasts, 'category-chart');
        
        // Render growth trends
        render_growth_chart(data.growth_trends, 'growth-chart');
    }
    
    function render_items_chart(items, element_id, title) {
        // Create dataset for chart
        let labels = [];
        let datasets = [];
        
        // Get all time periods
        if (items.length > 0 && items[0].forecast.length > 0) {
            labels = items[0].forecast.map(period => period.date);
        }
        
        // Create dataset for each item
        items.slice(0, 5).forEach(item => {
            datasets.push({
                name: item.item_name,
                values: item.forecast.map(period => period.amount)
            });
        });
        
        // Create chart
        new frappe.Chart("#" + element_id, {
            title: title,
            data: {
                labels: labels,
                datasets: datasets
            },
            type: 'line',
            height: 300,
            colors: ['#7cd6fd', '#743ee2', '#5e64ff', '#ffa00a', '#28a745']
        });
    }
    
    function render_category_chart(categories, element_id) {
        // Create dataset for chart
        let labels = categories.map(cat => cat.item_group);
        let historical = categories.map(cat => cat.historical_value);
        let forecast = categories.map(cat => cat.forecast_total_value);
        
        // Create chart
        new frappe.Chart("#" + element_id, {
            title: "Category Forecast",
            data: {
                labels: labels,
                datasets: [
                    {
                        name: "Historical",
                        values: historical
                    },
                    {
                        name: "Forecast",
                        values: forecast
                    }
                ]
            },
            type: 'bar',
            height: 300,
            colors: ['#7cd6fd', '#743ee2']
        });
    }
    
    function render_growth_chart(growth_data, element_id) {
        // Create dataset for chart
        let growth_items = growth_data.growth.map(item => ({
            name: item.item_code,
            value: item.value_growth
        }));
        
        // Create chart
        new frappe.Chart("#" + element_id, {
            title: "Growth Potential",
            data: {
                labels: growth_items.map(item => item.name),
                datasets: [
                    {
                        name: "Growth %",
                        values: growth_items.map(item => item.value)
                    }
                ]
            },
            type: 'bar',
            height: 300,
            colors: ['#28a745']
        });
    }
    
    function render_opportunities(data) {
        let $opportunities = $("#opportunities-list").empty();
        
        // Combine all opportunity types
        let all_opportunities = [
            ...data.bundle_opportunities.map(o => ({...o, type: "Bundle"})),
            ...data.seasonal_opportunities.map(o => ({...o, type: "Seasonal"})),
            ...data.segment_opportunities.map(o => ({...o, type: "Customer Segment"})),
            ...data.territory_opportunities.map(o => ({...o, type: "Territory"}))
        ];
        
        all_opportunities.forEach(opp => {
            let $card = $(`
                <div class="opportunity-card">
                    <div class="opportunity-type">${opp.type}</div>
                    <div class="opportunity-description">${opp.description}</div>
                </div>
            `);
            
            $card.appendTo($opportunities);
        });
    }
    
    // Initial load
    load_dashboard();
}
