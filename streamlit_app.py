import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram, linkage

# Load Data
df = pd.read_csv("Project Dataset.csv")
sd = df.sample(n=3001, random_state=55027)
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]
cat = sd[['Country', 'Import_Export', 'Payment_Terms']]

# Convert 'Date' column to datetime format
ncd['Date'] = pd.to_datetime(ncd['Date'], format="%d-%m-%Y")

# Streamlit Dashboard
st.title("Comprehensive Import/Export Data Dashboard")

# Sidebar for filters
st.sidebar.header("Filters")

# Country Selection
countries = cat['Country'].unique()
selected_country = st.sidebar.selectbox("Select a Country", countries)

# Filter data for the selected country
country_data = ncd[sd['Country'] == selected_country]
cat_data = cat[cat['Country'] == selected_country]

# Import/Export Selection
import_export_options = cat_data['Import_Export'].unique()
selected_import_export = st.sidebar.multiselect("Select Import/Export Type", import_export_options, default=import_export_options)

# Payment Terms Selection
payment_terms_options = cat_data['Payment_Terms'].unique()
selected_payment_terms = st.sidebar.multiselect("Select Payment Terms", payment_terms_options, default=payment_terms_options)

# Date Range Selection
date_min = ncd['Date'].min().date()
date_max = ncd['Date'].max().date()
selected_date_range = st.sidebar.date_input("Select Date Range", [date_min, date_max])

# Apply filters to the datasets
filtered_country_data = country_data[
    (country_data['Date'].dt.date >= selected_date_range[0]) &
    (country_data['Date'].dt.date <= selected_date_range[1]) &
    (cat_data['Import_Export'].isin(selected_import_export)) &
    (cat_data['Payment_Terms'].isin(selected_payment_terms))
]

# 1. Charts
st.subheader("Charts")

# Bar Chart
st.subheader("Bar Chart: Quantity by Import/Export")
quantity_by_import_export = filtered_country_data['Import_Export'].value_counts()
st.bar_chart(quantity_by_import_export)

# Line Chart
st.subheader("Line Chart: Value Over Time")
daily_value = filtered_country_data.groupby(filtered_country_data['Date'].dt.date)['Value'].sum().reset_index()
st.line_chart(daily_value.set_index('Date'))

# Pie Chart
st.subheader("Pie Chart: Distribution of Import/Export")
import_export_distribution = filtered_country_data['Import_Export'].value_counts()
st.plotly_chart(px.pie(values=import_export_distribution.values, names=import_export_distribution.index, title='Import/Export Distribution'))

# Area Chart
st.subheader("Area Chart: Quantity Over Time")
daily_quantity = filtered_country_data.groupby(filtered_country_data['Date'].dt.date)['Quantity'].sum().reset_index()
st.area_chart(daily_quantity.set_index('Date'))

# Scatter Plot
st.subheader("Scatter Plot: Quantity vs. Value")
fig, ax = plt.subplots()
ax.scatter(filtered_country_data['Quantity'], filtered_country_data['Value'])
ax.set_title('Quantity vs Value')
ax.set_xlabel('Quantity')
ax.set_ylabel('Value')
st.pyplot(fig)

# 2. Tables
st.subheader("Data Tables")
st.dataframe(filtered_country_data)
st.subheader("Summary Table")
st.table(filtered_country_data.describe())

# 3. Gauges and Meters
st.subheader("Single Value Gauge: Total Value")
total_value = filtered_country_data['Value'].sum()
st.metric("Total Value", f"${total_value:,.2f}")

# Radial Gauge (Simple Metric)
st.subheader("Radial Gauge: Total Quantity")
total_quantity = filtered_country_data['Quantity'].sum()
st.write(f"Total Quantity: {total_quantity:,.0f}")

# 4. Heat Maps
st.subheader("Heat Map: Quantity by Import/Export")
heat_data = filtered_country_data.groupby(['Import_Export', filtered_country_data['Date'].dt.date])['Quantity'].sum().unstack()
sns.heatmap(heat_data, annot=True, fmt='g', cmap='Blues')
st.pyplot()

# 5. Geographical Maps
st.subheader("Choropleth Map: Quantity by Country")
choropleth = px.choropleth(cat_data[cat_data['Country'] == selected_country], locations='Country', locationmode='country names',
                            color='Import_Export', hover_name='Country',
                            color_continuous_scale=px.colors.sequential.Plasma)
st.plotly_chart(choropleth)

# Bubble Map
st.subheader("Bubble Map: Quantity by Import/Export")
bubble_map = px.scatter_geo(cat_data[cat_data['Country'] == selected_country], locations="Country", size="Quantity", hover_name="Import_Export",
                             title="Bubble Map: Quantity by Import/Export")
st.plotly_chart(bubble_map)

# 6. Histogram
st.subheader("Histogram: Distribution of Quantity")
fig, ax = plt.subplots()
ax.hist(filtered_country_data['Quantity'], bins=20, color='skyblue', edgecolor='black')
ax.set_title('Quantity Distribution')
ax.set_xlabel('Quantity')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# 7. Bullet Graph
st.subheader("Bullet Graph: Value Target")
fig, ax = plt.subplots()
ax.barh(['Value'], [total_value], color='skyblue')
ax.axvline(x=filtered_country_data['Value'].mean(), color='red', label='Average Value')
ax.legend()
st.pyplot(fig)

# 8. Funnel Charts
st.subheader("Funnel Chart: Sales Funnel")
funnel_data = {'Stage': ['Leads', 'Qualified Leads', 'Proposals', 'Closed Deals'], 'Count': [1000, 600, 300, 150]}
funnel_df = pd.DataFrame(funnel_data)
st.bar_chart(funnel_df.set_index('Stage'))

# 9. Treemaps
st.subheader("Treemap: Import/Export Distribution")
treemap = px.treemap(cat_data[cat_data['Country'] == selected_country], path=['Import_Export'], values='Quantity', title="Treemap of Import/Export Types")
st.plotly_chart(treemap)

# 10. Sparklines
st.subheader("Sparklines: Daily Value")
spark_data = filtered_country_data.groupby(filtered_country_data['Date'].dt.date)['Value'].sum().reset_index()
st.line_chart(spark_data.set_index('Date'))

# 11. Word Clouds
st.subheader("Word Cloud: Most Common Payment Terms")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(cat_data['Payment_Terms']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot()

# 12. Network Graphs
st.subheader("Network Graph: Import/Export Relationships")
fig, ax = plt.subplots()
sns.countplot(x='Import_Export', data=cat_data[cat_data['Country'] == selected_country], ax=ax)
ax.set_title('Network of Import/Export')
plt.xticks(rotation=90)
st.pyplot(fig)

# 13. Cards
st.subheader("KPI Cards")
st.metric("Total Quantity", f"{filtered_country_data['Quantity'].sum():,.0f}")
st.metric("Average Value", f"${filtered_country_data['Value'].mean():,.2f}")

# 14. Gantt Charts
st.subheader("Gantt Chart: Project Timeline")
# Placeholder Gantt Chart
# Not Implemented

# 15. Box Plots
st.subheader("Box Plot: Quantity by Import/Export")
fig, ax = plt.subplots()
sns.boxplot(x='Import_Export', y='Quantity', data=filtered_country_data, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# 16. Waterfall Charts
st.subheader("Waterfall Chart: Value Changes")
# Sample Data for Waterfall Chart
waterfall_data = {'Stage': ['Start', 'Increase', 'Decrease', 'End'], 'Value': [0, 500, -300, 200]}
waterfall_df = pd.DataFrame(waterfall_data)
st.bar_chart(waterfall_df.set_index('Stage'))

# 17. Violin Plots
st.subheader("Violin Plot: Quantity Distribution by Import/Export")
fig, ax = plt.subplots()
sns.violinplot(x='Import_Export', y='Quantity', data=filtered_country_data, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# 18. Donut Charts
st.subheader("Donut Chart: Import/Export Distribution")
donut_data = filtered_country_data['Import_Export'].value_counts()
fig, ax = plt.subplots()
ax.pie(donut_data, labels=donut_data.index, autopct='%1.1f%%', startangle=90)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax.axis('equal')
st.pyplot(fig)

# 19. Stacked Bar/Column Charts
st.subheader("Stacked Bar Chart: Quantity by Import/Export")
stacked_data = filtered_country_data.groupby(['Date', 'Import_Export'])['Quantity'].sum().unstack()
stacked_data.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart: Quantity by Import/Export')
plt.xlabel('Date')
plt.ylabel('Quantity')
st.pyplot()

# 20. Radial Charts
st.subheader("Radial Chart: Quantity vs Value")
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
theta = np.linspace(0, 2 * np.pi, len(filtered_country_data), endpoint=False)
r = filtered_country_data['Quantity'].values
ax.fill(theta, r, color='blue', alpha=0.25)
ax.set_yticklabels([])
st.pyplot(fig)

# 21. Timeline Visualizations
st.subheader("Timeline Visualization: Import/Export Events")
timeline_data = pd.DataFrame({
    'Date': pd.to_datetime(['01-01-2023', '15-01-2023', '01-02-2023', '15-02-2023']),
    'Event': ['Event 1', 'Event 2', 'Event 3', 'Event 4']
})
timeline_data['Date'] = timeline_data['Date'].dt.strftime('%d-%m-%Y')
st.line_chart(timeline_data.set_index('Date'))

# 22. Matrix Charts
st.subheader("Matrix Chart: Quantity and Payment Terms")
matrix_data = pd.crosstab(cat_data['Payment_Terms'], cat_data['Import_Export'], values=filtered_country_data['Quantity'], aggfunc='sum')
sns.heatmap(matrix_data, annot=True, fmt='g', cmap='Blues')
st.pyplot()

# 23. Multi-Series Charts
st.subheader("Multi-Series Chart: Quantity and Value")
multi_series_data = filtered_country_data.groupby(filtered_country_data['Date'].dt.date).agg({'Quantity': 'sum', 'Value': 'sum'}).reset_index()
multi_series_data.plot(x='Date', y=['Quantity', 'Value'])
plt.title('Multi-Series Chart: Quantity and Value Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity / Value')
st.pyplot()

# 24. Comparison Charts
st.subheader("Comparison Chart: Quantity by Import/Export")
comparison_data = filtered_country_data.groupby('Import_Export')['Quantity'].sum().reset_index()
sns.barplot(x='Import_Export', y='Quantity', data=comparison_data)
plt.title('Comparison of Quantity by Import/Export Type')
plt.xlabel('Import/Export Type')
plt.ylabel('Quantity')
plt.xticks(rotation=45)
st.pyplot()

# 25. Dendrograms
st.subheader("Dendrogram: Clustering of Import/Export")
linked = linkage(cat_data['Import_Export'].value_counts().values.reshape(-1, 1), 'single')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=cat_data['Import_Export'].value_counts().index, distance_sort='descending', show_leaf_counts=True)
st.pyplot()

# Footer
st.write("This dashboard provides a comprehensive overview of the import/export data.")

# Run the app
if __name__ == "__main__":
    st.run()
