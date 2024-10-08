import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Load Data
df = pd.read_csv("Project Dataset.csv")
sd = df.sample(n=3001, random_state=55027)
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]
cat = sd[['Country', 'Import_Export', 'Payment_Terms']]

# Convert 'Date' column to datetime format with the specified format
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

# Print filtered columns for debugging
st.write("Filtered Columns in country_data:", country_data.columns)
st.write("Filtered Columns in cat_data:", cat_data.columns)

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
try:
    filtered_country_data = country_data[
        (country_data['Date'].dt.date >= selected_date_range[0]) &
        (country_data['Date'].dt.date <= selected_date_range[1]) &
        (cat_data['Import_Export'].isin(selected_import_export)) &
        (cat_data['Payment_Terms'].isin(selected_payment_terms))
    ]
except KeyError as e:
    st.error(f"KeyError: The column {e} is not found in the DataFrame.")
    st.stop()

# Visualizations

# 1. Bar Chart
st.subheader("Bar Chart: Quantity by Import/Export")
bar_data = filtered_country_data.groupby('Import_Export')['Quantity'].sum().reset_index()
sns.barplot(x='Import_Export', y='Quantity', data=bar_data)
plt.title('Total Quantity by Import/Export Type')
st.pyplot()

# 2. Line Chart
st.subheader("Line Chart: Quantity Over Time")
line_data = filtered_country_data.groupby('Date')['Quantity'].sum().reset_index()
plt.plot(line_data['Date'], line_data['Quantity'])
plt.title('Quantity Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity')
st.pyplot()

# 3. Pie Chart
st.subheader("Pie Chart: Quantity Distribution by Import/Export")
pie_data = filtered_country_data.groupby('Import_Export')['Quantity'].sum()
fig1, ax1 = plt.subplots()
ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures pie chart is circular.
st.pyplot(fig1)

# 4. Area Chart
st.subheader("Area Chart: Quantity Over Time")
plt.fill_between(line_data['Date'], line_data['Quantity'], color="skyblue", alpha=0.4)
plt.plot(line_data['Date'], line_data['Quantity'], color="Slateblue", alpha=0.6)
plt.title('Area Chart of Quantity Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity')
st.pyplot()

# 5. Scatter Plot
st.subheader("Scatter Plot: Quantity vs Value")
plt.scatter(filtered_country_data['Value'], filtered_country_data['Quantity'], alpha=0.5)
plt.title('Scatter Plot: Quantity vs Value')
plt.xlabel('Value')
plt.ylabel('Quantity')
st.pyplot()

# 6. Data Table
st.subheader("Data Table")
st.write(filtered_country_data)

# 7. Summary Table
summary_data = filtered_country_data.describe()
st.subheader("Summary Table")
st.write(summary_data)

# 8. Single Value Gauge
st.subheader("Single Value Gauge: Total Quantity")
total_quantity = filtered_country_data['Quantity'].sum()
st.metric(label="Total Quantity", value=total_quantity)

# 9. Radial Gauge
st.subheader("Radial Gauge: Total Value")
total_value = filtered_country_data['Value'].sum()
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
theta = np.linspace(0, 2 * np.pi, 100)
r = np.linspace(0, total_value, 100)
ax.fill(theta, r, color='blue', alpha=0.25)
ax.set_title('Total Value Gauge')
st.pyplot(fig)

# 10. 2D Heat Map
st.subheader("2D Heat Map: Quantity and Payment Terms")
heatmap_data = pd.crosstab(cat_data['Payment_Terms'], cat_data['Import_Export'], values=filtered_country_data['Quantity'], aggfunc='sum')
sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='Blues')
plt.title('2D Heat Map of Quantity by Payment Terms and Import/Export Type')
st.pyplot()

# 11. Histograms
st.subheader("Histogram: Distribution of Quantity")
plt.hist(filtered_country_data['Quantity'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Quantity Distribution')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
st.pyplot()

# 12. Funnel Chart Placeholder (Add your own implementation as needed)
st.subheader("Funnel Chart Placeholder")
st.write("Implement your funnel chart logic here.")

# 13. Treemaps Placeholder (Add your own implementation as needed)
st.subheader("Treemap Placeholder")
st.write("Implement your treemap logic here.")

# 14. Sparklines Placeholder (Add your own implementation as needed)
st.subheader("Sparklines Placeholder")
st.write("Implement your sparklines logic here.")

# 15. Word Clouds Placeholder (Add your own implementation as needed)
st.subheader("Word Cloud Placeholder")
st.write("Implement your word cloud logic here.")

# 16. Network Graphs Placeholder (Add your own implementation as needed)
st.subheader("Network Graphs Placeholder")
st.write("Implement your network graphs logic here.")

# 17. KPI Cards
st.subheader("KPI Cards")
st.metric(label="Average Value", value=filtered_country_data['Value'].mean())

# 18. Gantt Charts Placeholder (Add your own implementation as needed)
st.subheader("Gantt Chart Placeholder")
st.write("Implement your Gantt chart logic here.")

# 19. Box Plots
st.subheader("Box Plot: Quantity by Import/Export")
sns.boxplot(x='Import_Export', y='Quantity', data=filtered_country_data)
plt.title('Box Plot of Quantity by Import/Export Type')
st.pyplot()

# 20. Waterfall Charts Placeholder (Add your own implementation as needed)
st.subheader("Waterfall Chart Placeholder")
st.write("Implement your waterfall chart logic here.")

# 21. Violin Plots
st.subheader("Violin Plot: Quantity by Import/Export")
sns.violinplot(x='Import_Export', y='Quantity', data=filtered_country_data)
plt.title('Violin Plot of Quantity by Import/Export Type')
st.pyplot()

# 22. Donut Charts
st.subheader("Donut Chart: Quantity Distribution by Import/Export")
fig2, ax2 = plt.subplots()
size = 0.3
vals = pie_data
circular_sizes = [0.9, 0.6]  # Inner and outer radius
wedges, texts, autotexts = ax2.pie(vals, labels=vals.index, autopct='%.1f%%', startangle=90, radius=circular_sizes[0])
centre_circle = plt.Circle((0, 0), circular_sizes[1], color='white')
fig2.gca().add_artist(centre_circle)
ax2.axis('equal')
st.pyplot(fig2)

# 23. Stacked Bar/Column Charts
st.subheader("Stacked Bar Chart: Quantity by Import/Export")
stacked_data = filtered_country_data.groupby(['Date', 'Import_Export'])['Quantity'].sum().unstack()
stacked_data.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart: Quantity by Import/Export')
plt.xlabel('Date')
plt.ylabel('Quantity')
st.pyplot()

# 24. Comparison Charts
st.subheader("Comparison Chart: Quantity by Import/Export")
comparison_data = filtered_country_data.groupby('Import_Export')['Quantity'].sum().reset_index()
sns.barplot(x='Import_Export', y='Quantity', data=comparison_data)
plt.title('Comparison of Quantity by Import/Export Type')
plt.xlabel('Import/Export Type')
plt.ylabel('Quantity')
st.pyplot()

# 25. Dendrograms Placeholder (Add your own implementation as needed)
st.subheader("Dendrogram Placeholder")
st.write("Implement your dendrogram logic here.")
