import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px

# Load Data
df = pd.read_csv("Project Dataset.csv")
sd = df.sample(n=3001, random_state=55027)

ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]
cat = sd[['Country', 'Import_Export', 'Payment_Terms']]

# Check for required columns
required_columns = ['Quantity', 'Value', 'Date', 'Weight', 'Country', 'Import_Export', 'Payment_Terms']
if not all(col in sd.columns for col in required_columns):
    raise KeyError("Missing required columns in the dataset")


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
filtered_data = ncd[sd['Country'] == selected_country]
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
filtered_data = filtered_data[
    (filtered_data['Date'].dt.date >= selected_date_range[0]) &
    (filtered_data['Date'].dt.date <= selected_date_range[1])
]

# Further filter the data based on the Import/Export and Payment Terms selections
filtered_data = filtered_data[
    cat_data['Import_Export'].isin(selected_import_export) &
    cat_data['Payment_Terms'].isin(selected_payment_terms)
]

# Visualizations

# 1. Bar Chart
st.subheader("Bar Chart: Quantity by Import/Export")
bar_data = filtered_data.groupby('Import_Export')['Quantity'].sum().reset_index()
sns.barplot(x='Import_Export', y='Quantity', data=bar_data)
plt.title('Total Quantity by Import/Export Type')
st.pyplot()

# 2. Line Chart
st.subheader("Line Chart: Quantity Over Time")
line_data = filtered_data.groupby('Date')['Quantity'].sum().reset_index()
plt.plot(line_data['Date'], line_data['Quantity'])
plt.title('Quantity Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity')
st.pyplot()

# 3. Pie Chart
st.subheader("Pie Chart: Quantity Distribution by Import/Export")
pie_data = filtered_data.groupby('Import_Export')['Quantity'].sum()
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
plt.scatter(filtered_data['Value'], filtered_data['Quantity'], alpha=0.5)
plt.title('Scatter Plot: Quantity vs Value')
plt.xlabel('Value')
plt.ylabel('Quantity')
st.pyplot()

# 6. Data Table
st.subheader("Data Table")
st.write(filtered_data)

# 7. Summary Table
summary_data = filtered_data.describe()
st.subheader("Summary Table")
st.write(summary_data)

# 8. Single Value Gauge
st.subheader("Single Value Gauge: Total Quantity")
total_quantity = filtered_data['Quantity'].sum()
st.metric(label="Total Quantity", value=total_quantity)

# 9. Radial Gauge
st.subheader("Radial Gauge: Total Value")
total_value = filtered_data['Value'].sum()
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
theta = np.linspace(0, 2 * np.pi, 100)
r = np.linspace(0, total_value, 100)
ax.fill(theta, r, color='blue', alpha=0.25)
ax.set_title('Total Value Gauge')
st.pyplot(fig)

# 10. 2D Heat Map
st.subheader("2D Heat Map: Quantity and Payment Terms")
heatmap_data = pd.crosstab(cat_data['Payment_Terms'], cat_data['Import_Export'], values=filtered_data['Quantity'], aggfunc='sum')
sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='Blues')
plt.title('2D Heat Map of Quantity by Payment Terms and Import/Export Type')
st.pyplot()

# 11. Histograms
st.subheader("Histogram: Distribution of Quantity")
plt.hist(filtered_data['Quantity'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Quantity Distribution')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
st.pyplot()

# 12. Bullet Graph
st.subheader("Bullet Graph: Value Target vs Actual")
target_value = filtered_data['Value'].mean() + 1000  # Example target
plt.barh(['Actual Value'], [total_value], color='blue')
plt.barh(['Target Value'], [target_value], color='red', alpha=0.5)
plt.title('Bullet Graph')
plt.xlim(0, max(total_value, target_value) * 1.2)
st.pyplot()

# 13. Funnel Chart
st.subheader("Funnel Chart")
funnel_data = filtered_data.groupby('Import_Export')['Quantity'].sum().reset_index()
fig_funnel = px.funnel(funnel_data, x='Quantity', y='Import_Export', title='Funnel Chart: Quantity by Import/Export')
st.plotly_chart(fig_funnel)

# 14. Treemaps
st.subheader("Treemap: Quantity Distribution")
treemap_data = filtered_data.groupby('Import_Export')['Quantity'].sum().reset_index()
fig_treemap = px.treemap(treemap_data, path=['Import_Export'], values='Quantity', title='Treemap of Quantity by Import/Export')
st.plotly_chart(fig_treemap)

# 15. Sparklines
st.subheader("Sparklines")
sparkline_data = filtered_data.groupby('Date')['Quantity'].sum().reset_index()
st.line_chart(sparkline_data.set_index('Date'))

# 16. Word Clouds
st.subheader("Word Cloud of Countries")
wordcloud_data = ' '.join(cat['Country'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_data)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot()

# 17. Network Graphs
st.subheader("Network Graphs Placeholder")
# Implement your network graph logic here.

# 18. KPI Cards
st.subheader("KPI Cards")
st.metric(label="Average Value", value=filtered_data['Value'].mean())

# 19. Gantt Charts
st.subheader("Gantt Chart Placeholder")
# Implement your Gantt chart logic here.

# 20. Box Plots
st.subheader("Box Plot: Quantity by Import/Export")
sns.boxplot(x='Import_Export', y='Quantity', data=filtered_data)
plt.title('Box Plot of Quantity by Import/Export Type')
st.pyplot()

# 21. Waterfall Charts
st.subheader("Waterfall Chart Placeholder")
# Implement your waterfall chart logic here.

# 22. Violin Plots
st.subheader("Violin Plot: Quantity by Import/Export")
sns.violinplot(x='Import_Export', y='Quantity', data=filtered_data)
plt.title('Violin Plot of Quantity by Import/Export Type')
st.pyplot()

# 23. Donut Charts
st.subheader("Donut Chart: Quantity Distribution by Import/Export")
fig2, ax2 = plt.subplots()
size = 0.3
vals = pie_data
circular_sizes = [0.9, 0.6]  # Inner and outer radius
wedges, texts, autotexts = ax2.pie(vals, labels=vals.index, autopct='%.1f%%', startangle=90, radius=circular_sizes[0])
centre_circle = plt.Circle((0, 0), circular_sizes[1], color='white')
fig2.gca().add_artist(centre_circle)
ax2.axis('equal')  # Equal aspect ratio ensures pie chart is circular.
st.pyplot(fig2)

# 24. Stacked Bar/Column Charts
st.subheader("Stacked Bar Chart: Quantity by Payment Terms and Import/Export")
stacked_data = filtered_data.groupby(['Payment_Terms', 'Import_Export'])['Quantity'].sum().unstack()
stacked_data.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Quantity by Payment Terms and Import/Export')
plt.xlabel('Payment Terms')
plt.ylabel('Quantity')
st.pyplot()

# 25. Radial Charts
st.subheader("Radial Chart Placeholder")
# Implement your radial chart logic here.

# 26. Timeline Visualizations
st.subheader("Timeline Visualizations Placeholder")
# Implement your timeline logic here.

# 27. Matrix Charts
st.subheader("Matrix Chart Placeholder")
# Implement your matrix chart logic here.

# 28. Multi-Series Charts
st.subheader("Multi-Series Chart: Quantity by Date and Import/Export")
multi_series_data = filtered_data.pivot_table(values='Quantity', index='Date', columns='Import_Export', aggfunc='sum').fillna(0)
st.line_chart(multi_series_data)

# 29. Comparison Charts
st.subheader("Comparison Chart: Quantity by Import/Export")
comparison_data = filtered_data.groupby('Import_Export')['Quantity'].sum().reset_index()
plt.bar(comparison_data['Import_Export'], comparison_data['Quantity'], color=['orange', 'blue'])
plt.title('Comparison of Quantity by Import/Export')
st.pyplot()

# 30. Dendrogram
st.subheader("Dendrogram")
linkage_data = linkage(filtered_data[['Quantity', 'Value']], method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linkage_data)
plt.title('Dendrogram')
st.pyplot()
