import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram, linkage

# Load Data
df = pd.read_csv('Project Dataset.csv')
sd = df.sample(n=3001, random_state=55027)
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]
cat = sd[['Country', 'Import_Export', 'Shipping_Method', 'Payment_Terms']]

# Convert 'Date' column to datetime format
ncd['Date'] = pd.to_datetime(ncd['Date'], format="%d-%m-%Y")

# Streamlit Dashboard
st.title("Comprehensive Import/Export Data Dashboard")

# Sidebar for country selection
countries = cat['Country'].unique()
selected_country = st.sidebar.selectbox("Select a Country", countries)

# Filter data for the selected country
country_data = ncd[sd['Country'] == selected_country]
cat_data = cat[cat['Country'] == selected_country]

# 1. Charts
st.subheader("Charts")

# Bar Chart
st.subheader("Bar Chart: Quantity by Import/Export")
quantity_by_import_export = cat_data['Import_Export'].value_counts()
st.bar_chart(quantity_by_import_export)

# Line Chart
st.subheader("Line Chart: Value Over Time")
daily_value = country_data.groupby(country_data['Date'].dt.date)['Value'].sum().reset_index()
st.line_chart(daily_value.set_index('Date'))

# Pie Chart
st.subheader("Pie Chart: Distribution of Shipping Methods")
shipping_method_distribution = cat_data['Shipping_Method'].value_counts()
st.plotly_chart(px.pie(values=shipping_method_distribution.values, names=shipping_method_distribution.index, title='Shipping Method Distribution'))

# Area Chart
st.subheader("Area Chart: Quantity Over Time")
daily_quantity = country_data.groupby(country_data['Date'].dt.date)['Quantity'].sum().reset_index()
st.area_chart(daily_quantity.set_index('Date'))

# Scatter Plot
st.subheader("Scatter Plot: Quantity vs. Value")
fig, ax = plt.subplots()
ax.scatter(country_data['Quantity'], country_data['Value'])
ax.set_title('Quantity vs Value')
ax.set_xlabel('Quantity')
ax.set_ylabel('Value')
st.pyplot(fig)

# 2. Tables
st.subheader("Data Tables")
st.dataframe(country_data)
st.subheader("Summary Table")
st.table(country_data.describe())

# 3. Gauges and Meters
st.subheader("Single Value Gauge: Total Value")
total_value = country_data['Value'].sum()
st.metric("Total Value", f"${total_value:,.2f}")

# Radial Gauge (Simple Metric)
st.subheader("Radial Gauge: Total Quantity")
total_quantity = country_data['Quantity'].sum()
st.write(f"Total Quantity: {total_quantity:,.0f}")

# 4. Heat Maps
st.subheader("Heat Map: Quantity by Shipping Method")
heat_data = country_data.groupby(['Shipping_Method', country_data['Date'].dt.date])['Quantity'].sum().unstack()
sns.heatmap(heat_data, annot=True, fmt='g', cmap='Blues')
st.pyplot()

# 5. Geographical Maps
st.subheader("Choropleth Map: Quantity by Country")
choropleth = px.choropleth(cat_data, locations='Country', locationmode='country names',
                            color='Import_Export', hover_name='Country',
                            color_continuous_scale=px.colors.sequential.Plasma)
st.plotly_chart(choropleth)

# Bubble Map
st.subheader("Bubble Map: Quantity by Shipping Method")
bubble_map = px.scatter_geo(cat_data, locations="Country", size="Quantity", hover_name="Shipping_Method",
                             title="Bubble Map: Quantity by Shipping Method")
st.plotly_chart(bubble_map)

# 6. Histogram
st.subheader("Histogram: Distribution of Quantity")
fig, ax = plt.subplots()
ax.hist(country_data['Quantity'], bins=20, color='skyblue', edgecolor='black')
ax.set_title('Quantity Distribution')
ax.set_xlabel('Quantity')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# 7. Bullet Graph
st.subheader("Bullet Graph: Value Target")
# Simple Representation
fig, ax = plt.subplots()
ax.barh(['Value'], [total_value], color='skyblue')
ax.axvline(x=country_data['Value'].mean(), color='red', label='Average Value')
ax.legend()
st.pyplot(fig)

# 8. Funnel Charts
st.subheader("Funnel Chart: Sales Funnel")
# Sample Data
funnel_data = {'Stage': ['Leads', 'Qualified Leads', 'Proposals', 'Closed Deals'], 'Count': [1000, 600, 300, 150]}
funnel_df = pd.DataFrame(funnel_data)
st.bar_chart(funnel_df.set_index('Stage'))

# 9. Treemaps
st.subheader("Treemap: Shipping Method Distribution")
treemap = px.treemap(cat_data, path=['Shipping_Method'], values='Quantity', title="Treemap of Shipping Methods")
st.plotly_chart(treemap)

# 10. Sparklines
st.subheader("Sparklines: Daily Value")
spark_data = country_data.groupby(country_data['Date'].dt.date)['Value'].sum().reset_index()
st.line_chart(spark_data.set_index('Date'))

# 11. Word Clouds
st.subheader("Word Cloud: Most Common Payment Terms")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(cat_data['Payment_Terms']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot()

# 12. Network Graphs
st.subheader("Network Graph: Shipping Method Relationships")
# Simple Network Graph
sns.set(style="whitegrid")
fig, ax = plt.subplots()
sns.countplot(x='Shipping_Method', data=cat_data, ax=ax)
ax.set_title('Network of Shipping Methods')
plt.xticks(rotation=90)
st.pyplot(fig)

# 13. Cards
st.subheader("KPI Cards")
st.metric("Total Quantity", f"{country_data['Quantity'].sum():,.0f}")
st.metric("Average Value", f"${country_data['Value'].mean():,.2f}")

# 14. Gantt Charts
st.subheader("Gantt Chart: Project Timeline")
# Placeholder Gantt Chart
# Not Implemented

# 15. Box Plots
st.subheader("Box Plot: Quantity by Import/Export")
fig, ax = plt.subplots()
sns.boxplot(x='Import_Export', y='Quantity', data=cat_data, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# 16. Waterfall Charts
st.subheader("Waterfall Chart: Value Changes")
# Sample Data for Waterfall Chart
waterfall_data = {'Stage': ['Start', 'Increase', 'Decrease', 'End'], 'Value': [0, 500, -300, 200]}
waterfall_df = pd.DataFrame(waterfall_data)
st.bar_chart(waterfall_df.set_index('Stage'))

# 17. Violin Plots
st.subheader("Violin Plot: Quantity Distribution by Shipping Method")
fig, ax = plt.subplots()
sns.violinplot(x='Shipping_Method', y='Quantity', data=cat_data, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# 18. Donut Charts
st.subheader("Donut Chart: Import/Export Distribution")
donut_data = cat_data['Import_Export'].value_counts()
fig, ax = plt.subplots()
ax.pie(donut_data, labels=donut_data.index, autopct='%1.1f%%', startangle=90)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax.axis('equal')
st.pyplot(fig)

# 19. Stacked Bar/Column Charts
st.subheader("Stacked Bar Chart: Quantity by Shipping Method and Date")
stacked_data = country_data.groupby(['Shipping_Method', country_data['Date'].dt.date])['Quantity'].sum().unstack()
stacked_data.plot(kind='bar', stacked=True)
st.pyplot()

# 20. Radial Charts
st.subheader("Radial Chart: Quantity Distribution")
# Placeholder Radial Chart
# Not Implemented

# 21. Timeline Visualizations
st.subheader("Timeline Visualization: Events Over Time")
# Placeholder Timeline Visualization
# Not Implemented

# 22. Matrix Charts
st.subheader("Matrix Chart: Shipping Method vs. Import/Export")
matrix_data = country_data.pivot_table(values='Quantity', index='Shipping_Method', columns='Import_Export', aggfunc='sum')
sns.heatmap(matrix_data, annot=True, fmt='g', cmap='Blues')
st.pyplot()

# 23. Multi-Series Charts
st.subheader("Multi-Series Chart: Value and Quantity Over Time")
multi_series_data = daily_value.set_index('Date').join(daily_quantity.set_index('Date'), lsuffix='_Value', rsuffix='_Quantity')
multi_series_data.plot()
st.pyplot()

# 24. Comparison Charts
st.subheader("Comparison Chart: Quantity vs. Value")
fig, ax = plt.subplots()
ax.plot(country_data['Date'], country_data['Quantity'], label='Quantity', color='blue')
ax.plot(country_data['Date'], country_data['Value'], label='Value', color='orange')
ax.set_title('Comparison of Quantity and Value Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Quantity / Value')
ax.legend()
st.pyplot(fig)

# 25. Dendrograms
st.subheader("Dendrogram: Clustering of Shipping Methods")
linked = linkage(cat_data['Shipping_Method'].value_counts().values.reshape(-1, 1), 'single')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=cat_data['Shipping_Method'].value_counts().index, distance_sort='descending', show_leaf_counts=True)
st.pyplot()

# Footer
st.write("This dashboard provides a comprehensive overview of the import/export data.")

# Run the app
if __name__ == "__main__":
    st.run()

