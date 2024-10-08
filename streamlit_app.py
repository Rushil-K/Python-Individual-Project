import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# Load Data
df = pd.read_csv("Project Dataset.csv")
sd = df.sample(n=3001, random_state=55027)
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]
cat = sd[['Country', 'Import_Export', 'Shipping_Method', 'Payment_Terms']]

# Convert 'Date' column to datetime format
ncd['Date'] = pd.to_datetime(ncd['Date'])

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
st.pyplot(px.pie(values=shipping_method_distribution.values, names=shipping_method_distribution.index, title='Shipping Method Distribution'))

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

# Radial Gauge (Placeholder)
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

# Bubble Map (Placeholder)
st.subheader("Bubble Map: Quantity by Shipping Method")
bubble_map = px.scatter_geo(cat_data, locations="Country", size="Quantity", hover_name="Shipping_Method",
                             title="Bubble Map: Quantity by Shipping Method")
st.plotly_chart(bubble_map)

# Pin Map (Placeholder)
st.subheader("Pin Map: Shipping Method Locations")
# Add your implementation here for Pin Map

# 6. Histogram
st.subheader("Histogram: Distribution of Quantity")
fig, ax = plt.subplots()
ax.hist(country_data['Quantity'], bins=20, color='skyblue', edgecolor='black')
ax.set_title('Quantity Distribution')
ax.set_xlabel('Quantity')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# 7. Bullet Graph (Placeholder)
st.subheader("Bullet Graph: Value Target")
# Add your implementation here for Bullet Graph

# 8. Funnel Charts (Placeholder)
st.subheader("Funnel Chart: Sales Funnel")
# Add your implementation here for Funnel Chart

# 9. Treemaps (Placeholder)
st.subheader("Treemap: Shipping Method Distribution")
# Add your implementation here for Treemap

# 10. Sparklines (Placeholder)
st.subheader("Sparklines: Daily Value")
# Add your implementation here for Sparklines

# 11. Word Clouds
st.subheader("Word Cloud: Most Common Payment Terms")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(cat_data['Payment_Terms']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot()

# 12. Network Graphs (Placeholder)
st.subheader("Network Graph: Shipping Method Relationships")
# Add your implementation here for Network Graph

# 13. Cards
st.subheader("KPI Cards")
st.metric("Total Quantity", f"{country_data['Quantity'].sum():,.0f}")
st.metric("Average Value", f"${country_data['Value'].mean():,.2f}")

# 14. Gantt Charts (Placeholder)
st.subheader("Gantt Chart: Project Timeline")
# Add your implementation here for Gantt Chart

# 15. Box Plots
st.subheader("Box Plot: Quantity by Import/Export")
fig, ax = plt.subplots()
sns.boxplot(x='Import_Export', y='Quantity', data=cat_data, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# 16. Waterfall Charts (Placeholder)
st.subheader("Waterfall Chart: Value Changes")
# Add your implementation here for Waterfall Chart

# 17. Violin Plots
st.subheader("Violin Plot: Quantity Distribution by Shipping Method")
fig, ax = plt.subplots()
sns.violinplot(x='Shipping_Method', y='Quantity', data=cat_data, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# 18. Donut Charts (Placeholder)
st.subheader("Donut Chart: Import/Export Distribution")
# Add your implementation here for Donut Chart

# 19. Stacked Bar/Column Charts (Placeholder)
st.subheader("Stacked Bar Chart: Quantity by Shipping Method and Date")
# Add your implementation here for Stacked Bar Chart

# 20. Radial Charts (Placeholder)
st.subheader("Radial Chart: Quantity Distribution")
# Add your implementation here for Radial Chart

# 21. Timeline Visualizations (Placeholder)
st.subheader("Timeline Visualization: Events Over Time")
# Add your implementation here for Timeline Visualization

# 22. Matrix Charts (Placeholder)
st.subheader("Matrix Chart: Shipping Method vs. Import/Export")
# Add your implementation here for Matrix Chart

# 23. Multi-Series Charts (Placeholder)
st.subheader("Multi-Series Chart: Value and Quantity Over Time")
# Add your implementation here for Multi-Series Chart

# 24. Comparison Charts (Placeholder)
st.subheader("Comparison Chart: Quantity vs. Value")
# Add your implementation here for Comparison Chart

# 25. Dendrograms (Placeholder)
st.subheader("Dendrogram: Hierarchical Clustering")
# Add your implementation here for Dendrogram

# Run the app
if __name__ == '__main__':
    st.run()
