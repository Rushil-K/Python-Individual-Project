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
cat = sd[['Country', 'Product', 'Import_Export', 'Category', 'Port', 'Customs_Code', 'Shipping_Method', 'Supplier', 'Customer', 'Payment_Terms']]

# Specify the country for insights
selected_country = 'Your_Specified_Country'  # Replace with the country you want to analyze
ncd = ncd[sd['Country'] == selected_country]
cat = cat[sd['Country'] == selected_country]

# Streamlit Dashboard
st.title(f"Comprehensive Data Dashboard for {selected_country}")

# 1. Charts
st.subheader("Charts")

# Bar Chart
st.subheader("Bar Chart: Quantity by Product")
quantity_by_product = cat['Product'].value_counts()
st.bar_chart(quantity_by_product)

# Line Chart
st.subheader("Line Chart: Value Over Time")
daily_value = ncd.groupby(ncd['Date'].dt.date)['Value'].sum().reset_index()
st.line_chart(daily_value.set_index('Date'))

# Pie Chart
st.subheader("Pie Chart: Distribution of Categories")
category_distribution = cat['Category'].value_counts()
st.pyplot(px.pie(values=category_distribution.values, names=category_distribution.index, title='Category Distribution'))

# Area Chart
st.subheader("Area Chart: Quantity Over Time")
daily_quantity = ncd.groupby(ncd['Date'].dt.date)['Quantity'].sum().reset_index()
st.area_chart(daily_quantity.set_index('Date'))

# Scatter Plot
st.subheader("Scatter Plot: Quantity vs. Value")
fig, ax = plt.subplots()
ax.scatter(ncd['Quantity'], ncd['Value'])
ax.set_title('Quantity vs Value')
ax.set_xlabel('Quantity')
ax.set_ylabel('Value')
st.pyplot(fig)

# 2. Tables
st.subheader("Data Tables")
st.dataframe(ncd)
st.subheader("Summary Table")
st.table(ncd.describe())

# 3. Gauges and Meters
st.subheader("Single Value Gauge: Total Value")
total_value = ncd['Value'].sum()
st.metric("Total Value", f"${total_value:,.2f}")

# Radial Gauge (Placeholder)
st.subheader("Radial Gauge: Total Weight")
total_weight = ncd['Weight'].sum()
st.write(f"Total Weight: {total_weight:,.2f}")

# 4. Heat Maps
st.subheader("Heat Map: Quantity by Category")
heat_data = ncd.groupby(['Category', 'Date'])['Quantity'].sum().unstack()
sns.heatmap(heat_data, annot=True, fmt='g', cmap='Blues')
st.pyplot()

# 5. Geographical Maps
st.subheader("Choropleth Map: Quantity by Port")
choropleth = px.choropleth(cat, locations='Port', locationmode='country names', 
                            color='Quantity', hover_name='Port', 
                            color_continuous_scale=px.colors.sequential.Plasma)
st.plotly_chart(choropleth)

# Bubble Map (Placeholder)
st.subheader("Bubble Map: Quantity by Shipping Method")
bubble_map = px.scatter_geo(cat, locations="Port", size="Quantity", hover_name="Shipping_Method",
                             title="Bubble Map: Quantity by Shipping Method")
st.plotly_chart(bubble_map)

# Pin Map (Placeholder)
st.subheader("Pin Map: Port Locations")
# Add your implementation here for Pin Map

# 6. Histogram
st.subheader("Histogram: Distribution of Quantity")
fig, ax = plt.subplots()
ax.hist(ncd['Quantity'], bins=20, color='skyblue', edgecolor='black')
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
st.subheader("Treemap: Category Distribution")
# Add your implementation here for Treemap

# 10. Sparklines (Placeholder)
st.subheader("Sparklines: Daily Value")
# Add your implementation here for Sparklines

# 11. Word Clouds
st.subheader("Word Cloud: Most Common Suppliers")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(cat['Supplier']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot()

# 12. Network Graphs (Placeholder)
st.subheader("Network Graph: Supplier Relationships")
# Add your implementation here for Network Graph

# 13. Cards
st.subheader("KPI Cards")
st.metric("Total Quantity", f"{ncd['Quantity'].sum():,.0f}")
st.metric("Average Value", f"${ncd['Value'].mean():,.2f}")

# 14. Gantt Charts (Placeholder)
st.subheader("Gantt Chart: Project Timeline")
# Add your implementation here for Gantt Chart

# 15. Box Plots
st.subheader("Box Plot: Quantity by Category")
fig, ax = plt.subplots()
sns.boxplot(x='Category', y='Quantity', data=ncd, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# 16. Waterfall Charts (Placeholder)
st.subheader("Waterfall Chart: Value Changes")
# Add your implementation here for Waterfall Chart

# 17. Violin Plots
st.subheader("Violin Plot: Quantity Distribution by Category")
fig, ax = plt.subplots()
sns.violinplot(x='Category', y='Quantity', data=ncd, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# 18. Donut Charts (Placeholder)
st.subheader("Donut Chart: Category Distribution")
# Add your implementation here for Donut Chart

# 19. Stacked Bar/Column Charts (Placeholder)
st.subheader("Stacked Bar Chart: Quantity by Category and Date")
# Add your implementation here for Stacked Bar Chart

# 20. Radial Charts (Placeholder)
st.subheader("Radial Chart: Quantity Distribution")
# Add your implementation here for Radial Chart

# 21. Timeline Visualizations (Placeholder)
st.subheader("Timeline Visualization: Events Over Time")
# Add your implementation here for Timeline Visualization

# 22. Matrix Charts (Placeholder)
st.subheader("Matrix Chart: Category vs. Product")
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
