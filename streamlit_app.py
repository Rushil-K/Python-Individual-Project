import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud

# Load the pre-sampled data
df = pd.read_csv('/mnt/data/Project Dataset.csv')
sd = df.sample(n=3001, random_state=55027)

# Streamlit Dashboard Title
st.title("Comprehensive Dashboard")

# Sidebar Filters
st.sidebar.title("Filters")
selected_category = st.sidebar.multiselect('Select Category', sd['Category'].unique())
selected_supplier = st.sidebar.multiselect('Select Supplier', sd['Supplier'].unique())

# Filtering data
if selected_category:
    filtered_df = sd[sd['Category'].isin(selected_category)]
else:
    filtered_df = sd  # Use the sampled data if no filters applied

# KPI Cards
st.subheader("Key Performance Indicators")
total_value = filtered_df['Value'].sum()
total_quantity = filtered_df['Quantity'].sum()
top_product = filtered_df.groupby('Product')['Value'].sum().idxmax()

col1, col2, col3 = st.columns(3)
col1.metric(label="Total Value", value=f"${total_value:,.2f}")
col2.metric(label="Total Quantity", value=f"{total_quantity}")
col3.metric(label="Top Product", value=top_product)

# Summary Table
st.subheader("Summary Table")
st.dataframe(filtered_df.describe())

# 1. Bar Chart
st.subheader("Bar Chart - Value by Category")
bar_fig = px.bar(filtered_df, x='Category', y='Value', title='Value by Category', color='Category')
st.plotly_chart(bar_fig)

# 2. Line Chart
st.subheader("Line Chart - Value Over Time")
line_fig = px.line(filtered_df, x='Date', y='Value', title='Value Over Time')
st.plotly_chart(line_fig)

# 3. Pie Chart
st.subheader("Pie Chart - Product Distribution")
pie_fig = px.pie(filtered_df, values='Value', names='Product', title='Product Distribution by Value')
st.plotly_chart(pie_fig)

# 4. Area Chart
st.subheader("Area Chart - Quantity Over Time")
area_fig = px.area(filtered_df, x='Date', y='Quantity', title='Quantity Over Time')
st.plotly_chart(area_fig)

# 5. Scatter Plot
st.subheader("Scatter Plot - Quantity vs. Value")
scatter_fig = px.scatter(filtered_df, x='Quantity', y='Value', color='Category', title='Quantity vs. Value')
st.plotly_chart(scatter_fig)

# 6. Box Plot
st.subheader("Box Plot - Value Distribution by Category")
box_fig = px.box(filtered_df, x='Category', y='Value', title='Value Distribution by Category')
st.plotly_chart(box_fig)

# 7. Histogram
st.subheader("Histogram - Value Distribution")
plt.figure(figsize=(10, 5))
sns.histplot(filtered_df['Value'], bins=30, kde=True)
st.pyplot()

# 8. Donut Chart
st.subheader("Donut Chart - Payment Terms")
donut_data = filtered_df['Payment_Terms'].value_counts().reset_index()
donut_fig = px.pie(donut_data, names='index', values='Payment_Terms', title='Payment Terms Distribution')
donut_fig.update_traces(hole=0.4)
st.plotly_chart(donut_fig)

# 9. Stacked Bar Chart
st.subheader("Stacked Bar Chart - Value by Category and Payment Terms")
stacked_data = filtered_df.groupby(['Category', 'Payment_Terms']).sum().reset_index()
stacked_fig = px.bar(stacked_data, x='Category', y='Value', color='Payment_Terms', title='Stacked Bar Chart of Value by Category and Payment Terms', barmode='stack')
st.plotly_chart(stacked_fig)

# 10. Single Value Gauge
st.subheader("Single Value Gauge - Total Value")
gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=total_value,
    title={'text': "Total Value"},
    gauge={'axis': {'range': [0, total_value * 1.2]}, 'bar': {'color': "orange"}}))
st.plotly_chart(gauge_fig)

# 11. 2D Heat Map
st.subheader("2D Heat Map - Value by Category and Payment Terms")
heat_data = filtered_df.pivot_table(index='Category', columns='Payment_Terms', values='Value', aggfunc='sum')
sns.heatmap(heat_data, cmap="YlGnBu", annot=True)
st.pyplot()

# 12. Choropleth Map
st.subheader("Choropleth Map - Value by Country")
choropleth_fig = px.choropleth(
    filtered_df,
    locations='Country',
    locationmode='country names',
    color='Value',
    title='Choropleth Map of Value by Country'
)
st.plotly_chart(choropleth_fig)

# 13. Bubble Map
st.subheader("Bubble Map - Value by City")
bubble_fig = px.scatter_geo(
    filtered_df,
    locations='City',
    size='Value',
    hover_name='City',
    title='Bubble Map of Value by City'
)
st.plotly_chart(bubble_fig)

# 14. Funnel Chart
st.subheader("Funnel Chart - Sales Funnel")
funnel_data = filtered_df.groupby('Category').size().reset_index(name='Count')
funnel_fig = px.funnel(funnel_data, x='Count', y='Category', title='Sales Funnel by Category')
st.plotly_chart(funnel_fig)

# 15. Treemap
st.subheader("Treemap - Value by Category and Product")
treemap_fig = px.treemap(filtered_df, path=['Category', 'Product'], values='Value', title='Treemap of Value by Category and Product')
st.plotly_chart(treemap_fig)

# 16. Word Cloud
st.subheader("Word Cloud - Product Names")
wordcloud_data = " ".join(filtered_df['Product'].astype(str).values)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_data)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot()

# 17. Network Graph
st.subheader("Network Graph - Supplier-Customer Relationships")
G = nx.Graph()
for _, row in filtered_df.iterrows():
    G.add_edge(row['Supplier'], row['Customer'], weight=row['Value'])
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.3)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, edge_color='gray')
st.pyplot()

# 18. Gantt Chart
st.subheader("Gantt Chart - Order Timeline")
gantt_data = filtered_df[['Order_ID', 'Start_Date', 'End_Date', 'Product']]
gantt_fig = px.timeline(gantt_data, x_start='Start_Date', x_end='End_Date', y='Product', title='Gantt Chart of Orders')
st.plotly_chart(gantt_fig)

# 19. Waterfall Chart
st.subheader("Waterfall Chart - Sales by Category")
waterfall_data = filtered_df.groupby('Category')['Value'].sum().reset_index()
waterfall_fig = px.waterfall(waterfall_data, x='Category', y='Value', title='Waterfall Chart of Sales by Category')
st.plotly_chart(waterfall_fig)

# 20. Violin Plot
st.subheader("Violin Plot - Value Distribution by Category")
violin_fig = px.violin(filtered_df, y='Value', x='Category', box=True, points='all', title='Violin Plot of Value Distribution by Category')
st.plotly_chart(violin_fig)

# 21. Radial Chart
st.subheader("Radial Chart - Value by Payment Terms")
radial_fig = px.line_polar(filtered_df, r='Value', theta='Payment_Terms', line_close=True, title='Radial Chart of Value by Payment Terms')
st.plotly_chart(radial_fig)

# 22. Timeline Visualization
st.subheader("Timeline Visualization - Orders Over Time")
timeline_fig = px.timeline(filtered_df, x_start='Start_Date', x_end='End_Date', y='Product', title='Timeline of Orders')
st.plotly_chart(timeline_fig)

# 23. Matrix Chart
st.subheader("Matrix Chart - Value by Category and Payment Terms")
matrix_fig = px.imshow(filtered_df.pivot_table(index='Category', columns='Payment_Terms', values='Value', aggfunc='sum'))
st.plotly_chart(matrix_fig)

# 24. Multi-Series Chart
st.subheader("Multi-Series Chart - Value Over Time by Category")
multi_series_fig = px.line(filtered_df, x='Date', y='Value', color='Category', title='Multi-Series Chart of Value Over Time by Category')
st.plotly_chart(multi_series_fig)

# 25. Comparison Chart
st.subheader("Comparison Chart - Value by Category")
comparison_fig = px.bar(filtered_df, x='Category', y='Value', title='Comparison Chart of Value by Category', color='Category', barmode='group')
st.plotly_chart(comparison_fig)

# 26. Dendrogram
st.subheader("Dendrogram - Hierarchical Clustering of Categories")
sns.clustermap(filtered_df.groupby('Category').mean().drop(columns=['Value', 'Quantity']), cmap="viridis")
st.pyplot()

# Run the app using the command: streamlit run your_script.py
