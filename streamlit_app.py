import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud
import numpy as np
import geopandas as gpd

# Load and preprocess the data
df = pd.read_csv("Project Dataset.csv")
sample_data = df.sample(n=3001, random_state=55027)
sample_data['Date'] = pd.to_datetime(sample_data['Date'], format="%d-%m-%Y")

# Separate non-categorical and categorical data
non_categorical_data = sample_data[['Quantity', 'Value', 'Date', 'Weight']]
categorical_data = sample_data[['Country', 'Import_Export', 'Shipping_Method', 'Payment_Terms']]

# Streamlit layout
st.title("Comprehensive Dashboard")

# Sidebar for filters
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=categorical_data['Country'].unique(),
    default=categorical_data['Country'].unique()  # Default to all countries
)

# Filter data based on selected countries
filtered_data = sample_data[sample_data['Country'].isin(selected_countries)]

# Bar Chart
bar_fig = px.bar(filtered_data, x='Country', y='Value', title="Bar Chart: Value by Country")
st.plotly_chart(bar_fig)

# Line Chart
line_fig = px.line(filtered_data, x='Date', y='Value', title="Line Chart: Value over Time")
st.plotly_chart(line_fig)

# Pie Chart
pie_fig = px.pie(filtered_data, names='Country', values='Value', title="Pie Chart: Value Distribution by Country")
st.plotly_chart(pie_fig)

# Area Chart
area_fig = px.area(filtered_data, x='Date', y='Value', title="Area Chart: Value over Time")
st.plotly_chart(area_fig)

# Scatter Plot
scatter_fig = px.scatter(filtered_data, x='Weight', y='Value', color='Country', title="Scatter Plot: Weight vs Value")
st.plotly_chart(scatter_fig)

# Histogram
histogram_fig = px.histogram(filtered_data, x='Value', title="Histogram: Value Distribution")
st.plotly_chart(histogram_fig)

# Heat Map
heatmap_data = filtered_data.pivot_table(index='Country', columns='Import_Export', values='Value', aggfunc='sum').fillna(0)
heatmap_fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Viridis'))
heatmap_fig.update_layout(title='Heatmap: Value by Country and Import/Export', xaxis_title='Import/Export', yaxis_title='Country')
st.plotly_chart(heatmap_fig)

# Choropleth Map
geo_df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
merged = geo_df.set_index('name').join(filtered_data.groupby('Country').sum())
choropleth_fig = px.choropleth(merged, geojson=geo_df.geometry, locations=merged.index, color='Value',
                                title="Choropleth Map: Value by Country", color_continuous_scale="Viridis")
choropleth_fig.update_geos(fitbounds="locations")
st.plotly_chart(choropleth_fig)

# Bubble Map
bubble_fig = px.scatter_geo(filtered_data, locations='Country', size='Value', title='Bubble Map: Value by Country', hover_name='Country')
st.plotly_chart(bubble_fig)

# Box Plot
box_fig = px.box(filtered_data, x='Country', y='Value', title="Box Plot: Value by Country")
st.plotly_chart(box_fig)

# Word Cloud
wordcloud_data = ' '.join(filtered_data['Country'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_data)
st.image(wordcloud.to_array(), caption='Word Cloud of Countries', use_column_width=True)

# KPI Cards
total_value = filtered_data['Value'].sum()
total_quantity = filtered_data['Quantity'].sum()
kpi_col1, kpi_col2 = st.columns(2)
kpi_col1.metric(label="Total Value", value=f"{total_value:.2f}")
kpi_col2.metric(label="Total Quantity", value=f"{total_quantity:.2f}")

# Funnel Chart (Example Data)
funnel_fig = go.Figure()
funnel_fig.add_trace(go.Funnel(
    name="Funnel",
    y=["Stage 1", "Stage 2", "Stage 3", "Stage 4"],
    x=[100, 80, 50, 20]
))
funnel_fig.update_layout(title="Funnel Chart")
st.plotly_chart(funnel_fig)

# Waterfall Chart (Example Data)
waterfall_fig = go.Figure(go.Waterfall(
    x=["Start", "Step 1", "Step 2", "End"],
    y=[500, -200, 300, 600],
    measure=["total", "increase", "decrease", "total"]
))
waterfall_fig.update_layout(title="Waterfall Chart")
st.plotly_chart(waterfall_fig)

# Radial Chart (Example Data)
radial_fig = px.pie(filtered_data, values='Value', names='Country', title="Radial Chart")
st.plotly_chart(radial_fig)

# Donut Chart
donut_fig = px.pie(filtered_data, values='Value', names='Country', hole=0.4, title="Donut Chart")
st.plotly_chart(donut_fig)

# To run the app, use the command: streamlit run <filename>.py

