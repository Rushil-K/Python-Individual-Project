import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import networkx as nx

# Load the dataset
df = pd.read_csv('Project Dataset.csv')

# Sample DataFrame based on given instruction
sd = df.sample(n=3001, random_state=55027)
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]
cat = sd[['Country', 'Product', 'Import_Export', 'Category', 'Port', 'Customs_Code', 'Shipping_Method', 'Supplier', 'Customer', 'Payment_Terms']]

# Dashboard Title
st.title("Comprehensive Data Visualization Dashboard: Import Export")

# Sidebar for filters
st.sidebar.title("Filters")
selected_country = st.sidebar.multiselect('Select Country', df['Country'].unique())
selected_product = st.sidebar.multiselect('Select Product', df['Product'].unique())

filtered_df = df[(df['Country'].isin(selected_country)) & (df['Product'].isin(selected_product))] if selected_country and selected_product else df

# Visualization options
st.sidebar.title("Visualizations")
visualization_type = st.sidebar.selectbox(
    'Select Visualization Type', 
    ['Bar Chart', 'Line Chart', 'Pie Chart', 'Scatter Plot', 'Area Chart', 'Heat Map', 
     'Choropleth Map', 'Histogram', 'Bullet Graph', 'Treemap', 'Box Plot', 'Violin Plot', 
     'Donut Chart', 'Waterfall Chart', 'Funnel Chart', 'Radar Chart', 'Stacked Bar', 
     'Timeline', 'Comparison Chart', 'Dendrogram', 'Word Cloud', 'Network Graph', 
     'KPI Cards', 'Summary Tables']
)

# Bar Chart
if visualization_type == 'Bar Chart':
    st.subheader("Bar Chart")
    fig = px.bar(filtered_df, x='Quantity', y='Value', color='Product', title="Bar Chart")
    st.plotly_chart(fig)

# Line Chart
elif visualization_type == 'Line Chart':
    st.subheader("Line Chart")
    fig = px.line(filtered_df, x='Date', y='Value', color='Payment_Terms', title="Line Chart")
    st.plotly_chart(fig)

# Pie Chart
elif visualization_type == 'Pie Chart':
    st.subheader("Pie Chart")
    fig = px.pie(filtered_df, values='Value', names='Category', title="Pie Chart")
    st.plotly_chart(fig)

# Scatter Plot
elif visualization_type == 'Scatter Plot':
    st.subheader("Scatter Plot")
    fig = px.scatter(filtered_df, x='Quantity', y='Value', color='Import_Export', title="Scatter Plot")
    st.plotly_chart(fig)

# Area Chart
elif visualization_type == 'Area Chart':
    st.subheader("Area Chart")
    fig = px.area(filtered_df, x='Date', y='Value', color='Import_Export', title="Area Chart")
    st.plotly_chart(fig)

# Heat Map
elif visualization_type == 'Heat Map':
    st.subheader("Heat Map")
    heat_data = filtered_df.pivot_table(index='Shipping_Method', columns='Category', values='Value', aggfunc='mean')
    sns.heatmap(heat_data, cmap="YlGnBu")
    st.pyplot()

# Choropleth Map
elif visualization_type == 'Choropleth Map':
    st.subheader("Choropleth Map")
    fig = px.choropleth(filtered_df, locations="Country", locationmode='country names', color='Value',
                        hover_name="Country", title="Choropleth Map")
    st.plotly_chart(fig)

# Histogram
elif visualization_type == 'Histogram':
    st.subheader("Histogram")
    fig = px.histogram(filtered_df, x='Value', title="Histogram")
    st.plotly_chart(fig)

# Bullet Graph (simplified using a bar graph)
elif visualization_type == 'Bullet Graph':
    st.subheader("Bullet Graph (simplified)")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = filtered_df['Value'].sum(),
        title = {'text': "Total Value"},
        delta = {'reference': 50000},
        gauge = {'axis': {'range': [None, 100000]}}))
    st.plotly_chart(fig)

# Treemap
elif visualization_type == 'Treemap':
    st.subheader("Treemap")
    fig = px.treemap(filtered_df, path=['Category', 'Payment_Terms'], values='Value', title="Treemap")
    st.plotly_chart(fig)

# Box Plot
elif visualization_type == 'Box Plot':
    st.subheader("Box Plot")
    fig = px.box(filtered_df, x='Quantity', y='Value', color='Category', title="Box Plot")
    st.plotly_chart(fig)

# Violin Plot
elif visualization_type == 'Violin Plot':
    st.subheader("Violin Plot")
    fig = px.violin(filtered_df, y='Value', x='Quantity', color='Category', title="Violin Plot")
    st.plotly_chart(fig)

# Donut Chart
elif visualization_type == 'Donut Chart':
    st.subheader("Donut Chart")
    fig = px.pie(filtered_df, values='Value', names='Category', hole=0.3, title="Donut Chart")
    st.plotly_chart(fig)

# Waterfall Chart
elif visualization_type == 'Waterfall Chart':
    st.subheader("Waterfall Chart")
    fig = go.Figure(go.Waterfall(
        x = filtered_df['Country'].unique(),
        measure = ["relative"]*len(filtered_df['Country'].unique()),
        y = filtered_df.groupby('Country')['Value'].sum(),
        base = 0))
    fig.update_layout(title="Waterfall Chart")
    st.plotly_chart(fig)

# Funnel Chart
elif visualization_type == 'Funnel Chart':
    st.subheader("Funnel Chart")
    fig = px.funnel(filtered_df, x='Payment_Terms', y='Value', color='Category', title="Funnel Chart")
    st.plotly_chart(fig)

# Radar Chart
elif visualization_type == 'Radar Chart':
    st.subheader("Radar Chart")
    categories = ['Quantity', 'Value', 'Weight']
    fig = go.Figure()
    for country in filtered_df['Country'].unique():
        radar_data = filtered_df[filtered_df['Country'] == country][categories].mean().values.flatten()
        fig.add_trace(go.Scatterpolar(r=radar_data, theta=categories, fill='toself', name=country))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Radar Chart")
    st.plotly_chart(fig)

# Stacked Bar Chart
elif visualization_type == 'Stacked Bar':
    st.subheader("Stacked Bar Chart")
    fig = px.bar(filtered_df, x='Category', y='Value', color='Payment_Terms', title="Stacked Bar Chart")
    st.plotly_chart(fig)

# Timeline Visualization
elif visualization_type == 'Timeline':
    st.subheader("Timeline Visualization")
    fig = px.timeline(filtered_df, x_start='Date', x_end='Date', y='Category', color='Payment_Terms', title="Timeline")
    st.plotly_chart(fig)


# Word Cloud
elif visualization_type == 'Word Cloud':
    st.subheader("Word Cloud")
    text = " ".join(cat['Product'].astype(str).values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()

# Network Graph (simplified version)
elif visualization_type == 'Network Graph':
    st.subheader("Network Graph")
    G = nx.Graph()
    G.add_nodes_from(cat['Shipping_Method'].unique())
    G.add_edges_from(zip(cat['Supplier'], cat['Payment_Terms']))
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=10)
    st.pyplot()

# KPI Cards
elif visualization_type == 'KPI Cards':
    st.subheader("KPI Cards")
    total_value = filtered_df['Value'].sum()
    total_quantity = filtered_df['Quantity'].sum()
    st.metric(label="Total Value", value=f"${total_value:,.2f}")
    st.metric(label="Total Quantity", value=f"{total_quantity}")

# Summary Tables
elif visualization_type == 'Summary Tables':
    st.subheader("Summary Table")
    st.write(filtered_df.describe())

# Run the app using the command: streamlit run your_script.py

