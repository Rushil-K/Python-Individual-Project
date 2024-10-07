import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Load Data
data = pd.read_csv("Project Dataset.csv")

# Sample Data
sd = data.sample(n=3001, random_state=55027)
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]
catd = sd[['Country', 'Product', 'Import_Export', 'Category', 'Port', 'Customs_Code', 'Shipping_Method', 'Supplier', 'Customer', 'Payment_Terms']]

# Streamlit Title
st.title('Data Analysis Dashboard')

# Slicers
st.sidebar.header("Filters")
country_filter = st.sidebar.multiselect("Select Countries", options=catd['Country'].unique())
product_filter = st.sidebar.multiselect("Select Products", options=catd['Product'].unique())
category_filter = st.sidebar.multiselect("Select Categories", options=catd['Category'].unique())

# Filter data based on selections
filtered_data = catd.copy()
if country_filter:
    filtered_data = filtered_data[filtered_data['Country'].isin(country_filter)]
if product_filter:
    filtered_data = filtered_data[filtered_data['Product'].isin(product_filter)]
if category_filter:
    filtered_data = filtered_data[filtered_data['Category'].isin(category_filter)]

# Display Sample Data
st.subheader('Sample Data')
st.dataframe(filtered_data)

# Normality Tests
st.subheader('Normality Tests')
for col in ['Quantity', 'Value', 'Weight']:
    stat, p = stats.shapiro(ncd[col])
    st.write(f"**Shapiro-Wilk Test for {col} -** Statistic: {stat}, p-value: {p}")
    
    # Visualize with histogram and Q-Q plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.histplot(ncd[col], kde=True, ax=ax[0])
    ax[0].set_title(f'Histogram of {col}')
    
    stats.probplot(ncd[col], dist="norm", plot=ax[1])
    ax[1].set_title(f'Q-Q Plot of {col}')
    
    st.pyplot(fig)

# Linear Regression
st.subheader('Linear Regression Analysis')
X = ncd[['Quantity', 'Weight']]
y = ncd['Value']

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Plotting Linear Regression
fig = go.Figure()
fig.add_trace(go.Scatter(x=X['Quantity'], y=y, mode='markers', name='Actual Data', marker=dict(color='blue')))
fig.add_trace(go.Scatter(x=X['Quantity'], y=y_pred_lin, mode='lines', name='Linear Regression', line=dict(color='red')))
fig.update_layout(title='Linear Regression: Value as Dependent Variable', xaxis_title='Quantity', yaxis_title='Value')
st.plotly_chart(fig)

# Polynomial Regression
st.subheader('Polynomial Regression Analysis')

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Plotting Polynomial Regression
fig = go.Figure()
fig.add_trace(go.Scatter(x=X['Quantity'], y=y, mode='markers', name='Actual Data', marker=dict(color='blue')))
fig.add_trace(go.Scatter(x=X['Quantity'], y=y_pred_poly, mode='lines', name='Polynomial Regression', line=dict(color='green')))
fig.update_layout(title='Polynomial Regression: Value as Dependent Variable', xaxis_title='Quantity', yaxis_title='Value')
st.plotly_chart(fig)

# Box Plots
st.subheader('Box-Whisker Plots')
fig = go.Figure()
for col in ['Quantity', 'Value', 'Weight']:
    fig.add_trace(go.Box(y=ncd[col], name=col, boxmean='sd'))

fig.update_layout(title='Box-Whisker Plot for Quantity, Value, and Weight')
st.plotly_chart(fig)

# Scatter Plots
st.subheader('Scatter Plots')
fig = go.Figure()
fig.add_trace(go.Scatter(x=ncd['Quantity'], y=ncd['Value'], mode='markers', name='Quantity vs Value', marker=dict(color='blue')))
fig.add_trace(go.Scatter(x=ncd['Quantity'], y=ncd['Weight'], mode='markers', name='Quantity vs Weight', marker=dict(color='red')))
fig.add_trace(go.Scatter(x=ncd['Value'], y=ncd['Weight'], mode='markers', name='Value vs Weight', marker=dict(color='green')))
fig.update_layout(title='Scatter Plots', xaxis_title='X-axis', yaxis_title='Y-axis')
st.plotly_chart(fig)

# Monthly Trends
st.subheader('Monthly Trends')
ncd['Date'] = pd.to_datetime(ncd['Date'], format='%d-%m-%Y')
monthly_data = ncd.resample('M', on='Date').mean()

# Plot for Quantity
fig = go.Figure()
fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['Quantity'], mode='lines+markers', name='Average Quantity'))
fig.update_layout(title='Monthly Trend of Quantity', xaxis_title='Month', yaxis_title='Average Quantity')
st.plotly_chart(fig)

# Correlation Matrix
st.subheader('Correlation Matrix')
correlation_matrix = ncd[['Quantity', 'Value', 'Weight']].corr()
st.write(correlation_matrix)

# Correlation Visualization
fig = px.imshow(correlation_matrix, text_auto=True, title='Correlation Heatmap')
st.plotly_chart(fig)

# Categorical Data Analysis
st.subheader('Categorical Data Analysis')
# Minimum and Maximum Frequencies
min_freqs = []
max_freqs = []
for col in filtered_data.columns:
    value_counts = filtered_data[col].value_counts()
    min_freqs.append(value_counts.min())
    max_freqs.append(value_counts.max())

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Minimum Frequencies
ax[0].bar(filtered_data.columns, min_freqs)
ax[0].set_xlabel('Variables')
ax[0].set_ylabel('Minimum Frequency')
ax[0].set_title('Minimum Frequencies')
ax[0].set_xticklabels(filtered_data.columns, rotation=45)

# Maximum Frequencies
ax[1].bar(filtered_data.columns, max_freqs)
ax[1].set_xlabel('Variables')
ax[1].set_ylabel('Maximum Frequency')
ax[1].set_title('Maximum Frequencies')
ax[1].set_xticklabels(filtered_data.columns, rotation=45)

plt.tight_layout()
st.pyplot(fig)

# Pie and Bar Charts for Categorical Variables
variables = ['Import_Export', 'Category', 'Shipping_Method', 'Payment_Terms']
for var in variables:
    value_counts = filtered_data[var].value_counts()
    
    # Pie Chart
    pie_fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values, hole=.3)])
    pie_fig.update_layout(title_text=f'Pie Chart of {var}')
    st.plotly_chart(pie_fig)

    # Bar Chart
    bar_fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values)])
    bar_fig.update_layout(title_text=f'Bar Chart of {var}', xaxis_title=var, yaxis_title='Frequency')
    st.plotly_chart(bar_fig)

# Import-Export Network Graph
st.subheader('Import-Export Network Graph')
graph = nx.DiGraph()
countries = catd['Country'].unique()
graph.add_nodes_from(countries)

for _, row in catd.iterrows():
    if row['Import_Export'] == 'Import':
        graph.add_edge(row['Country'], 'Your Country')  # Replace 'Your Country' with the actual country
    elif row['Import_Export'] == 'Export':
        graph.add_edge('Your Country', row['Country'])  # Replace 'Your Country' with the actual country

pos = nx.spring_layout(graph)
edge_x = []
edge_y = []
for edge in graph.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_text = []
for node in graph.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=node_text,
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(graph.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append(f'{adjacencies[0]} (# of connections: {len(adjacencies[1])})')

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                     title='Import-Export Network Graph',
                     titlefont_size=16,
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

st.plotly_chart(fig)

# Distribution of Categorical Variables
st.subheader('Distribution of Categorical Variables')
categorical_vars = ['Shipping_Method', 'Supplier', 'Customer', 'Payment_Terms']

for var in categorical_vars:
    fig = px.bar(catd, x=var, title=f'Distribution of {var}',
                  color=var,
                  color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(fig)

# 3D Choropleth Maps
st.subheader('3D Choropleth Maps for Top Imports and Exports')
top_imports = data.sort_values(by='Value', ascending=False).head(100)
top_exports = data.sort_values(by='Value', ascending=False).head(100)

fig_imports = px.choropleth(top_imports,
                             locations='Country',
                             locationmode='country names',
                             color='Value',
                             hover_name='Product',
                             title='Top 100 Imports by Value',
                             projection='orthographic',
                             width=1000,
                             height=800)

fig_exports = px.choropleth(top_exports,
                             locations='Country',
                             locationmode='country names',
                             color='Value',
                             hover_name='Product',
                             title='Top 100 Exports by Value',
                             projection='orthographic',
                             width=1000,
                             height=800)

st.plotly_chart(fig_imports)
st.plotly_chart(fig_exports)

# Animation for Category vs Value over Time
st.subheader('Animated Bar Chart for Category vs Value Over Time')
fig = px.bar(sd, x='Category', y='Value', color='Category', animation_frame='Date')
st.plotly_chart(fig)

# Scatter Plot for Quantity vs Value
st.subheader('Scatter Plot for Quantity vs Value')
fig = px.scatter(sd, x='Quantity', y='Value', color='Country', hover_data=['Product'])
st.plotly_chart(fig)

# Sankey Diagram
st.subheader('Trade Flow Sankey Diagram')
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="green", width=0.5),
        label=["Country A", "Country B", "Product X", "Product Y"],
        color="blue"
    ),
    link=dict(
        source=[0, 1, 0, 2, 3, 3],  # indices correspond to labels
        target=[2, 3, 3, 4, 4, 5],
        value=[8, 4, 2, 8, 4, 2]
    ))])

fig.update_layout(title_text="Trade Flow Sankey Diagram", font_size=10)
st.plotly_chart(fig)

# Sunburst Chart
st.subheader('Trade Distribution by Category (Sunburst)')
if 'Value' not in catd.columns:
    sunburst_data = pd.merge(catd, sd[['Category', 'Value']], on=['Category'], how='left')
else:
    sunburst_data = catd  # 'catd' already has 'Value'

fig = px.sunburst(sunburst_data, path=['Category'], values='Value',
                  title='Trade Distribution by Category')

fig.update_layout(width=800, height=600)
st.plotly_chart(fig)
