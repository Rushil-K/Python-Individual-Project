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

# Sample Data
data = pd.read_csv("Project Dataset.csv")
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
monthly_data = ncd.resample('ME', on='Date').mean()

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

# Categorical Histogram
st.subheader('Categorical Histograms')
for column in catd[['Import_Export', 'Category', 'Shipping_Method', 'Payment_Terms']].columns:
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    sns.countplot(x=column, data=catd)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the current figure

# Create a world map
st.subheader('World Map of Countries')
countries = sd['Country'].unique().tolist()
fig = px.choropleth(
    locations=countries,
    locationmode='country names',
    color=sd['Country'].value_counts().sort_index(),  # Color based on frequency
    color_continuous_scale='Viridis',  # Choose a color scale
    title='Interactive Map of Countries present in the dataset'
)

# Update layout for larger size
fig.update_layout(
    width=1000,  # Adjust width as desired
    height=600   # Adjust height as desired
)

# Show the map
st.plotly_chart(fig)

# Import-Export Movements on World Map
country_movements = pd.crosstab(catd['Country'], catd['Import_Export'])
country_movements['Total'] = country_movements.sum(axis=1)  # Calculate total movements for each country

fig = go.Figure(data=go.Choropleth(
    locations=country_movements.index,
    z=country_movements['Total'],
    locationmode='country names',
    colorscale='Reds',  # Choose a suitable color scale
        colorbar_title="Total Movements",
))

fig.update_layout(
    title_text='Import-Export Movements on World Map',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

# Update layout for larger size
fig.update_layout(
    width=1000,  # Adjust width as desired
    height=600   # Adjust height as desired
)

# Show the map
st.plotly_chart(fig)

# Network Graph Visualization (if applicable)
st.subheader('Network Graph of Relationships')
G = nx.Graph()
# Example: Add edges between Supplier and Customer
for index, row in catd.iterrows():
    G.add_edge(row['Supplier'], row['Customer'])

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
plt.title('Network Graph of Suppliers and Customers')
st.pyplot(plt.gcf())
plt.clf()  # Clear the current figure

# Conclusion
st.subheader('Conclusion')
st.write("This dashboard provides a comprehensive analysis of the import-export dataset, including statistical tests, visualizations, and interactive charts to understand the relationships and trends within the data.")

# Footer
st.write("Developed by Rushil Kohli")
