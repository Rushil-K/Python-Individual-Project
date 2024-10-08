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

# Cached filtered data
cached_filtered_data = {}

def get_filtered_data(country_filter, product_filter, category_filter):
  key = (tuple(country_filter), tuple(product_filter), tuple(category_filter))
  if key not in cached_filtered_data:
    filtered_data = catd.copy()
    if country_filter:
      filtered_data = filtered_data[filtered_data['Country'].isin(country_filter)]
    if product_filter:
      filtered_data = filtered_data[filtered_data['Product'].isin(product_filter)]
    if category_filter:
      filtered_data = filtered_data[filtered_data['Category'].isin(category_filter)]
    cached_filtered_data[key] = filtered_data
  return cached_filtered_data[key]

# Display Sample Data
st.subheader('Sample Data')
filtered_data = get_filtered_data(country_filter, product_filter, category_filter)
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
  st.plotly(pie_fig)
  
