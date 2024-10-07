import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.express as px

# Load Data
data = pd.read_csv("Project Dataset.csv")

# Sample Data
sd = data.sample(n=3001, random_state=55027)
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]
catd = sd[['Country', 'Product', 'Import_Export', 'Category', 'Port', 'Customs_Code', 'Shipping_Method', 'Supplier', 'Customer', 'Payment_Terms']]

# Streamlit Title
st.title('Data Analysis Dashboard')

# Slicers
country_filter = st.sidebar.multiselect("Select Countries", options=catd['Country'].unique())
product_filter = st.sidebar.multiselect("Select Products", options=catd['Product'].unique())
category_filter = st.sidebar.multiselect("Select Categories", options=catd['Category'].unique())

# Filter data based on selections
if country_filter:
    catd = catd[catd['Country'].isin(country_filter)]
if product_filter:
    catd = catd[catd['Product'].isin(product_filter)]
if category_filter:
    catd = catd[catd['Category'].isin(category_filter)]

# Display Sample Data
st.subheader('Sample Data')
st.dataframe(catd)

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

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X['Quantity'], y, color='blue', label='Actual Data')
ax.plot(X['Quantity'], y_pred_lin, color='red', label='Linear Regression')
ax.set_xlabel('Quantity')
ax.set_ylabel('Value')
ax.set_title('Linear Regression: Value as Dependent Variable')
ax.legend()
st.pyplot(fig)

# Polynomial Regression
st.subheader('Polynomial Regression Analysis')

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X['Quantity'], y, color='blue', label='Actual Data')
ax.plot(X['Quantity'], y_pred_poly, color='green', label='Polynomial Regression')
ax.set_xlabel('Quantity')
ax.set_ylabel('Value')
ax.set_title('Polynomial Regression: Value as Dependent Variable')
ax.legend()
st.pyplot(fig)

# Box Plots
st.subheader('Box-Whisker Plots')
fig, ax = plt.subplots(figsize=(10, 6))
ncd.boxplot(column=['Quantity', 'Value', 'Weight'], showmeans=True, ax=ax)
ax.set_title('Box-Whisker Plot for Quantity, Value, and Weight')
st.pyplot(fig)

# Scatter Plots
st.subheader('Scatter Plots')
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].scatter(ncd['Quantity'], ncd['Value'], alpha=0.5)
ax[0].set_xlabel('Quantity')
ax[0].set_ylabel('Value')
ax[0].set_title('Quantity vs Value')

ax[1].scatter(ncd['Quantity'], ncd['Weight'], alpha=0.5)
ax[1].set_xlabel('Quantity')
ax[1].set_ylabel('Weight')
ax[1].set_title('Quantity vs Weight')

ax[2].scatter(ncd['Value'], ncd['Weight'], alpha=0.5)
ax[2].set_xlabel('Value')
ax[2].set_ylabel('Weight')
ax[2].set_title('Value vs Weight')

st.pyplot(fig)

# Monthly Trends
st.subheader('Monthly Trends')
ncd['Date'] = pd.to_datetime(ncd['Date'], format='%d-%m-%Y')
monthly_data = ncd.resample('M', on='Date').mean()

# Plot for Quantity
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(monthly_data.index, monthly_data['Quantity'], label='Average Quantity', marker='o')
ax.set_xlabel('Month')
ax.set_ylabel('Average Quantity')
ax.set_title('Monthly Trend of Quantity')
ax.legend()
st.pyplot(fig)

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
for col in catd.columns:
    value_counts = catd[col].value_counts()
    min_freqs.append(value_counts.min())
    max_freqs.append(value_counts.max())

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Minimum Frequencies
ax[0].bar(catd.columns, min_freqs)
ax[0].set_xlabel('Variables')
ax[0].set_ylabel('Minimum Frequency')
ax[0].set_title('Minimum Frequencies')
ax[0].set_xticklabels(catd.columns, rotation=45)

# Maximum Frequencies
ax[1].bar(catd.columns, max_freqs)
ax[1].set_xlabel('Variables')
ax[1].set_ylabel('Maximum Frequency')
ax[1].set_title('Maximum Frequencies')
ax[1].set_xticklabels(catd.columns, rotation=45)

plt.tight_layout()
st.pyplot(fig)

# Pie and Bar Charts for Categorical Variables
variables = ['Import_Export', 'Category', 'Shipping_Method', 'Payment_Terms']
for var in variables:
    value_counts = catd[var].value_counts()
    
    fig = plt.figure(figsize=(12, 5))
    
    # Pie Chart
    plt.subplot(1, 2, 1)
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'Pie Chart of {var}')
    
    # Bar Chart
    plt.subplot(1, 2, 2)
    plt.bar(value_counts.index, value_counts.values)
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Bar Chart of {var}')
    plt.xticks(rotation=45)

    st.pyplot(fig)

# Choropleth Map for Countries
st.subheader('Interactive Map of Countries')
country_counts = catd['Country'].value_counts()
fig = px.choropleth(
    locations=country_counts.index,
    locationmode='country names',
    color=country_counts.values,
    color_continuous_scale='Viridis',
    title='Interactive Map of Countries Present in the Dataset'
)
st.plotly_chart(fig)

# Run the app
if __name__ == "__main__":
    st.run()

