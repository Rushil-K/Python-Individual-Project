import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Title of the app
st.title('Data Analysis and Visualization App')

# Load dataset
data = pd.read_csv("Project Dataset.csv")

# Sample data
sd = data.sample(n=3001, random_state=55027)
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]
catd = sd[['Country', 'Product', 'Import_Export', 'Category', 'Port', 'Customs_Code', 'Shipping_Method', 'Supplier', 'Customer', 'Payment_Terms']]

# Normality Tests
st.header('Normality Tests')
for col in ['Quantity', 'Value', 'Weight']:
    st.subheader(f'Shapiro-Wilk Test for {col}')
    stat, p = stats.shapiro(ncd[col])
    st.write(f'Statistic: {stat}, p-value: {p}')

    # Histogram
    fig, ax = plt.subplots()
    sns.histplot(ncd[col], kde=True, ax=ax)
    ax.set_title(f'Histogram of {col}')
    st.pyplot(fig)

    # Q-Q Plot
    fig, ax = plt.subplots()
    stats.probplot(ncd[col], dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot of {col}')
    st.pyplot(fig)

# Linear Regression
st.header('Linear Regression Analysis')
cases = ['Value', 'Quantity', 'Weight']
for case in cases:
    st.subheader(f'Linear Regression with {case} as dependent variable')
    if case == 'Value':
        X = ncd[['Quantity', 'Weight']]
        y = ncd['Value']
    elif case == 'Quantity':
        X = ncd[['Value', 'Weight']]
        y = ncd['Quantity']
    else:
        X = ncd[['Value', 'Quantity']]
        y = ncd['Weight']
    
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred_lin = lin_reg.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X.iloc[:, 0], y, color='blue')
    ax.plot(X.iloc[:, 0], y_pred_lin, color='red', label='Linear Regression')
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(case)
    ax.set_title(f'Linear Regression: {case} as Dependent Variable')
    ax.legend()
    st.pyplot(fig)

# Polynomial Regression
st.header('Polynomial Regression Analysis')
for case in cases:
    st.subheader(f'Polynomial Regression with {case} as dependent variable')
    if case == 'Value':
        X = ncd[['Quantity', 'Weight']]
        y = ncd['Value']
    elif case == 'Quantity':
        X = ncd[['Value', 'Weight']]
        y = ncd['Quantity']
    else:
        X = ncd[['Value', 'Quantity']]
        y = ncd['Weight']
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    y_pred_poly = poly_reg.predict(X_poly)

    fig, ax = plt.subplots()
    ax.scatter(X.iloc[:, 0], y, color='blue')
    ax.plot(X.iloc[:, 0], y_pred_poly, color='green', label='Polynomial Regression')
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(case)
    ax.set_title(f'Polynomial Regression: {case} as Dependent Variable')
    ax.legend()
    st.pyplot(fig)

# Monthly Trends
st.header('Monthly Trends')
ncd['Date'] = pd.to_datetime(ncd['Date'], format='%d-%m-%Y')
ncd = ncd.sort_values(by='Date')
monthly_data = ncd.resample('ME', on='Date').mean()

for metric in ['Quantity', 'Value', 'Weight']:
    fig, ax = plt.subplots()
    ax.plot(monthly_data.index, monthly_data[metric], marker='o')
    ax.set_xlabel('Monthly')
    ax.set_ylabel(f'Average {metric}')
    ax.set_title(f'Monthly Trend of {metric}')
    st.pyplot(fig)

# Correlation Analysis
st.header('Correlation Analysis')
correlation_matrix = ncd[['Quantity', 'Value', 'Weight']].corr()
st.write('Correlation Matrix:')
st.write(correlation_matrix)

# Correlation Heatmap
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, ax=ax)
st.pyplot(fig)


# Interactive Plots
st.header('Interactive Visualizations')
# World Map for Countries
countries = sd['Country'].unique().tolist()
country_counts = sd['Country'].value_counts()
fig = px.choropleth(locations=countries,
                     locationmode='country names',
                     color=country_counts,
                     color_continuous_scale='Viridis',
                     title='Interactive Map of Countries in the Dataset')
st.plotly_chart(fig)

# Interactive Box Plots
for column in ['Value', 'Weight', 'Quantity']:
    fig = px.box(catd, y=column, title=f'Interactive Box Plot of {column}')
    st.plotly_chart(fig)

# Interactive Scatter Plot
fig = px.scatter(catd, x='Quantity', y='Value', color='Weight',
                 hover_data=['Date'], title='Interactive Scatter Plot of Quantity vs Value')
st.plotly_chart(fig)

# Country Movements
st.header('Country Import-Export Movements')
country_movements = pd.crosstab(catd['Country'], catd['Import_Export'])
country_movements['Total'] = country_movements.sum(axis=1)
fig = go.Figure(data=go.Choropleth(
    locations=country_movements.index,
    z=country_movements['Total'],
    locationmode='country names',
    colorscale='Reds',
    colorbar_title="Total Movements",
))

fig.update_layout(
    title_text='Import-Export Movements on World Map',
    geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular')
)
st.plotly_chart(fig)

