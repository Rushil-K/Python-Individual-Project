import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go

# Read the CSV file (directly using file path instead of upload)
data = pd.read_csv('Project Dataset.csv')

# Random sampling of 3001 rows from the dataset
sd = data.sample(n=3001, random_state=55027)

# Selecting specific columns
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]

# Visualization
st.header("Normality Tests and Visualizations")

# Shapiro-Wilk test for normality
for col in ['Quantity', 'Value', 'Weight']:
    stat, p = stats.shapiro(ncd[col])
    st.write(f"Shapiro-Wilk Test for {col} - Statistic: {stat}, p-value: {p}")

    # Visualize with histogram and Q-Q plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(ncd[col], kde=True, ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')

    stats.probplot(ncd[col], dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot of {col}')

    st.pyplot(fig)

# Kolmogorov-Smirnov test for normality
for col in ['Quantity', 'Value', 'Weight']:
    ks_stat, ks_p = stats.kstest(ncd[col], 'norm', args=(ncd[col].mean(), ncd[col].std()))
    st.write(f"Kolmogorov-Smirnov Test for {col} - KS Statistic: {ks_stat}, p-value: {ks_p}")

    # Visualize with histogram and Q-Q plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(ncd[col], kde=True, ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')

    stats.probplot(ncd[col], dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot of {col}')

    st.pyplot(fig)

# Anderson-Darling test for normality
for col in ['Quantity', 'Value', 'Weight']:
    result = stats.anderson(ncd[col])
    st.write(f"Anderson-Darling Test for {col} - Statistic: {result.statistic}, "
             f"Critical Values: {result.critical_values}, "
             f"Significance Levels: {result.significance_level}")

    # Visualize with histogram and Q-Q plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(ncd[col], kde=True, ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')

    stats.probplot(ncd[col], dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot of {col}')

    st.pyplot(fig)

# Jarque-Bera Test for normality
for col in ['Quantity', 'Value', 'Weight']:
    jb_stat, jb_p = stats.jarque_bera(ncd[col])
    st.write(f"Jarque-Bera Test for {col} - Statistic: {jb_stat}, p-value: {jb_p}")

    # Visualize with histogram and Q-Q plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(ncd[col], kde=True, ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')

    stats.probplot(ncd[col], dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot of {col}')

    st.pyplot(fig)

# Linear Regression
st.header("Linear Regression Analysis")

# Case 1: Value as dependent variable
X = ncd[['Quantity', 'Weight']]
y = ncd['Value']

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X['Quantity'], y, color='blue', label='Data Points')
ax.plot(X['Quantity'], y_pred_lin, color='red', label='Linear Regression')
ax.set_xlabel('Quantity')
ax.set_ylabel('Value')
ax.set_title('Case 1: Linear Regression')
ax.legend()
st.pyplot(fig)

# Case 2: Quantity as dependent variable
X = ncd[['Value', 'Weight']]
y = ncd['Quantity']

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X['Value'], y, color='blue', label='Data Points')
ax.plot(X['Value'], y_pred_lin, color='red', label='Linear Regression')
ax.set_xlabel('Value')
ax.set_ylabel('Quantity')
ax.set_title('Case 2: Linear Regression')
ax.legend()
st.pyplot(fig)

# Case 3: Weight as dependent variable
X = ncd[['Value', 'Quantity']]
y = ncd['Weight']

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X['Value'], y, color='blue', label='Data Points')
ax.plot(X['Value'], y_pred_lin, color='red', label='Linear Regression')
ax.set_xlabel('Value')
ax.set_ylabel('Weight')
ax.set_title('Case 3: Linear Regression')
ax.legend()
st.pyplot(fig)

# Polynomial Regression
st.header("Polynomial Regression Analysis")

# Case 1: Value as dependent variable
X = ncd[['Quantity', 'Weight']]
y = ncd['Value']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X['Quantity'], y, color='blue', label='Data Points')
ax.plot(X['Quantity'], y_pred_poly, color='green', label='Polynomial Regression')
ax.set_xlabel('Quantity')
ax.set_ylabel('Value')
ax.set_title('Case 1: Polynomial Regression')
ax.legend()
st.pyplot(fig)

# Case 2: Quantity as dependent variable
X = ncd[['Value', 'Weight']]
y = ncd['Quantity']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X['Value'], y, color='blue', label='Data Points')
ax.plot(X['Value'], y_pred_poly, color='green', label='Polynomial Regression')
ax.set_xlabel('Value')
ax.set_ylabel('Quantity')
ax.set_title('Case 2: Polynomial Regression')
ax.legend()
st.pyplot(fig)

# Case 3: Weight as dependent variable
X = ncd[['Value', 'Quantity']]
y = ncd['Weight']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X['Value'], y, color='blue', label='Data Points')
ax.plot(X['Value'], y_pred_poly, color='green', label='Polynomial Regression')
ax.set_xlabel('Value')
ax.set_ylabel('Weight')
ax.set_title('Case 3: Polynomial Regression')
ax.legend()
st.pyplot(fig)

# Boxplot
st.header("Boxplot Analysis")
fig, ax = plt.subplots(figsize=(10, 6))
ncd.boxplot(column=['Quantity', 'Value', 'Weight'], showmeans=True, ax=ax)
ax.set_title('Box-Whisker Plot for Quantity, Value, and Weight')
st.pyplot(fig)

for column in ncd.select_dtypes(include=np.number).columns:
    fig, ax = plt.subplots()
    ncd.boxplot(column=[column], showmeans=True, ax=ax)
    ax.set_title(f'Box-Whisker Plot for {column}')
    st.pyplot(fig)

# Pairplot
st.header("Pairplot Analysis")
fig = sns.pairplot(ncd)
st.pyplot(fig)

# Scatter Plots
st.header("Scatter Plot Analysis")
fig, ax = plt.subplots()
ax.scatter(ncd['Quantity'], ncd['Value'], alpha=0.5)
ax.set_xlabel('Quantity')
ax.set_ylabel('Value')
ax.set_title('Scatter Plot of Quantity vs Value')
st.pyplot(fig)

fig, ax = plt.subplots()
ax.scatter(ncd['Quantity'], ncd['Weight'], alpha=0.5)
ax.set_xlabel('Quantity')
ax.set_ylabel('Weight')
ax.set_title('Scatter Plot of Quantity vs Weight')
st.pyplot(fig)

fig, ax = plt.subplots()
ax.scatter(ncd['Value'], ncd['Weight'], alpha=0.5)
ax.set_xlabel('Value')
ax.set_ylabel('Weight')
ax.set_title('Scatter Plot of Value vs Weight')
st.pyplot(fig)

# Monthly Trends Visualization
st.header("Monthly Trends Analysis")
ncd['Date'] = pd.to_datetime(ncd['Date'], format='%d-%m-%Y')
ncd['Month'] = ncd['Date'].dt.to_period('M')

# Grouping and plotting
grouped = ncd.groupby('Month')[['Quantity', 'Value', 'Weight']].sum().reset_index()

fig = px.line(grouped, x='Month', y='Quantity', title='Monthly Trends of Quantity')
st.plotly_chart(fig)

fig = px.line(grouped, x='Month', y='Value', title='Monthly Trends of Value')
st.plotly_chart(fig)

fig = px.line(grouped, x='Month', y='Weight', title='Monthly Trends of Weight')
st.plotly_chart(fig)

# Network Graph (using a subset of the dataset)
st.header("Network Graph")
sampled_data = ncd[['Quantity', 'Value']].sample(50)

G = nx.Graph()
for idx, row in sampled_data.iterrows():
    G.add_edge(row['Quantity'], row['Value'])

pos = nx.spring_layout(G)
fig, ax = plt.subplots(figsize=(8, 6))
nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
ax.set_title('Network Graph of Sampled Data')
st.pyplot(fig)
