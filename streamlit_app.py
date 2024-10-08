import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.express as px
import networkx as nx

# Load the dataset
data = pd.read_csv("Project Dataset.csv")

# Sample the data
sd = data.sample(n=3001, random_state=55027)

# Ensure Date column is in datetime format
sd['Date'] = pd.to_datetime(sd['Date'], errors='coerce')

# Check if there are any missing values after conversion
sd.dropna(subset=['Date'], inplace=True)

# Select numerical columns
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]

# Visualize distributions and test for normality
for col in ['Quantity', 'Value', 'Weight']:
    # Shapiro-Wilk test for normality
    stat, p = stats.shapiro(ncd[col])
    print(f"Shapiro-Wilk Test for {col} - Statistic: {stat}, p-value: {p}")

    # Histogram and Q-Q Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(ncd[col], kde=True)
    plt.title(f'Histogram of {col}')

    plt.subplot(1, 2, 2)
    stats.probplot(ncd[col], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {col}')
    
    plt.show()

# Linear Regression Analysis
# Case 1: Value as dependent variable
X = ncd[['Quantity', 'Weight']]
y = ncd['Value']

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X['Quantity'], y, color='blue', alpha=0.5, label='Actual Values')
plt.scatter(X['Quantity'], y_pred_lin, color='red', label='Predicted Values', alpha=0.5)
plt.xlabel('Quantity')
plt.ylabel('Value')
plt.title('Case 1: Linear Regression (Value ~ Quantity + Weight)')
plt.legend()
plt.show()

# Polynomial Regression Analysis
# Case 1: Value as dependent variable
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X['Quantity'], y, color='blue', alpha=0.5, label='Actual Values')
plt.scatter(X['Quantity'], y_pred_poly, color='green', label='Polynomial Regression', alpha=0.5)
plt.xlabel('Quantity')
plt.ylabel('Value')
plt.title('Case 1: Polynomial Regression (Value ~ Quantity + Weight)')
plt.legend()
plt.show()

# Box-Whisker Plot
ncd.boxplot(column=['Quantity', 'Value', 'Weight'], showmeans=True)
plt.title('Box-Whisker Plot for Quantity, Value, and Weight')
plt.show()

# Monthly trend analysis
monthly_data = ncd.resample('M', on='Date').mean()

plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['Quantity'], label='Average Quantity', marker='o')
plt.xlabel('Monthly')
plt.ylabel('Average Quantity')
plt.title('Monthly Trend of Quantity')
plt.legend()
plt.show()

# Correlation Analysis
correlation_matrix = ncd[['Quantity', 'Value', 'Weight']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Categorical Data Analysis (ensure your dataset has a categorical column)
if 'Category' in ncd.columns:
    category_counts = ncd['Category'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('Category Counts')
    plt.xticks(rotation=90)
    plt.show()

# Geospatial Analysis
# Check if 'Country' and 'Value' columns exist for choropleth
if 'Country' in ncd.columns and 'Value' in ncd.columns:
    fig = px.choropleth(ncd, 
                        locations='Country', 
                        locationmode='country names',
                        color='Value',
                        hover_name='Country',
                        title='Choropleth Map of Values by Country')
    fig.show()

# Network Graph
if 'Country' in ncd.columns and 'Partner Country' in ncd.columns and 'Value' in ncd.columns:
    G = nx.from_pandas_edgelist(ncd, source='Country', target='Partner Country', edge_attr='Value', create_using=nx.DiGraph())
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title('Network Graph of Import/Export Relationships')
    plt.show()
else:
    print("Network graph requires 'Country', 'Partner Country', and 'Value' columns.")
