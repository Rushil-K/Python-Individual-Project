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

# Select numerical columns
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]

# Visualization
# Shapiro-Wilk test for normality
for col in ['Quantity', 'Value', 'Weight']:
    stat, p = stats.shapiro(ncd[col])
    print(f"Shapiro-Wilk Test for {col} -\n Statistic: {stat}, p-value: {p}\n")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(ncd[col], kde=True)
    plt.title(f'Histogram of {col}')
    
    plt.subplot(1, 2, 2)
    stats.probplot(ncd[col], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {col}')
    
    plt.show()

# Kolmogorov-Smirnov test for normality
for col in ['Quantity', 'Value', 'Weight']:
    ks_stat, ks_p = stats.kstest(ncd[col], 'norm', args=(ncd[col].mean(), ncd[col].std()))
    print(f"Kolmogorov-Smirnov Test for {col} -\n KS Statistic: {ks_stat}, p-value: {ks_p}\n")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(ncd[col], kde=True)
    plt.title(f'Histogram of {col}')
    
    plt.subplot(1, 2, 2)
    stats.probplot(ncd[col], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {col}')
    
    plt.show()

# Anderson-Darling test for normality
for col in ['Quantity', 'Value', 'Weight']:
    result = stats.anderson(ncd[col])
    print(f"Anderson-Darling Test for {col} -\n Statistic: {result.statistic},\n Critical Values: {result.critical_values},\n Significance Levels: {result.significance_level}\n")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(ncd[col], kde=True)
    plt.title(f'Histogram of {col}')
    
    plt.subplot(1, 2, 2)
    stats.probplot(ncd[col], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {col}')
    
    plt.show()

# Jarque-Bera Test for normality
for col in ['Quantity', 'Value', 'Weight']:
    jb_stat, jb_p = stats.jarque_bera(ncd[col])
    print(f"Jarque-Bera Test for {col} -\n Statistic: {jb_stat}, p-value: {jb_p}\n")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(ncd[col], kde=True)
    plt.title(f'Histogram of {col}')
    
    plt.subplot(1, 2, 2)
    stats.probplot(ncd[col], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {col}')
    
    plt.show()

# Linear Regression
# Case 1: Value as dependent variable
X = ncd[['Quantity', 'Weight']]
y = ncd['Value']

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X['Quantity'], y, color='blue')
plt.plot(X['Quantity'], y_pred_lin, color='red', label='Linear Regression')
plt.xlabel('Quantity')
plt.ylabel('Value')
plt.title('Case 1: Linear Regression')
plt.legend()
plt.show()

# Case 2: Quantity as dependent variable
X = ncd[['Value', 'Weight']]
y = ncd['Quantity']

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X['Value'], y, color='blue')
plt.plot(X['Value'], y_pred_lin, color='red', label='Linear Regression')
plt.xlabel('Value')
plt.ylabel('Quantity')
plt.title('Case 2: Linear Regression')
plt.legend()
plt.show()

# Case 3: Weight as dependent variable
X = ncd[['Value', 'Quantity']]
y = ncd['Weight']

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X['Value'], y, color='blue')
plt.plot(X['Value'], y_pred_lin, color='red', label='Linear Regression')
plt.xlabel('Value')
plt.ylabel('Weight')
plt.title('Case 3: Linear Regression')
plt.legend()
plt.show()

# Polynomial Regression
# Case 1: Value as dependent variable
X = ncd[['Quantity', 'Weight']]
y = ncd['Value']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X['Quantity'], y, color='blue')
plt.plot(X['Quantity'], y_pred_poly, color='green', label='Polynomial Regression')
plt.xlabel('Quantity')
plt.ylabel('Value')
plt.title('Case 1: Polynomial Regression')
plt.legend()
plt.show()

# Case 2: Quantity as dependent variable
X = ncd[['Value', 'Weight']]
y = ncd['Quantity']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X['Value'], y, color='blue')
plt.plot(X['Value'], y_pred_poly, color='green', label='Polynomial Regression')
plt.xlabel('Value')
plt.ylabel('Quantity')
plt.title('Case 2: Polynomial Regression')
plt.legend()
plt.show()

# Case 3: Weight as dependent variable
X = ncd[['Value', 'Quantity']]
y = ncd['Weight']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X['Value'], y, color='blue')
plt.plot(X['Value'], y_pred_poly, color='green', label='Polynomial Regression')
plt.xlabel('Value')
plt.ylabel('Weight')
plt.title('Case 3: Polynomial Regression')
plt.legend()
plt.show()

# Box-Whisker Plot
ncd.boxplot(column=['Quantity', 'Value', 'Weight'], showmeans=True)
plt.title('Box-Whisker Plot for Quantity, Value, and Weight')
plt.show()

for column in ncd.select_dtypes(include=np.number).columns:
    plt.figure()  # Create a new figure for each plot
    ncd.boxplot(column=[column], showmeans=True)
    plt.title(f'Box-Whisker Plot for {column}')
    plt.show()

sns.pairplot(ncd)
plt.show()

plt.scatter(ncd['Quantity'], ncd['Value'], alpha=0.5)
plt.xlabel('Quantity')
plt.ylabel('Value')
plt.title('Scatter Plot of Quantity vs Value')
plt.show()

plt.scatter(ncd['Quantity'], ncd['Weight'], alpha=0.5)
plt.xlabel('Quantity')
plt.ylabel('Weight')
plt.title('Scatter Plot of Quantity vs Weight')
plt.show()

plt.scatter(ncd['Value'], ncd['Weight'], alpha=0.5)
plt.xlabel('Value')
plt.ylabel('Weight')
plt.title('Scatter Plot of Value vs Weight')
plt.show()

# Monthly trend analysis
ncd['Date'] = pd.to_datetime(ncd['Date'], format='%d-%m-%Y')
ncd = ncd.sort_values(by='Date')

monthly_data = ncd.resample('ME', on='Date').mean()

plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['Quantity'], label='Quantity Trend', marker='o')
plt.xlabel('Monthly')
plt.ylabel('Average Quantity')
plt.title('Monthly Trend of Quantity')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['Value'], label='Value Trend', marker='o')
plt.xlabel('Monthly')
plt.ylabel('Average Value')
plt.title('Monthly Trend of Value')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['Weight'], label='Weight Trend', marker='o')
plt.xlabel('Monthly')
plt.ylabel('Average Weight')
plt.title('Monthly Trend of Weight')
plt.legend()
plt.show()

# Yearly trend analysis
yearly_data = ncd.resample('YE', on='Date').mean()

plt.figure(figsize=(10, 6))
plt.plot(yearly_data.index, yearly_data['Quantity'], label='Quantity Trend', marker='o')
plt.xlabel('Yearly')
plt.ylabel('Average Quantity')
plt.title('Yearly Trend of Quantity')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(yearly_data.index, yearly_data['Value'], label='Value Trend', marker='o')
plt.xlabel('Yearly')
plt.ylabel('Average Value')
plt.title('Yearly Trend of Value')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(yearly_data.index, yearly_data['Weight'], label='Weight Trend', marker='o')
plt.xlabel('Yearly')
plt.ylabel('Average Weight')
plt.title('Yearly Trend of Weight')
plt.legend()
plt.show()

# Correlation Analysis
correlation_matrix = ncd[['Quantity', 'Value', 'Weight']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Categorical Data Analysis
category_counts = ncd['Category'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Category Counts')
plt.xticks(rotation=90)
plt.show()

# Geospatial Analysis
# Choropleth Map
# You might need to adjust based on your dataset for country mapping
fig = px.choropleth(ncd, 
                    locations='Country', 
                    locationmode='country names',
                    color='Value',
                    hover_name='Country',
                    title='Choropleth Map of Values by Country')
fig.show()

# Network Graph
G = nx.from_pandas_edgelist(ncd, source='Country', target='Partner Country', edge_attr='Value', create_using=nx.DiGraph())
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=700)
nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title('Network Graph of Import/Export Relationships')
plt.show()
