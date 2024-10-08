import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import statsmodels.api as sm
import networkx as nx

# 1. Load and Cache Data
@st.cache_data
def load_data():
    data = pd.read_csv("Project Dataset.csv")
    return data

data = load_data()

# 2. Sample and filter the data
sd = data.sample(n=3001, random_state=55027)
ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]
catd = sd[['Country','Product','Import_Export','Category','Port','Customs_Code','Shipping_Method','Supplier','Customer','Payment_Terms']]

# 3. Sidebar Filters
st.sidebar.header("Data Filters")

# Date filter
min_date, max_date = st.sidebar.date_input('Select Date Range:', [sd['Date'].min(), sd['Date'].max()])
sd = sd[(sd['Date'] >= pd.to_datetime(min_date)) & (sd['Date'] <= pd.to_datetime(max_date))]

# Filter for categorical fields
selected_country = st.sidebar.multiselect('Select Country:', options=sd['Country'].unique(), default=sd['Country'].unique())
selected_product = st.sidebar.multiselect('Select Product:', options=sd['Product'].unique(), default=sd['Product'].unique())
selected_import_export = st.sidebar.multiselect('Select Import/Export:', options=sd['Import_Export'].unique(), default=sd['Import_Export'].unique())

# Apply filters to the data
sd_filtered = sd[(sd['Country'].isin(selected_country)) & 
                 (sd['Product'].isin(selected_product)) & 
                 (sd['Import_Export'].isin(selected_import_export))]

# 4. Visualization of numeric data
st.header("Visualizations of Numeric Data")

# Shapiro-Wilk Test for Normality with Visuals
st.subheader("Normality Test Results & Visualizations")

for col in ['Quantity', 'Value', 'Weight']:
    stat, p = stats.shapiro(ncd[col])
    st.write(f"Shapiro-Wilk Test for {col}: Statistic={stat:.4f}, p-value={p:.4f}")
    
    # Histogram & Q-Q plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(ncd[col], kde=True, ax=ax[0])
    ax[0].set_title(f'Histogram of {col}')
    stats.probplot(ncd[col], dist="norm", plot=ax[1])
    ax[1].set_title(f'Q-Q Plot of {col}')
    st.pyplot(fig)

# Box Plot Section
st.subheader("Box Plots")
st.plotly_chart(px.box(ncd, y='Value', title='Box Plot of Value'))
st.plotly_chart(px.box(ncd, y='Weight', title='Box Plot of Weight'))
st.plotly_chart(px.box(ncd, y='Quantity', title='Box Plot of Quantity'))

# Pairplot
st.subheader("Pair Plot")
st.write("Scatter plots between Quantity, Value, and Weight.")
sns.pairplot(ncd)
st.pyplot()

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr_matrix = ncd.corr()
st.plotly_chart(px.imshow(corr_matrix.values,
                 x=corr_matrix.columns,
                 y=corr_matrix.index,
                 title='Interactive Correlation Heatmap'))

# 5. Linear & Polynomial Regression
st.header("Regression Analysis")

# Linear Regression (Example: Quantity vs Value)
st.subheader("Linear Regression: Quantity vs Value")
X = ncd[['Quantity', 'Weight']]
y = ncd['Value']
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

fig, ax = plt.subplots()
ax.scatter(X['Quantity'], y, color='blue')
ax.plot(X['Quantity'], y_pred_lin, color='red', label='Linear Regression')
ax.set_xlabel('Quantity')
ax.set_ylabel('Value')
ax.set_title('Linear Regression')
st.pyplot(fig)

# Polynomial Regression (Degree=2)
st.subheader("Polynomial Regression")
degree = st.slider("Select Degree for Polynomial Regression:", min_value=1, max_value=5, value=2)
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

fig, ax = plt.subplots()
ax.scatter(X['Quantity'], y, color='blue')
ax.plot(X['Quantity'], y_pred_poly, color='green', label=f'Polynomial Regression (degree={degree})')
ax.set_xlabel('Quantity')
ax.set_ylabel('Value')
ax.set_title('Polynomial Regression')
st.pyplot(fig)

# 6. Categorical Data Visualizations
st.header("Categorical Data Visualizations")

# Pie and Bar Charts
for var in ['Import_Export', 'Category', 'Shipping_Method', 'Payment_Terms']:
    st.subheader(f"{var} Distribution")
    value_counts = catd[var].value_counts()
    
    fig = plt.figure(figsize=(10, 5))
    
    # Pie chart
    plt.subplot(1, 2, 1)
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'Pie Chart of {var}')
    
    # Bar chart
    plt.subplot(1, 2, 2)
    plt.bar(value_counts.index, value_counts.values)
    plt.title(f'Bar Chart of {var}')
    st.pyplot(fig)

# Country-specific visualizations
st.subheader("Interactive World Map of Import-Export Movements")
countries = sd['Country'].unique().tolist()
fig = px.choropleth(locations=countries, locationmode='country names',
                    color=sd['Country'].value_counts().sort_index(),
                    title='Interactive Map of Countries')
st.plotly_chart(fig)

# Network Graph
st.subheader("Import-Export Network Graph")
graph = nx.DiGraph()
countries = catd['Country'].unique()
graph.add_nodes_from(countries)

# Add edges (import-export relationships)
for _, row in catd.iterrows():
    if row['Import_Export'] == 'Import':
        graph.add_edge(row['Country'], 'Your Country')  # Replace with relevant country
    elif row['Import_Export'] == 'Export':
        graph.add_edge('Your Country', row['Country'])  # Replace with relevant country

# Network Graph Visual
pos = nx.spring_layout(graph)
fig_network = go.Figure(data=[go.Scatter(x=[pos[node][0] for node in graph.nodes()],
                                         y=[pos[node][1] for node in graph.nodes()],
                                         mode='markers+text', text=[node for node in graph.nodes()])])
st.plotly_chart(fig_network)

