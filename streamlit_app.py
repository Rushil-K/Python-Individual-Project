import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats
import numpy as np
import networkx as nx

# Load data (replace 'data.csv' with your actual file or data source)
ncd = pd.read_csv('Project Dataset.csv')
ncd['Date'] = pd.to_datetime(ncd['Date'], format='%d-%m-%Y')
ncd = ncd.sort_values(by='Date')

# Sidebar for selection
st.sidebar.header('Visualization Settings')
selected_visual = st.sidebar.selectbox(
    'Select visualization',
    ['Normality Tests', 'Linear Regression', 'Polynomial Regression', 'Correlation Heatmap', 
     'Box Plots', 'Pair Plot', 'Scatter Plots', 'Time Series (Monthly)', 'Time Series (Yearly)',
     'Categorical Analysis', 'Import-Export Network Graph', 'Interactive Heatmap', 
     'Interactive Box Plots', 'Interactive Scatter Plots', 'World Map of Countries']
)

# Normality Tests
if selected_visual == 'Normality Tests':
    st.header('Normality Tests')
    cols = ['Quantity', 'Value', 'Weight']

    for col in cols:
        st.subheader(f"Shapiro-Wilk Test for {col}")
        stat, p = stats.shapiro(ncd[col])
        st.write(f"Statistic: {stat}, p-value: {p}")

        st.subheader(f"Kolmogorov-Smirnov Test for {col}")
        ks_stat, ks_p = stats.kstest(ncd[col], 'norm', args=(ncd[col].mean(), ncd[col].std()))
        st.write(f"KS Statistic: {ks_stat}, p-value: {ks_p}")

        st.subheader(f"Anderson-Darling Test for {col}")
        result = stats.anderson(ncd[col])
        st.write(f"Statistic: {result.statistic}, Critical Values: {result.critical_values}")

        st.subheader(f"Jarque-Bera Test for {col}")
        jb_stat, jb_p = stats.jarque_bera(ncd[col])
        st.write(f"Statistic: {jb_stat}, p-value: {jb_p}")

        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(ncd[col], kde=True, ax=ax[0])
        ax[0].set_title(f'Histogram of {col}')
        stats.probplot(ncd[col], dist="norm", plot=ax[1])
        ax[1].set_title(f'Q-Q Plot of {col}')
        st.pyplot(fig)

# Linear Regression
if selected_visual == 'Linear Regression':
    st.header('Linear Regression')
    
    case = st.selectbox("Select Case:", ["Value as Dependent", "Quantity as Dependent", "Weight as Dependent"])
    if case == "Value as Dependent":
        X = ncd[['Quantity', 'Weight']]
        y = ncd['Value']
    elif case == "Quantity as Dependent":
        X = ncd[['Value', 'Weight']]
        y = ncd['Quantity']
    else:
        X = ncd[['Value', 'Quantity']]
        y = ncd['Weight']
    
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred = lin_reg.predict(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X.iloc[:, 0], y, color='blue')
    plt.plot(X.iloc[:, 0], y_pred, color='red', label='Linear Regression')
    plt.xlabel(X.columns[0])
    plt.ylabel(y.name)
    plt.title(f'{y.name} vs {X.columns[0]}')
    plt.legend()
    st.pyplot(plt)

# Polynomial Regression
if selected_visual == 'Polynomial Regression':
    st.header('Polynomial Regression')
    
    case = st.selectbox("Select Case:", ["Value as Dependent", "Quantity as Dependent", "Weight as Dependent"])
    if case == "Value as Dependent":
        X = ncd[['Quantity', 'Weight']]
        y = ncd['Value']
    elif case == "Quantity as Dependent":
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

    plt.figure(figsize=(10, 6))
    plt.scatter(X.iloc[:, 0], y, color='blue')
    plt.plot(X.iloc[:, 0], y_pred_poly, color='green', label='Polynomial Regression')
    plt.xlabel(X.columns[0])
    plt.ylabel(y.name)
    plt.title(f'{y.name} vs {X.columns[0]}')
    plt.legend()
    st.pyplot(plt)

# Correlation Heatmap
if selected_visual == 'Correlation Heatmap':
    st.header('Correlation Heatmap')
    correlation_matrix = ncd[['Quantity', 'Value', 'Weight']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)

# Box Plots
if selected_visual == 'Box Plots':
    st.header('Box Plots for Quantity, Value, and Weight')
    ncd.boxplot(column=['Quantity', 'Value', 'Weight'], showmeans=True)
    st.pyplot(plt)

# Pair Plot
if selected_visual == 'Pair Plot':
    st.header('Pair Plot')
    sns.pairplot(ncd[['Quantity', 'Value', 'Weight']])
    st.pyplot(plt)

# Scatter Plots
if selected_visual == 'Scatter Plots':
    st.header('Scatter Plots')
    x_axis = st.selectbox('Select X-axis variable', ['Quantity', 'Value', 'Weight'])
    y_axis = st.selectbox('Select Y-axis variable', ['Quantity', 'Value', 'Weight'])
    plt.scatter(ncd[x_axis], ncd[y_axis], alpha=0.5)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{x_axis} vs {y_axis}')
    st.pyplot(plt)

# Time Series (Monthly and Yearly)
if selected_visual in ['Time Series (Monthly)', 'Time Series (Yearly)']:
    st.header(f'{selected_visual} Trend')
    freq = 'M' if selected_visual == 'Time Series (Monthly)' else 'Y'
    resampled_data = ncd.resample(freq, on='Date').mean()

    col = st.selectbox('Select Column:', ['Quantity', 'Value', 'Weight'])
    plt.figure(figsize=(10, 6))
    plt.plot(resampled_data.index, resampled_data[col], marker='o')
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.title(f'{selected_visual} Trend of {col}')
    st.pyplot(plt)

# Categorical Analysis
if selected_visual == 'Categorical Analysis':
    st.header('Categorical Analysis')
    variables = ['Import_Export', 'Category', 'Shipping_Method', 'Payment_Terms']
    
    for var in variables:
        value_counts = ncd[var].value_counts()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Pie chart
        ax[0].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        ax[0].set_title(f'Pie Chart of {var}')
        
        # Bar chart
        ax[1].bar(value_counts.index, value_counts.values)
        ax[1].set_title(f'Bar Chart of {var}')
        ax[1].tick_params(axis='x', rotation=45)
        
        st.pyplot(fig)

# Import-Export Network Graph
if selected_visual == 'Import-Export Network Graph':
    st.header('Import-Export Network Graph')
    
    graph = nx.DiGraph()
    countries = ncd['Country'].unique()
    graph.add_nodes_from(countries)
    
    for _, row in ncd.iterrows():
        if row['Import_Export'] == 'Import':
            graph.add_edge(row['Country'], 'Your Country')
        else:
            graph.add_edge('Your Country', row['Country'])

    pos = nx.spring_layout(graph)
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue', ax=ax)
    st.pyplot(fig)

# Interactive Heatmap
if selected_visual == 'Interactive Heatmap':
    st.header('Interactive Correlation Heatmap')
    corr_matrix = ncd[['Quantity', 'Value', 'Weight']].corr()
    fig = px.imshow(corr_matrix, text_auto=True, title='Interactive Heatmap')
    st.plotly_chart(fig)

# Interactive Box Plots
if selected_visual == 'Interactive Box Plots':
    st.header('Interactive Box Plots')
    figv = px.box(ncd, y='Value', title='Interactive Box Plot of Value')
    st.plotly_chart(figv)
    
    figw = px.box(ncd, y='Weight', title='Interactive Box Plot of Weight')
    st.plotly_chart(figw)
    
    # Interactive Box Plot of Quantity
    figq = px.box(ncd, y='Quantity', title='Interactive Box Plot of Quantity')
    st.plotly_chart(figq)  # Display the box plot in Streamlit

# Interactive Scatter Plot
if selected_visual == 'Interactive Scatter Plots':
    fig = px.scatter(ncd, x='Quantity', y='Value', color='Weight',
                     hover_data=['Date'], title='Interactive Scatter Plot')
    st.plotly_chart(fig)  # Display the scatter plot in Streamlit

# Interactive Monthly Quantity Trend (Line Plot)
if selected_visual == 'Interactive Monthly Line Plots':
    figq = px.line(monthly_data, x=monthly_data.index, y='Quantity',
                   title='Interactive Monthly Quantity Trend')
    st.plotly_chart(figq)  # Display the line plot for Quantity in Streamlit
    
    # Interactive Monthly Value Trend (Line Plot)
    figv = px.line(monthly_data, x=monthly_data.index, y='Value',
                   title='Interactive Monthly Value Trend')
    st.plotly_chart(figv)  # Display the line plot for Value in Streamlit
    
    # Interactive Monthly Weight Trend (Line Plot)
    figw = px.line(monthly_data, x=monthly_data.index, y='Weight',
                   title='Interactive Monthly Weight Trend')
    st.plotly_chart(figw)  # Display the line plot for Weight in Streamlit

# Create lists to store minimum and maximum frequencies
min_freqs = []
max_freqs = []

for col in catd.columns:
    value_counts = catd[col].value_counts()
    min_freqs.append(value_counts.min())
    max_freqs.append(value_counts.max())

# Minimum and Maximum Frequencies Bar Charts
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Minimum Frequencies
axes[0].bar(catd.columns, min_freqs)
axes[0].set_xlabel('Variables')
axes[0].set_ylabel('Minimum Frequency')
axes[0].set_title('Minimum Frequencies')
axes[0].tick_params(axis='x', rotation=45)

# Maximum Frequencies
axes[1].bar(catd.columns, max_freqs)
axes[1].set_xlabel('Variables')
axes[1].set_ylabel('Maximum Frequency')
axes[1].set_title('Maximum Frequencies')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
st.pyplot(fig)

# List of variables for pie and bar charts
variables = ['Import_Export', 'Category', 'Shipping_Method', 'Payment_Terms']

# Pie and Bar Charts for Variables
for var in variables:
    value_counts = catd[var].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Pie chart
    axes[0].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title(f'Pie Chart of {var}')

    # Bar chart
    axes[1].bar(value_counts.index, value_counts.values)
    axes[1].set_xlabel(var)
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Bar Chart of {var}')
    axes[1].tick_params(axis='x', rotation=45, ha='right')

    plt.tight_layout()
    st.pyplot(fig)

# Histograms for Categorical Variables
for column in catd.columns:
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=catd)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Plotly Choropleth Map
countries = sd['Country'].unique().tolist()
fig = px.choropleth(
    locations=countries,
    locationmode='country names',
    color=sd['Country'].value_counts().sort_index(),
    color_continuous_scale='Viridis',
    title='Interactive Map of Countries in the Dataset'
)
fig.update_layout(width=1000, height=600)
st.plotly_chart(fig)

# Plotly Choropleth Map for Import-Export Movements
country_movements = pd.crosstab(catd['Country'], catd['Import_Export'])
country_movements['Total'] = country_movements.sum(axis=1)
fig = go.Figure(data=go.Choropleth(
    locations=country_movements.index,
    z=country_movements['Total'],
    locationmode='country names',
    colorscale='Reds',
    colorbar_title="Total Movements",
))
fig.update_layout(title_text='Import-Export Movements on World Map', width=1000, height=600)
st.plotly_chart(fig)

# Plotly Network Graph for Import-Export
graph = nx.DiGraph()
graph.add_nodes_from(catd['Country'].unique())

for _, row in catd.iterrows():
    if row['Import_Export'] == 'Import':
        graph.add_edge(row['Country'], 'Your Country')
    elif row['Import_Export'] == 'Export':
        graph.add_edge('Your Country', row['Country'])

pos = nx.spring_layout(graph)
edge_x, edge_y = [], []
for edge in graph.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
node_x, node_y, node_text = [], [], []
for node in graph.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
    marker=dict(showscale=True, colorscale='YlGnBu', size=10, line_width=2)
)
fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(title='Import-Export Network Graph', width=1000, height=600)
st.plotly_chart(fig)

# Plotly Sunburst Chart
if 'Value' not in catd.columns:
    sunburst_data = pd.merge(catd, sd[['Category', 'Value']], on=['Category'], how='left')
else:
    sunburst_data = catd
fig = px.sunburst(sunburst_data, path=['Category'], values='Value', title='Trade Distribution by Category')
fig.update_layout(width=800, height=600)
st.plotly_chart(fig)

