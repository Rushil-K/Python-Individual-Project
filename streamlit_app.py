import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("Project Dataset.csv")
    sd = data.sample(n=3001, random_state=55027)
    ncd = sd[['Quantity', 'Value', 'Date', 'Weight']]
    return ncd, sd

ncd, sd = load_data()

# Dashboard Layout
st.title("Data Analysis Dashboard")

st.sidebar.header("Navigation")
pages = ["Summary", "Normality Tests", "Regression Analysis", "Interactive Visualizations"]
page = st.sidebar.selectbox("Choose a section", pages)

if page == "Summary":
    st.subheader("Dataset Summary")
    st.write(ncd.describe())
    st.write("Correlation Matrix:")
    st.write(ncd.corr())
    st.write(sns.pairplot(ncd))

    # Boxplots
    st.subheader("Boxplots for Quantity, Value, and Weight")
    st.write(sns.boxplot(data=ncd[['Quantity', 'Value', 'Weight']], showmeans=True))
    
elif page == "Normality Tests":
    st.subheader("Normality Tests")
    
    for col in ['Quantity', 'Value', 'Weight']:
        # Shapiro-Wilk Test
        stat, p = stats.shapiro(ncd[col])
        st.write(f"Shapiro-Wilk Test for {col}: Statistic={stat}, p-value={p}")

        # Q-Q Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        sns.histplot(ncd[col], kde=True, ax=ax1)
        stats.probplot(ncd[col], dist="norm", plot=ax2)
        st.pyplot(fig)

elif page == "Regression Analysis":
    st.subheader("Linear and Polynomial Regression")

    # Case 1: Value as dependent variable (Linear Regression)
    X = ncd[['Quantity', 'Weight']]
    y = ncd['Value']

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred_lin = lin_reg.predict(X)

    fig_lin, ax_lin = plt.subplots()
    ax_lin.scatter(X['Quantity'], y, color='blue')
    ax_lin.plot(X['Quantity'], y_pred_lin, color='red', label='Linear Regression')
    ax_lin.set_xlabel('Quantity')
    ax_lin.set_ylabel('Value')
    ax_lin.set_title('Linear Regression')
    st.pyplot(fig_lin)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    y_pred_poly = poly_reg.predict(X_poly)

    fig_poly, ax_poly = plt.subplots()
    ax_poly.scatter(X['Quantity'], y, color='blue')
    ax_poly.plot(X['Quantity'], y_pred_poly, color='green', label='Polynomial Regression')
    ax_poly.set_xlabel('Quantity')
    ax_poly.set_ylabel('Value')
    ax_poly.set_title('Polynomial Regression')
    st.pyplot(fig_poly)

elif page == "Interactive Visualizations":
    st.subheader("Interactive Visualizations")

    # Correlation Heatmap
    fig_corr = px.imshow(ncd.corr().values,
                         x=ncd.columns,
                         y=ncd.columns,
                         title="Correlation Heatmap")
    st.plotly_chart(fig_corr)

    # Interactive Scatter Plot
    fig_scatter = px.scatter(ncd, x='Quantity', y='Value', color='Weight',
                             hover_data=['Date'], title='Quantity vs Value')
    st.plotly_chart(fig_scatter)

    # Monthly Trend Line Plot
    ncd['Date'] = pd.to_datetime(ncd['Date'], format='%d-%m-%Y')
    monthly_data = ncd.resample('M', on='Date').mean()

    fig_monthly = px.line(monthly_data, x=monthly_data.index, y='Quantity', title='Monthly Quantity Trend')
    st.plotly_chart(fig_monthly)

    # Interactive World Map
    fig_map = px.choropleth(locations=sd['Country'].unique(),
                            locationmode='country names',
                            color=sd['Country'].value_counts(),
                            title='Countries Present in Dataset')
    st.plotly_chart(fig_map)

    # Import-Export Network Graph
    graph = nx.DiGraph()
    countries = sd['Country'].unique()
    graph.add_nodes_from(countries)
    for _, row in sd.iterrows():
        if row['Import_Export'] == 'Import':
            graph.add_edge(row['Country'], 'Your Country')
        elif row['Import_Export'] == 'Export':
            graph.add_edge('Your Country', row['Country'])

    pos = nx.spring_layout(graph)
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(x=[], y=[], mode='markers', hoverinfo='text', marker=dict(showscale=True))
    fig_network = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='Import-Export Network'))
    st.plotly_chart(fig_network)
