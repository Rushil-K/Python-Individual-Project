import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

if uploaded_file is not None:
    # Read the CSV file
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

    # Monthly Trends
    st.header("Monthly Trends")
    ncd['Date'] = pd.to_datetime(ncd['Date'], format='%d-%m-%Y')
    ncd = ncd.sort_values(by='Date')
    monthly_data = ncd.resample('ME', on='Date').mean()

    for col in ['Quantity', 'Value', 'Weight']:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(monthly_data.index, monthly_data[col], label=f'{col} Trend', marker='o')
        ax.set_xlabel('Monthly')
        ax.set_ylabel(f'Average {col}')
        ax.set_title(f'Monthly Trend of {col}')
        ax.legend()
        st.pyplot(fig)

    # Yearly Trends
    st.header("Yearly Trends")
    yearly_data = ncd.resample('YE', on='Date').mean()

    for col in ['Quantity', 'Value', 'Weight']:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(yearly_data.index, yearly_data[col], label=f'{col} Trend', marker='o')
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Average {col}')
        ax.set_title(f'Yearly Trend of {col}')
        ax.legend()
        st.pyplot(fig)
      
    # Correlation visualization heatmap
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    # Interactive Correlation Heatmap with Plotly
    st.subheader("Interactive Correlation Heatmap")
    corr_matrix_plotly = pd.DataFrame(correlation_matrix)
    fig = px.imshow(corr_matrix_plotly.values,
                    x=corr_matrix_plotly.columns,
                    y=corr_matrix_plotly.index,
                    title='Interactive Correlation Heatmap')
    st.plotly_chart(fig)

    # Interactive Box Plots
    st.subheader("Interactive Box Plots")
    figv = px.box(ncd, y='Value', title='Interactive Box Plot of Value')
    st.plotly_chart(figv)

    figw = px.box(ncd, y='Weight', title='Interactive Box Plot of Weight')
    st.plotly_chart(figw)

    figq = px.box(ncd, y='Quantity', title='Interactive Box Plot of Quantity')
    st.plotly_chart(figq)

    # Interactive Scatter Plot
    st.subheader("Interactive Scatter Plot")
    scatter_fig = px.scatter(ncd, x='Quantity', y='Value', color='Weight',
                              hover_data=['Date'], title='Interactive Scatter Plot')
    st.plotly_chart(scatter_fig)

    # Monthly Data for Trends
    monthly_data = ncd.resample('ME', on='Date').mean()

    # Interactive Monthly Trend Plots
    st.subheader("Interactive Monthly Trends")
    figq = px.line(monthly_data, x=monthly_data.index, y='Quantity',
                   title='Interactive Monthly Quantity Trend')
    st.plotly_chart(figq)

    figv = px.line(monthly_data, x=monthly_data.index, y='Value',
                   title='Interactive Monthly Value Trend')
    st.plotly_chart(figv)

    figw = px.line(monthly_data, x=monthly_data.index, y='Weight',
                   title='Interactive Monthly Weight Trend')
    st.plotly_chart(figw)

    # Selecting specific columns
    catd = sd[['Country', 'Product', 'Import_Export', 'Category', 
                'Port', 'Customs_Code', 'Weight', 
                'Shipping_Method', 'Supplier', 'Customer', 
                'Payment_Terms']]

    # Create lists to store minimum and maximum frequencies
    min_freqs = []
    max_freqs = []

    # Calculate min and max frequencies for each categorical column
    for col in catd.columns:
        value_counts = catd[col].value_counts()
        min_freqs.append(value_counts.min())
        max_freqs.append(value_counts.max())

    # Create separate bar charts for minimum and maximum frequencies
    st.header("Frequency Analysis")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Minimum Frequencies
    ax[0].bar(catd.columns, min_freqs)
    ax[0].set_xlabel('Variables')
    ax[0].set_ylabel('Minimum Frequency')
    ax[0].set_title('Minimum Frequencies')
    ax[0].tick_params(axis='x', rotation=45)

    # Maximum Frequencies
    ax[1].bar(catd.columns, max_freqs)
    ax[1].set_xlabel('Variables')
    ax[1].set_ylabel('Maximum Frequency')
    ax[1].set_title('Maximum Frequencies')
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    # List of variables for pie and bar charts
    variables = ['Import_Export', 'Category', 'Shipping_Method', 'Payment_Terms']

    # Iterate over the specified variables to create pie and bar charts
    for var in variables:
        # Count the frequency of each category
        value_counts = catd[var].value_counts()

        # Create a figure for pie and bar charts
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Pie chart
        ax[0].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        ax[0].set_title(f'Pie Chart of {var}')

        # Bar chart
        ax[1].bar(value_counts.index, value_counts.values)
        ax[1].set_xlabel(var)
        ax[1].set_ylabel('Frequency')
        ax[1].set_title(f'Bar Chart of {var}')
        ax[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

    # Create histograms for each categorical variable
    st.header("Histograms of Categorical Variables")
    for column in catd.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=column, data=catd)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()
        st.pyplot(plt)

    # Create an interactive world map showing the countries present in the dataset
    st.header("Interactive Map of Countries")
    countries = sd['Country'].unique().tolist()
    
    # Create a world map
    fig = px.choropleth(
        locations=countries,
        locationmode='country names',
        color=sd['Country'].value_counts().sort_index(),  # Color based on frequency
        color_continuous_scale='Viridis',  # Choose a color scale
        title='Interactive Map of Countries Present in the Dataset'
    )

    # Update layout for larger size
    fig.update_layout(width=1000, height=600)
    st.plotly_chart(fig)

    # Cross-tabulation for Import-Export movements
    catd = sd[['Country', 'Product', 'Import_Export', 'Category', 
                'Port', 'Customs_Code', 'Shipping_Method', 
                'Supplier', 'Customer', 'Payment_Terms']]

    country_movements = pd.crosstab(catd['Country'], catd['Import_Export'])
    country_movements['Total'] = country_movements.sum(axis=1)  # Calculate total movements for each country

    # Create an interactive choropleth map for import-export movements
    st.header("Import-Export Movements on World Map")
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
        ),
        width=1000,
        height=600
    )
    st.plotly_chart(fig)

    
    # Create a directed graph
    st.header("Import-Export Network Graph")
    graph = nx.DiGraph()
    countries = catd['Country'].unique()
    graph.add_nodes_from(countries)

    for _, row in catd.iterrows():
        if row['Import_Export'] == 'Import':
            graph.add_edge(row['Country'], 'Your Country')  # Replace 'Your Country'
        elif row['Import_Export'] == 'Export':
            graph.add_edge('Your Country', row['Country'])  # Replace 'Your Country'

    pos = nx.spring_layout(graph)

    # Create edge traces
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    # Create node traces
    node_x, node_y, node_text = [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                            marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True, color=[],
                                        size=10, colorbar=dict(thickness=15, title='Node Connections',
                                                               xanchor='left', titleside='right'),
                                        line_width=2))

    # Color nodes based on degree
    node_adjacencies = []
    for node, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='Import-Export Network Graph', titlefont_size=16,
                                                                    showlegend=False, hovermode='closest',
                                                                    margin=dict(b=20, l=5, r=5, t=40),
                                                                    xaxis=dict(showgrid=False, zeroline=False,
                                                                               showticklabels=False),
                                                                    yaxis=dict(showgrid=False, zeroline=False,
                                                                               showticklabels=False)))
    st.plotly_chart(fig)

    # Visualize distributions of categorical variables
    st.header("Distributions of Categorical Variables")
    categorical_vars = ['Shipping_Method', 'Supplier', 'Customer', 'Payment_Terms']
    for var in categorical_vars:
        fig = px.bar(catd, x=var, title=f'Distribution of {var}', color=var, color_discrete_sequence=px.colors.qualitative.Dark24)
        st.plotly_chart(fig)

    # Create 3D choropleth maps for top 100 imports and exports by value
    st.header("Top 100 Imports and Exports by Value")
    top_imports = data.sort_values(by='Value', ascending=False).head(100)
    top_exports = data.sort_values(by='Value', ascending=False).head(100)

    fig_imports = px.choropleth(top_imports, locations='Country', locationmode='country names', color='Value',
                                hover_name='Product', title='Top 100 Imports by Value', projection='orthographic',
                                width=1000, height=800)
    fig_exports = px.choropleth(top_exports, locations='Country', locationmode='country names', color='Value',
                                hover_name='Product', title='Top 100 Exports by Value', projection='orthographic',
                                width=1000, height=800)

    st.plotly_chart(fig_imports)
    st.plotly_chart(fig_exports)

    # Scatter plot of Quantity vs Value with Country as color
    st.header("Scatter Plot of Quantity vs Value")
    fig = px.scatter(sd, x='Quantity', y='Value', color='Country', hover_data=['Product'])
    st.plotly_chart(fig)

    # Trade Flow Sankey Diagram
    st.header("Trade Flow Sankey Diagram")
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="green", width=0.5),
            label=["Country A", "Country B", "Product X", "Product Y"],
            color="blue"),
        link=dict(source=[0, 1, 0, 2, 3, 3], target=[2, 3, 3, 4, 4, 5], value=[8, 4, 2, 8, 4, 2]))])
    
    fig.update_layout(title_text="Trade Flow Sankey Diagram", font_size=10)
    st.plotly_chart(fig)

    # Sunburst chart for Trade Distribution by Category
    st.header("Trade Distribution by Category")
    if 'Value' not in catd.columns:
        sunburst_data = pd.merge(catd, sd[['Category', 'Value']], on=['Category'], how='left')
    else:
        sunburst_data = catd

    fig = px.sunburst(sunburst_data, path=['Category'], values='Value', title='Trade Distribution by Category')
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig)
