import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv("Project Dataset.csv")
    sample_data = df.sample(n=3001, random_state=55027)
    return sample_data

# Main function
def main():
    st.title("Comprehensive Dashboard")
    
    # Load data
    df = load_data()
    
    # Format date
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    
    # Split into categorical and non-categorical data
    non_categorical_data = df[['Quantity', 'Value', 'Date', 'Weight']]
    categorical_data = df[['Country', 'Import_Export', 'Shipping_Method', 'Payment_Terms']]
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_country = st.sidebar.multiselect("Select Country:", options=categorical_data['Country'].unique())
    selected_import_export = st.sidebar.multiselect("Select Import/Export:", options=categorical_data['Import_Export'].unique())
    
    if selected_country:
        df = df[df['Country'].isin(selected_country)]
    if selected_import_export:
        df = df[df['Import_Export'].isin(selected_import_export)]

    # 1. Bar Chart
    st.subheader("Bar Chart")
    bar_data = df['Country'].value_counts()
    fig_bar = px.bar(bar_data, x=bar_data.index, y=bar_data.values, labels={'x': 'Country', 'y': 'Count'})
    st.plotly_chart(fig_bar)

    # 2. Line Chart
    st.subheader("Line Chart")
    line_data = df.groupby('Date').sum().reset_index()
    fig_line = px.line(line_data, x='Date', y='Value', title='Total Value Over Time')
    st.plotly_chart(fig_line)

    # 3. Pie Chart
    st.subheader("Pie Chart")
    pie_data = df['Shipping_Method'].value_counts()
    fig_pie = px.pie(pie_data, names=pie_data.index, values=pie_data.values, title='Shipping Method Distribution')
    st.plotly_chart(fig_pie)

    # 4. Area Chart
    st.subheader("Area Chart")
    area_data = df.groupby('Date').sum().reset_index()
    fig_area = px.area(area_data, x='Date', y='Value', title='Total Value Area Chart')
    st.plotly_chart(fig_area)

    # 5. Scatter Plot
    st.subheader("Scatter Plot")
    fig_scatter = px.scatter(df, x='Quantity', y='Value', color='Country', title='Quantity vs Value Scatter Plot')
    st.plotly_chart(fig_scatter)

    # 6. Data Table
    st.subheader("Data Table")
    st.dataframe(df)

    # 7. Summary Table
    st.subheader("Summary Table")
    summary = df.describe()
    st.dataframe(summary)

    # 8. Single Value Gauge
    st.subheader("Single Value Gauge")
    total_value = df['Value'].sum()
    st.metric(label="Total Value", value=f"${total_value:,.2f}")

    # 9. Heat Map
    st.subheader("Heat Map")
    heat_map_data = df.pivot_table(index='Country', columns='Import_Export', values='Value', aggfunc='sum')
    sns.heatmap(heat_map_data, annot=True, fmt=".0f")
    plt.title('Heat Map of Value by Country and Import/Export')
    st.pyplot(plt)

    # 10. Histogram
    st.subheader("Histogram")
    plt.figure(figsize=(10, 5))
    plt.hist(df['Value'], bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Value')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    # 11. Donut Chart
    st.subheader("Donut Chart")
    donut_data = df['Payment_Terms'].value_counts()
    fig_donut = px.pie(donut_data, names=donut_data.index, values=donut_data.values, hole=0.4, title='Payment Terms Distribution')
    st.plotly_chart(fig_donut)

    # 12. Word Cloud
    st.subheader("Word Cloud")
    text = " ".join(df['Country'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # 13. Box Plot
    st.subheader("Box Plot")
    fig_box = px.box(df, x='Country', y='Value', title='Box Plot of Value by Country')
    st.plotly_chart(fig_box)

    # 14. Funnel Chart (Requires Plotly Express)
    st.subheader("Funnel Chart")
    funnel_data = df.groupby('Shipping_Method').size().reset_index(name='counts')
    fig_funnel = px.funnel(funnel_data, x='Shipping_Method', y='counts', title='Funnel Chart of Shipping Methods')
    st.plotly_chart(fig_funnel)

    # 15. Treemap (Requires Plotly Express)
    st.subheader("Treemap")
    fig_treemap = px.treemap(df, path=['Country', 'Import_Export'], values='Value', title='Treemap of Value by Country and Import/Export')
    st.plotly_chart(fig_treemap)

    # 16. Sparklines
    st.subheader("Sparklines")
    sparklines_data = df.groupby('Date').sum()['Value']
    st.line_chart(sparklines_data)

    # 17. Waterfall Chart
    st.subheader("Waterfall Chart")
    waterfall_data = df.groupby('Date')['Value'].sum().reset_index()
    waterfall_data['Previous Value'] = waterfall_data['Value'].shift(1).fillna(0)
    waterfall_data['Change'] = waterfall_data['Value'] - waterfall_data['Previous Value']
    fig_waterfall = px.waterfall(waterfall_data, x='Date', y='Change', title='Waterfall Chart of Value')
    st.plotly_chart(fig_waterfall)

    # 18. Violin Plot
    st.subheader("Violin Plot")
    fig_violin = px.violin(df, y='Value', box=True, points="all", title='Violin Plot of Value')
    st.plotly_chart(fig_violin)

    # 19. Radial Chart
    st.subheader("Radial Chart")
    radial_data = df['Country'].value_counts()
    fig_radial = px.line_polar(radial_data, r=radial_data.values, theta=radial_data.index, line_close=True, title='Radial Chart of Countries')
    st.plotly_chart(fig_radial)

    # 20. Matrix Chart
    st.subheader("Matrix Chart")
    matrix_data = df.pivot_table(index='Country', columns='Payment_Terms', values='Value', aggfunc='sum')
    sns.heatmap(matrix_data, annot=True, fmt=".0f")
    plt.title('Matrix Chart of Value by Country and Payment Terms')
    st.pyplot(plt)

# Run the app
if __name__ == '__main__':
    main()

