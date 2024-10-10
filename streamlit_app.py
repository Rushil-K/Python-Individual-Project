import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
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

    # Sidebar filters
    st.sidebar.header("Filters")
    selected_country = st.sidebar.multiselect("Select Country:", options=df['Country'].unique())
    selected_import_export = st.sidebar.multiselect("Select Import/Export:", options=df['Import_Export'].unique())

    if selected_country:
        df = df[df['Country'].isin(selected_country)]
    if selected_import_export:
        df = df[df['Import_Export'].isin(selected_import_export)]

    # Split into categorical and non-categorical data
    categorical_data = df[['Country', 'Import_Export', 'Shipping_Method', 'Payment_Terms']]

    # Use tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Charts", "Statistics", "Advanced Visualizations"])

    # Tab 1: Basic Charts
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bar Chart: Country Distribution")
            bar_data = df['Country'].value_counts()
            fig_bar = px.bar(bar_data, x=bar_data.index, y=bar_data.values, labels={'x': 'Country', 'y': 'Count'})
            st.plotly_chart(fig_bar)

        with col2:
            st.subheader("Pie Chart: Shipping Method Distribution")
            pie_data = df['Shipping_Method'].value_counts()
            fig_pie = px.pie(pie_data, names=pie_data.index, values=pie_data.values)
            st.plotly_chart(fig_pie)

        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Line Chart: Total Value Over Time")
            line_data = df.groupby('Date').sum(numeric_only=True).reset_index()
            fig_line = px.line(line_data, x='Date', y='Value')
            st.plotly_chart(fig_line)

        with col4:
            st.subheader("Scatter Plot: Quantity vs Value")
            fig_scatter = px.scatter(df, x='Quantity', y='Value', color='Country')
            st.plotly_chart(fig_scatter)

    # Tab 2: Summary Statistics
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Box Plot: Value by Country")
            fig_box = px.box(df, x='Country', y='Value')
            st.plotly_chart(fig_box)

        with col2:
            st.subheader("Histogram of Value")
            plt.figure(figsize=(5, 4))
            plt.hist(df['Value'], bins=30, color='blue', alpha=0.7)
            st.pyplot(plt)

        st.subheader("Key Metrics")
        col3, col4 = st.columns(2)

        with col3:
            total_value = df['Value'].sum()
            st.metric("Total Value", f"${total_value:,.2f}")

        with col4:
            avg_value = df['Value'].mean()
            st.metric("Average Value", f"${avg_value:,.2f}")

    # Tab 3: Advanced Visualizations
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Treemap of Value by Country and Import/Export")
            fig_treemap = px.treemap(df, path=['Country', 'Import_Export'], values='Value')
            st.plotly_chart(fig_treemap)

        with col2:
            st.subheader("Waterfall Chart of Value Over Time")
            waterfall_data = df.groupby('Date').sum(numeric_only=True).reset_index()
            waterfall_data['Previous Value'] = waterfall_data['Value'].shift(1).fillna(0)
            waterfall_data['Change'] = waterfall_data['Value'] - waterfall_data['Previous Value']
            waterfall_data['Total'] = waterfall_data['Change'].cumsum()

            fig_waterfall = go.Figure(go.Waterfall(
                name="Waterfall",
                orientation="v",
                x=waterfall_data['Date'],
                y=waterfall_data['Change'],
                connector={"line": {"color": "gray"}},
            ))

            fig_waterfall.update_layout(title="Waterfall Chart of Value", xaxis_title="Date", yaxis_title="Change in Value")
            st.plotly_chart(fig_waterfall)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Violin Plot: Value Distribution")
            fig_violin = px.violin(df, y='Value', box=True, points="all")
            st.plotly_chart(fig_violin)

        with col4:
            st.subheader("Stacked Bar Chart: Value by Country and Import/Export")
            stacked_data = df.groupby(['Country', 'Import_Export']).sum(numeric_only=True).reset_index()
            fig_stacked = px.bar(stacked_data, x='Country', y='Value', color='Import_Export')
            st.plotly_chart(fig_stacked)

# Run the app
if __name__ == '__main__':
    main()
