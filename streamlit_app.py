import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv("Project Dataset.csv")
    sample_data = df.sample(n=3001, random_state=55027)
    return sample_data

# Function to create a Radar Chart
def create_radar_chart(df, categories, title):
    # Prepare data
    values = df[categories].mean().values.flatten().tolist()
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop" and append the start to the end.
    values += values[:1]
    angles += angles[:1]

    # Draw the radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=angles,
        fill='toself',
        name='Average Values',
        marker=dict(color='blue')
    ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )),
        showlegend=True
    )

    return fig

# Main function
def main():
    st.set_page_config(layout="wide")  # Set the layout to wide for better organization
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

    # Create sections with expandable containers
    with st.expander("📊 Visualizations Overview", expanded=True):
        # Organizing the visualizations in two columns
        col1, col2, col3 = st.columns(3)

        # Bar Chart
        with col1:
            st.subheader("Bar Chart")
            bar_data = df['Country'].value_counts()
            fig_bar = px.bar(bar_data, x=bar_data.index, y=bar_data.values, labels={'x': 'Country', 'y': 'Count'})
            st.plotly_chart(fig_bar, key="bar_chart_1")

        # Line Chart
        with col1:
            st.subheader("Line Chart")
            line_data = df.groupby('Date').sum(numeric_only=True).reset_index()
            fig_line = px.line(line_data, x='Date', y='Value', title='Total Value Over Time')
            st.plotly_chart(fig_line, key="line_chart_1")

        # Pie Chart
        with col1:
            st.subheader("Pie Chart")
            pie_data = df['Shipping_Method'].value_counts()
            fig_pie = px.pie(pie_data, names=pie_data.index, values=pie_data.values, title='Shipping Method Distribution')
            st.plotly_chart(fig_pie, key="pie_chart_1")

        # Area Chart
        with col1:
            st.subheader("Area Chart")
            area_data = df.groupby('Date').sum(numeric_only=True).reset_index()
            fig_area = px.area(area_data, x='Date', y='Value', title='Total Value Area Chart')
            st.plotly_chart(fig_area, key="area_chart_1")

        # Scatter Plot
        with col1:
            st.subheader("Scatter Plot")
            fig_scatter = px.scatter(df, x='Quantity', y='Value', color='Country', title='Quantity vs Value Scatter Plot')
            st.plotly_chart(fig_scatter, key="scatter_plot_1")

        # Histogram
        with col2:
            st.subheader("Histogram")
            plt.figure(figsize=(5, 4))
            plt.hist(df['Value'], bins=30, color='blue', alpha=0.7)
            plt.title('Histogram of Value')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            st.pyplot(plt)

        # Single Value Gauge
        with col2:
            st.subheader("Single Value Gauge")
            total_value = df['Value'].sum()
            st.metric(label="Total Value", value=f"${total_value:,.2f}")

        # Box Plot
        with col2:
            st.subheader("Box Plot")
            fig_box = px.box(df, x='Country', y='Value', title='Box Plot of Value by Country')
            st.plotly_chart(fig_box, key="box_plot_1")

        # Word Cloud
        with col2:
            st.subheader("Word Cloud")
            text = " ".join(df['Country'].tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(5, 4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

        # Treemap
        with col3:
            st.subheader("Treemap")
            fig_treemap = px.treemap(df, path=['Country', 'Import_Export'], values='Value', title='Treemap of Value by Country and Import/Export')
            st.plotly_chart(fig_treemap, key="treemap_1")

        # Violin Plot
        with col3:
            st.subheader("Violin Plot")
            fig_violin = px.violin(df, y='Value', box=True, points="all", title='Violin Plot of Value')
            st.plotly_chart(fig_violin, key="violin_plot_1")

        # Funnel Chart
        with col3:
            st.subheader("Funnel Chart")
            funnel_data = df.groupby('Shipping_Method').size().reset_index(name='counts')
            fig_funnel = px.funnel(funnel_data, x='Shipping_Method', y='counts', title='Funnel Chart of Shipping Methods')
            st.plotly_chart(fig_funnel, key="funnel_chart_1")

        # Waterfall Chart
        with col3:
            st.subheader("Waterfall Chart")
            waterfall_data = df.groupby('Date').sum(numeric_only=True).reset_index()
            waterfall_data['Previous Value'] = waterfall_data['Value'].shift(1).fillna(0)
            waterfall_data['Change'] = waterfall_data['Value'] - waterfall_data['Previous Value']
            waterfall_data['Total'] = waterfall_data['Change'].cumsum()
            
            fig_waterfall = go.Figure()
            fig_waterfall.add_trace(go.Waterfall(
                name="Waterfall",
                orientation="v",
                x=waterfall_data['Date'],
                y=waterfall_data['Change'],
                textposition="outside",
                text=waterfall_data['Change'].apply(lambda x: f"{x:,.2f}"),
                connector={"line": {"color": "gray"}},
            ))

            fig_waterfall.update_layout(title="Waterfall Chart of Value", xaxis_title="Date", yaxis_title="Change in Value")
            st.plotly_chart(fig_waterfall, key="waterfall_chart_1")

        # Sparklines
        with col3:
            st.subheader("Sparklines")
            sparklines_data = df.groupby('Date').sum(numeric_only=True)['Value']
            st.line_chart(sparklines_data)

        # Radar Chart
        with col2:
            st.subheader("Radar Chart of Average Metrics")
            radar_categories = ['Quantity', 'Value', 'Weight']  # Select your metrics
            radar_fig = create_radar_chart(df, radar_categories, "Average Metrics by Category")
            st.plotly_chart(radar_fig, key="radar_chart_1")

        # Stacked Bar Chart
        with col3:
            st.subheader("Stacked Bar Chart")
            stacked_data = df.groupby(['Country', 'Import_Export']).sum(numeric_only=True).reset_index()
            fig_stacked = px.bar(stacked_data, x='Country', y='Value', color='Import_Export', title='Stacked Bar Chart of Value by Country')
            st.plotly_chart(fig_stacked, key="stacked_bar_chart_1")

# Run the app
if __name__ == '__main__':
    main()
