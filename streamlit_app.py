import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Project Dataset.csv")
    sample_data = df.sample(n=3001, random_state=55027)
    return sample_data

# Main function
def main():
    st.title("Comprehensive Dashboard: Import/Export Analysis of various Countries")

    # Load data
    df = load_data()

    # Format date
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

    # Sidebar filters
    st.sidebar.header("Filters")
    selected_country = st.sidebar.multiselect("Select Country:", options=df['Country'].unique())
    selected_import_export = st.sidebar.multiselect("Select Import/Export:", options=df['Import_Export'].unique())

    # Sort Top/Bottom countries based on trade value
    st.sidebar.subheader("Sort Countries by Value")
    sort_order = st.sidebar.selectbox("Select Order", ["Top", "Bottom"])
    num_countries = st.sidebar.slider("Number of Countries", min_value=1, max_value=20, value=5)

    # Filter and sort data based on the selected order and number of countries
    country_value_data = df.groupby('Country').sum(numeric_only=True).reset_index()
    if sort_order == "Top":
        country_value_data = country_value_data.nlargest(num_countries, 'Value')
    else:
        country_value_data = country_value_data.nsmallest(num_countries, 'Value')

    # Apply country and import/export filters
    if selected_country:
        df = df[df['Country'].isin(selected_country)]
    if selected_import_export:
        df = df[df['Import_Export'].isin(selected_import_export)]

    # Use tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["World Map", "Charts", "Key Metrics", "Advanced Visualizations", "Managerial Insights"])

    # Tab 1: World Map Visualization
    with tab1:
        st.subheader("World Map: Trade Value by Country")
        world_map_data = country_value_data

        fig_map = px.choropleth(world_map_data,
                                locations='Country',
                                locationmode='country names',
                                color='Value',
                                hover_name='Country',
                                color_continuous_scale=px.colors.sequential.Plasma,
                                title="Trade Value by Country")

        st.plotly_chart(fig_map)

    # Tab 2: Basic Charts
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bar Chart: Selected Country Distribution")
            fig_bar = px.bar(country_value_data, x='Country', y='Value', labels={'x': 'Country', 'y': 'Value'})
            st.plotly_chart(fig_bar)

        with col2:
            st.subheader("Pie Chart: Shipping Method Distribution")
            pie_data = df['Shipping_Method'].value_counts()
            fig_pie = px.pie(pie_data, names=pie_data.index, values=pie_data.values)
            st.plotly_chart(fig_pie)

    # Continue with the rest of your tabs (Tab 3, Tab 4, Tab 5)...

# Run the app
if __name__ == '__main__':
    main()
