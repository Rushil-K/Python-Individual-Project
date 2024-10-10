import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

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

    # Tab layout for better organization
    tab1, tab2, tab3 = st.tabs(["Key Visuals", "Advanced Analysis", "Summary"])

    # Tab 1: Key Visuals
    with tab1:
        st.header("Key Visuals")

        # Bar Chart: Country Distribution
        st.subheader("Bar Chart: Country Distribution")
        bar_data = df['Country'].value_counts()
        fig_bar = px.bar(bar_data, x=bar_data.index, y=bar_data.values,
                         labels={'x': 'Country', 'y': 'Count'},
                         color=bar_data.index,  # Adding colors
                         title="Country-wise Distribution",
                         template="plotly_white",
                         text_auto=True)

        fig_bar.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig_bar.update_layout(margin=dict(t=50, b=50, l=50, r=50), title_x=0.5)
        st.plotly_chart(fig_bar)

        # Line Chart: Total Value Over Time
        st.subheader("Line Chart: Total Value Over Time")
        line_data = df.groupby('Date').sum(numeric_only=True).reset_index()
        fig_line = px.line(line_data, x='Date', y='Value',
                           labels={'Date': 'Date', 'Value': 'Total Value'},
                           title="Total Value Over Time",
                           template="plotly_white",
                           color_discrete_sequence=["#1f77b4"],
                           markers=True)
        
        fig_line.update_traces(line=dict(width=3), marker=dict(size=8))
        fig_line.update_layout(margin=dict(t=50, b=50, l=50, r=50), title_x=0.5)
        st.plotly_chart(fig_line)

        st.markdown("---")

    # Tab 2: Advanced Analysis
    with tab2:
        st.header("Advanced Visualizations")

        # Treemap: Value by Country and Import/Export
        st.subheader("Treemap: Value by Country and Import/Export")
        fig_treemap = px.treemap(df, path=['Country', 'Import_Export'], values='Value',
                                 color='Value', color_continuous_scale='Viridis',
                                 title="Treemap of Value by Country and Import/Export")
        fig_treemap.update_layout(margin=dict(t=50, b=50, l=50, r=50), title_x=0.5)
        st.plotly_chart(fig_treemap)

        # Violin Plot: Value Distribution
        st.subheader("Violin Plot: Value Distribution")
        fig_violin = px.violin(df, y='Value', box=True, points="all", color_discrete_sequence=["#ff7f0e"],
                               title="Violin Plot of Value Distribution")
        fig_violin.update_layout(margin=dict(t=50, b=50, l=50, r=50), title_x=0.5)
        st.plotly_chart(fig_violin)

        st.markdown("---")

    # Tab 3: Summary
    with tab3:
        st.header("Summary Statistics")

        # Total Value Gauge
        st.subheader("Key Metric: Total Value")
        total_value = df['Value'].sum()
        st.metric("Total Value", f"${total_value:,.2f}")

        # Box Plot: Value by Country
        st.subheader("Box Plot: Value by Country")
        fig_box = px.box(df, x='Country', y='Value', color='Country',
                         title="Box Plot of Value by Country",
                         template="plotly_white")
        fig_box.update_layout(margin=dict(t=50, b=50, l=50, r=50), title_x=0.5)
        st.plotly_chart(fig_box)

        st.markdown("---")

# Run the app
if __name__ == '__main__':
    main()
