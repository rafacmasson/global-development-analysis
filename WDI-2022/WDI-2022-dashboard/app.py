import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page Configurations ---
st.set_page_config(
    page_title="WDI 2022 - Global Development Analysis",
    page_icon="🌍",
    layout="wide"
)

# --- Title and Introduction ---
st.title("🌍 WDI 2022: Global Development Analysis")
st.markdown("""
This dashboard presents the clustering of countries based on socioeconomic development indicators from the year 2022, using data from the **World Bank**.
Two clustering algorithms were applied: **K-Means** (K=3) and **HDBSCAN**.
""")

# --- Loading Data ---
@st.cache_data
def load_data():
    # File path relative to this script's location
    file_path = os.path.join(os.path.dirname(__file__), '..', 'WDI-2022-clustering', 'WDI2022_clusters.csv')
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at: {file_path}")
        return None

df = load_data()

# --- Dashboard Layout ---
if df is not None:
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select View:",
        ("Dashboard", "Analysis (K-Means vs. HDBSCAN)")
    )
    
    if page == "Dashboard":
        st.sidebar.header("Settings")
        
        # 1. Algorithm Selection
        selected_algorithm = st.sidebar.radio(
            "Select Clustering Algorithm:",
            ("K-Means (3 Clusters)", "HDBSCAN (Density)")
        )
        
        # Define which column to use based on selection
        if "K-Means" in selected_algorithm:
            cluster_col = "Cluster_KMeans"
            title_suffix = "K-Means Method"
        else:
            cluster_col = "Cluster_HDBSCAN"
            title_suffix = "HDBSCAN Method"
            
        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head())
        
        # --- Geographic Visualization ---
        st.markdown(f"### 🗺️ Global Distribution of Clusters ({title_suffix})")
        
        # Treat cluster as discrete (categorical) for the map, to have distinct colors
        df_map = df.copy()
        df_map[cluster_col] = df_map[cluster_col].astype(str)
        
        fig_map = px.choropleth(
            df_map, 
            locations="Country Code",
            color=cluster_col,
            hover_name="Country Name",
            title=f"Countries by Cluster - {title_suffix}",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        # Update map layout to be bigger and cleaner
        fig_map.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
        # --- Cluster Analysis ---
        st.markdown(f"### 📈 Cluster Profiles ({title_suffix})")
        st.write("Average indicators for each cluster group. This helps understand the defining characteristics of each segment.")
        
        # Calculate means of numeric columns grouped by the selected cluster
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        # Remove the other cluster column so it doesn't show up in means
        cols_to_drop = ['Cluster_KMeans', 'Cluster_HDBSCAN']
        numeric_df = df[numeric_cols].drop(columns=[c for c in cols_to_drop if c != cluster_col], errors='ignore')
        
        cluster_profiles = numeric_df.groupby(cluster_col).mean()
        
        # Apply background gradient styled df
        st.dataframe(cluster_profiles.style.background_gradient(cmap='coolwarm', axis=0).format("{:.2f}"))
        
        # --- Countries by Cluster ---
        st.markdown("### 📝 Countries in Each Cluster")
        st.write("Select a cluster to view the countries that belong to it.")
        
        # Get distinct clusters, sorted
        clusters = sorted(df[cluster_col].unique().tolist())
        
        tabs = st.tabs([f"Cluster {c}" for c in clusters])
        
        for i, cluster_id in enumerate(clusters):
            with tabs[i]:
                countries = df[df[cluster_col] == cluster_id]['Country Name'].sort_values().tolist()
                if cluster_id == -1 and "HDBSCAN" in selected_algorithm:
                    st.info("Cluster -1 represents NOISE / OUTLIERS in the HDBSCAN algorithm.")
                
                st.write(f"**Total Countries:** {len(countries)}")
                # Show in a bulleted list format
                st.markdown("\n".join([f"- {country}" for country in countries]))

    elif page == "Analysis (K-Means vs. HDBSCAN)":
        # --- Analysis Section ---
        st.markdown("## 📋 Analysis of Clustering Results (K-Means vs. HDBSCAN)")
        st.markdown("This section provides a summary and comparison of the clustering results obtained using K-Means and HDBSCAN.")
        
        st.markdown("### K-Means Analysis")
        st.markdown("""
Based on the Elbow Method and Silhouette Analysis, K=3 was chosen as the optimal number of clusters for the K-Means algorithm.

- **Cluster 0:** Characterized by lower values across most development indicators (e.g., lower GDP per capita, higher child mortality, lower life expectancy). This cluster appears to represent countries with lower levels of socioeconomic development.
- **Cluster 1:** Represents a transitional group with moderate values for most indicators. These countries are generally more developed than those in Cluster 0 but less so than those in Cluster 2.
- **Cluster 2:** Shows higher values for development indicators (e.g., higher GDP per capita, lower child mortality, higher life expectancy). This cluster likely represents more developed countries.

The K-Means approach provides a clear partitioning of countries into three distinct groups based on the chosen indicators. The heat map of cluster profiles clearly illustrates the average characteristics of each cluster.
        """)

        st.markdown("### HDBSCAN Analysis")
        st.markdown("""
HDBSCAN identified three clusters (-1, 0, and 1), with cluster -1 representing noise points (outliers) that do not fit well into any specific cluster.

- **Cluster -1 (Noise):** This group contains countries that are significantly different from the core clusters, likely due to extreme values in some indicators.
- **Cluster 0:** Similar to a sub-group identified by K-Means, this cluster appears to contain countries with relatively low development indicators, although it is a smaller group than the K-Means Cluster 0.
- **Cluster 1:** This is the largest cluster and seems to group countries with higher development indicators, similar to K-Means Cluster 2 and some countries from K-Means Cluster 1.

HDBSCAN's strength lies in its ability to identify noise and clusters of varying densities. The noise cluster (-1) highlights countries that are outliers in the dataset based on the chosen features. The resulting clusters are not as evenly distributed as with K-Means, with a large number of countries falling into the noise category.
        """)
        
        st.markdown("### Comparison and Interpretation")
        st.markdown("""
- **K-Means** successfully divided the countries into 3 distinct and well-balanced clusters, which clearly represent different levels of socioeconomic development.
- **HDBSCAN**, on the other hand, identified 2 main clusters but struggled with the continuous nature of the socioeconomic data. It resulted in a very large number of outliers (Cluster -1) and a significantly small Cluster 0 (which seemingly represented countries with lower indicators). Because HDBSCAN relies on dense regions to form clusters, it failed to categorize many countries that have unique or dispersed indicator profiles.
        """)
        
        st.markdown("### Conclusion")
        st.info("Given the distribution of global development data, the **K-Means algorithm provided a much more precise, interpretable, and valuable clustering solution** compared to HDBSCAN. Therefore, the K-Means clusters were chosen as the final grouping to be used in the interactive dashboard.")


else:
    st.warning("Please ensure the 'WDI2022_clusters.csv' file was successfully generated in the 'WDI-2022-clustering' folder.")
