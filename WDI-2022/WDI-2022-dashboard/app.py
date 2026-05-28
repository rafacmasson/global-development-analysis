import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

# --- Page Configurations ---
st.set_page_config(
    page_title="WDI 2022 - Global Development Analysis",
    page_icon="🌍",
    layout="wide"
)

# --- Premium Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Premium card container styles */
    .premium-card {
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        border: 1px solid rgba(128, 128, 128, 0.15);
        background-color: var(--background-secondary-color, rgba(255, 255, 255, 0.8));
    }
    
    .premium-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.1);
    }
    
    /* Tier Contextual Borders and Glows */
    .tier-0-card {
        border-left: 6px solid #e43f5a !important;
        background: linear-gradient(135deg, rgba(228, 63, 90, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
    }
    
    .tier-1-card {
        border-left: 6px solid #00b4d8 !important;
        background: linear-gradient(135deg, rgba(0, 180, 216, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
    }
    
    .tier-2-card {
        border-left: 6px solid #2ca02c !important;
        background: linear-gradient(135deg, rgba(44, 160, 44, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
    }
    
    .card-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        font-weight: 600;
        color: #7f8c8d;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .card-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-color, #2c3e50);
        margin-bottom: 12px;
    }
    
    .card-desc {
        font-size: 0.95rem;
        color: var(--text-color, #7f8c8d);
        opacity: 0.85;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

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

# --- Loading ML Models ---
@st.cache_resource
def load_ml_models():
    models_path = os.path.join(os.path.dirname(__file__), 'models')
    try:
        classifier = joblib.load(os.path.join(models_path, 'rf_classifier.joblib'))
        regressor = joblib.load(os.path.join(models_path, 'rf_regressor.joblib'))
        feature_lists = joblib.load(os.path.join(models_path, 'feature_columns.joblib'))
        return classifier, regressor, feature_lists
    except FileNotFoundError:
        return None

df = load_data()

# --- Dashboard Layout ---
if df is not None:
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select View:",
        ("Dashboard", "Analysis (K-Means vs. HDBSCAN)", "Predictive Simulator (ML)")
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

    elif page == "Predictive Simulator (ML)":
        st.markdown("## 🔮 Interactive Machine Learning Simulator")
        st.markdown("""
        Experience global development forecasting in real-time! This simulator uses the optimized **Random Forest** models trained on World Development Indicators (WDI) 2022 data.
        
        *   **Regression Model (R² = 0.82):** Predicts country-level **Life Expectancy** based on socioeconomic and infrastructure indicators.
        *   **Classification Model (Accuracy = 95%):** Classifies the simulated country's **Development Cluster** (Tier 0, 1, or 2).
        """)
        
        # Load ML models
        models = load_ml_models()
        if models is None:
            st.error("ML Models could not be loaded. Please ensure the models were trained and saved in the 'models' directory.")
        else:
            classifier, regressor, feature_lists = models
            class_features = feature_lists['classifier_features']
            reg_features = feature_lists['regressor_features']
            
            st.markdown("### 🛠️ Step 1: Configure Socioeconomic Profile")
            st.write("Adjust the sliders in each category to define your simulated country's characteristics. Sliders are pre-configured with the actual minimum, maximum, and average values from the global dataset.")
            
            col1, col2 = st.columns(2)
            inputs = {}
            
            with col1:
                with st.expander("📈 Economic & Growth Indicators", expanded=True):
                    # GDP per capita
                    inputs['gdp_per_capita_usd'] = st.slider(
                        "GDP per Capita (USD)",
                        min_value=float(df['gdp_per_capita_usd'].min()),
                        max_value=float(df['gdp_per_capita_usd'].max()),
                        value=float(df['gdp_per_capita_usd'].mean()),
                        step=100.0,
                        help="Gross Domestic Product per capita in current US Dollars."
                    )
                    
                    # GDP growth
                    inputs['gdp_growth_annual_percent'] = st.slider(
                        "Annual GDP Growth (%)",
                        min_value=float(df['gdp_growth_annual_percent'].min()),
                        max_value=float(df['gdp_growth_annual_percent'].max()),
                        value=float(df['gdp_growth_annual_percent'].mean()),
                        step=0.1,
                        help="Annual percentage growth of GDP."
                    )
                    
                    # Agriculture value added
                    inputs['agriculture_value_added_percent_gdp'] = st.slider(
                        "Agriculture Value Added (% of GDP)",
                        min_value=float(df['agriculture_value_added_percent_gdp'].min()),
                        max_value=float(df['agriculture_value_added_percent_gdp'].max()),
                        value=float(df['agriculture_value_added_percent_gdp'].mean()),
                        step=0.5,
                        help="Share of GDP coming from agriculture, forestry, and fishing."
                    )
                    
                    # Industry value added
                    inputs['industry_value_added_percent_gdp'] = st.slider(
                        "Industry Value Added (% of GDP)",
                        min_value=float(df['industry_value_added_percent_gdp'].min()),
                        max_value=float(df['industry_value_added_percent_gdp'].max()),
                        value=float(df['industry_value_added_percent_gdp'].mean()),
                        step=0.5,
                        help="Share of GDP coming from mining, manufacturing, construction, and utilities."
                    )

                with st.expander("⚖️ Institutions, Stability & Population", expanded=True):
                    # Corruption Perception
                    inputs['corruption_perception_estimate'] = st.slider(
                        "Corruption Perception Index (Estimate)",
                        min_value=float(df['corruption_perception_estimate'].min()),
                        max_value=float(df['corruption_perception_estimate'].max()),
                        value=float(df['corruption_perception_estimate'].mean()),
                        step=0.1,
                        help="Governance indicator scoring corruption perception (higher is cleaner/more honest)."
                    )
                    
                    # Government Stability
                    inputs['stability_of_government_estimate'] = st.slider(
                        "Political Stability Estimate",
                        min_value=float(df['stability_of_government_estimate'].min()),
                        max_value=float(df['stability_of_government_estimate'].max()),
                        value=float(df['stability_of_government_estimate'].mean()),
                        step=0.1,
                        help="Political stability and absence of violence/terrorism estimate."
                    )
                    
                    # Total Population
                    inputs['population_total'] = st.slider(
                        "Total Population",
                        min_value=int(df['population_total'].min()),
                        max_value=1500000000, # Cap at 1.5B for slider ease
                        value=int(df['population_total'].mean()),
                        step=5000000,
                        format="%d",
                        help="Total population of the simulated country."
                    )
            
            with col2:
                with st.expander("⚡ Infrastructure & Urban Development", expanded=True):
                    # Access to electricity
                    inputs['access_to_electricity_percent'] = st.slider(
                        "Access to Electricity (% of population)",
                        min_value=float(df['access_to_electricity_percent'].min()),
                        max_value=100.0,
                        value=float(df['access_to_electricity_percent'].mean()),
                        step=1.0,
                        help="Percentage of the population with access to electricity."
                    )
                    
                    # Urban population
                    inputs['urban_population_percent'] = st.slider(
                        "Urban Population (% of total)",
                        min_value=float(df['urban_population_percent'].min()),
                        max_value=100.0,
                        value=float(df['urban_population_percent'].mean()),
                        step=1.0,
                        help="Percentage of population living in urban areas."
                    )
                    
                    # Mobile cellular
                    inputs['mobile_cellular_subscriptions_per_100_people'] = st.slider(
                        "Mobile Subscriptions per 100 People",
                        min_value=float(df['mobile_cellular_subscriptions_per_100_people'].min()),
                        max_value=float(df['mobile_cellular_subscriptions_per_100_people'].max()),
                        value=float(df['mobile_cellular_subscriptions_per_100_people'].mean()),
                        step=1.0,
                        help="Mobile cellular subscriptions per 100 individuals."
                    )

                with st.expander("🏥 Health, Environment & Demographics", expanded=True):
                    # Health expenditure
                    inputs['health_expenditure_pct_gdp'] = st.slider(
                        "Health Expenditure (% of GDP)",
                        min_value=float(df['health_expenditure_pct_gdp'].min()),
                        max_value=float(df['health_expenditure_pct_gdp'].max()),
                        value=float(df['health_expenditure_pct_gdp'].mean()),
                        step=0.5,
                        help="Total health expenditure as a percentage of GDP."
                    )
                    
                    # Fertility rate
                    inputs['fertility_rate_total'] = st.slider(
                        "Fertility Rate (births per woman)",
                        min_value=float(df['fertility_rate_total'].min()),
                        max_value=float(df['fertility_rate_total'].max()),
                        value=float(df['fertility_rate_total'].mean()),
                        step=0.1,
                        help="Total fertility rate (births per woman)."
                    )
                    
                    # Population 65+
                    inputs['population_65_plus_percent'] = st.slider(
                        "Population aged 65+ (%)",
                        min_value=float(df['population_65_plus_percent'].min()),
                        max_value=float(df['population_65_plus_percent'].max()),
                        value=float(df['population_65_plus_percent'].mean()),
                        step=0.5,
                        help="Percentage of total population aged 65 years and older."
                    )
                    
                    # CO2 emissions
                    inputs['co2_emissions_per_capita'] = st.slider(
                        "CO2 Emissions (metric tons per capita)",
                        min_value=float(df['co2_emissions_per_capita'].min()),
                        max_value=float(df['co2_emissions_per_capita'].max()),
                        value=float(df['co2_emissions_per_capita'].mean()),
                        step=0.1,
                        help="Carbon dioxide emissions per capita."
                    )
            
            # --- Live Predictions (Runs immediately as sliders change) ---
            st.markdown("### 🔮 Step 2: Live Forecast & Comparison")
            
            # 2.1 Reconstruct features for Regressor
            reg_input_df = pd.DataFrame([inputs])
            reg_input_df = reg_input_df[reg_features]
            
            # Run Regression prediction
            predicted_life_expectancy = float(regressor.predict(reg_input_df)[0])
            
            # 2.2 Reconstruct features for Classifier
            class_inputs = inputs.copy()
            class_inputs['life_expectancy_at_birth'] = predicted_life_expectancy
            class_input_df = pd.DataFrame([class_inputs])
            class_input_df = class_input_df[class_features]
            
            # Run Classification prediction
            predicted_cluster = int(classifier.predict(class_input_df)[0])
            
            # Display Results in Custom CSS Cards
            st.markdown("#### 📊 Real-Time Forecast Results")
            
            cluster_names = {
                0: "Tier 0 (Lower Development)",
                1: "Tier 1 (Medium Development)",
                2: "Tier 2 (High Development)"
            }
            
            tier_class = f"tier-{predicted_cluster}-card"
            
            # Build the custom HTML Cards
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(f"""
                <div class="premium-card {tier_class}">
                    <div class="card-title">Predicted Life Expectancy</div>
                    <div class="card-value">{predicted_life_expectancy:.2f} Years</div>
                    <div class="card-desc">
                        Your simulated country is predicted to have an average life expectancy of <b>{predicted_life_expectancy:.2f} years</b>.
                        Longevity is heavily driven by basic infrastructure (electricity), GDP per capita, and health parameters.
                        <br><br>
                        <i>World Avg: {df['life_expectancy_at_birth'].mean():.2f} years (Difference: {predicted_life_expectancy - df['life_expectancy_at_birth'].mean():+.2f})</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                cluster_desc = {
                    0: "Low urbanization, lower access to basic infrastructure like electricity, higher fertility rates, and lower GDP per capita.",
                    1: "A transitional state with moderate economic stability, developing infrastructure, growing urban centers, and middle-tier longevity.",
                    2: "Highly urbanized populations, high GDP per capita, advanced infrastructure, very high mobile subscriptions, and maximum health support."
                }
                
                st.markdown(f"""
                <div class="premium-card {tier_class}">
                    <div class="card-title">Predicted Development Cluster</div>
                    <div class="card-value">{cluster_names.get(predicted_cluster)}</div>
                    <div class="card-desc">
                        The Random Forest Classifier places this profile into <b>{cluster_names.get(predicted_cluster)}</b>.
                        <br><br>
                        <b>Profile Characteristics:</b> {cluster_desc.get(predicted_cluster)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- Visualizations (Gauge & Bar Charts) ---
            st.markdown("#### 📈 Deep-Dive Comparative Visualizations")
            
            # Row with a gauge and comparison bars
            viz_col1, viz_col2, viz_col3 = st.columns([1.2, 1, 1])
            
            # Calculate actual cluster averages
            actual_cluster_profile = df.groupby('Cluster_KMeans')[['gdp_per_capita_usd', 'life_expectancy_at_birth']].mean()
            
            with viz_col1:
                # Gauge Chart for Longevity
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_life_expectancy,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Predicted Life Expectancy (Years)", 'font': {'size': 16, 'family': 'Outfit'}},
                    number = {'suffix': " yrs", 'font': {'size': 32, 'family': 'Outfit'}},
                    gauge = {
                        'axis': {'range': [50, 85], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#1f4068" if predicted_cluster != 2 else "#2ca02c"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [50, 62], 'color': 'rgba(228, 63, 90, 0.25)'},   # Low (Warm Red)
                            {'range': [62, 74], 'color': 'rgba(255, 193, 7, 0.25)'},   # Medium (Yellow)
                            {'range': [74, 85], 'color': 'rgba(44, 160, 44, 0.25)'}    # High (Emerald Green)
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 3},
                            'thickness': 0.75,
                            'value': df['life_expectancy_at_birth'].mean() # World Avg marker
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=240, 
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with viz_col2:
                # GDP per capita comparison chart
                fig_gdp = px.bar(
                    x=["Simulated", "Tier 0 Avg", "Tier 1 Avg", "Tier 2 Avg"],
                    y=[
                        inputs['gdp_per_capita_usd'],
                        actual_cluster_profile.loc[0, 'gdp_per_capita_usd'],
                        actual_cluster_profile.loc[1, 'gdp_per_capita_usd'],
                        actual_cluster_profile.loc[2, 'gdp_per_capita_usd']
                    ],
                    labels={'x': '', 'y': 'GDP per Capita (USD)'},
                    color=["Simulated Country", "Tier 0 Average", "Tier 1 Average", "Tier 2 Average"],
                    color_discrete_map={
                        "Simulated Country": "#e43f5a" if predicted_cluster == 0 else ("#00b4d8" if predicted_cluster == 1 else "#2ca02c"),
                        "Tier 0 Average": "rgba(228, 63, 90, 0.4)",
                        "Tier 1 Average": "rgba(0, 180, 216, 0.4)",
                        "Tier 2 Average": "rgba(44, 160, 44, 0.4)"
                    },
                    title="GDP per Capita (USD)"
                )
                fig_gdp.update_layout(
                    showlegend=False,
                    height=240,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_gdp, use_container_width=True)
                
            with viz_col3:
                # Life expectancy comparison chart
                fig_life = px.bar(
                    x=["Simulated", "Tier 0 Avg", "Tier 1 Avg", "Tier 2 Avg"],
                    y=[
                        predicted_life_expectancy,
                        actual_cluster_profile.loc[0, 'life_expectancy_at_birth'],
                        actual_cluster_profile.loc[1, 'life_expectancy_at_birth'],
                        actual_cluster_profile.loc[2, 'life_expectancy_at_birth']
                    ],
                    labels={'x': '', 'y': 'Life Expectancy (Years)'},
                    color=["Simulated Country", "Tier 0 Average", "Tier 1 Average", "Tier 2 Average"],
                    color_discrete_map={
                        "Simulated Country": "#e43f5a" if predicted_cluster == 0 else ("#00b4d8" if predicted_cluster == 1 else "#2ca02c"),
                        "Tier 0 Average": "rgba(228, 63, 90, 0.4)",
                        "Tier 1 Average": "rgba(0, 180, 216, 0.4)",
                        "Tier 2 Average": "rgba(44, 160, 44, 0.4)"
                    },
                    title="Life Expectancy (Years)"
                )
                fig_life.update_layout(
                    showlegend=False,
                    height=240,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_life, use_container_width=True)
            
            # Comparison table below the graphs
            st.markdown("#### 🗺️ Multi-Indicator Comparison Grid")
            st.write("How your simulated country compares numerically to actual WDI 2022 cluster averages:")
            
            comparison_df = pd.DataFrame({
                "GDP per Capita (USD)": [
                    inputs['gdp_per_capita_usd'],
                    actual_cluster_profile.loc[0, 'gdp_per_capita_usd'],
                    actual_cluster_profile.loc[1, 'gdp_per_capita_usd'],
                    actual_cluster_profile.loc[2, 'gdp_per_capita_usd']
                ],
                "Life Expectancy (Years)": [
                    predicted_life_expectancy,
                    actual_cluster_profile.loc[0, 'life_expectancy_at_birth'],
                    actual_cluster_profile.loc[1, 'life_expectancy_at_birth'],
                    actual_cluster_profile.loc[2, 'life_expectancy_at_birth']
                ]
            }, index=["Simulated Country", "Actual Tier 0 Avg", "Actual Tier 1 Avg", "Actual Tier 2 Avg"])
            
            st.dataframe(comparison_df.style.highlight_max(axis=0, color='rgba(44, 160, 44, 0.15)').format({
                "GDP per Capita (USD)": "${:,.2f}",
                "Life Expectancy (Years)": "{:.2f} years"
            }), use_container_width=True)

else:
    st.warning("Please ensure the 'WDI2022_clusters.csv' file was successfully generated in the 'WDI-2022-clustering' folder.")
