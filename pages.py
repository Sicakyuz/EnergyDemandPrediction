import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from Data_Overview import main_data
from Time_series import main_City
from TR_general import main_TR
import os

# Define the path relative to your script or working directory
image_path = os.path.join('PHOTOS', 'roadmap.png')
video_path = os.path.join('VIDEOS', 'VIDEO-2024-04-10-01-50-33.mp4')

# Function to render the home page
def home_page():
    """Render the Home page."""
    st.title("Turkey Energy Overview")
    if os.path.exists(image_path):
        st.image(image_path, use_column_width='always')
    else:
        st.error(f"Image file not found: {image_path}")
    st.markdown("""
        ## 
        This dashboard is designed to analyze Turkey's energy consumption.
        You can perform detailed data analysis, forecasting, and anomaly detection.
    """)

# Function to render the contact page

def data_overview_page():
    """Render the Data Overview page."""
    if 'data' in st.session_state:
        st.title("Data Overview")
        st.dataframe(st.session_state['data'].head())
        main_data()
def modeling_page():
    """Render the Modeling page."""
    tab1, tab2 = st.tabs(["Turkey Energy Consumption Forecast", "City Energy Consumption Forecast"])

    with tab1:
        st.subheader("Turkey Energy Consumption Forecast")
        main_TR()  # Turkey forecast fonksiyonunu çağırır

    with tab2:
        st.subheader("City Energy Consumption Forecast")
        if 'data' in st.session_state:
            st.dataframe(st.session_state['data'].head())  # Veri tablosunu gösterir
        main_City()  # City forecast fonksiyonunu çağırır

def map_page():
    """Render the Results page."""
    selected_subtab = st.sidebar.radio("Options", ["Energy consumption", "Energy Production Forecast"])
    st.header("**🌟 Turkey's Energy Map**")
    # Usage Instructions with Emojis
    st.subheader("🗺️ How to Use the Map Functionality")
    # Custom CSS to style the expander and text colors
    st.markdown("""
            <style>
            /* Expander Header Styling */
            .css-1v3fvcr .css-1e6f6z4 {
                background-color: #333; /* Dark background for the expander header */
                color: #38cebb; /* Teal color for the expander header text */
            }

            /* Expander Inner Content Styling */
            .css-1v3fvcr .css-1n3zhzy {
                background-color: #2b2b2b; /* Dark grey background for inner content */
                color: #e0e0e0; /* Light grey text color for better readability */
                border-radius: 5px;
                padding: 10px;
            }

            /* Expander Inner Content Text Color */
            .css-1v3fvcr .css-1n3zhzy h2, 
            .css-1v3fvcr .css-1n3zhzy h3 {
                color: #38cebb; /* Teal color for section headers */
            }
            </style>
        """, unsafe_allow_html=True)
    # Usage Instructions with Emojis in Expanders
    with st.expander("🗺️ How to Use the Map Functionality", expanded=True):
        st.markdown("""
                Welcome to the map feature of the Turkey Energy Overview application! This interactive map helps you visualize energy consumption and production forecasts across Turkey. Here’s a quick guide on how to use it effectively:

                <span style="color:#38cebb">**1. Navigating to the Map Page** 🧭</span>
                - **Select Map View**: From the sidebar on the left, choose between "Map View" and "Energy Production Forecast" to switch between different map types.

                <span style="color:#38cebb">**2. Understanding the Maps** 🌍</span>
                - **Energy Consumption Map**:
                  - **Purpose**: Displays actual energy consumption data across Turkey.
                  - **Markers**: Represent different locations with their total energy consumption.
                  - **Tooltip** 💬: Hover over a marker to see the location name and total energy consumption.
                  - **Popup** 📊: Click on a marker to view detailed consumption data for that location.

                - **Energy Production Forecast Map**:
                  - **Purpose**: Shows forecasted energy production data for various regions.
                  - **Markers**: Indicate locations with estimated energy production.
                  - **Tooltip** 💬: Hover over a marker to see the location name and estimated production.
                  - **Popup** 📊: Click on a marker to view the forecasted energy production or note if no data is available.

                <span style="color:#38cebb">**3. Switching Between Views** 🔄</span>
                - Use the sidebar to select **"Map View"** for consumption data or **"Energy Production Forecast"** for production forecasts.

                <span style="color:#38cebb">**4. Tips for Effective Use** 💡</span>
                - **Zoom and Pan**: Use the zoom controls or scroll wheel to explore specific areas. Drag the map to pan and view different regions.
                - **Clustered Markers**: Markers are grouped into clusters for easier readability. Zooming in will reveal individual markers.

                <span style="color:#38cebb">**5. Need More Help?** 🤔</span>
                - **Contact Us**: For questions or further assistance, please visit the Contact page to get in touch with us.
            """, unsafe_allow_html=True)
    if selected_subtab == "Energy consumption":
        st.header("🌟 Turkey's Energy Map: Energy Consumption")
        turkey_map = folium.Map(location=[39.9334, 32.8597], zoom_start=6)
        marker_cluster = MarkerCluster().add_to(turkey_map)

        for index, row in st.session_state['data'].iterrows():
            tooltip_text = f"{row['ILLER']}: {row['TUKETIM_GENEL_TOPLAM']} kWh"
            folium.Marker(
                location=[row['ENLEM'], row['BOYLAM']],
                popup=f"Total Consumption: {row['TUKETIM_GENEL_TOPLAM']} kWh",
                tooltip=tooltip_text
            ).add_to(marker_cluster)

        folium_static(turkey_map, width=1450, height=700)

    elif selected_subtab == "Energy Production Forecast":
        st.header("🌟Energy Production Forecast Map")
        energy_forecast_map = folium.Map(location=[39.9334, 32.8597], zoom_start=6)
        marker_cluster = MarkerCluster().add_to(energy_forecast_map)

        for index, row in st.session_state['data'].iterrows():
            tooltip_text = f"{row['ILLER']}: Estimated Production"
            folium.Marker(
                location=[row['ENLEM'], row['BOYLAM']],
                popup=f"Estimated Production: {row.get('TUKETIM_TAHMIN', 'No Info')}",
                tooltip=tooltip_text
            ).add_to(marker_cluster)

        folium_static(energy_forecast_map, width=1450, height=700)

def contact_page():
    """Render the Contact page."""
    st.title("Contact")
    st.markdown("""
        ## Contact Information
        TYou can reach us via contact form.
    """)

    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Your Message")
        submitted = st.form_submit_button("Send")
        if submitted:
            st.success("Your message has been sent successfully.")
