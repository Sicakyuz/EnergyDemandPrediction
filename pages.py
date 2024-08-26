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
    selected_subtab = st.sidebar.radio("Options", ["Map View", "Energy Production Forecast"])
    if selected_subtab == "Map View":
        st.header("Turkey Map: Energy Consumption")
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
        st.header("Energy Production Forecast Map")
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
        This section contains contact information and a form.
        You can reach out to one of the team members below.
    """)
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.error(f"Video file not found: {video_path}")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write("Person 1")
    with col2:
        st.write("Person 2")
    with col3:
        st.write("Person 3")
    with col4:
        st.write("Person 4")
    with col5:
        st.write("Person 5")

    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Your Message")
        submitted = st.form_submit_button("Send")
        if submitted:
            st.success("Your message has been sent successfully.")
