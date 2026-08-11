# ==========================================
# IMPORTS & INITIAL CONFIGURATION
# ==========================================
import pickle
import pandas as pd
import streamlit as st

# Page Configuration (Sets title, layout, and sidebar defaults)
st.set_page_config(
    page_title="Bollywood Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default fallback thumbnail image URL if a song lacks art in the dataset
DEFAULT_THUMBNAIL = "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=500&auto=format&fit=crop&q=60"


# ==========================================
# PAGE DEFINITIONS
# ==========================================

def home_page():
    """
    Renders the Home page displaying system goals, usage steps, 
    key feature breakdowns, and tech stack details.
    """
    st.title("🎵 Bollywood Music Recommendation System")
    st.caption("Discover your next favorite track powered by Content-Based Machine Learning.")

    st.divider()

    # Split main layout into two uneven columns (Objective/Usage vs. Tech Stack)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🎯 Objective
        The aim of this project is to recommend songs to users based on their selected song or genre. The system provides quick and personalized music suggestions through a simple Streamlit web interface.

        ### 📖 How to Use
        1. Go to Recommendation
        2. Enter a song
        3. Click Recommend

        ### ✨ Core Features
        * 🔍 **Smart Search:** Type or select any track from the dataset dropdown.
        * 🎙️ **Artist Metadata:** Track exact artist details along with title results.
        * 🖼️ **Visual Album Art:** View high-resolution song thumbnails for selected and recommended items.
        * ⚡ **Top 5 Vector Match:** Instant cosine-similarity computation to surface identical vibe tracks.
        """)

    with col2:
        st.info("""
        ### 🛠️ Tech Stack
        * **Python**
        * **Pandas & NumPy**
        * **Scikit-learn**
        * **Streamlit Framework**
        """)

    st.divider()
    st.success("💡 **Quickstart:** Open the sidebar menu and navigate to **Recommendation** to start searching!")


def recommendation_page():
    """
    Renders the core Recommendation page: loads cached dataset models,
    handles track selection input, processes vector similarity matches, 
    and displays top 5 recommended tracks in a grid view.
    """
    st.title("🎧 Music Recommendation")
    st.caption("Select a song you love and let vector similarity do the rest.")

    # Cache dataset loading to prevent repetitive disk reads on rerun
    @st.cache_resource
    def load_data():
        df = pickle.load(open("song_df.pkl", "rb"))
        similarity = pickle.load(open("similarity.pkl", "rb"))
        return df, similarity

    # Error handling for missing pre-trained data files
    try:
        df, similarity = load_data()
    except FileNotFoundError:
        st.error("⚠️ Error: `song_df.pkl` or `similarity.pkl` missing from root folder.")
        st.stop()

    def recommend(song_title):
        """
        Calculates cosine similarity distance scores for the selected song 
        and returns details of the top 5 nearest vector matches.
        """
        song_title = song_title.lower()
        try:
            # Find matching index of the selected song
            index = df[df["song_name"].str.lower() == song_title].index[0]
        except IndexError:
            return []
        
        # Retrieve vector distance array for the target index
        distance = similarity[index]
        
        # Sort vector indices by similarity score descending; select top 5 matches (excluding index 0 self-match)
        song_list = sorted(
            list(enumerate(distance)),
            reverse=True,
            key=lambda x: x[1]
        )[1:6]

        recommendations = []
        for i, score in song_list:
            # Resolve thumbnail with fallback checking
            thumb = df.iloc[i]["thumbnail"] if ("thumbnail" in df.columns and pd.notna(df.iloc[i]["thumbnail"])) else DEFAULT_THUMBNAIL
            recommendations.append({
                "song": df.iloc[i]["song_name"],
                "artist": df.iloc[i]["artist"],
                "thumbnail": thumb
            })
        return recommendations

    # Search section inside a structured card layout
    with st.container():
        song_list_options = df["song_name"].values
        user_input = st.selectbox(
            "Search or select a song:",
            options=song_list_options,
            index=None,
            placeholder="Type to search songs..."
        )

        submit_btn = st.button("Get Recommendations", type="primary", use_container_width=True)

    # Process recommendation computation when user clicks button
    if submit_btn:
        if not user_input:
            st.warning("Please select or enter a song name first!")
            st.stop()

        try:
            # Extract target track metadata
            idx = df[df["song_name"].str.lower() == user_input.lower()].index[0]
            current_song = df.iloc[idx]["song_name"]
            current_artist = df.iloc[idx]["artist"]
            current_thumb = df.iloc[idx]["thumbnail"] if ("thumbnail" in df.columns and pd.notna(df.iloc[idx]["thumbnail"])) else DEFAULT_THUMBNAIL

            st.divider()
            
            # Selected Song Hero Section
            st.subheader("Target Track")
            hero_col1, hero_col2 = st.columns([1, 4])
            with hero_col1:
                st.image(current_thumb, use_container_width=True)
            with hero_col2:
                st.title(current_song)
                st.markdown(f"**Artist:** `{current_artist}`")
                st.caption("Showing 5 top tracks matching the vector profile of this song.")
                
        except IndexError:
            st.error("Selected track could not be resolved in dataset indices.")
            st.stop()

        st.divider()
        st.subheader("🔥 Top 5 Recommended Songs")

        # Execute recommendation calculation
        results = recommend(user_input)

        if not results:
            st.info("No close vector matches located for this track.")
        else:
            # Display recommendations in a stylish 5-column grid layout
            cols = st.columns(5)
            for rank, (col, item) in enumerate(zip(cols, results), 1):
                with col:
                    st.markdown(f"""
                    <div class="song-card">
                        <div class="card-img-container">
                            <img src="{item['thumbnail']}" alt="Cover art">
                        </div>
                        <span class="rank-badge">#{rank} Recommendation</span>
                        <div class="song-title" title="{item['song']}">{item['song']}</div>
                        <div class="artist-name" title="{item['artist']}">{item['artist']}</div>
                    </div>
                    """, unsafe_allow_html=True)


def about_page():
    """
    Renders the About page containing system architecture highlights, 
    team member credits, and project outcome metrics.
    """
    st.title("ℹ️ About the Project")
    
    st.markdown("""
    ### 📌 Project Overview
    This platform computes **Cosine Similarity** over multi-dimensional feature vectors (derived from audio attributes, tempo, valence, and genres) to recommend similar tracks from an extensive Bollywood catalog.
    """)
    
    st.divider()
    
    st.subheader("👩‍💻 Team Members")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Developer", value="Harshada")
    with col2:
        st.metric(label="Developer", value="Ruchita")
    with col3:
        st.metric(label="Source collecter", value="Sanjana")
        
    st.divider()
    
    st.markdown("""
    ### 🎯 System Highlights
    * **Low Latency:** Precomputed similarity arrays stored in serialized `.pkl` format for real-time querying.
    * **Interactive UI:** Dynamic UI layout built with custom CSS integration on top of Streamlit components.
    """)

    st.divider()

    st.markdown("""
    ### ✅Project Outcome
    The project successfully recommends songs similar to the selected song through a simple and interactive Streamlit interface.It demonstrates the practical use of machine learning, data preprocessing, 
    and web application development in a music recommendation system.
    """)


# ==========================================
# STREAMLIT MULTI-PAGE NAVIGATION SETUP
# ==========================================

# Define pages for multi-page routing
pages = [
    st.Page(home_page, title="Home", icon="🏠", default=True),
    st.Page(recommendation_page, title="Recommendation", icon="🎧"),
    st.Page(about_page, title="About", icon="ℹ️")
]

# Initialize Streamlit navigation controller
pg = st.navigation(pages)
pg.run()