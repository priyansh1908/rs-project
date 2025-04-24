import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set page config with better styling
st.set_page_config(
    page_title="Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    :root {
        --primary-color: #FF9800;
        --primary-dark: #F57C00;
        --primary-light: #FFB74D;
        --primary-bg: #FFF3E0;
        --background-dark: #121212;
        --card-dark: #1E1E1E;
        --text-primary: #FFFFFF;
        --text-secondary: #FFE0B2;
        --border-color: #FF9800;
        --hover-color: #FFA726;
    }
    
    /* Filter Section Improvements */
    .filter-container {
        background-color: var(--card-dark);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border: 1px solid var(--border-color);
    }
    
    .filter-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--primary-color);
    }
    
    .filter-header-icon {
        font-size: 1.5rem;
        color: var(--primary-color);
    }
    
    .filter-header-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .filter-section {
        background-color: rgba(255, 152, 0, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .filter-section:hover {
        background-color: rgba(255, 152, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .filter-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: var(--primary-light);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .filter-description {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
        line-height: 1.5;
        opacity: 0.9;
    }
    
    /* Slider Improvements */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div > div {
        background-color: var(--primary-color) !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: var(--primary-light) !important;
        border: 2px solid var(--primary-color) !important;
    }
    
    /* Button Styles */
    .stButton>button {
        background: linear-gradient(45deg, var(--primary-color), var(--hover-color));
        color: var(--text-primary);
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 4px rgba(255, 152, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(45deg, var(--hover-color), var(--primary-color));
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 152, 0, 0.3);
    }
    
    /* Card Styles */
    .restaurant-card {
        background: linear-gradient(145deg, var(--card-dark), #262626);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(255, 152, 0, 0.1);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .restaurant-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(255, 152, 0, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(145deg, var(--card-dark), #262626);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(255, 152, 0, 0.1);
        border: 1px solid var(--border-color);
        height: 100%;
    }
    
    .cuisine-tag {
        background-color: var(--primary-dark);
        color: var(--text-primary);
        padding: 0.4rem 1rem;
        border-radius: 25px;
        margin: 0.3rem;
        font-size: 0.9rem;
        box-shadow: 0 2px 4px rgba(255, 152, 0, 0.1);
        border: 1px solid var(--primary-color);
        transition: all 0.3s ease;
    }
    
    .cuisine-tag:hover {
        background-color: var(--hover-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 152, 0, 0.2);
    }
    
    /* Info Box */
    .info-box {
        background-color: rgba(255, 152, 0, 0.1);
        border-left: 4px solid var(--primary-color);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        background-color: var(--card-dark) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--primary-color) !important;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: var(--card-dark) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }
    
    .stMultiSelect > div > div:hover {
        border-color: var(--primary-color) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title with better styling
st.markdown("<h1 style='text-align: center; color: var(--text-primary);'>🍽️ Restaurant Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: var(--text-secondary); font-size: 1.2rem;'>Find restaurants with similar cuisines and dining experiences!</p>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        # Read only necessary columns
        df = pd.read_csv('zomato.csv', 
                        usecols=['name', 'cuisines', 'reviews_list', 'rate', 
                                'location', 'approx_cost(for two people)', 'rest_type'])
        
        # Basic cleaning
        df = df.dropna()
        df = df[df['rate'] != 'NEW']
        df = df[df['rate'] != '-']
        
        # Process ratings
        df['rate'] = df['rate'].str.replace('/5', '').astype(float)
        
        # Clean cost
        df['cost'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '').astype(float)
        
        # Clean restaurant types
        df['rest_type'] = df['rest_type'].fillna('').str.lower()
        
        # Clean cuisines
        df['cuisines'] = df['cuisines'].fillna('').str.lower()
        
        # Take only restaurants with significant number of reviews and reset index
        df['review_length'] = df['reviews_list'].str.len()
        df = df.sort_values('review_length', ascending=False)
        df = df.head(3000)  # Increased to 3000 restaurants for better variety
        df = df.reset_index(drop=True)  # Reset index to avoid index issues
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def create_similarity_matrix(df):
    try:
        # Create TF-IDF matrix with focus on cuisines and restaurant types
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        
        # Combine cuisines, restaurant type, and reviews for better recommendations
        df['features'] = (
            df['cuisines'] + ' ' + 
            df['rest_type'] + ' ' + 
            df['reviews_list'].fillna('').str.slice(0, 1000)
        )
        
        # Create TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(df['features'])
        
        # Calculate similarity scores
        return cosine_similarity(tfidf_matrix)
    except Exception as e:
        st.error(f"Error creating similarity matrix: {str(e)}")
        return None

def get_recommendations(df, restaurant_name, similarity_matrix):
    try:
        # Find the index of the restaurant
        restaurant_idx = df.index[df['name'] == restaurant_name].tolist()[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[restaurant_idx]))
        
        # Sort restaurants by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 20 similar restaurants (we'll filter duplicates from these)
        sim_scores = sim_scores[1:21]  # Get more than needed to account for duplicates
        
        # Get restaurant indices
        restaurant_indices = [i[0] for i in sim_scores]
        
        # Get recommendations and remove duplicates
        recommendations = df.iloc[restaurant_indices][['name', 'cuisines', 'rest_type', 'rate', 'cost', 'location']]
        
        # Remove any potential duplicate restaurants (same name or very similar names)
        recommendations = recommendations.drop_duplicates(subset=['name'])
        
        # Ensure the selected restaurant is not in recommendations
        recommendations = recommendations[recommendations['name'] != restaurant_name]
        
        # Take only top 10 after filtering
        recommendations = recommendations.head(10)
        
        # Rename columns for display
        recommendations = recommendations.rename(columns={
            'name': 'Restaurant',
            'cuisines': 'Cuisines',
            'rest_type': 'Restaurant Type',
            'rate': 'Rating',
            'cost': 'Cost for Two',
            'location': 'Location'
        })
        
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None

def display_cuisine_tags(cuisines):
    tags = [f"<span class='cuisine-tag'>{cuisine.strip()}</span>" for cuisine in cuisines.split(',')]
    return " ".join(tags)

# Main application
try:
    # Create sidebar with improved filters
    with st.sidebar:
        st.markdown("""
            <div class='filter-section'>
                <div class='filter-title'>
                    <span>⭐</span>
                    <span>Rating Filter</span>
                </div>
                <div class='filter-description'>
                    Set the minimum rating to ensure quality recommendations.
                </div>
                <div class='filter-content'>
        """, unsafe_allow_html=True)
        min_rating = st.slider("Minimum Rating", 1.0, 5.0, 3.0, 0.1, label_visibility="collapsed")
        
        st.markdown("""
                </div>
            </div>
            
            <div class='filter-section'>
                <div class='filter-title'>
                    <span>💰</span>
                    <span>Budget Filter</span>
                </div>
                <div class='filter-description'>
                    Define your budget range for two people.
                </div>
                <div class='filter-content'>
        """, unsafe_allow_html=True)
        max_cost = st.slider("Maximum Cost", 100, 5000, 2000, 100, label_visibility="collapsed")
        
        st.markdown("""
                </div>
            </div>
            
            <div class='filter-section'>
                <div class='filter-title'>
                    <span>🍽️</span>
                    <span>Cuisine Preferences</span>
                </div>
                <div class='filter-description'>
                    Select your preferred cuisines for better matches.
                </div>
                <div class='filter-content'>
        """, unsafe_allow_html=True)
        
        # Add cuisine filter
        df = load_data()
        if df is not None:
            all_cuisines = set()
            for cuisines in df['cuisines'].str.split(','):
                all_cuisines.update([c.strip() for c in cuisines])
            selected_cuisines = st.multiselect(
                "Select Cuisines",
                sorted(list(all_cuisines)),
                default=[],
                label_visibility="collapsed"
            )
        
        st.markdown("""
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    if df is not None and len(df) > 0:
        # Create similarity matrix
        similarity_matrix = create_similarity_matrix(df)
        
        if similarity_matrix is not None:
            # Create restaurant selector
            col1, col2 = st.columns([3, 1])
            with col1:
                restaurant_list = sorted(df['name'].unique())
                selected_restaurant = st.selectbox('Select a restaurant:', restaurant_list)
            
            with col2:
                st.write("")
                st.write("")
                if st.button('Get Recommendations', key='recommend_button'):
                    pass
            
            if st.session_state.get('recommend_button', False):
                with st.spinner('Finding similar restaurants...'):
                    # Get recommendations
                    recommendations = get_recommendations(df, selected_restaurant, similarity_matrix)
                    
                    if recommendations is not None:
                        # Filter recommendations based on sidebar criteria
                        recommendations = recommendations[
                            (recommendations['Rating'] >= min_rating) & 
                            (recommendations['Cost for Two'] <= max_cost)
                        ]
                        
                        # Apply cuisine filter if any cuisines are selected
                        if selected_cuisines:
                            recommendations = recommendations[
                                recommendations['Cuisines'].apply(
                                    lambda x: any(cuisine in x.lower() for cuisine in selected_cuisines)
                                )
                            ]
                        
                        # Display restaurant info in a card
                        st.markdown("<h2 class='section-title'>Selected Restaurant</h2>", unsafe_allow_html=True)
                        restaurant_info = df[df['name'] == selected_restaurant].iloc[0]
                        
                        # Create metrics in a row
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h3 style='color: var(--text-primary);'>Rating</h3>
                                <h2 style='color: var(--text-primary);'>{restaurant_info['rate']:.1f}/5</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h3 style='color: var(--text-primary);'>Cost for Two</h3>
                                <h2 style='color: var(--text-primary);'>₹{restaurant_info['cost']:.0f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h3 style='color: var(--text-primary);'>Location</h3>
                                <h4 style='color: var(--text-secondary);'>{restaurant_info['location']}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                        with col4:
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h3 style='color: var(--text-primary);'>Restaurant Type</h3>
                                <h4 style='color: var(--text-secondary);'>{restaurant_info['rest_type'].title()}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display cuisines with tags
                        st.markdown("<h3 class='section-title'>Cuisines</h3>", unsafe_allow_html=True)
                        st.markdown(display_cuisine_tags(restaurant_info['cuisines']), unsafe_allow_html=True)
                        
                        # Display recommendations in cards
                        st.markdown("<h2 class='section-title'>Recommended Restaurants</h2>", unsafe_allow_html=True)
                        for idx, row in recommendations.iterrows():
                            st.markdown(f"""
                            <div class='restaurant-card'>
                                <h3 style='color: var(--text-primary);'>{row['Restaurant']}</h3>
                                <p style='color: var(--text-secondary);'><strong>Restaurant Type:</strong> {row['Restaurant Type'].title()}</p>
                                <p style='color: var(--text-secondary);'><strong>Cuisines:</strong> {display_cuisine_tags(row['Cuisines'])}</p>
                                <p style='color: var(--text-secondary);'><strong>Rating:</strong> {row['Rating']:.1f}/5</p>
                                <p style='color: var(--text-secondary);'><strong>Cost for Two:</strong> ₹{row['Cost for Two']:.0f}</p>
                                <p style='color: var(--text-secondary);'><strong>Location:</strong> {row['Location']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display recommendation explanation
                        st.markdown("""
                        <div class='info-box'>
                            <h4 style='color: var(--text-primary);'>💡 How recommendations work:</h4>
                            <ul style='color: var(--text-secondary);'>
                                <li>Similar cuisines and food preferences</li>
                                <li>Restaurant type and dining experience</li>
                                <li>Customer reviews and ratings</li>
                                <li>Location and price range</li>
                            </ul>
                            <p style='color: var(--text-secondary); font-size: 0.9rem;'>
                                Note: Showing recommendations from top 30000 most-reviewed restaurants for better accuracy.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.error("Failed to load the dataset. Please check if the file exists and is properly formatted.")

except Exception as e:
    st.error(f"""
    ⚠️ An error occurred while running the app. Please make sure:
    1. The 'zomato.csv' file is in the same directory as this script
    2. The file contains all required columns
    3. The file is not corrupted
    
    Error details: {str(e)}
    """)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True) 