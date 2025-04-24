import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
import gc
import sys
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Zomato Restaurant Recommender",
    page_icon="🍽️",
    layout="wide"
)

# Add title and description
st.title("🍽️ Zomato Restaurant Recommender")
st.markdown("""
This app recommends restaurants based on their reviews and ratings. 
Enter a restaurant name to find similar restaurants in Bangalore!
""")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None
if 'cosine_similarities' not in st.session_state:
    st.session_state.cosine_similarities = None

def load_and_process_data():
    """Load and process the data efficiently"""
    try:
        # Read the dataset with only necessary columns
        columns_to_keep = ['name', 'rate', 'votes', 'location', 'rest_type', 
                          'cuisines', 'approx_cost(for two people)', 'reviews_list']
        
        # Read data in chunks
        chunksize = 5000  # Reduced chunk size
        chunks = []
        for chunk in pd.read_csv("zomato.csv", usecols=columns_to_keep, chunksize=chunksize):
            # Basic cleaning on each chunk
            chunk = chunk.drop_duplicates()
            chunk = chunk.dropna(how='any')
            chunks.append(chunk)
            gc.collect()
        
        # Combine chunks
        zomato = pd.concat(chunks, ignore_index=True)
        
        # Free memory
        del chunks
        gc.collect()
        
        # Rename columns
        zomato = zomato.rename(columns={
            'approx_cost(for two people)': 'cost'
        })
        
        # Process cost
        zomato['cost'] = zomato['cost'].astype(str)
        zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',', '.'))
        zomato['cost'] = zomato['cost'].astype(float)
        
        # Process ratings
        zomato = zomato.loc[zomato.rate != 'NEW']
        zomato = zomato.loc[zomato.rate != '-'].reset_index(drop=True)
        zomato.rate = zomato.rate.apply(lambda x: x.replace('/5', '') if type(x) == str else x)
        zomato.rate = zomato.rate.str.strip().astype('float')
        
        # Calculate mean rating
        restaurants = list(zomato['name'].unique())
        zomato['Mean Rating'] = 0.0
        
        # Process in smaller batches to save memory
        batch_size = 500  # Reduced batch size
        for i in range(0, len(restaurants), batch_size):
            batch = restaurants[i:i+batch_size]
            for restaurant in batch:
                mask = zomato['name'] == restaurant
                zomato.loc[mask, 'Mean Rating'] = float(zomato.loc[mask, 'rate'].mean())
            gc.collect()
        
        # Normalize ratings
        scaler = MinMaxScaler(feature_range=(1, 5))
        zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)
        
        # Free memory
        del scaler
        gc.collect()
        
        return zomato
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_similarity_matrix(zomato):
    """Create TF-IDF matrix and cosine similarities"""
    try:
        # Use a smaller vocabulary size
        tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=2,  # Increased min_df to reduce vocabulary size
            max_df=0.95,  # Added max_df to filter out very common words
            stop_words='english',
            max_features=10000  # Limit vocabulary size
        )
        
        # Process reviews in chunks
        reviews = zomato['reviews_list'].values
        tfidf_matrix = tfidf.fit_transform(reviews)
        
        # Free memory
        del reviews
        gc.collect()
        
        # Calculate cosine similarities
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Free memory
        del tfidf
        gc.collect()
        
        return tfidf_matrix, cosine_similarities
    except Exception as e:
        st.error(f"Error creating similarity matrix: {str(e)}")
        return None, None

def get_recommendations(restaurant_name, zomato, cosine_similarities):
    """Get restaurant recommendations"""
    try:
        # Find the index of the restaurant
        idx = zomato[zomato['name'] == restaurant_name].index[0]
        
        # Get similarity scores
        score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
        
        # Get top 10 similar restaurants
        top10_indexes = list(score_series.iloc[1:11].index)
        
        # Create result dataframe
        result_df = pd.DataFrame(columns=['Restaurant', 'Cuisines', 'Average Cost for Two', 'Rating', 'Location'])
        
        for idx in top10_indexes:
            restaurant = zomato.iloc[idx]
            result_df = pd.concat([result_df, pd.DataFrame({
                'Restaurant': [restaurant['name']],
                'Cuisines': [restaurant['cuisines']],
                'Average Cost for Two': [f"₹{restaurant['cost']}"],
                'Rating': [f"{restaurant['Mean Rating']:.1f}/5.0"],
                'Location': [restaurant['location']]
            })])
        
        return result_df
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None

# Main application logic
def main():
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner('Loading data...'):
            zomato = load_and_process_data()
            if zomato is not None:
                st.session_state.zomato = zomato
                st.session_state.data_loaded = True
            else:
                st.error("Failed to load data. Please check if the zomato.csv file exists and is properly formatted.")
                st.stop()
    
    # Create similarity matrix if not already created
    if st.session_state.tfidf_matrix is None:
        with st.spinner('Creating similarity matrix...'):
            tfidf_matrix, cosine_similarities = create_similarity_matrix(st.session_state.zomato)
            if tfidf_matrix is not None:
                st.session_state.tfidf_matrix = tfidf_matrix
                st.session_state.cosine_similarities = cosine_similarities
            else:
                st.error("Failed to create similarity matrix. Please try again.")
                st.stop()
    
    # Create the Streamlit interface
    st.sidebar.header("Search Options")
    
    # Get unique restaurant names
    restaurant_names = sorted(st.session_state.zomato['name'].unique())
    
    # Create search box
    selected_restaurant = st.sidebar.selectbox(
        "Select a restaurant",
        restaurant_names,
        index=0
    )
    
    # Add a search button
    if st.sidebar.button("Find Similar Restaurants"):
        with st.spinner('Finding similar restaurants...'):
            recommendations = get_recommendations(
                selected_restaurant,
                st.session_state.zomato,
                st.session_state.cosine_similarities
            )
            
            if recommendations is not None:
                st.subheader(f"Restaurants similar to {selected_restaurant}")
                st.dataframe(
                    recommendations,
                    column_config={
                        "Restaurant": st.column_config.TextColumn("Restaurant Name"),
                        "Cuisines": st.column_config.TextColumn("Cuisines"),
                        "Average Cost for Two": st.column_config.TextColumn("Cost for Two"),
                        "Rating": st.column_config.TextColumn("Rating"),
                        "Location": st.column_config.TextColumn("Location")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.error("No recommendations found. Please try another restaurant.")
    
    # Add some statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Statistics")
    st.sidebar.write(f"Total Restaurants: {len(st.session_state.zomato['name'].unique())}")
    st.sidebar.write(f"Average Rating: {st.session_state.zomato['Mean Rating'].mean():.2f}/5.0")
    st.sidebar.write(f"Average Cost for Two: ₹{st.session_state.zomato['cost'].mean():.2f}")
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 