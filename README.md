# Zomato Restaurant Recommender

A **Streamlit web application** that recommends similar restaurants based on reviews and ratings using **content-based filtering**.

## Features

- Interactive web interface for easy navigation
- Restaurant search and recommendation functionality
- Detailed restaurant information including:
  - Cuisines
  - Average cost for two
  - Ratings
  - Location
- Dataset statistics to give insights into the data

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/priyansh1908/rs-project.git


## Usage

1. Make sure you have the `zomato.csv` file in the same directory as the application
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## How it Works

The application uses content-based filtering to recommend restaurants:
1. It analyzes restaurant reviews using TF-IDF (Term Frequency-Inverse Document Frequency)
2. Calculates similarity scores between restaurants
3. Recommends the top 10 most similar restaurants based on the selected restaurant

## Data Source

The dataset used in this application is from Zomato, containing information about restaurants in Bangalore, India.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
>>>>>>> c0a769e (rs project)
