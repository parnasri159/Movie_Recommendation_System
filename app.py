import streamlit as st
from recommender import MovieRecommender
from analysis import predict_rating, get_trends, get_correlations, get_sentiment_scores, get_top_movies, get_runtime_impact, get_movies_by_actor
from viz import plot_revenue_trends, plot_correlation_heatmap, plot_runtime_impact
import pandas as pd
import pickle

# Cache recommender
@st.cache_resource
def load_recommender():
    return MovieRecommender()

rec = load_recommender()
movies_list = rec.df['title'].sort_values().tolist()

# Watchlist
def load_watchlist():
    try:
        with open('watchlist.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return []

def save_watchlist(watchlist):
    with open('watchlist.pkl', 'wb') as f:
        pickle.dump(watchlist, f)

# Session state
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = load_watchlist()
if 'current_movie' not in st.session_state:
    st.session_state['current_movie'] = None
if 'initial_movie' not in st.session_state:
    st.session_state['initial_movie'] = None

# Sexy UI CSS
st.markdown("""
<style>
.movie-card { border: 1px solid #ddd; border-radius: 10px; padding: 10px; text-align: center; background-color: #1f1f1f; }
.movie-card:hover { box-shadow: 0 4px 8px rgba(255,255,255,0.2); transition: 0.3s; }
.stButton > button { width: 100%; background-color: #4CAF50; color: white; }
</style>
""", unsafe_allow_html=True)

# App config
st.set_page_config(page_title="Movie Explorer", layout="wide")
page = st.sidebar.selectbox("Navigate", ["Home", "Recommender", "Analytics", "Watchlist"])

def show_movie_details(movie_title):
    movie_data = rec.df[rec.df['title'] == movie_title].iloc[0]
    st.subheader(f"{movie_title} ({movie_data['year']})")
    col1, col2 = st.columns([1, 2])
    with col1:
        poster_url = f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}" if movie_data['poster_path'] and not pd.isna(movie_data['poster_path']) else "https://via.placeholder.com/300x450?text=No+Poster"
        st.image(poster_url, width=300)
    with col2:
        st.write(f"Overview: {movie_data['overview']}")
        st.write(f"Vote Average: {movie_data['vote_average']}/10")
        st.write(f"Popularity: {movie_data['popularity']:.2f}")
        st.write(f"Revenue: ${movie_data['revenue']:,}")
        st.write(f"Budget: ${movie_data['budget']:,}")
        st.write(f"Runtime: {movie_data['runtime']} min")
        st.write(f"Genres: {', '.join(movie_data['genres'])}")
        st.write(f"Companies: {', '.join(movie_data['production_companies'])}")
        st.write(f"Tagline: {movie_data['tagline']}")
        st.write(f"Status: {movie_data['status']}")
    
    # OMDb details
    with st.spinner("Fetching details..."):
        details = rec.get_movie_details(movie_title)
    st.subheader("Actors/Actresses")
    st.write(", ".join(details['actors']) if details['actors'] else "Not available")
    st.subheader("Director")
    st.write(details['director'] or "Not available")
    
    # Trailer embed
    st.subheader("Trailer")
    if details['trailer_id']:
        st.video(f"https://www.youtube.com/watch?v={details['trailer_id']}")
    else:
        st.write("No trailer available. Search on YouTube.")
    
    # Sentiment
    sentiment = rec.sia.polarity_scores(movie_data['overview'])['compound']
    st.subheader("Sentiment Score (Overview)")
    st.write(f"{sentiment:.2f} (Positive if >0, Negative if <0)")
    
    # Recommendations
    st.subheader("Recommended Movies")
    sentiment_filter = st.slider("Minimum Sentiment Score for Recommendations", -1.0, 1.0, 0.0, key=f"sentiment_{movie_title}")
    with st.spinner("Fetching recommendations..."):
        recs = rec.get_recommendations(movie_title, n=10, min_sentiment=sentiment_filter)
    cols = st.columns(5)
    for i, rec_movie in enumerate(recs):
        title = rec_movie['title']
        rec_data = rec.df[rec.df['title'] == title].iloc[0]
        with cols[i % 5]:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            poster_url = f"https://image.tmdb.org/t/p/w200{rec_data['poster_path']}" if rec_data['poster_path'] and not pd.isna(rec_data['poster_path']) else "https://via.placeholder.com/200x300?text=No+Poster"
            st.image(poster_url, caption=title, use_container_width=True)
            if st.button("Get Details", key=f"rec_detail_{title}_{i}"):
                st.session_state['current_movie'] = title
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Add to Watchlist", key=f"watchlist_{movie_title}"):
        if movie_title not in [m['title'] for m in st.session_state['watchlist']]:
            st.session_state['watchlist'].append(movie_data.to_dict())
            save_watchlist(st.session_state['watchlist'])
            st.success(f"Added {movie_title} to watchlist!")
    
    if st.session_state['initial_movie'] and st.session_state['current_movie'] != st.session_state['initial_movie']:
        if st.button("Back to Original Movie", key=f"back_{movie_title}"):
            st.session_state['current_movie'] = st.session_state['initial_movie']
            st.rerun()

if page == "Home":
    st.title("Welcome to Movie Explorer App")
    st.write("Explore movies: Recommend, Analyze, Save to Watchlist!")
    st.write(f"Loaded {len(rec.df)} movies.")
    st.write("Use the sidebar to navigate. If posters/trailers fail, use VPN or DNS (8.8.8.8).")

elif page == "Recommender":
    st.title("Movie Recommender")
    search = st.text_input("Type Movie Name")
    col1, col2 = st.columns(2)
    with col1:
        genres = st.multiselect("Filter by Genre", ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'])
    with col2:
        year = st.number_input("Filter by Year", 1990, 2025, value=None, step=1)
    filtered_movies = [m for m in movies_list if search.lower() in m.lower()] if search else movies_list
    if genres:
        filtered_movies = [m for m in filtered_movies if any(g in rec.df[rec.df['title'] == m]['genres'].iloc[0] for g in genres)]
    if year:
        filtered_movies = [m for m in filtered_movies if rec.df[rec.df['title'] == m]['year'].iloc[0] == year]
    selected_movie = st.selectbox("Select Movie", filtered_movies, key="movie_select")
    
    if selected_movie:
        st.session_state['initial_movie'] = selected_movie
        if not st.session_state['current_movie']:
            st.session_state['current_movie'] = selected_movie
        show_movie_details(st.session_state['current_movie'])

elif page == "Analytics":
    st.title("Movie Analytics")
    st.subheader("Revenue Trends Over Time")
    st.plotly_chart(plot_revenue_trends(get_trends()))
    st.subheader("Feature Correlations")
    st.plotly_chart(plot_correlation_heatmap(get_correlations()))
    st.subheader("Top Movies")
    genre = st.selectbox("Select Genre", ['All'] + ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'])
    year = st.number_input("Select Year", 1990, 2025, value=None, step=1)
    sort_by = st.selectbox("Sort By", ['popularity', 'vote_average'])
    top_movies = get_top_movies(genre if genre != 'All' else None, year, sort_by)
    st.table(pd.DataFrame(top_movies))
    st.subheader("Runtime Impact")
    st.plotly_chart(plot_runtime_impact(get_runtime_impact()))
    st.subheader("Search Movies by Actor")
    actor = st.text_input("Enter Actor Name (e.g., Tom Hanks)")
    if actor and st.button("Search Actor"):
        actor_movies = get_movies_by_actor(actor, st.secrets.get("OMDB_API_KEY", ""))
        if actor_movies:
            st.table(pd.DataFrame(actor_movies))
        else:
            st.warning("No movies found or OMDb key missing.")
    st.subheader("Predict Movie Rating")
    revenue = st.number_input("Revenue ($)", 0, 1000000000, 100000000)
    popularity = st.number_input("Popularity", 0.0, 1000.0, 50.0)
    runtime = st.number_input("Runtime (min)", 0, 300, 120)
    genres = st.multiselect("Genres for Prediction", ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'])
    if st.button("Predict"):
        pred = predict_rating({'revenue': revenue, 'popularity': popularity, 'runtime': runtime, 'genres': genres})
        st.write(f"Predicted Rating: {pred:.2f}/10")

elif page == "Watchlist":
    st.title("Your Watchlist")
    if st.session_state['watchlist']:
        st.table(pd.DataFrame(st.session_state['watchlist']))
        if st.button("Export Watchlist as CSV"):
            pd.DataFrame(st.session_state['watchlist']).to_csv('watchlist.csv', index=False)
            st.success("Exported to watchlist.csv")
        if st.button("Clear Watchlist"):
            st.session_state['watchlist'] = []
            save_watchlist([])
            st.success("Watchlist cleared")
    else:
        st.write("Your watchlist is empty.")