import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st

# def preprocess_text(text):
#     #stop_words = set(stopwords.words('english')) #stopwords to remove the words like is,the,am...
#     #lowering all the text to maintain to uniformity followed by it 
#     #makes a translation table which removes the all punctuation signs
#     #string.punctuation: contains all the punctuation
#     text = text.lower().translate(str.maketrans('', '', string.punctuation))

#     #tokenizing the all the strings to get a list ["a","b"...]

#     tokens = word_tokenize(text)

#     #returns the joined string from the tokens(list of words) by removing stopwords
#     return ' '.join([word for word in tokens if word not in stop_words])

class MovieRecommender:
    #constructor
    def __init__(self): #whenever the class is created the constructor created and loads
        #necessary infos..
        self.df = pd.read_pickle('processed_movies.pkl') #Loads the preprocessed movies file 
        #soup represents the single string created by joining all the words- vertically we are joining

        self.df['soup'] = self.df.apply(self.create_soup, axis=1)
        #vectorizing using TFID(Term frequncy Inverse)
        self.vectorizer = TfidfVectorizer(stop_words='english',max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['soup'])
        self.sia = SentimentIntensityAnalyzer()

    #creates a soup column with all the info combined
    def create_soup(self, row):
        #for row it joins all the genres which we actually parsed them into the lists of strings
        genres = ' '.join(row['genres'])
        companies = ' '.join(row['production_companies'])
        lang = row['original_language']
        overview = row['overview'].lower()
        return f"{genres} {companies} {lang} {overview}"
    
    #recommends the movies based on the cosine simillarity - Content based recommendation
    def get_recommendations(self, title, n=10, min_sentiment=0): #default recommendation is 10
        if title not in self.df['title'].values:
            return [] #checks for the movie title in the all the movies
        idx = self.df[self.df['title'] == title].index[0]
        movie_vec = self.tfidf_matrix[idx]
        # Compute similarity (O(N))
        sim_scores = cosine_similarity(movie_vec, self.tfidf_matrix).flatten()
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        rec_df = self.df.iloc[movie_indices].copy()
        rec_df['sentiment'] = rec_df['overview'].apply(lambda o: self.sia.polarity_scores(o)['compound'])
        rec_df = rec_df[rec_df['sentiment'] > min_sentiment][:n]
        return rec_df[['title', 'vote_average', 'popularity', 'year']].to_dict('records')

    def get_movie_details(self, title):
        api_key = st.secrets.get("OMDB_API_KEY", "")
        details = {'actors': [], 'director': '', 'trailer_id': None}
        if api_key:
            url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
            response = requests.get(url)
            if response.status_code == 200 and response.json().get('Response') == 'True':
                data = response.json()
                details['actors'] = data.get('Actors', '').split(', ')
                details['director'] = data.get('Director', '')
        # YouTube trailer ID
        yt_key = st.secrets.get("YOUTUBE_API_KEY", "")
        if yt_key:
            yt_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={title}+official+trailer&type=video&maxResults=1&key={yt_key}"
            yt_response = requests.get(yt_url)
            if yt_response.status_code == 200:
                yt_data = yt_response.json()
                if yt_data['items']:
                    details['trailer_id'] = yt_data['items'][0]['id']['videoId']
        if not details['trailer_id']:
            details['trailer_id'] = 'dQw4w9WgXcQ'  # Fallback video ID (Rick Roll for testing; replace with real fallback)
        return details
    
if __name__=='__main__':
    import streamlit as st  # For st.secrets in testing
    rec = MovieRecommender()
    print(rec.get_recommendations('Inception', n=5))
    print(rec.get_movie_details('Inception'))  