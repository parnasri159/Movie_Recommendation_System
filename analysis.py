import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import requests
import streamlit as st

def load_data():
    return pd.read_pickle('processed_movies.pkl')

def predict_rating(features): #here features is a dictionary
    df = load_data() #loading the data
    mlb = MultiLabelBinarizer() #mlb actually converts list of labels -> one hot encoder
    # ['Action','Drama'] => [0,1,1,0]....
    #transforms the genre column into matrix (binary)

    genres_onehot = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_, index=df.index)
    #genres onhot will be a 2d matrix like having labels as columns -> 1 present and 0 absent
    #creating X (independent set) by merging these columns with genres_onehot
    X = pd.concat([df[['revenue', 'popularity', 'runtime']], genres_onehot], axis=1)
    y = df['vote_average'] #dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #splitting into train and test dataset
    model = LinearRegression() #initializing model simple linear regression model
    model.fit(X_train, y_train) #fittting the train data
    #converting the input genres into one hot -> as changed earlier it will return genre else []
    #transform method we have to use. as we already fitted the earlier data
    genres_input = pd.DataFrame(mlb.transform([features.get('genres', [])]), columns=mlb.classes_)
    # here get method used to fetch the respective data out of the dictionary else 0 is returned.
    input_df = pd.DataFrame([[features.get('revenue', 0), features.get('popularity', 0), features.get('runtime', 0)]], 
                            columns=['revenue', 'popularity', 'runtime'])
    #full df -> by concatenating
    input_full = pd.concat([input_df, genres_input], axis=1).reindex(columns=X.columns, fill_value=0)
    return round(model.predict(input_full)[0],2) #returns prediction

def get_trends(): #returns the trends 
    df = load_data()
    #uses groupby function it to group by year based on the revenue and mean
    trends = df.groupby('year')['revenue'].mean().reset_index()
    return trends.to_dict('records') #returns the trends as a form of dict

def get_correlations(): #getting the correlations based on the budget revenue popularity vote avg runtime
    df = load_data()
    corr = df[['budget', 'revenue', 'popularity', 'vote_average', 'runtime']].corr()
    corr=corr.round(2)
    return corr.to_dict() #returns the ans in the form of dictionary

def genre_popularity_over_time(): #getting the genre popularity over the time
    df = load_data()
    df_exploded = df.explode('genres') #it explodes like for a movie multiple genres
    #then it will convert the movie for each genre one row
    #like movie1 ["action",'drama]
    #after exploding it will be movie1 action and movie 1 drama 
    genre_trends = df_exploded.groupby(['year', 'genres'])['popularity'].mean().unstack().fillna(0) #unstack opens up the genres to be
    #the new columns and then replaces the NA values with 0
    return genre_trends

def extract_keywords_from_overview(movie_title, n=10):#n represents the keywords to return
    df = load_data() 
    overview = df[df['title'] == movie_title]['overview'].values[0] #extracts the overview from the df where movie title matches 
    stop_words = set(stopwords.words('english')) #here stopwords are used to remove unwanted words
    tokens = word_tokenize(overview.lower().translate(str.maketrans('', '', string.punctuation))) #same preprocessing: lower->remove punctuations (translate) -> tokenize
    freq = FreqDist([w for w in tokens if w not in stop_words]) #builds a freq distribution 
    #like ['king':3,'man':2...]
    return freq.most_common(n) #and returns the most common 

def get_sentiment_scores():
    df = load_data() #load data
    sia = SentimentIntensityAnalyzer() #initializing the SIA
    #create sentiment column-> overview apply lambda-> returns polarity scores in range of -1(Negative) to 1(Positive)
    df['sentiment'] = df['overview'].apply(lambda o: sia.polarity_scores(o)['compound'])
    #returns scores and title in the form of dict
    df['sentiment']=df['sentiment'].apply(lambda x: round(x,2))
    return df[['title', 'sentiment']].to_dict('records')

def get_top_movies(genre=None, year=None, sort_by='popularity', n=10): #Top movies in particular year and genre
    df = load_data() 
    #if genre is provided then df['genre'] -> list of genre...check the genre in the list
    if genre:
        df = df[df['genres'].apply(lambda x: genre in x)]
    #year-> extract that df where df['year']==year
    if year:
        df = df[df['year'] == year]
    #returning the dictionary containing these 4 -> sorted wrt to popularity(descending-> top movies first)
    return df[['title', 'popularity', 'vote_average', 'year']].sort_values(sort_by, ascending=False)[:n].to_dict('records')

def get_runtime_impact():
    df = load_data()
    # cutting dataframe into the frames with bins givens where labels also provided
    df['runtime_bin'] = pd.cut(df['runtime'], bins=[0, 90, 120, 150, 180, 300], labels=['<90', '90-120', '120-150', '150-180', '>180'])
    #grouping by runtime_bin->
    impact = df.groupby('runtime_bin',observed=False)[['popularity', 'revenue']].mean().reset_index()
    impact=impact.round(2)
    return impact.to_dict('records')

def get_movies_by_actor(actor_name, api_key=''):
    df = load_data()
    api_key = st.secrets.get("OMDB_API_KEY", "")
    movies = []
    if api_key:
        url = f"http://www.omdbapi.com/?s={actor_name}&type=movie&apikey={api_key}"
        response = requests.get(url)
        if response.status_code == 200 and response.json().get('Response') == 'True':
            for item in response.json().get('Search', []):
                title = item['Title']
                if title in df['title'].values:
                    movies.append(df[df['title'] == title][['title', 'year', 'vote_average']].iloc[0].to_dict())
    return movies[:10]

if __name__ == '__main__':
    print(get_sentiment_scores()[:5])
    print(get_top_movies(genre='Action', n=5))
    print(get_runtime_impact())