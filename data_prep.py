import pandas as pd 
import numpy as np 
from datetime import datetime

def load_clean_data(file_path='tmdb_movies.csv'):
    # load dataset
    df=pd.read_csv(file_path)
    print(f'Original Shape:{df.shape}') #printing original shape

    #dropping duplicates
    df.drop_duplicates(subset=['id'],inplace=True)

    # Handle missing values
    df['title']=df['title'].fillna('Unknown')
    df['overview']=df['overview'].fillna('')
    df['genres']=df['genres'].fillna('')
    df['production_companies']=df['production_companies'].fillna('')
    df['original_language']=df['original_language'].fillna('en')
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['tagline']=df['tagline'].fillna('No tagline')
    df['status']=df['status'].fillna('Released')
    df.dropna(subset=['release_date'], inplace=True)

    # Replace zeros with medians
    df['budget'] = df['budget'].replace(0, np.nan)
    df['revenue'] = df['revenue'].replace(0, np.nan)
    df['budget']=df['budget'].fillna(df['budget'].median())
    df['revenue']=df['revenue'].fillna(df['revenue'].median())
    df['runtime']=df['runtime'].fillna(df['runtime'].median())

    #parsing lists : input-> Action,comedy... output->['Action','comedy'] 
    #Making Lists if empty return the empty list
    df['genres'] = df['genres'].apply(lambda x: x.split(',') if x else [])
    df['production_companies'] = df['production_companies'].apply(lambda x: x.split(',') if x else [])

    #Extrating 2 new features from the existing ones
    #Succes Ratio-> gives the ratio of revenue/budget and it will give 0 if budget=0 
    # {as we replaced NaN with the 0 in place of revenue and budget }
    df['year'] = df['release_date'].dt.year
    df['success_ratio'] = np.where(df['budget'] > 0, df['revenue'] / df['budget'], 0)

    #filtering in order to reduce the response time
    # final shape - ~20k to 30k

    df=df[(df['vote_count']>200) & (df['year']>1990)]
    print(f'Filtered Shape: {df.shape}')

    #save the files 
    df.to_pickle('processed_movies.pkl')
    return df 

if __name__ == '__main__':
    load_clean_data()

       