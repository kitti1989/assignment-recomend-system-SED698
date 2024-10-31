import streamlit as st
import pickle
import pandas as pd
from surprise import SVD

with open('recommendation_movie_svd_66130701703.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

st.title("Movie Recommendation System")
st.write("Enter your user ID to get movie recommendations.")

user_id = st.number_input("User ID", min_value=1, step=1)

if st.button("Get Recommendations"):
    # Filter out movies the user has already rated
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]

    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    top_recommendations = sorted_predictions[:10]

    st.write(f"\nTop 10 movie recommendations for User {user_id}:")
    for recommendation in top_recommendations:
        movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
        st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")

