import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

movies = pd.DataFrame({
    'title': ['The Dark Knight', 'Inception', 'Interstellar',
              'The Avengers', 'Iron Man', 'Spider-Man',
              'The Godfather', 'Pulp Fiction', 'Fight Club',
              'Toy Story', 'Finding Nemo', 'The Lion King'],
    'genres': ['action crime drama', 'action sci-fi thriller', 'sci-fi drama adventure',
               'action sci-fi superhero', 'action sci-fi superhero', 'action superhero adventure',
               'crime drama thriller', 'crime drama thriller', 'drama thriller',
               'animation comedy adventure', 'animation comedy adventure', 'animation drama adventure']
})

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])
similarity = cosine_similarity(tfidf_matrix)

movie = st.selectbox("Select a movie:", movies['title'].tolist())

if st.button("Recommend"):
    idx = movies[movies['title'] == movie].index[0]
    scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:4]
    st.subheader("You might also like:")
    for i, _ in scores:
        st.write(f"🎬 {movies['title'][i]}")
