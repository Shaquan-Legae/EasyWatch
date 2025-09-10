import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Sample movie data
movies_data = {
    'title': [
        'The Matrix', 'Inception', 'The Dark Knight', 'Pulp Fiction', 
        'The Shawshank Redemption', 'Interstellar', 'Fight Club', 
        'Forrest Gump', 'The Godfather', 'Avatar'
    ],
    'genres': [
        'Action Sci-Fi', 'Sci-Fi Thriller', 'Action Crime Thriller',
        'Crime Drama', 'Drama', 'Sci-Fi Drama', 'Drama Thriller',
        'Drama Romance', 'Crime Drama', 'Action Sci-Fi Adventure'
    ]
}

# Create a DataFrame
df = pd.DataFrame(movies_data)

# Precompute TF-IDF matrix and cosine similarity for efficiency
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, df, cosine_sim, num_recommendations=5):
    idx = df[df['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return None, f"Movie '{title}' not found in the database."
    
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist(), None

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html', movies=df['title'].tolist())

# Route for getting recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie_title')
    recommendations, error = get_recommendations(movie_title, df, cosine_sim, 3)
    if error:
        return jsonify({'error': error})
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)