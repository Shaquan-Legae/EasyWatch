import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK with the service account
cred = credentials.Certificate('secrets/serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)
app.secret_key = 'MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC9RIWSOqUR7lgp\nopAzbY8mqdy0bpiD5QK6ToUvQTxiE0wGTLIogC+DlGwu/8x0c+ayc2sz+ynlXS3s\n/V6ArDRW3V8pxK/aiWJiksQLz7RxtW64NsH3kNDFkI3FbH4V2FswaxXIg8E14gUb\n731Wow/Vegrev7a65Snl2sr2arSQi38qUNio3FYojDgeJ4ukLQam8XHTYPL0Pejw\nukc7GxZxlngfJQvHhoZxrmUG3pQkDB7OmHK+UExpGfp2ouXKgFDxrDFOTXIF380r\nHQrv2urn06x8Dn+CYo1ijN7WhHDzJdPx8iuwhLj8jMXM077kXx0cMJ8hgQbjz6gm\noLIV2P/9AgMBAAECggEADF2/xHAkfOlvxTij51hPNB2BGCDpwiRiaz3aId5HTbkp\nDHhSY6cMc7Js5x07hJhWCG7WIE1WzW+rIoLje9DEkrBgGWCKhOLZFMu2F3d4uL08\nALlDLyO9IEtzl+Sg0FiUjLNSdwl0xEqCPvME43ZAAJi6wRcRn2B9vlDvVe5e4dq+\nqRzxNrTbq6ZsxaIZasD6YkS0mH0lY3UrFVZkhKZ/EcLWYbAhjod9V6afRNZaPeBS\n2tkitWVhVwks8YmEuzsmCjslNCPQkP9N5T+N3vTgnUrLiJ1wDCCd71DQvfuiBiun\npkwVLGvr1ueGCHcVAvuX5qIqBHE/ifaMWdImPiuvAQKBgQDkyyhywA7RixJJMA+q\nC1QxN2JlpIzZVYmbAbyMf5vC8YCBbmf4vZ2ScDQzan3vOhcchUEnuX5lLMwo714n\n86U4ZhfZSLyFqdDYj/IwV4dfWGMIL9z0ROSyK/0orLKZ0jl/GCngNxh7o+JYAdSt\nQy4yY9w6bMAoC6toFab0C9xW3QKBgQDTxiGuv6qUugxDGqQTDFQK7cKH3eUWHcY1\nJvs6Icn0bTMCvE5lveuXY+LDR5/zsnFRfdU6IKysQzVOy6KWk05JILt6JFq3OMr8\ndvhyqGxxAW0BW8yWRER1dnyTrewlOEIxlJE0XfnLlex+xcnFlD3HExRiZeGDqMlb\nqjwZnFhroQKBgQCxs7Fo3w8jZZcATVn9QutThqbgN1xGeY91W3Xs0jhSw2yCGxSa\nN/wo/wksXiwOINpOhplCl2o6fv7bVH+XHEBZe8JOO5ZhYrIZYkRDk9hLD8VrWHGP\nL+tJD62DaA8YNhX+RvBPe7uCXJmyrUlYgXNiI/mrT0g4UkgBgb+4kXXD6QKBgEGc\nELNjDiYlvcbMaii8mQ0JvEr7pA3GC7JL8WmBLmBbtIIUdPVwcZzlhUua1SNbFWB5\n66WmmdiEue8/h4++83IUggDFYpWBMuIkubRMGcyo9GyHVEr5u3voyY+3QoIFe/yA\nWrwuoUVBnZNpE6ny03DpqFcT2VpA4KFVjGki1wIBAoGBAIW3kAHvbTq7dApdAmQd\nNxmznOX/O3EtyknmlXNjjTR+ZR+S/FmpdsKBOTuBvnHim7gf5iHKX/rVhFtcT69c'  # Replace with a secure key in production

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

# Route for the loading page
@app.route('/')
def loading():
    return render_template('loading.html')

# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email and password:
            user_ref = db.collection('users').document(email).get()
            if user_ref.exists:
                user_data = user_ref.to_dict()
                if user_data.get('password') == password:  # Warning: Plaintext comparison; use hashing in production
                    session['user'] = email
                    session['dob'] = user_data.get('dob')
                    return redirect(url_for('index'))
                return render_template('login.html', error="Incorrect password.")
            return render_template('login.html', error="Email not found.")
        return render_template('login.html', error="Please fill in all fields.")
    return render_template('login.html')

# Route for the signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        dob = request.form.get('dob')
        if email and password and dob:
            # Check if user already exists
            user_ref = db.collection('users').document(email).get()
            if user_ref.exists:
                return render_template('signup.html', error="Email already registered.")
            # Store user data in Firestore
            user_ref = db.collection('users').document(email)
            user_ref.set({
                'email': email,
                'password': password,  # Warning: Store passwords hashed in production (e.g., with bcrypt)
                'dob': dob,
                'created_at': firestore.SERVER_TIMESTAMP
            })
            session['user'] = email
            session['dob'] = dob
            return redirect(url_for('index'))
        return render_template('signup.html', error="Please fill in all fields.")
    return render_template('signup.html')

# Route for the homepage
@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))  # Changed from signup to login
    return render_template('index.html', movies=df['title'].tolist())

# Route for getting recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user' not in session:
        return jsonify({'error': 'Please log in first.'})
    movie_title = request.form.get('movie_title')
    recommendations, error = get_recommendations(movie_title, df, cosine_sim, 3)
    if error:
        return jsonify({'error': error})
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)