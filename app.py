
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import ast
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOVIES_FILE = os.path.join(BASE_DIR, "tmdb_5000_movies.csv")
CREDITS_FILE = os.path.join(BASE_DIR, "tmdb_5000_credits_small.csv")
POSTERS_FILE = os.path.join(BASE_DIR, "movie_posters.csv")


def load_data():

    movies = pd.read_csv(MOVIES_FILE)
    credits = pd.read_csv(CREDITS_FILE)

    # Merge credits
    movies = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        how="left"
    )

    # --------------------------------------------------------
    # Extract cast
    # --------------------------------------------------------

    def get_cast(cast_text, number=5):
        try:
            if pd.isna(cast_text):
                return ""

            cast_list = json.loads(cast_text)

            names = [
                person.get("name", "")
                for person in cast_list[:number]
            ]

            return ", ".join([x for x in names if x])

        except:
            return ""

    movies["cast_names"] = movies["cast"].apply(get_cast)

    # --------------------------------------------------------
    # Clean genres
    # --------------------------------------------------------

    movies["genres"] = movies["genres"].fillna("")

    def get_genres(value):
        try:
            data = json.loads(value)
            return ", ".join([x["name"] for x in data])
        except:
            return ""

    movies["genre_names"] = movies["genres"].apply(get_genres)

    # --------------------------------------------------------
    # Keywords
    # --------------------------------------------------------

    def get_keywords(value):
        try:
            data = json.loads(value)
            return " ".join([x["name"] for x in data])
        except:
            return ""

    if "keywords" in movies.columns:
        movies["keyword_names"] = movies["keywords"].fillna("").apply(get_keywords)
    else:
        movies["keyword_names"] = ""

    # --------------------------------------------------------
    # Overview
    # --------------------------------------------------------

    movies["overview"] = movies["overview"].fillna("")

    # --------------------------------------------------------
    # Recommendation features
    # --------------------------------------------------------

    movies["features"] = (
        movies["genre_names"].fillna("") + " " +
        movies["keyword_names"].fillna("") + " " +
        movies["overview"].fillna("") + " " +
        movies["cast_names"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )

    feature_matrix = vectorizer.fit_transform(movies["features"])

    model = NearestNeighbors(
        n_neighbors=11,
        metric="cosine"
    )

    model.fit(feature_matrix)

    # --------------------------------------------------------
    # Posters
    # --------------------------------------------------------

    try:
        posters = pd.read_csv(POSTERS_FILE)

        poster_map = dict(
            zip(
                posters["id"].astype(str),
                posters["poster_path"]
            )
        )

        movies["poster"] = movies["id"].astype(str).map(poster_map)

    except:
        movies["poster"] = ""

    return movies, feature_matrix, model


movies, feature_matrix, model = load_data()


# ============================================================
# Mood mappings
# ============================================================

MOOD_MAP = {
    "comedy": ["Comedy"],
    "funny": ["Comedy"],
    "emotional": ["Drama", "Romance"],
    "sad": ["Drama"],
    "romantic": ["Romance"],
    "love": ["Romance"],
    "scary": ["Horror", "Thriller"],
    "horror": ["Horror"],
    "thriller": ["Thriller"],
    "exciting": ["Action", "Adventure"],
    "action": ["Action"],
    "adventure": ["Adventure"],
    "fantasy": ["Fantasy"],
    "science fiction": ["Science Fiction"],
    "sci-fi": ["Science Fiction"],
    "crime": ["Crime"],
    "mystery": ["Mystery"],
    "animation": ["Animation"],
    "family": ["Family"],
    "kids": ["Family"],
    "documentary": ["Documentary"],
    "music": ["Music"],
    "war": ["War"],
    "history": ["History"]
}


# ============================================================
# Helper functions
# ============================================================

def movie_details(row, similarity=None):

    year = ""

    try:
        if pd.notna(row["release_date"]):
            year = str(row["release_date"])[:4]
    except:
        pass

    rating = round(float(row["vote_average"]), 1) \
        if pd.notna(row["vote_average"]) else 0

    votes = int(row["vote_count"]) \
        if pd.notna(row["vote_count"]) else 0

    popularity = round(float(row["popularity"]), 1) \
        if pd.notna(row["popularity"]) else 0

    genres = row.get("genre_names", "")

    if similarity is not None:

        similarity_percent = round(
            (1 - similarity) * 100,
            1
        )

        if similarity_percent >= 70:
            explanation = (
                "Strong content similarity based on genres, "
                "keywords, description and cast."
            )

        elif similarity_percent >= 50:
            explanation = (
                "This movie shares several characteristics "
                "with your selected movie."
            )

        else:
            explanation = (
                "This movie has some content similarities "
                "with your selected movie."
            )

    else:
        similarity_percent = None
        explanation = ""

    return {
        "id": int(row["id"]),
        "title": row["title"],
        "year": year,
        "genres": genres,
        "rating": rating,
        "votes": votes,
        "popularity": popularity,
        "cast": row.get("cast_names", ""),
        "overview": row.get("overview", ""),
        "poster": row.get("poster", ""),
        "similarity": similarity_percent,
        "explanation": explanation
    }


def recommend_movie(title, count=6):

    matches = movies[
        movies["title"].str.lower() == title.lower()
    ]

    if matches.empty:
        matches = movies[
            movies["title"].str.lower().str.contains(
                title.lower(),
                na=False
            )
        ]

    if matches.empty:
        return []

    index = matches.index[0]

    distances, indices = model.kneighbors(
        feature_matrix[index],
        n_neighbors=count + 1
    )

    results = []

    for distance, idx in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        row = movies.iloc[idx]

        results.append(
            movie_details(
                row,
                distance
            )
        )

    return results


def mood_recommendations(
    mood="",
    genre="All Genres",
    min_rating=6.0,
    count=10
):

    data = movies.copy()

    # Mood filter
    mood_genres = []

    if mood:
        mood_key = mood.lower().strip()

        if mood_key in MOOD_MAP:
            mood_genres = MOOD_MAP[mood_key]

    if mood_genres:

        data = data[
            data["genre_names"].apply(
                lambda x: any(
                    g.lower() in x.lower()
                    for g in mood_genres
                )
            )
        ]

    # Genre filter
    if genre and genre != "All Genres":

        data = data[
            data["genre_names"].str.contains(
                genre,
                case=False,
                na=False
            )
        ]

    # Rating filter
    data = data[
        data["vote_average"].fillna(0) >= float(min_rating)
    ]

    # Sort
    data = data.sort_values(
        ["vote_average", "popularity"],
        ascending=False
    )

    results = []

    for _, row in data.head(count).iterrows():

        results.append(
            movie_details(row)
        )

    return results


# ============================================================
# Routes
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def api_recommend():

    data = request.get_json() or {}

    title = data.get("title", "")
    count = int(data.get("count", 6))

    if not title:
        return jsonify({
            "success": False,
            "message": "Please enter a movie name."
        })

    results = recommend_movie(
        title,
        count
    )

    if not results:
        return jsonify({
            "success": False,
            "message": "Movie not found. Try another movie name."
        })

    return jsonify({
        "success": True,
        "results": results
    })


@app.route("/api/mood", methods=["POST"])
def api_mood():

    data = request.get_json() or {}

    mood = data.get("mood", "")
    genre = data.get("genre", "All Genres")
    rating = float(data.get("rating", 6.0))
    count = int(data.get("count", 10))

    results = mood_recommendations(
        mood,
        genre,
        rating,
        count
    )

    return jsonify({
        "success": True,
        "results": results
    })


@app.route("/api/search", methods=["POST"])
def api_search():

    data = request.get_json() or {}

    query = data.get("query", "").strip()
    year = data.get("year", "All Years")

    if not query:
        return jsonify({
            "success": False,
            "message": "Please enter a movie name."
        })

    data_movies = movies.copy()

    # Exact title first
    exact = data_movies[
        data_movies["title"].str.lower() == query.lower()
    ]

    if not exact.empty:
        results_data = exact

    else:
        results_data = data_movies[
            data_movies["title"].str.lower().str.startswith(
                query.lower(),
                na=False
            )
        ]

    # Year filter
    if year != "All Years":

        results_data = results_data[
            results_data["release_date"]
            .fillna("")
            .astype(str)
            .str[:4] == str(year)
        ]

    results = []

    for _, row in results_data.head(10).iterrows():

        results.append(
            movie_details(row)
        )

    if not results:
        return jsonify({
            "success": False,
            "message": (
                "😔 No matching movies found. "
                "💡 Try another movie name or select 'All Years'."
            )
        })

    return jsonify({
        "success": True,
        "results": results
    })


# Vercel uses this Flask application
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8501,
        debug=True
    )
