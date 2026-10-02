
import streamlit as st
import pandas as pd
import json
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Smart Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

/* Main title */
.main-title {
    font-size: 42px !important;
    font-weight: 700 !important;
    text-align: center;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px !important;
    margin-bottom: 25px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1rem;
}

/* Dropdowns */
.stSelectbox > div > div {
    border-radius: 10px;
}

/* Search box */
.stTextInput > div > div > input {
    border-radius: 10px;
}

/* Alerts */
.stAlert {
    border-radius: 10px;
}

/* Posters */
div[data-testid="stImage"] img {
    border-radius: 12px;
}

/* Movie cards */
.movie-card {
    padding: 15px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 20px;
}

.movie-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="main-title">
🎬 Smart Movie Recommendation System
</div>

<div class="subtitle">
Discover movies based on your favorite movie, mood, genre, and content similarity 🤖🍿
</div>
""", unsafe_allow_html=True)




st.markdown("""

""", unsafe_allow_html=True)







# ============================================================
# LOAD MOVIE DATA + CAST
# ============================================================

@st.cache_data
def load_data():

    movies = pd.read_csv(
        "/content/tmdb_5000_movies.csv"
    )

    credits = pd.read_csv(
        "/content/tmdb_5000_credits.csv"
    )

    posters = pd.read_csv(
        "/content/movie_posters.csv"
    )

    movies = movies.merge(
        posters[["id", "poster_path"]],
        on="id",
        how="left"
    )

    # Merge movie data with credits
    movies = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        how="left"
    )

    # Remove duplicate titles
    movies = movies.drop_duplicates(
        subset="title_x"
    ).reset_index(drop=True)

    # Use the correct movie title
    movies["title"] = movies["title_x"]

    # Fill missing values
    movies["overview"] = movies["overview"].fillna("")
    movies["genres"] = movies["genres"].fillna("[]")
    movies["keywords"] = movies["keywords"].fillna("[]")
    movies["cast"] = movies["cast"].fillna("[]")

    # Release year
    movies["release_date"] = pd.to_datetime(
        movies["release_date"],
        errors="coerce"
    )

    movies["year"] = movies["release_date"].dt.year

    return movies


movies = load_data()


# ============================================================
# CONVERT JSON DATA
# ============================================================

def get_names(text):

    try:

        data = json.loads(text)

        return " ".join(
            item["name"]
            for item in data
        )

    except:

        try:

            data = ast.literal_eval(text)

            return " ".join(
                item["name"]
                for item in data
            )

        except:

            return ""


def get_cast(text):

    try:

        data = json.loads(text)

        return ", ".join(
            person["name"]
            for person in data[:5]
        )

    except:

        try:

            data = ast.literal_eval(text)

            return ", ".join(
                person["name"]
                for person in data[:5]
            )

        except:

            return "Not available"


movies["genre_text"] = movies[
    "genres"
].apply(get_names)


movies["keyword_text"] = movies[
    "keywords"
].apply(get_names)


movies["cast_names"] = movies[
    "cast"
].apply(get_cast)


# ============================================================
# CREATE ML FEATURES
# ============================================================

movies["features"] = (
    movies["genre_text"] + " " +
    movies["keyword_text"] + " " +
    movies["overview"]
)

movies["features"] = movies[
    "features"
].fillna("")


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

feature_matrix = vectorizer.fit_transform(
    movies["features"]
)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

model = NearestNeighbors(
    n_neighbors=6,
    metric="cosine"
)

model.fit(feature_matrix)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_movie(movie_title):

    movie_index = movies[
        movies["title"] == movie_title
    ].index[0]

    distances, indices = model.kneighbors(
        feature_matrix[movie_index],
        n_neighbors=6
    )

    recommendations = []

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        movie = movies.iloc[index]

        similarity = max(
            0,
            (1 - distance) * 100
        )

        recommendations.append({

            "title": movie["title"],

            "year": movie["year"],

            "genres": movie["genre_text"],

            "overview": movie["overview"],

            "rating": movie["vote_average"],

            "votes": movie["vote_count"],

            "popularity": movie["popularity"],

            "cast_names": movie["cast_names"],

            "poster_path": movie["poster_path"],

            "similarity": similarity
        })

    return recommendations


# ============================================================
# MOOD MAPPING
# ============================================================

mood_mapping = {

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
# DISPLAY MOVIE
# ============================================================

def display_movie(movie, similarity=None):

    st.markdown(
        '<div class="movie-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="movie-title">🎬 '
        + str(movie["title"])
        + '</div>',
        unsafe_allow_html=True
    )

    # Poster and movie information
    poster_col, info_col = st.columns([1, 3])

    with poster_col:

        poster = movie.get(
            "poster_path",
            None
        )

        if (
            isinstance(poster, str)
            and poster.startswith("http")
        ):

            st.image(
                poster,
                width=220
            )

        else:

            st.info(
                "🖼️ Poster not available"
            )

    with info_col:

        col1, col2 = st.columns(2)

        with col1:

            if pd.notna(movie["year"]):

                st.write(
                    "📅 **Release Year:**",
                    int(movie["year"])
                )

            else:

                st.write(
                    "📅 **Release Year:** Not available"
                )

            if movie["genres"]:

                st.write(
                    "🎭 **Genres:**",
                    movie["genres"]
                )

            else:

                st.write(
                    "🎭 **Genres:** Not available"
                )

            rating = movie.get(
                "rating",
                movie.get("vote_average", None)
            )

            if (
                pd.notna(rating)
                and float(rating) > 0
            ):

                st.write(
                    f"⭐ **Rating:** "
                    f"{float(rating):.1f}/10"
                )

            else:

                st.write(
                    "⭐ **Rating:** Not available"
                )

        with col2:

            cast = movie.get(
                "cast_names",
                "Not available"
            )

            st.write(
                "👥 **Main Cast:**",
                cast
            )

            votes = movie.get(
                "votes",
                movie.get("vote_count", None)
            )

            if (
                pd.notna(votes)
                and float(votes) > 0
            ):

                st.write(
                    f"🗳️ **Votes:** "
                    f"{int(votes):,}"
                )

            else:

                st.write(
                    "🗳️ **Votes:** Not available"
                )

            popularity = movie.get(
                "popularity",
                None
            )

            if pd.notna(popularity):

                st.write(
                    f"📈 **Popularity:** "
                    f"{float(popularity):.1f}"
                )

            else:

                st.write(
                    "📈 **Popularity:** Not available"
                )

            if similarity is not None:

                st.write(
                    f"🎯 **Similarity:** "
                    f"{similarity:.1f}%"
                )

    # Recommendation explanation

    if similarity is not None:

        st.write(
            "💡 **Why this movie is recommended:**"
        )

        if similarity >= 70:

            reason = (
                "It has strong content similarity based on "
                "genres, keywords, and movie description."
            )

        elif similarity >= 50:

            reason = (
                "It shares several characteristics with "
                "your selected movie."
            )

        else:

            reason = (
                "It has some content similarities with "
                "your selected movie."
            )

        st.info(reason)

    # Movie overview

    st.write(
        "📝 **About the Movie:**"
    )

    overview = movie.get(
        "overview",
        ""
    )

    if (
        isinstance(overview, str)
        and overview.strip()
    ):

        st.write(
            overview
        )

    else:

        st.info(
            "Movie description is not available."
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )



# ============================================================
# TITLE
# ============================================================


st.divider()


# ============================================================
# SIMILAR MOVIES
# ============================================================

st.header(
    "🎬 1. Find Similar Movies"
)

movie_list = movies[
    "title"
].sort_values().tolist()

selected_movie = st.selectbox(
    "🎥 Select a movie you like:",
    movie_list
)


if st.button(
    "✨ Recommend Movies",
    key="recommend_button"
):

    selected_data = movies[
        movies["title"] == selected_movie
    ].iloc[0]

    selected_display = {

        "title": selected_data["title"],

        "year": selected_data["year"],

        "genres": selected_data["genre_text"],

        "overview": selected_data["overview"],

        "rating": selected_data["vote_average"],

        "votes": selected_data["vote_count"],

        "popularity": selected_data["popularity"],

        "cast_names": selected_data["cast_names"],

        "poster_path": selected_data["poster_path"]
    }

    st.subheader(
        "🎥 Your Selected Movie"
    )

    display_movie(
        selected_display
    )

    st.divider()

    st.subheader(
        "🍿 Movies You May Like"
    )

    recommendations = recommend_movie(
        selected_movie
    )

    for movie in recommendations:

        display_movie(
            movie,
            movie["similarity"]
        )


# ============================================================
# MOOD SEARCH
# ============================================================

st.divider()

st.header(
    "💭 2. Search by Mood or Genre"
)

st.write(
    "Try: comedy, emotional, romantic, "
    "scary, action, adventure, fantasy, "
    "crime, mystery, animation, etc."
)

search = st.text_input(
    "🔎 What kind of movie are you looking for?",
    placeholder="Example: comedy"
)


if st.button(
    "🔎 Find Movies",
    key="mood_button"
):

    if not search.strip():

        st.warning(
            "Please enter a mood or genre."
        )

    else:

        search_lower = (
            search.lower().strip()
        )

        selected_genres = mood_mapping.get(
            search_lower,
            [search]
        )

        results = movies[
            movies["genre_text"].apply(
                lambda x: any(
                    genre.lower() in x.lower()
                    for genre in selected_genres
                )
            )
        ]

        results = results.head(10)

        if len(results) == 0:

            st.warning(
                "No movies found. "
                "Try another mood or genre."
            )

        else:

            st.success(
                f"🎉 Found {len(results)} movies "
                f"for '{search}'."
            )

            for _, movie in results.iterrows():

                movie_display = {

                    "title": movie["title"],

                    "year": movie["year"],

                    "genres": movie["genre_text"],

                    "overview": movie["overview"],

                    "rating": movie["vote_average"],

                    "votes": movie["vote_count"],

                    "popularity": movie["popularity"],

                    "cast_names": movie["cast_names"],

                    "poster_path": movie["poster_path"]
                }

                display_movie(
                    movie_display
                )


# ============================================================
# ABOUT PROJECT
# ============================================================


st.divider()

st.caption(
    "🎬 Smart Movie Recommendation System | "
    "Machine Learning + Python + Streamlit"
)

# ============================================================
# 3. PERSONALIZED RECOMMENDATIONS
# ============================================================

st.markdown("---")

st.header("🎯 Personalized Movie Recommendations")

st.write(
    "Choose your preferred genre and minimum rating "
    "to discover movies you may like."
)

# Clean genre options for users
genre_options = [
    "All Genres",
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
    "War"
]

# Genre selection
selected_genre = st.selectbox(
    "🎭 Choose a Genre",
    genre_options
)

# Minimum rating
minimum_rating = st.selectbox(
    "⭐ Minimum Rating",
    [5.0, 6.0, 7.0, 8.0, 9.0],
    index=1
)

# Number of movies
number_of_movies = st.slider(
    "🎬 Number of Movies",
    min_value=5,
    max_value=20,
    value=10
)

# Filter movies
personalized_movies = movies.copy()

if selected_genre != "All Genres":

    personalized_movies = personalized_movies[
        personalized_movies["genres"]
        .fillna("")
        .str.contains(
            selected_genre,
            case=False,
            regex=False
        )
    ]

# Rating filter
personalized_movies = personalized_movies[
    pd.to_numeric(
        personalized_movies["vote_average"],
        errors="coerce"
    ).fillna(0) >= minimum_rating
]

# Sort results
personalized_movies = personalized_movies.sort_values(
    by=["vote_average", "popularity"],
    ascending=False
)

# Select number of movies
personalized_movies = personalized_movies.head(
    number_of_movies
)

if len(personalized_movies) > 0:

    st.success(
        f"🎉 Found {len(personalized_movies)} movies for you!"
    )

    for _, movie in personalized_movies.iterrows():

        movie_data = movie.to_dict()

        display_movie(movie_data)

else:

    st.warning(
        "😔 No movies found with these preferences. "
        "Try lowering the minimum rating or choosing another genre."
    )

# ============================================================
# 4. SEARCH MOVIES
# ============================================================

st.markdown("---")

st.header("🔍 Search for a Movie")

search_text = st.text_input(
    "🎬 Enter a movie name",
    placeholder="Example: Avatar, Titanic, Batman..."
)

# Year filter
year_values = pd.to_numeric(
    movies["year"],
    errors="coerce"
).dropna().astype(int)

year_options = ["All Years"] + sorted(
    year_values.unique(),
    reverse=True
)

selected_year = st.selectbox(
    "📅 Release Year",
    year_options
)

# Search movies
if search_text.strip():

    search_query = search_text.strip().lower()

    # First check for an exact movie title
    exact_results = movies[
        movies["title"]
        .fillna("")
        .str.lower()
        .eq(search_query)
    ]

    # If exact title exists, show it
    if len(exact_results) > 0:

        search_results = exact_results

    else:

        # Otherwise find titles that start with the search text
        search_results = movies[
            movies["title"]
            .fillna("")
            .str.lower()
            .str.startswith(search_query)
        ]

    # Apply year filter
    if selected_year != "All Years":

        search_results = search_results[
            pd.to_numeric(
                search_results["year"],
                errors="coerce"
            ) == int(selected_year)
        ]

    search_results = search_results.head(10)

    if len(search_results) > 0:

        st.success(
            f"🎉 Found {len(search_results)} matching movies!"
        )

        for _, movie in search_results.iterrows():

            movie_data = movie.to_dict()

            display_movie(movie_data)

    else:

        st.warning(
            "😔 No matching movies found."
        )

        st.info(
            "💡 Try another movie name or select "
            "'All Years' from the Release Year dropdown."
        )
