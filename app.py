"""
TMDB + IMDb Movie Recommendation System - Streamlit Application

A poster-based, content-based movie recommendation web application
with recursive exploration and explainable recommendations.
"""

import streamlit as st
import pandas as pd
import importlib
from recommender import MovieRecommender

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* === Title button styled as a heading text === */
    .st-key-home_link {
        text-align: center;
    }
    .st-key-home_link button {
        background: transparent !important;
        border: none !important;
        color: #E50914 !important;
        font-size: 2.5rem !important;
        font-weight: bold !important;
        padding: 5px 15px !important;
        box-shadow: none !important;
        margin: 0 auto !important;
        display: block !important;
        width: auto !important;
        cursor: pointer !important;
    }
    .st-key-home_link button:hover {
        color: #ff6b6b !important;
        background: transparent !important;
        box-shadow: none !important;
        transform: scale(1.02);
    }
    .st-key-home_link button:focus,
    .st-key-home_link button:active {
        box-shadow: none !important;
        border: none !important;
        outline: none !important;
        background: transparent !important;
    }

    .subtitle {
        text-align: center;
        opacity: 0.6;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        color: #E50914;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 25px 0 15px 0;
        border-bottom: 2px solid #E50914;
        padding-bottom: 8px;
    }

    /* === Movie text (theme-adaptive: inherits Streamlit's text color) === */
    .movie-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin: 8px 0 4px 0;
        line-height: 1.3;
        /* No color - inherits theme */
    }
    .movie-meta {
        font-size: 0.8rem;
        opacity: 0.65;
        margin: 2px 0;
    }

    /* === Generic poster-button (the button IS the poster image) === */
    /* Targets any button whose key starts with "poster_" via .st-key-poster_ class */
    [class*="st-key-poster_"] button {
        background-size: cover !important;
        background-position: center !important;
        background-color: #1a1a2e !important;
        width: 100% !important;
        aspect-ratio: 2 / 3 !important;
        min-height: 280px !important;
        border: 2px solid transparent !important;
        border-radius: 8px !important;
        color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s !important;
        cursor: pointer !important;
        font-size: 0 !important;
    }
    [class*="st-key-poster_"] button:hover {
        transform: scale(1.1) !important;
    }
    [class*="st-key-poster_"] button:focus,
    [class*="st-key-poster_"] button:active {
        outline: none !important;
        color: transparent !important;
    }
    /* Hide any text/icons inside the poster button */
    [class*="st-key-poster_"] button p,
    [class*="st-key-poster_"] button div {
        opacity: 0 !important;
        color: transparent !important;
    }

    /* === Badges === */
    .similarity-badge {
        background: linear-gradient(90deg, #E50914, #ff6b6b);
        color: white !important;
        padding: 3px 10px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 5px;
    }
    .rating-badge {
        background: linear-gradient(90deg, #f39c12, #f1c40f);
        color: #1a1a2e !important;
        padding: 3px 10px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
    }
    .overview-text {
        font-size: 0.9rem;
        line-height: 1.5;
        margin-top: 10px;
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750/1a1a2e/E50914?text=No+Poster"
CACHE_VERSION = 7


@st.cache_resource
def load_recommender(version=CACHE_VERSION):
    """Load and cache the movie recommender system."""
    with st.spinner("Loading movie database..."):
        import preprocessing
        import recommender as rec_module
        importlib.reload(preprocessing)
        importlib.reload(rec_module)
        df = preprocessing.preprocess_data(
            filepath="TMDB  IMDB Movies Dataset.csv",
            sample_size=15000,
            min_votes=500
        )
        recommender_instance = rec_module.MovieRecommender(df)
    return recommender_instance


def get_poster_url(movie: dict) -> str:
    """Get poster URL with fallback to placeholder."""
    poster_url = movie.get("poster_url")
    if poster_url and pd.notna(poster_url):
        return poster_url
    return PLACEHOLDER_POSTER


def render_clickable_poster(movie: dict, key_prefix: str, show_similarity: bool = False) -> bool:
    """
    Render the poster image AS the button (background-image).
    The whole poster is clickable. No separate button below.
    """
    poster_url = get_poster_url(movie)
    movie_idx = movie["index"]
    title = movie["title"]
    title_display = title[:35] + ('...' if len(title) > 35 else '')

    # Key starts with "poster_" so global CSS catches it
    btn_key = f"poster_{key_prefix}_{movie_idx}"

    # Inject the poster URL as the button's background image
    st.markdown(
        f'<style>div.st-key-{btn_key} button {{ background-image: url("{poster_url}") !important; }}</style>',
        unsafe_allow_html=True
    )

    # The poster IS the button — click anywhere on it to navigate
    clicked = st.button(" ", key=btn_key, help=f"Click to explore {title}")

    # Title (plain text, theme-adaptive)
    st.markdown(f'<div class="movie-title">{title_display}</div>', unsafe_allow_html=True)

    # Metadata
    if show_similarity:
        release = movie['release_date'][:10] if movie['release_date'] != 'Unknown' else 'Unknown'
        st.markdown(
            f'<span class="similarity-badge"> {movie["similarity_score"]}% like</span>'
            f'<span class="rating-badge">⭐ {movie["vote_average"]}</span>'
            f'<div class="movie-meta">published:{release}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="movie-meta">⭐ {movie["vote_average"]} · {movie["genres"][:35]}</div>',
            unsafe_allow_html=True
        )

    return clicked


def render_selected_movie_detail(movie: dict):
    """Render detailed view of the selected movie."""
    poster_url = get_poster_url(movie)

    st.markdown('<div class="section-header">🎬 Selected Movie</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(poster_url, use_container_width=True)

    with col2:
        st.markdown(f"## {movie['title']}")

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Rating", f"⭐ {movie['vote_average']}/10")
        with mcol2:
            release = movie['release_date'][:10] if movie['release_date'] != "Unknown" else "Unknown"
            st.metric("Release", release)
        with mcol3:
            genre_count = len(movie.get('genres_list', []))
            st.metric("Genres", f"{genre_count} genres")

        st.markdown(f"**Genres:** {movie['genres']}")

        if movie.get('directors'):
            st.markdown(f"**Director(s):** {movie['directors']}")

        if movie.get('cast'):
            cast_preview = ", ".join(movie['cast'].split(", ")[:5])
            st.markdown(f"**Cast:** {cast_preview}")

        st.markdown("---")
        st.markdown("**Overview:**")
        st.markdown(f"<div class='overview-text'>{movie['overview']}</div>", unsafe_allow_html=True)


def main():
    recommender = load_recommender()
    all_movies = recommender.get_all_movies()

    if "selected_movie_idx" not in st.session_state:
        st.session_state.selected_movie_idx = None
    if "history" not in st.session_state:
        st.session_state.history = []

    # === Clickable title (styled as a heading via .st-key- CSS) ===
    if st.button("🎬 Movie Recommender", key="home_link"):
        st.session_state.selected_movie_idx = None
        st.session_state.history = []
        st.rerun()

    st.markdown(
        '<p class="subtitle">TMDB + IMDb · Content-Based Recommendations · Click any poster to explore</p>',
        unsafe_allow_html=True
    )

    # === MOVIE SELECTED: detail + recommendations ===
    if st.session_state.selected_movie_idx is not None:
        movie_info = recommender.get_movie_info(st.session_state.selected_movie_idx)

        if movie_info is None:
            st.error("Movie not found. Returning to browse.")
            st.session_state.selected_movie_idx = None
            st.rerun()
            return

        if st.session_state.history:
            history_display = " → ".join(st.session_state.history[-5:])
            st.info(f"📍 **Navigation:** {history_display}")

        render_selected_movie_detail(movie_info)

        st.markdown('<div class="section-header">🎯 Top 5 Recommendations</div>', unsafe_allow_html=True)
        st.caption("Click any poster to explore similar movies")

        recommendations = recommender.get_recommendations_by_index(
            st.session_state.selected_movie_idx,
            top_n=5
        )

        if recommendations:
            cols = st.columns(5)
            for i, rec in enumerate(recommendations):
                with cols[i]:
                    clicked = render_clickable_poster(rec, key_prefix=f"rec_{i}", show_similarity=True)

                    with st.expander("💡 Why?"):
                        st.markdown(f"*{rec['explanation']}*")
                        detail = recommender.explain_recommendation(
                            st.session_state.selected_movie_idx, rec['index']
                        )
                        if detail['common_genres']:
                            st.markdown(f"**Genres:** {', '.join(detail['common_genres'])}")
                        if detail['common_keywords']:
                            st.markdown(f"**Themes:** {', '.join(detail['common_keywords'][:6])}")

                    if clicked:
                        title_short = rec['title'][:25]
                        if not st.session_state.history or st.session_state.history[-1] != title_short:
                            st.session_state.history.append(title_short)
                        st.session_state.selected_movie_idx = rec['index']
                        st.rerun()
        else:
            st.warning("No recommendations found for this movie.")

        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; opacity:0.5; font-size:0.8rem; padding:20px;">'
            '<b>Movie Recommender</b> · TF-IDF · Cosine Similarity · Content-Based Filtering'
            '</div>',
            unsafe_allow_html=True
        )

    # === HOMEPAGE: BROWSE MOVIES ===
    else:
        st.markdown("---")

        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "🔍 Search movies",
                placeholder="Type movie name (e.g., Inception, Avatar)...",
                key="search_input"
            )
        with col2:
            sort_options = ["Popularity", "Rating", "Title (A-Z)", "Release Date"]
            sort_by = st.selectbox("Sort by", sort_options, index=0)

        st.markdown('<div class="section-header">🎬 Browse Movies</div>', unsafe_allow_html=True)
        st.caption("Click any poster to see recommendations")

        display_df = all_movies.copy()

        if search_query:
            display_df = display_df[
                display_df["title"].str.lower().str.contains(search_query.lower(), na=False)
            ]
            st.success(f"Found **{len(display_df)}** movies matching '{search_query}'")

        if sort_by == "Popularity":
            display_df = display_df.sort_values("popularity", ascending=False)
        elif sort_by == "Rating":
            display_df = display_df.sort_values("vote_average", ascending=False)
        elif sort_by == "Title (A-Z)":
            display_df = display_df.sort_values("title")
        elif sort_by == "Release Date":
            display_df = display_df.sort_values("release_date", ascending=False)

        movies_per_page = 20
        total_movies = len(display_df)
        total_pages = max((total_movies + movies_per_page - 1) // movies_per_page, 1)

        if "page" not in st.session_state:
            st.session_state.page = 0
        if st.session_state.page >= total_pages:
            st.session_state.page = 0

        _, pcol, _ = st.columns([1, 2, 1])
        with pcol:
            p1, p2, p3, p4, p5 = st.columns(5)
            with p1:
                if st.button("⏮️", disabled=(st.session_state.page == 0),
                             key="first", use_container_width=True):
                    st.session_state.page = 0
                    st.rerun()
            with p2:
                if st.button("◀️", disabled=(st.session_state.page == 0),
                             key="prev", use_container_width=True):
                    st.session_state.page -= 1
                    st.rerun()
            with p3:
                st.markdown(
                    f"<div style='text-align:center; padding:8px;'><b>{st.session_state.page + 1} / {total_pages}</b></div>",
                    unsafe_allow_html=True
                )
            with p4:
                if st.button("▶️", disabled=(st.session_state.page >= total_pages - 1),
                             key="next", use_container_width=True):
                    st.session_state.page += 1
                    st.rerun()
            with p5:
                if st.button("⏭️", disabled=(st.session_state.page >= total_pages - 1),
                             key="last", use_container_width=True):
                    st.session_state.page = total_pages - 1
                    st.rerun()

        start_idx = st.session_state.page * movies_per_page
        end_idx = min(start_idx + movies_per_page, total_movies)
        page_movies = display_df.iloc[start_idx:end_idx]

        if len(page_movies) == 0:
            st.warning("No movies found. Try a different search.")
        else:
            cols_per_row = 5
            rows = [page_movies.iloc[i:i + cols_per_row]
                    for i in range(0, len(page_movies), cols_per_row)]

            for row_idx, row_movies in enumerate(rows):
                cols = st.columns(cols_per_row)
                for col_idx, (df_idx, movie) in enumerate(row_movies.iterrows()):
                    with cols[col_idx]:
                        movie_dict = {
                            "title": movie["title"],
                            "genres": movie["genres_display"],
                            "vote_average": movie["vote_average"],
                            "release_date": movie["release_date"],
                            "poster_url": movie.get("poster_url"),
                            "movie_id": movie["id"] if "id" in movie else df_idx,
                            "index": df_idx
                        }
                        if render_clickable_poster(movie_dict, key_prefix=f"browse_{row_idx}_{col_idx}"):
                            st.session_state.history = [movie["title"][:25]]
                            st.session_state.selected_movie_idx = df_idx
                            st.rerun()

        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; opacity:0.5; font-size:0.8rem; padding:20px;">'
            '<b>Movie Recommender</b> · TF-IDF · Cosine Similarity · Explainable AI<br>'
            'Data: TMDB + IMDb'
            '</div>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
