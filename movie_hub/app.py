from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from movie_utils import (
    decade_breakdown,
    genre_breakdown,
    load_movie_data,
    recommend_by_preferences,
    recommend_by_title,
    top_by_rating,
    user_activity_summary,
)

st.set_page_config(
    page_title="Movie hub",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
        [data-testid="stTabList"] {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    button[data-testid="stTab"] {
        flex-grow: 1;
        justify-content: center;
    }

    button[data-testid="stTab"] p {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1rem;
    }
    .soft-card {
        border: 1px solid rgba(120,120,120,0.18);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: rgba(255,255,255,0.03);
    }
    .tiny-note {
        color: #666;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_movie_data()


def fmt_int(x: float) -> str:
    return f"{int(x):,}"


def draw_bar(series: pd.Series, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def draw_hist(series: pd.Series, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.hist(series, bins=15)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def mood_thresholds(mood: str) -> tuple[int, float]:
    if mood == "Crowd favorites":
        return 120, 4.0
    if mood == "Hidden gems":
        return 20, 3.7
    return 50, 3.5


def show_movie_summary_card(movie_row: pd.Series) -> None:
    left, mid, right = st.columns([0.01, 2.0, 0.01])
   
    with mid:
        st.markdown(
            f"""
            <div class="soft-card" style="text-align: center;">
                <div style="font-size:1.25rem; font-weight:700; margin-bottom:0.35rem;">
                    {movie_row['title']}
                </div>
                <div style="margin-bottom: 0.2rem;"><strong>Genres:</strong> {movie_row['genres']}</div>
                <div style="margin-bottom: 0.2rem;"><strong>Average rating:</strong> {movie_row['rating_mean']:.2f} </div>
                <div style="margin-bottom: 0.2rem;"><strong>Votes:</strong> {fmt_int(movie_row['rating_count'])}</div>
                <div><strong>Year:</strong> {"N/A" if pd.isna(movie_row['year']) else int(movie_row['year'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.markdown(
    """
    <h1 style='text-align: center;'>
        🍿 Movie Hub 🍿
    </h1>
    """,
    unsafe_allow_html=True)

    try:
        data = get_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run `python download_data.py` first, then reopen the app.")
        st.stop()

    merged = data.merged.copy()
    ratings = data.ratings
    summary = user_activity_summary(ratings)

    genre_options = ["All"] + sorted(
        merged["genres"].fillna("Unknown").str.get_dummies(sep="|").columns.tolist()
    )
    top_titles = sorted(merged["title"].tolist())

    tab1, tab2, tab3, = st.tabs(["Dashboard", "Recommender", "Preference"])

    with tab1:
        st.subheader("Dashboard")
        st.write("A quick overview of the dataset and the most popular patterns inside it.")

        dash_a, dash_b = st.columns([1.1, 0.9])
        with dash_a:
            focus_genre = st.selectbox(
                "Focus genre",
                genre_options,
                index=0,
                key="dash_genre",
            )
        with dash_b:
            min_votes_dash = st.slider(
                "Minimum votes:",
                1,
                330,
                50,
                1,
                key="dash_min_votes",
            )

        if focus_genre != "All":
            dashboard_view = merged[merged["genres"].str.contains(focus_genre, case=False, na=False)].copy()
        else:
            dashboard_view = merged.copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Movies", fmt_int(dashboard_view["movieId"].nunique()))
        c2.metric("Ratings", fmt_int(summary["ratings"]))
        c3.metric("Users", fmt_int(summary["users"]))
        c4.metric("Avg ratings per user", f"{summary['avg_ratings_per_user']:.1f}")

        left, right = st.columns([1.1, 0.9])
        with left:
            top_movies = top_by_rating(dashboard_view, n=10, min_votes=min_votes_dash)
            st.markdown("#### Top-rated movies in this view")
            st.dataframe(top_movies, use_container_width=True, hide_index=True)
        with right:
            genre_counts = genre_breakdown(dashboard_view).head(10)
            st.markdown("#### Genre frequency")
            st.pyplot(draw_bar(genre_counts, "Most common genres", "Genre", "Count"))

        b1, b2 = st.columns(2)
        with b1:
            decade_counts = decade_breakdown(dashboard_view)
            if not decade_counts.empty:
                st.markdown("#### Movies by decade")
                st.pyplot(draw_bar(decade_counts, "Movies by decade", "Decade", "Count"))
            else:
                st.info("No year information available for the selected view.")
        with b2:
            avg_rating_hist = dashboard_view["rating_mean"].replace(0, pd.NA).dropna()
            st.markdown("#### Average rating distribution")
            st.pyplot(draw_hist(avg_rating_hist, "Average rating distribution", "Average rating", "Number of movies"))


    with tab2:
        st.subheader("Recommender")
        st.write("Pick one movie and get similar titles based on genre and popularity profile.")

        r1, r2 = st.columns([1.1, 0.9])
        with r1:
            picked_title = st.selectbox("Seed movie", top_titles, key="rec_title")
        with r2:
            rec_count = st.slider("Recommendations:", 3, 15, 8, 1, key="rec_count")

        selected_row = merged.loc[merged["title"] == picked_title].iloc[0]
        show_movie_summary_card(selected_row)

        recs = recommend_by_title(merged, data.similarity, picked_title, n=rec_count)
        st.markdown(f"#### Recommended for: {picked_title}")
        st.dataframe(recs, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Preference")
        st.write("A small preference-based area for testing different recommendation styles.")

        t1, t2 = st.columns([1.1, 0.9])
        with t1:
            preference_genre = st.selectbox("Preferred genre", genre_options, key="taste_genre")
        with t2:
            mood = st.radio(
                "Recommendation mood:",
                ["Balanced", "Crowd favorites", "Hidden gems"],
                horizontal=True,
                key="taste_mood",
            )

        base_votes, base_rating = mood_thresholds(mood)
        recs2 = recommend_by_preferences(
            merged,
            genre=preference_genre,
            min_votes=base_votes,
            min_rating=base_rating,
            n=8,
        )

        if recs2.empty:
            st.warning("No movies matched the current taste profile. Try a different genre or mood.")
        else:
            st.dataframe(recs2, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()