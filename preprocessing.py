"""
TMDB + IMDb Merged Dataset Preprocessing Module

This module handles:
- Loading and cleaning the merged TMDB + IMDb movies dataset
- Parsing comma-separated fields (genres, keywords)
- Creating combined text features for recommendation
"""

import pandas as pd
import re
from typing import List, Optional


def load_data(filepath: str = "TMDB  IMDB Movies Dataset.csv") -> pd.DataFrame:
    """Load the TMDB + IMDb merged movies dataset."""
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df)} movies from dataset.")
    return df


def parse_comma_separated(text: str) -> List[str]:
    """
    Parse a comma-separated string and return list of items.
    
    Args:
        text: Comma-separated string
        
    Returns:
        List of individual items
    """
    if pd.isna(text) or text == "":
        return []
    
    items = [item.strip() for item in str(text).split(",")]
    return [item for item in items if item]


def clean_text(text: str) -> str:
    """
    Clean and normalize text for vectorization.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned lowercase text with normalized whitespace
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and format genre names from comma-separated column."""
    df["genres_list"] = df["genres"].apply(parse_comma_separated)
    df["genres_str"] = df["genres_list"].apply(lambda x: " ".join(x))
    return df


def extract_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and format keyword names from comma-separated column."""
    df["keywords_list"] = df["keywords"].apply(parse_comma_separated)
    df["keywords_str"] = df["keywords_list"].apply(lambda x: " ".join(x))
    return df


def create_content_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined text features for content-based recommendation.
    
    Combines: genres + overview + keywords
    """
    df["overview_clean"] = df["overview"].apply(clean_text)
    df["genres_clean"] = df["genres_str"].apply(clean_text)
    df["keywords_clean"] = df["keywords_str"].apply(clean_text)
    
    df["content_text"] = (
        df["genres_clean"] + " " + 
        df["overview_clean"] + " " + 
        df["keywords_clean"]
    )
    
    df["content_text"] = df["content_text"].apply(lambda x: x.strip())
    
    return df


def format_genres_display(genres_list: List[str]) -> str:
    """Format genres for display (comma-separated)."""
    if not genres_list:
        return "Unknown"
    return ", ".join(genres_list)


def get_poster_url(poster_path: str, size: str = "w500") -> str:
    """
    Generate full TMDB poster URL from poster path.
    
    Args:
        poster_path: TMDB poster path (e.g., /oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg)
        size: Image size (w92, w154, w185, w342, w500, w780, original)
        
    Returns:
        Full URL to the poster image
    """
    if pd.isna(poster_path) or not poster_path:
        return None
    
    base_url = f"https://image.tmdb.org/t/p/{size}"
    return f"{base_url}{poster_path}"


def preprocess_data(
    filepath: str = "TMDB  IMDB Movies Dataset.csv",
    sample_size: Optional[int] = None,
    min_votes: int = 100
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for the TMDB + IMDb merged dataset.
    
    Args:
        filepath: Path to the CSV file
        sample_size: Optional limit on number of movies to process
        min_votes: Minimum vote count to include a movie (quality filter)
        
    Returns:
        Preprocessed DataFrame ready for recommendation
    """
    df = load_data(filepath)
    
    df = df[df["vote_count"] >= min_votes].copy()
    print(f"After vote filter (>={min_votes} votes): {len(df)} movies")
    
    if sample_size and len(df) > sample_size:
        df = df.nlargest(sample_size, "vote_count").copy()
        print(f"Limited to top {sample_size} movies by vote count")
    
    df = extract_genres(df)
    df = extract_keywords(df)
    
    df = create_content_features(df)
    
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["release_date"] = df["release_date"].fillna("Unknown")
    df["title"] = df["title"].fillna("Unknown Title")
    df["overview"] = df["overview"].fillna("No overview available.")
    
    df["genres_display"] = df["genres_list"].apply(format_genres_display)
    
    df["poster_url"] = df["poster_path"].apply(get_poster_url)
    
    if "directors" in df.columns:
        df["directors"] = df["directors"].fillna("")
    if "cast" in df.columns:
        df["cast"] = df["cast"].fillna("")
    
    df = df[df["content_text"].str.len() > 10].reset_index(drop=True)
    
    print(f"Final dataset: {len(df)} movies ready for recommendation.")
    
    return df


if __name__ == "__main__":
    df = preprocess_data(sample_size=10000)
    print("\nSample preprocessed data:")
    print(df[["title", "genres_display", "vote_average", "release_date"]].head(10))
    print(f"\nContent text sample:\n{df['content_text'].iloc[0][:200]}...")
    print(f"\nPoster URL sample: {df['poster_url'].iloc[0]}")
