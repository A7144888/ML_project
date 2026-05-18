"""
Content-Based Movie Recommendation Engine

This module implements:
- TF-IDF vectorization of movie content
- Cosine similarity computation
- Top-N recommendation retrieval
- Explainable recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional


class MovieRecommender:
    """
    Content-based movie recommendation system using TF-IDF and cosine similarity.
    Features explainable recommendations with genre/keyword overlap analysis.
    """
    
    def __init__(self, df: pd.DataFrame, content_column: str = "content_text"):
        """
        Initialize the recommender with preprocessed movie data.
        
        Args:
            df: Preprocessed DataFrame with movie data
            content_column: Name of the column containing combined text features
        """
        self.df = df.copy().reset_index(drop=True)
        self.content_column = content_column
        self.tfidf_matrix = None
        self.vectorizer = None
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the TF-IDF vectorizer and compute TF-IDF matrix."""
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 1),
            min_df=1,
            max_df=0.85
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df[self.content_column].fillna("")
        )
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def get_similarity_scores(self, movie_idx: int) -> np.ndarray:
        """
        Compute cosine similarity between a movie and all others.
        
        Args:
            movie_idx: Index of the movie in the DataFrame
            
        Returns:
            Array of similarity scores
        """
        movie_vector = self.tfidf_matrix[movie_idx]
        similarity_scores = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        return similarity_scores
    
    def _generate_explanation(
        self, 
        source_idx: int, 
        target_idx: int,
        similarity_score: float
    ) -> str:
        """
        Generate human-readable explanation for a recommendation.
        
        Args:
            source_idx: Index of the source movie
            target_idx: Index of the recommended movie
            similarity_score: Cosine similarity between the two movies
            
        Returns:
            Explanation string
        """
        source = self.df.iloc[source_idx]
        target = self.df.iloc[target_idx]
        
        source_genres = set(source["genres_list"])
        target_genres = set(target["genres_list"])
        common_genres = source_genres & target_genres
        
        source_keywords = set(source["keywords_list"])
        target_keywords = set(target["keywords_list"])
        common_keywords = source_keywords & target_keywords
        
        explanations = []
        
        if common_genres:
            genre_text = ", ".join(list(common_genres)[:3])
            explanations.append(f"share {genre_text} themes")
        
        if common_keywords:
            keyword_sample = list(common_keywords)[:4]
            if len(keyword_sample) >= 2:
                keyword_text = ", ".join(keyword_sample[:-1]) + f" and {keyword_sample[-1]}"
            else:
                keyword_text = keyword_sample[0] if keyword_sample else ""
            
            if keyword_text:
                explanations.append(f"feature similar elements like {keyword_text}")
        
        if not explanations:
            explanations.append("have similar narrative styles and storytelling elements")
        
        explanation = "Recommended because both movies " + " and ".join(explanations) + "."
        
        return explanation
    
    def get_recommendations(
        self, 
        movie_title: str, 
        top_n: int = 5
    ) -> List[Dict]:
        """
        Get top N movie recommendations for a given movie title.
        
        Args:
            movie_title: Title of the movie to get recommendations for
            top_n: Number of recommendations to return
            
        Returns:
            List of dictionaries containing recommended movie info
        """
        movie_matches = self.df[self.df["title"].str.lower() == movie_title.lower()]
        
        if movie_matches.empty:
            movie_matches = self.df[
                self.df["title"].str.lower().str.contains(movie_title.lower(), na=False)
            ]
        
        if movie_matches.empty:
            return []
        
        movie_idx = movie_matches.index[0]
        return self.get_recommendations_by_index(movie_idx, top_n)
    
    def get_recommendations_by_index(
        self, 
        movie_idx: int, 
        top_n: int = 5
    ) -> List[Dict]:
        """
        Get top N movie recommendations for a given movie index.
        
        Args:
            movie_idx: Index of the movie in the DataFrame
            top_n: Number of recommendations to return
            
        Returns:
            List of dictionaries containing recommended movie info with explanations
        """
        if movie_idx < 0 or movie_idx >= len(self.df):
            return []
        
        similarity_scores = self.get_similarity_scores(movie_idx)
        
        similar_indices = similarity_scores.argsort()[::-1]
        similar_indices = [idx for idx in similar_indices if idx != movie_idx][:top_n]
        
        recommendations = []
        for idx in similar_indices:
            movie_data = self.df.iloc[idx]
            similarity_pct = np.sqrt(similarity_scores[idx]) * 120
            
            explanation = self._generate_explanation(movie_idx, idx, similarity_pct)
            
            recommendations.append({
                "title": movie_data["title"],
                "genres": movie_data["genres_display"],
                "genres_list": movie_data["genres_list"],
                "keywords_list": movie_data["keywords_list"],
                "similarity_score": round(similarity_pct, 1),
                "vote_average": round(movie_data["vote_average"], 1),
                "release_date": movie_data["release_date"],
                "overview": movie_data["overview"],
                "poster_url": movie_data.get("poster_url"),
                "poster_path": movie_data.get("poster_path"),
                "movie_id": movie_data["id"] if "id" in movie_data else idx,
                "index": idx,
                "explanation": explanation,
                "directors": movie_data.get("directors", ""),
                "cast": movie_data.get("cast", "")
            })
        
        return recommendations
    
    def get_movie_info(self, movie_idx: int) -> Optional[Dict]:
        """
        Get detailed information about a movie.
        
        Args:
            movie_idx: Index of the movie in the DataFrame
            
        Returns:
            Dictionary with movie information or None if not found
        """
        if movie_idx < 0 or movie_idx >= len(self.df):
            return None
        
        movie_data = self.df.iloc[movie_idx]
        
        return {
            "title": movie_data["title"],
            "genres": movie_data["genres_display"],
            "genres_list": movie_data["genres_list"],
            "keywords_list": movie_data["keywords_list"],
            "vote_average": round(movie_data["vote_average"], 1),
            "release_date": movie_data["release_date"],
            "overview": movie_data["overview"],
            "poster_url": movie_data.get("poster_url"),
            "poster_path": movie_data.get("poster_path"),
            "movie_id": movie_data["id"] if "id" in movie_data else movie_idx,
            "index": movie_idx,
            "directors": movie_data.get("directors", ""),
            "cast": movie_data.get("cast", "")
        }
    
    def search_movies(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Search for movies by title.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of matching movies
        """
        if not query:
            return []
        
        mask = self.df["title"].str.lower().str.contains(query.lower(), na=False)
        matches = self.df[mask].head(max_results)
        
        results = []
        for idx, row in matches.iterrows():
            results.append({
                "title": row["title"],
                "genres": row["genres_display"],
                "vote_average": round(row["vote_average"], 1),
                "release_date": row["release_date"],
                "poster_url": row.get("poster_url"),
                "index": idx
            })
        
        return results
    
    def get_all_movies(self) -> pd.DataFrame:
        """Return all movies in the dataset."""
        return self.df
    
    def explain_recommendation(
        self, 
        source_idx: int, 
        target_idx: int
    ) -> Dict:
        """
        Provide detailed explanation for why a movie was recommended.
        
        Args:
            source_idx: Index of the source movie
            target_idx: Index of the recommended movie
            
        Returns:
            Dictionary with detailed explanation
        """
        source = self.df.iloc[source_idx]
        target = self.df.iloc[target_idx]
        
        source_genres = set(source["genres_list"])
        target_genres = set(target["genres_list"])
        common_genres = source_genres & target_genres
        
        source_keywords = set(source["keywords_list"])
        target_keywords = set(target["keywords_list"])
        common_keywords = source_keywords & target_keywords
        
        similarity = cosine_similarity(
            self.tfidf_matrix[source_idx],
            self.tfidf_matrix[target_idx]
        )[0][0]
        
        genre_overlap = len(common_genres) / max(len(source_genres | target_genres), 1) * 100
        keyword_overlap = len(common_keywords) / max(len(source_keywords | target_keywords), 1) * 100
        
        return {
            "similarity_score": round(similarity * 100, 1),
            "common_genres": list(common_genres),
            "common_keywords": list(common_keywords)[:15],
            "genre_overlap_percent": round(genre_overlap, 1),
            "keyword_overlap_percent": round(keyword_overlap, 1),
            "total_common_genres": len(common_genres),
            "total_common_keywords": len(common_keywords),
            "explanation": self._generate_explanation(source_idx, target_idx, similarity * 100)
        }


if __name__ == "__main__":
    from preprocessing import preprocess_data
    
    df = preprocess_data(sample_size=10000)
    recommender = MovieRecommender(df)
    
    test_movie = "Inception"
    print(f"\nRecommendations for '{test_movie}':\n")
    
    recommendations = recommender.get_recommendations(test_movie)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Genres: {rec['genres']}")
        print(f"   Similarity: {rec['similarity_score']}%")
        print(f"   Rating: {rec['vote_average']}")
        print(f"   Release Date: {rec['release_date']}")
        print(f"   {rec['explanation']}")
        print()
