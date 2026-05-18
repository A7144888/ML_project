# Movie Recommendation System

A **poster-based, content-based movie recommendation** web application using the **TMDB + IMDb Merged Dataset**.

## Features

- **Movie Poster Gallery**: Browse 15,000+ movies with real poster images from TMDB
- **Content-Based Recommendations**: Get top 5 similar movies based on genres, overview, and keywords
- **Explainable AI**: Each recommendation includes a human-readable explanation of why it was suggested
- **Recursive Exploration**: Click any recommended movie to discover more similar films (infinite browsing)
- **Search & Sort**: Find movies by title and sort by popularity, rating, title, or release date
- **Rich Metadata**: View directors, cast, ratings, and release dates

## Technical Stack

- **Framework**: Streamlit
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Similarity Metric**: Cosine Similarity
- **Language**: Python 3.8+
- **Dataset**: TMDB + IMDb Merged Movies Dataset (~438,000 movies)

## Project Structure

```
ML_final/
├── app.py                          # Streamlit web application
├── preprocessing.py                # Data cleaning and text normalization
├── recommender.py                  # TF-IDF vectorization and recommendation engine
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── TMDB  IMDB Movies Dataset.csv   # Dataset (required)
```

## Installation

### 1. Clone/Download the Project

Ensure all files are in the same directory along with `TMDB  IMDB Movies Dataset.csv`.

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## How It Works

### Data Preprocessing

1. **Load Dataset**: Read `TMDB  IMDB Movies Dataset.csv`
2. **Quality Filter**: Keep movies with ≥500 votes (configurable)
3. **Sample Selection**: Use top 15,000 movies by vote count for performance
4. **Text Cleaning**: Lowercase, remove punctuation, normalize whitespace
5. **Feature Construction**: Combine `genres + overview + keywords` into a single text field

### Recommendation Engine

1. **TF-IDF Vectorization**: Convert text features into numerical vectors
   - Max 5000 features
   - 1-2 word n-grams
   - English stop words removed

2. **Cosine Similarity**: Compute similarity between movie vectors

3. **Top-N Selection**: Return the 5 most similar movies (excluding the query movie)

4. **Explainability**: Generate human-readable explanations based on:
   - Common genres
   - Shared keywords/themes
   - Narrative similarity

### User Interface

1. **Browse Section**: Grid of movie posters with pagination
2. **Selected Movie Detail**: Full information about the chosen movie
3. **Recommendations**: Top 5 similar movies with explanations
4. **Recursive Navigation**: Click any recommendation to explore further

## Dataset Information

**Source**: TMDB + IMDb Merged Movies Dataset (Kaggle)

### Fields Used

| Field | Source | Description |
|-------|--------|-------------|
| title | TMDB/IMDb | Movie name |
| genres | TMDB | Genre categories |
| overview | TMDB/IMDb | Plot description |
| keywords | TMDB | Associated keywords |
| vote_average | TMDB/IMDb | Average rating (0-10) |
| vote_count | TMDB/IMDb | Number of votes |
| release_date | TMDB | Release date (YYYY-MM-DD) |
| poster_path | TMDB | Poster image path |
| directors | IMDb | Director names |
| cast | IMDb | Main cast members |

## Example Output

When selecting "Inception", you'll see recommendations like:

```
1. Interstellar
   Genres: Adventure, Drama, Science Fiction
   Similarity: 91%
   Average Rating: 8.4
   Release Date: 2014-11-05
   
   💡 Recommended because both movies share Science Fiction, Drama themes
      and feature similar elements like dream, space, and time.

2. The Matrix
   Genres: Action, Science Fiction
   Similarity: 85%
   Average Rating: 8.2
   Release Date: 1999-03-30
   
   💡 Recommended because both movies share Science Fiction, Action themes
      and feature similar elements like virtual reality and philosophy.
```

## Configuration Options

In `preprocessing.py`, you can adjust:

```python
preprocess_data(
    filepath="TMDB  IMDB Movies Dataset.csv",
    sample_size=15000,     # Number of movies to include (None for all)
    min_votes=500          # Minimum vote count filter
)
```

## Constraints

This system:
- ❌ Does NOT use collaborative filtering
- ❌ Does NOT use neural networks
- ❌ Does NOT require backend servers
- ❌ Does NOT scrape websites dynamically
- ✅ Is lightweight and runs locally
- ✅ Is fully explainable
- ✅ Uses real TMDB poster images

## Troubleshooting

### "FileNotFoundError: TMDB IMDB Movies Dataset.csv"

Ensure the dataset file is in the same directory as the Python files. Note the filename has two spaces: `TMDB  IMDB Movies Dataset.csv`.

### "ModuleNotFoundError"

Run `pip install -r requirements.txt` to install all dependencies.

### Slow First Load

The first load takes 30-60 seconds to:
1. Load and filter the dataset
2. Build the TF-IDF model

Subsequent interactions are fast due to Streamlit caching.

### Poster Images Not Loading

The system uses TMDB's image CDN. If posters don't load:
- Check your internet connection
- Some older movies may not have poster images (placeholder will be shown)

## Performance Notes

- **Dataset Size**: 15,000 movies (filtered from 438,000)
- **TF-IDF Matrix**: ~15,000 x 5,000 sparse matrix
- **Recommendation Time**: < 100ms per query
- **Memory Usage**: ~500MB

## License

This project is for educational purposes. The dataset is provided by TMDB and IMDb under their respective terms of use.

---

**Built with** ❤️ **using Python, Streamlit, and scikit-learn**
