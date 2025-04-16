import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
books = pd.read_csv('data/books.csv')
ratings = pd.read_csv('data/ratings.csv')

# Merge books and ratings data
data = pd.merge(ratings, books, on='book_id')

# Create a user-book matrix
user_book_matrix = data.pivot_table(index='user_id', columns='title', values='rating')

# Fill NaN values with 0
user_book_matrix.fillna(0, inplace=True)

# Calculate cosine similarity between books
cosine_sim = cosine_similarity(user_book_matrix.T)

# Create a DataFrame for cosine similarity
similarity_df = pd.DataFrame(cosine_sim, index=user_book_matrix.columns, columns=user_book_matrix.columns)

def recommend_books(book_title, num_recommendations=5):
    """Recommend books based on a given book title."""
    if book_title not in similarity_df.index:
        return f"'{book_title}' not found in the dataset."
    
    recommendations = similarity_df[book_title].sort_values(ascending=False)[1:num_recommendations+1]
    return recommendations

# Example usage
if __name__ == "__main__":
    print("Welcome to the Book Recommendation System!")
    user_input = input("Enter a book title you like: ")
    recommendations = recommend_books(user_input)
    
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print("\nBooks you might like:")
        for book, score in recommendations.items():
            print(f"{book} (Similarity Score: {score:.2f})")