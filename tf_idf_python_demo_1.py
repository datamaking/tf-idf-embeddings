from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus: list of documents (strings)
documents = [
    "the quick brown fox jumps over the lazy dog",
    "never jump over the lazy dog quickly",
    "bright foxes leap over lazy dogs in summer"
]

documents = [
    "the quick brown fox jumps over the lazy dog",
    "never jump over the lazy dog quickly",
    "bright foxes leap over lazy dogs in summer",
    "new_word_1 new_word_2 new_word_3"
]


# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Learn vocabulary and idf, then transform the documents into TF-IDF features
tfidf_matrix = vectorizer.fit_transform(documents)

# Retrieve the feature names (terms)
feature_names = vectorizer.get_feature_names_out()
print("Feature names:", feature_names)

# Convert the sparse matrix to a dense array for display
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())

tfidf_matrix_array = tfidf_matrix.toarray()

print(tfidf_matrix_array.shape)
