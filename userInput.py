import joblib  # For loading model and vectorizer

def load_model_and_vectorizer(model_path, vectorizer_path):
    """Load the trained model and vectorizer from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_artist(lyrics, model, vectorizer):
    """Predict the artist of the given lyrics."""
    # Preprocess the lyrics similar to how the training data was preprocessed
    # Note: Ensure to perform the same preprocessing steps here as you did for your training data
    lyrics = clean_lyrics(lyrics, "Unknown Artist")  # Use a placeholder since the artist is unknown
    # Vectorize the preprocessed lyrics
    lyrics_tfidf = vectorizer.transform([lyrics])
    # Predict using the trained model
    prediction = model.predict(lyrics_tfidf)
    return prediction[0]  # Return the predicted artist name

def main():
    # Load the model and vectorizer
    model, vectorizer = load_model_and_vectorizer('model_path.pkl', 'vectorizer_path.pkl')
    
    # Prompt the user to input song lyrics
    user_input = input("Enter song lyrics to predict the artist: ")
    
    # Predict the artist
    predicted_artist = predict_artist(user_input, model, vectorizer)
    
    # Display the predicted artist
    print(f"The predicted artist is: {predicted_artist}")

if __name__ == "__main__":
    main()
