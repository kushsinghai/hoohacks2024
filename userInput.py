import joblib  # For loading model and vectorizer
from getLyrics import clean_lyrics

def load_model_and_vectorizer(model_path, vectorizer_path):
    """Load the trained model and vectorizer from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_artist(lyrics, model, vectorizer):
    """Predict the artist of the given lyrics."""
    # Assume clean_lyrics function is defined elsewhere in your script
    lyrics = clean_lyrics(lyrics, "Unknown Artist")  # Use a placeholder since the artist is unknown
    lyrics_tfidf = vectorizer.transform([lyrics])  # Vectorize the preprocessed lyrics
    prediction = model.predict(lyrics_tfidf)  # Predict using the trained model
    return prediction[0]  # Return the predicted artist name

def main():
    model, vectorizer = load_model_and_vectorizer(
        r"C:\Users\and2b\Documents\hoohacks2024-1\naive_bayes_model.pkl",
        r"C:\Users\and2b\Documents\hoohacks2024-1\tfidf_vectorizer.pkl"
    )

    # Prompt the user to input the path to a text file with song lyrics
    file_path = input("Enter the path to the text file containing the song lyrics: ")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lyrics = file.read()  # Read the contents of the file (the lyrics)

        # Predict the artist
        predicted_artist = predict_artist(lyrics, model, vectorizer)

        # Display the predicted artist
        print(f"The predicted artist is: {predicted_artist}")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    main()
