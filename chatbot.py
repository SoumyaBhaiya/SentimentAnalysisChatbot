import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier


def load_sentiment140_data(filepath):
    data = []
    with open(filepath, encoding="latin-1") as file:
        for line in file:
            label, _, _, text = line.strip().split(",", 3)
            label = int(label.replace('"', ''))
            data.append((text, label))
    return data

data = load_sentiment140_data("SentimentAnalysisData.csv")

# Shuffle split data into training and testing set
import random
random.shuffle(data)

train_data = data[:1600000]
test_data = data[1600000:]

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    return [word for word in words if word.isalpha() and word not in stop_words]

#  preprocessing to tha training and test dataa
train_set = [(preprocess_text(text), label) for text, label in train_data]
test_set = [(preprocess_text(text), label) for text, label in test_data]

def create_word_features(words):
    return dict([(word, True) for word in words])

training_features = [(create_word_features(text), label) for (text, label) in train_set]
classifier = NaiveBayesClassifier.train(training_features)

def classify_sentiment(text):
    user_input_preprocessed = preprocess_text(text)
    features = create_word_features(user_input_preprocessed)
    sentiment = classifier.classify(features)
    return sentiment

def sentiment_label(sentiment):
    if sentiment == 0:
        return "negative"
    elif sentiment == 4:
        return "positive"
    elif sentiment == 2:
        return "neutral"
    else:
        return "unknown"

def chatbot():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Identify neutral sentiment
        if classify_sentiment(user_input) == 2:
            print("Chatbot: Neutral sentiment.")
        else:
            # Adjust "not fine" to negative sentiment
            if "not fine" in user_input.lower():
                print("Chatbot: Negative sentiment.")
            else:
                sentiment = classify_sentiment(user_input)
                print(f"Chatbot: {sentiment_label(sentiment).capitalize()} sentiment.")

if __name__ == "__main__":
    chatbot()
