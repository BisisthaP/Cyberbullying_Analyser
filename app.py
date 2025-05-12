import pickle 
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import emoji
from wordcloud import STOPWORDS
import streamlit as st

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 
                  'im', 'll', 'y', 've', 'u', 'ur', 'don', 
                  'p', 't', 's', 'aren', 'kp', 'o', 'kat', 
                  'de', 're', 'amp', 'will'])

def preprocess_input_text(tweet):
    tweet = re.sub(r"won\'t", "will not", tweet)
    tweet = re.sub(r"can\'t", "can not", tweet)
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would",tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'m", " am", tweet)
    tweet = re.sub('[^a-zA-Z]',' ',tweet)
    tweet = re.sub(r'[^\x00-\x7f]','',tweet)
    tweet = emoji.replace_emoji(tweet, replace='')
    tweet = " ".join([stemmer.stem(word) for word in tweet.split()])
    tweet = [lemmatizer.lemmatize(word) for word in tweet.split() if not word in set(STOPWORDS)]
    tweet = ' '.join(tweet)
    return tweet

def predict_sentiment(text):
    try:
        with open('lr_model.pkl','rb') as model:
            loaded_model = pickle.load(model)
        with open('tfidf.pkl','rb') as file:
            tfidf_file = pickle.load(file)
        cleaned_text = preprocess_input_text(text)
        text_tf = tfidf_file.transform([text])
        predicted_label = loaded_model.predict(text_tf)[0]

        # Map the numerical label back to the sentiment category
        label_mapping = {0: "not_cyberbullying", 1: "gender", 2: "ethnicity", 3: "religion", 4: "age"}
        predicted_sentiment = label_mapping.get(predicted_label)

        return predicted_sentiment
    except FileNotFoundError:
        return "Error: Model or TF-IDF vectorizer file not found."
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    st.write("By Bisistha Patra")
    st.title("Cyberbullying Sentiment Analysis")
    st.subheader("Enter a text to predict the cyberbullying type")

    user_input = st.text_area("Text entered here -", "")

    if st.button("Predict Sentiment"):
        if user_input:
            prediction = predict_sentiment(user_input)
            st.subheader("Prediction:")
            # Define the mapping from sentiment to image file
            image_mapping = {
                "age": "age.jpg",
                "gender": "gender.jpg",
                "ethnicity": "ethinic.jpg",
                "religion": "religion.jpg",
                "not_cyberbullying": "not.jpg"  
            }
            image = image_mapping.get(prediction)
            if image:
                st.image(image, caption=f"Predicted Sentiment: {image.split('.')[0]}", width=500)
            else:
                st.warning("Could not determine the predicted sentiment.")
        
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == '__main__':
    main()