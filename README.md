# Cyberbullying Sentiment Analysis Web Application

## Overview

This web application utilizes a machine learning model to predict the sentiment of user-provided text, specifically focusing on identifying different types of cyberbullying. The backend is powered by a Logistic Regression model trained on a dataset of tweets labeled with categories such as age, ethnicity, gender, religion-based cyberbullying, and non-cyberbullying. The user interface is built using Streamlit, providing an intuitive way to input text and visualize the prediction as a corresponding image.

## Features

* **Real-time Sentiment Prediction:** Users can input any text and receive an immediate prediction of its sentiment category related to cyberbullying.
* **Visual Output:** Instead of text labels, the application displays a relevant image corresponding to the predicted sentiment (e.g., an image representing "age" if the model predicts age-based cyberbullying).
* **User-Friendly Interface:** Built with Streamlit for a simple and interactive user experience.
* **Preprocessed Input:** The application automatically preprocesses the user's input text using techniques like URL removal, hashtag removal, lowercasing, handling contractions, stemming, and lemmatization to ensure consistency with the model's training data.

## Technologies Used
* There are total 6 models used - **Logistic Regression, SVM, Decision Trees, Random Forest, KNN, Naive Bayes**

* **Python:** The primary programming language.
* **Streamlit:** For creating the interactive web application.
* **scikit-learn (sklearn):** For the machine learning model (Logistic Regression) and text vectorization (TF-IDF).
* **pandas:** For data manipulation (used during model training).
* **NLTK (Natural Language Toolkit):** For text preprocessing tasks (tokenization, stemming, lemmatization, stop word removal).
* **pickle:** For saving and loading the trained machine learning model and TF-IDF vectorizer.
* **re (Regular Expressions):** For text cleaning.
* **emoji:** For handling and removing emojis from text.

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ensure Model and Vectorizer Files Exist:** 
* Make sure you have the trained model (`cyberbullying_lr_model.pkl`) and the TF-IDF vectorizer (`cyberbullying_tfidf_vectorizer.pkl`) files in the same directory as your Streamlit application script (`app.py`). These files are generated after training the machine learning model.

4.  **Ensure Image Files Exist:** 
* Place the image files corresponding to the sentiment labels (`age.jpg`, `gender.jpg`, `ethnicity.jpg`, `religion.jpg`, `not_cyberbullying.jpg`) in the same directory as `app.py`.

## Running the Application

1.  Open your terminal or command prompt.
2.  Navigate to the directory containing your `app.py` file.
3.  Run the Streamlit application using the following command:
    ```bash
    streamlit run app.py
    ```

This will automatically open the application in your web browser.

## Usage

1.  Once the application is running in your browser, you will see a text area labeled "Tweet text here".
2.  Enter the text you want to analyze for cyberbullying sentiment in this area.
3.  Click the "Predict Sentiment" button.
4.  Below the button, the application will display an image corresponding to the predicted sentiment of the input text.

## Model Training (Brief Overview)

The underlying machine learning model (Logistic Regression) was trained on a dataset of tweets labeled with different categories of cyberbullying. The text data was preprocessed using various techniques, including:

* Removal of URLs, mentions, and hashtags.
* Lowercasing.
* Handling of contractions.
* Stemming and Lemmatization to reduce words to their root form.
* Removal of stop words.
* TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert the text data into numerical vectors suitable for the machine learning model.

The trained model and the fitted TF-IDF vectorizer were then saved using the `pickle` library for deployment in this Streamlit application.

## Contributing

Contributions to this project are welcome. Please feel free to submit pull requests or open issues for any bugs, enhancements, or suggestions.

## License

This project is licensed under the MIT License.

## Acknowledgements

* The developers of Streamlit for providing an excellent framework for building web applications for machine learning.
* The creators of scikit-learn, NLTK, pandas, and other libraries used in this project.
* The contributors to the cyberbullying dataset used for training the model.

## Contact
* **Author** - **Bisistha Patra**
* **GitHub** - 
* **Email** - patrabisistha@gmail.com