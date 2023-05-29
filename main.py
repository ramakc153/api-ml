from fastapi import FastAPI, Form
import uvicorn
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
data = pd.read_csv('dataset/model.csv')


# Download stopwords
nltk.download('stopwords')

def preProcess_data(text):
    # Cleaning the data
    text = text.lower()
    new_text = re.sub('[^a-zA-Z0-9\s]', '', text)
    new_text = re.sub('rt', '', new_text)
    
    # Hapus Tanda Baca
    remove = string.punctuation
    translator = str.maketrans(remove, ' ' * len(remove))
    new_text = new_text.translate(translator)
    
    # Hapus ASCII dan UNICODE
    new_text = new_text.encode('ascii', 'ignore').decode('utf-8')
    new_text = re.sub(r'[^\x00-\x7f]', 'r', new_text)
    
    # Remove newline
    new_text = new_text.replace('\n', ' ')
    
    # Tokenisasi kata
    tokens = word_tokenize(new_text)
    
    # Combine kata-kata penting
    combined_tokens = combine_important_words(tokens)
    
    # Hapus stopwords
    stopwords_ind = stopwords.words('indonesian')
    filtered_tokens = [word for word in combined_tokens if word.lower() not in stopwords_ind]
    
    # Gabungkan kembali kata-kata menjadi string
    new_text = ' '.join(filtered_tokens)
    
    return new_text

def combine_important_words(tokens):
    combined_tokens = []
    skip_next = False
    for i in range(len(tokens) - 1):
        if skip_next:
            skip_next = False
            continue
        if tokens[i] in ['tidak', 'kurang']:
            combined_tokens.append(tokens[i] + '_' + tokens[i+1])
            skip_next = True
        else:
            combined_tokens.append(tokens[i])
    if not skip_next:
        combined_tokens.append(tokens[-1])
    return combined_tokens

data['clean_text'] = data['Sentences'].apply(preProcess_data)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['clean_text'].values)

def my_pipeline(text):
    text_new = preProcess_data(text)
    print("text_new", text_new)
    X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    X = pad_sequences(X, maxlen=100, padding='post')
    return X

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/predict')
async def predict(request: dict):
    text = request.get('text')
    clean_text = my_pipeline(text)  # Cleaning and preprocessing of the texts
    loaded_model = tf.keras.models.load_model('model/model.h5')  # Loading the saved model
    predictions = loaded_model.predict(clean_text)  # Making predictions
    print("predict", predictions)
    probability = max(predictions.tolist()[0])  # Probability of maximum prediction
    
    if probability > 0.5:
        t_sentiment = 'Positif'
    else:
        t_sentiment = 'Negatif'
 
    return {  # Returning a dictionary as endpoint
        "Kalimat": text,
        "Sentimen": t_sentiment,
        "Hasil": probability
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
