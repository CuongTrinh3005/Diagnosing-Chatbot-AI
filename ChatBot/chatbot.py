import nltk
import numpy as np
import json
import pickle
from tensorflow.keras.models import load_model
import random
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from keras.models import Sequential
from keras.layers import Dense, Dropout

# Load data
with open('intents.json') as file:
    data = json.load(file)

try:
    with open('information.pickle', 'rb') as file:
        vocab, classes, training = pickle.load(file)
        print("Vocabulary: ", vocab)
        print("Classes: ", classes)
        print("Training: ", training)
except:
    vocab = []
    classes = []
    documents = []
    ignore = ['?']

    # Create vocabulary, documents and labels(classes):
    for intent in data['intents']:
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern)
            vocab.extend(word)
            documents.append((word, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Stem and sort vocabulary:
    vocab = [stemmer.stem(w.lower()) for w in vocab if w not in ignore]
    vocab = sorted(list(set(vocab)))
    print(vocab)
    classes = sorted(list(set(classes)))
    print(classes)
    print(documents)

    print("Number of documents: ", len(documents))
    print("Number of classes: ", len(classes))
    print("Number of words in vocabulary: ", len(vocab))

    # Create training data
    training = []
    output_empty = [0 for _ in range(len(classes))]

    # Create bag of words in order to map with the label
    for doc in documents:
        bag = []
        pattern_word = doc[0]
        pattern_word = [stemmer.stem(w.lower()) for w in pattern_word]
        for item in vocab:
            bag.append(1) if item in pattern_word else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

# Save data:
with open('information.pickle', 'wb') as f:
    pickle.dump((vocab, classes, training), f)

training = np.array(training)

# Split data to train and label mapping
train_x = list(training[:, 0])
train_y = list(training[:, 1])

try:
    model = load_model('chatbot.h5')
except:
    # Build model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(len(train_x[0]), )))   # train_x[0] is bag of words
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit model
    model.fit(np.asarray(train_x), np.asarray(train_y), epochs=1024, verbose=0)

# Save model
model.save('chatbot.h5')
# Clean user sentence
def create_bag_of_words(sentence):
    bag = [0 for _ in range(len(vocab))]
    word_of_sentence = nltk.word_tokenize(sentence)
    stemmed_word = [stemmer.stem(w.lower()) for w in word_of_sentence]
    for i, w in enumerate(vocab):
        if w in stemmed_word:
            bag[i] = 1

    return np.array([bag])

def chat():
    stop = ['exit', 'quit']
    print("Starting ChatBot, type exit to quit")
    while True:
        request = input("User: ")
        if request.lower() in stop:
            break
        result = model.predict(create_bag_of_words(request))
        tag = classes[np.argmax(result)]

        for content in data['intents']:
            if content['tag'] == tag:
                response = content['responses']
        print("Bot: ", random.choice(response))

chat()