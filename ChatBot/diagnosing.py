import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load data as csv format file into a data frame
def create_dataset(file):
    data_org = pd.read_csv(file)
    print("No. of rows: %s, No. of columns: %s" % (data_org.shape[0], data_org.shape[1]))

    X, y = data_org.values[:, :-1], data_org.values[:, -1]
    X = X.astype('float32')

    le = LabelEncoder()
    y = le.fit_transform(y)

    y_inv = le.inverse_transform(y)

    dic_code = dict()
    for index in range(len(y_inv)):
        dic_code[y[index]] = y_inv[index]
    return X, y, dic_code

features, labels, dic_code = create_dataset("Training.csv")
with open('dictionary_diseases.pickle', 'wb') as f:
    pickle.dump(dic_code, f)

# Explore symptoms
def print_simptoms(data):
    print("List of symptoms of diseases: ")
    symptoms = data.columns.tolist()[:-1]
    pattern = ''
    for item in symptoms:
        item = item.replace("_", " ")
        pattern += '"' + item + '"' + ', '
    return pattern

# Split the data set
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
print("Training set's shape = %s, Training label shape = %s" % (x_train.shape, y_train.shape))
print("Test set's shape = %s, Test label shape = %s" % (x_test.shape, y_test.shape))
n_features = x_test.shape[1]
print("Input features: ", n_features)

# Build a model
# Step 1: Define a model
def build_model():
    model = Sequential()
    model.add(Dense(12, activation='relu', kernel_initializer='he_normal', input_shape=(n_features, )))
    #model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    #model.add(Dropout(0.3))
    model.add(Dense(len(dic_code), activation='softmax'))

    # Step 2: Compile a model
    opt = Adam(lr=1e-5, decay=1e-5)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
# Step 3: Fit model
'''hist = model.fit(x_train, y_train, epochs=340, batch_size=64)

def plot_diagrams():
    plt.title("Loss curve")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(hist.history['loss'], label = 'loss')
    plt.show()

# Step 4: Evaluate model
loss, acc = model.evaluate(x_test, y_test)
print("Loss = %.2f and Accuracy = %.3f" % (loss, acc))

# Step 5: Make predictions
rows_fed = np.random.choice([0, 1], size=(1, 132), p=[1./3, 2./3])
y_hat = model.predict([rows_fed])

#print("Predicted: ", predicted)
print("Predicted:", dic_code[np.argmax(y_hat)])

plot_diagrams()
model.save('Diagnosing.h5')'''
print(dic_code)