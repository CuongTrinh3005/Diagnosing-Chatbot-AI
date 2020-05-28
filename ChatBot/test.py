import pandas as pd
import re
from tensorflow.keras.models import load_model
import numpy as np
import pickle

with open('dictionary_diseases.pickle', 'rb') as file:
    dic_code = pickle.load(file)

data = pd.read_csv("Training.csv")
u_input = 'I feel headache, joint pain, itching, muscle wasting'

def create_symptoms_list(dataframe):
    symptoms_org = dataframe.columns.tolist()[:-1]
    symptoms = []
    for symptom in symptoms_org:
        symptom = symptom.replace('_', ' ')
        symptoms.append(symptom)

    return symptoms

def create_one_hot_symptoms(symptoms_list, user_input):
    o_input = [0 for _ in range(len(symptoms_list))]
    for index, item in enumerate(symptoms_list):
        y = re.findall(item, user_input)
        if y:
            print(y, index)
            o_input[index] = 1
    return o_input

symptoms = create_symptoms_list(data)
print(symptoms)

o_input = create_one_hot_symptoms(symptoms, u_input)
print("One hot input: ", o_input)
o_input = np.array(o_input)
o_input = o_input.reshape(1, 132)
print("One hot input shape: ", o_input.shape)

# Load model to predict
diagnose_model = load_model('Diagnosing.h5')
disease = diagnose_model.predict(o_input)
print(dic_code[np.argmax(disease)])