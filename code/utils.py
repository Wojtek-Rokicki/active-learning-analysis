import pickle
import uuid
import os

from config import MODELS_PATH

try:
    os.makedirs(MODELS_PATH)
except FileExistsError:
    pass

def save_model(model):
    unique_filename = str(uuid.uuid4())+'.pkl'
    with open(MODELS_PATH+unique_filename, 'wb') as file:
        pickle.dump(model, file)
    return unique_filename

def load_model(filename):
    # Load from file
    with open(MODELS_PATH+filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model