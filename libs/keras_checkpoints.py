from keras.models import model_from_json


def save_model(model, name):
    '''Saves a model in hard-disk'''
    model_json = model.to_json()
    with open(name + '.json', "w") as json_file:
        json_file.write(model_json)
        json_file.close()
    model.save_weights(name + '.h5')
    print("Saved model to disk")

def load_model(name):
    '''Loads a model from hard-disk (DOES NOT COMPILES THE MODEL)'''
    # load json and create model
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + '.h5')
    print("Loaded model from disk. Don't forget to compile if you need to retrain it.")
    return loaded_model