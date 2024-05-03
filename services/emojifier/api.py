from keras.models import model_from_json
import emoji
import numpy as np
import pandas as pd

emoji_dictionary = {"0": "\u2764\uFE0F",    
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:",
                   }

embeddings = {}
with open('glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        embeddings[word] = coeffs

def getOutputEmbeddings(X):
    embedding_matrix_output = np.zeros((X.shape[0],10,50))
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]
    return embedding_matrix_output

with open("model.json", "r") as file:
    model = model_from_json(file.read())
model.load_weights("model.h5")

def predict(x):
    X = pd.Series([x])
    emb_X = getOutputEmbeddings(X)
    p = model.predict(emb_X)
    print(' '.join(X[0]))
    print(emoji.emojize(emoji_dictionary[str(np.argmax(p[0]))]))

if __name__=='__main__':
    print(predict("hello"))












