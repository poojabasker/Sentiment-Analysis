import pandas as pd
import preprocessor as p
import numpy as np 
import pandas as pd 
import emoji
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from tqdm import tqdm
import pymysql.cursors
from keras.models import model_from_yaml

def get_sentiment():
    # Connect to the database
    connection = pymysql.connect(host='remotemysql.com',
                                    user='DdhABhaLIk',
                                    password='OYIGc9RYkB',
                                    database='DdhABhaLIk',
                                    cursorclass=pymysql.cursors.DictCursor)

    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM RetrainData"
        cursor.execute(sql)
        result = cursor.fetchall()

    df1 = pd.DataFrame.from_dict(result)

    df2 = pd.read_csv("text_emotion.csv")
    data = df2.append(df1,ignore_index=True)

    if len(data)==0:
        data = df2

    misspell_data = pd.read_csv("aspell.txt",sep=":",names=["correction","misspell"])
    misspell_data.misspell = misspell_data.misspell.str.strip()
    misspell_data.misspell = misspell_data.misspell.str.split(" ")
    misspell_data = misspell_data.explode("misspell").reset_index(drop=True)
    misspell_data.drop_duplicates("misspell",inplace=True)
    miss_corr = dict(zip(misspell_data.misspell, misspell_data.correction))

    def misspelled_correction(val):
        for x in val.split(): 
            if x in miss_corr.keys(): 
                val = val.replace(x, miss_corr[x]) 
        return val
    data["clean_content"] = data.content.apply(lambda x : misspelled_correction(x))

    #Contractions
    contractions = pd.read_csv("contractions.csv")
    cont_dic = dict(zip(contractions.Contraction, contractions.Meaning))

    def cont_to_meaning(val): 
  
        for x in val.split(): 
            if x in cont_dic.keys(): 
                val = val.replace(x, cont_dic[x]) 
        return val

    data.clean_content = data.clean_content.apply(lambda x : cont_to_meaning(x))

    data["clean_content"]=data.content.apply(lambda x : p.clean(x))

    #Punctuations and emojis
    def punctuation(val): 
        punctuations = '''()-[]{};:'"\,<>./@#$%^&_~'''
        for x in val.lower(): 
            if x in punctuations: 
                val = val.replace(x, " ") 
        return val

    data.clean_content = data.clean_content.apply(lambda x : ' '.join(punctuation(emoji.demojize(x)).split()))

    #Removing empty comments
    data = data[data.clean_content != ""]
    data.sentiment.value_counts()

    sent_to_id  = {"empty":0, "sadness":1,"enthusiasm":2,"neutral":3,"worry":4,"surprise":5,"love":6,"fun":7,"hate":8,"happiness":9,"boredom":10,"relief":11,"anger":12}

    data["sentiment_id"] = data['sentiment'].map(sent_to_id)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data.sentiment_id)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)

    X_train, X_test, y_train, y_test = train_test_split(data.clean_content,Y, random_state=1995, test_size=0.2, shuffle=True)

    token = text.Tokenizer(num_words=None)
    max_len = 160
    Epoch = 5
    token.fit_on_texts(list(X_train) + list(X_test))
    X_train_pad = sequence.pad_sequences(token.texts_to_sequences(X_train), maxlen=max_len)
    X_test_pad = sequence.pad_sequences(token.texts_to_sequences(X_test), maxlen=max_len)

    w_idx = token.word_index

    embed_dim = 160
    lstm_out = 250

    model = Sequential()
    model.add(Embedding(len(w_idx) +1 , embed_dim,input_length = X_test_pad.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(keras.layers.core.Dense(13, activation='softmax'))
    #adam rmsprop 
    model.compile(loss = "categorical_crossentropy", optimizer='adam',metrics = ['accuracy'])

    batch_size = 32

    model.fit(X_train_pad, y_train, epochs = Epoch, batch_size=batch_size,validation_data=(X_test_pad, y_test))

    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights("model.h5")
    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights("model.h5")
    print("done")

    return