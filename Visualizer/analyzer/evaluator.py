import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model

MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
def data_creation():
  with open('token.json') as json_file:
      data = json.load(json_file)
  df = pd.read_csv('data.csv',sep=",", encoding='cp1252')
  X = []
  for i in df.index:
    S = ''
    for j in df.iloc[i,1].split(' '):
      if j in data:
        S += j+' '
    X.append(S)
  return X,df

def RNN_output(X):
    with open('tokenizer.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    trained_model = load_model('RNN-Model.h5')
    preds = trained_model.predict(X,batch_size=16)
    pred_labels = preds.argmax(axis=1)
    Prediction = []
    for i in pred_labels:
      if i == 0:
        Prediction.append('Negative Comment')
      else:
        Prediction.append('Positive Comment')
    return Prediction
def CNN_output(X):
    with open('tokenizer.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    trained_model = load_model('CNN-Model.h5')
    preds = trained_model.predict(X,batch_size=16)
    pred_labels = preds.argmax(axis=1)
    Prediction = []
    for i in pred_labels:
      if i == 0:
        Prediction.append('Negative Comment')
      else:
        Prediction.append('Positive Comment')
    return Prediction
def Bert_output(X):
    with open('tokenizer.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    trained_model = load_model('BERT-Model.h5')
    preds = trained_model.predict(X,batch_size=16)
    pred_labels = preds.argmax(axis=1)
    Prediction = []
    for i in pred_labels:
      if i == 0:
        Prediction.append('Negative Comment')
      else:
        Prediction.append('Positive Comment')
    return Prediction
def predictor(X,Y,name='Default'):
  df['Predictions'+name] = Y
  df.to_csv('out.csv')
  pos = []
  neg = []
  for i in range(len(Y)):
    if Y[i] == 'Positive Comment':
      pos.append(X[i])
    else :
      neg.append(X[i])
  p = WC(pos,'Positive Sentiment Words')
  n  = WC(neg,'Negative Sentiment Words')
  plt.figure(figsize = (40, 30), facecolor = None)
  plt.rcParams.update({'font.size': 22})
  plt.subplot(2, 2, 1)
  plt.bar(['Positive','Negative'], [len(pos),len(neg)],color=['green','blue'])
  plt.rc('xtick', labelsize=45) 
  plt.rc('ytick', labelsize=60) 
  plt.title('Bar Chart ',fontsize = 30)
  plt.subplot(2, 2, 3)
  plt.imshow(p)
  plt.axis("off")
  plt.title('Positive Sentiment Words',fontsize = 30)
  plt.subplot(2, 2, 4)
  plt.imshow(n)
  plt.axis("off")
  plt.title('Negative Sentiment Words',fontsize = 30)
  plt.savefig("../public/assets/"+name+".png")
def WC(X,Y):
  from wordcloud import WordCloud, STOPWORDS
  import matplotlib.pyplot as plt
  import pandas as pd
  comment_words = ''
  stopwords = set(STOPWORDS)
  for val in X:
      val = str(val)
      tokens = val.split()
      for i in range(len(tokens)):
          tokens[i] = tokens[i].lower()
      comment_words += " ".join(tokens)+" "
  wordcloud = WordCloud(width = 800, height = 800,
                  background_color ='white',
                  stopwords = stopwords,
                  min_font_size = 10).generate(comment_words)
  return wordcloud

X,df=data_creation()

predictor(X,RNN_output(X),'RNN')

predictor(X,CNN_output(X),'CNN')

predictor(X,Bert_output(X),'BERT')