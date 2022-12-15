# -*- coding: utf-8 -*-
"""codigo de classificacao.ipynb

## Realizando os imports necessários
"""

# Commented out IPython magic to ensure Python compatibility.
# !pip install wordcloud --user
# !pip install nltk --user
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# %matplotlib inline

"""## Carregando datasets
depressives -> apenas tweets depressivos  
random -> tweets aleatórios que não são depressivos  
dataset -> depressives + random, para o treinamento da rede
   
"""

# lendo datasets
depressives = pd.read_csv('datasets/depressives.csv', index_col=0)
random = pd.read_csv('datasets/random_tweets.csv', index_col=0)

# adicionando coluna de classificação
depressives['depressive'] = 1
random['depressive'] = 0
dataset = pd.concat([depressives, random])

# montando dataset de palavras com 2000 mil tweets
words = pd.read_csv('datasets/dataset.csv', engine='python', error_bad_lines=False)
words = pd.concat([words['tweet'], dataset['tweet']])
word_embedding_dataset = pd.DataFrame(words)

"""## Definindo stopwords em pt-br"""

stopwords = '''
eu de a o que e do da em um para é com não uma os no se na por mais as dos como mas foi ao ele das tem à seu sua ou ser 
quando muito há nos já está eu também só pelo pela até isso ela entre era depois sem mesmo aos ter seus quem nas me 
esse eles estão você tinha foram essa num nem suas meu às minha têm numa pelos elas havia seja qual será nós tenho lhe 
deles essas esses pelas este fosse dele tu te vocês vos lhes meus minhas teu tua teus tuas nosso nossa nossos nossas dela 
delas esta estes estas aquele aquela aqueles aquelas isto aquilo estou está estamos estão estive esteve estivemos
estiveram estava estávamos estavam estivera estivéramos esteja estejamos estejam estivesse estivéssemos estivessem
estiver estivermos estiverem hei há havemos hão houve houvemos houveram houvera houvéramos haja hajamos hajam houvesse
houvéssemos houvessem houver houvermos houverem houverei houverá houveremos houverão houveria houveríamos houveriam 
sou somos são era éramos eram fui foi fomos foram fora fôramos seja sejamos sejam fosse fôssemos fossem for  formos
forem serei será seremos serão seria seríamos seriam tenho tem temos tém tinha tínhamos tinham tive teve tivemos
tiveram tivera tivéramos tenha tenhamos tenham tivesse tivéssemos tivessem tiver tivermos tiverem terei terá 
teremos terão teria teríamos teriam pra O q https http A de
'''
stopwords = re.split(r'\W', stopwords)

"""## Gerando wordcloud do vocabulário utilizado no tweets
### tweets depressivos
"""

depressive_words = ' '.join(list(dataset[dataset['depressive'] == 1]['tweet']))
for word in stopwords:
    depressive_words = depressive_words.replace(' '+word+' ', '')
depressive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="Reds").generate(depressive_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(depressive_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

"""### Tweets aleatórios"""

positive_words = ' '.join(list(dataset[dataset['depressive'] == 0]['tweet']))
for word in stopwords:
    positive_words = positive_words.replace(' '+word+' ', '')
positive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="Blues").generate(positive_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(positive_wc)
plt.axis('off'), 
plt.tight_layout(pad = 0)
plt.show()

"""## Dividindo dados de treino e de teste"""

X_train, X_test, y_train, y_test = train_test_split(dataset['tweet'], dataset['depressive'], test_size=0.2)

"""## Transformando palavras em tokens e removendo stop words para word embedding"""

import nltk
import string
from nltk.tokenize import word_tokenize

tweet_lines = list()
lines = word_embedding_dataset['tweet'].values.tolist()

for line in lines:   
    tokens = word_tokenize(str(line))
    # coloca todas as palavras para minusculo
    tokens = [w.lower() for w in tokens]
    # retira a pontuacao    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # retira palavras nao alfabéticos
    words = [word for word in stripped if word.isalpha()]
    # filtra as stopwords  
    words = [w for w in words if not w in stopwords]
    tweet_lines.append(words)
len(tweet_lines)

"""## Criando modelo de palavras"""

import gensim 
# cria um modelo para setar os valores dos parametros

EMBEDDING_DIM = 100
# train word2vec model
model = gensim.models.Word2Vec(sentences=tweet_lines, size=EMBEDDING_DIM, window=5, workers=4, min_count=5)
# tamanho do vocabulario
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

"""## Salvando modelo"""

filename = 'models/tweetsc.txt'

"""## Carregando modelo e testando similaridade de palavras no modelo"""

# carrega modelo treinado previamente
model.wv.most_similar('queijo')

"""## Transformando palavras em token para o treino da rede"""

tweet_lines = list()
lines = dataset['tweet'].values.tolist()

for line in lines:   
    tokens = word_tokenize(str(line))
    # coloca todas as palavras para minusculo
    tokens = [w.lower() for w in tokens]
    # retira a pontuacao    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # retira palavras nao alfabéticos
    words = [word for word in stripped if word.isalpha()]
    # filtra as stopwords  
    words = [w for w in words if not w in stopwords]
    tweet_lines.append(words)
len(tweet_lines)

"""## Carregando modelo de palavras"""

import os

embeddings_index = {}
f = open(os.path.join('', filename),  encoding = "utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()

max_length = 100  # cria outra opções para o tamanho das sentenças

"""## Preparando os dados para treinar a rede
### Transformando os vetores de tokens em único tamanho
"""

from keras.preprocessing import image

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

VALIDATION_SPLIT = 0.2

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(tweet_lines)
sequences = tokenizer_obj.texts_to_sequences(tweet_lines)

# pad sequences
word_index = tokenizer_obj.word_index


tweet_pad = pad_sequences(sequences, maxlen=max_length)
sentiment =  dataset['depressive'].values
print('forma de tweets:', tweet_pad.shape)
print('forma de sentimentos :', sentiment.shape)

# divide a base para treinamento e validação
indices = np.arange(tweet_pad.shape[0])
np.random.shuffle(indices)
tweet_pad = tweet_pad[indices]
sentiment = sentiment[indices]
num_validation_samples = int(VALIDATION_SPLIT * tweet_pad.shape[0])

X_train_pad = tweet_pad[:-num_validation_samples]
y_train = sentiment[:-num_validation_samples]
X_test_pad = tweet_pad[-num_validation_samples:]
y_test = sentiment[-num_validation_samples:]

print('X_train_pad tensor:', X_train_pad.shape)
print('y_train tensor:', y_train.shape)

print('X_test_pad tensor:', X_test_pad.shape)
print('y_test tensor:', y_test.shape)

"""## Criando matriz"""

EMBEDDING_DIM =100
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

"""## Definindo camadas da rede e treinando

"""

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.initializers import Constant
from keras.layers import Conv1D, Dense, Input, LSTM, Embedding, Dropout, Activation, GlobalMaxPooling1D

# define modelo
model = Sequential()
# carrega pre treinamento
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_length,
                            trainable=False)

model.add(embedding_layer)

model.add(Dropout(0.2))

model.add(Conv1D(filters=250, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalMaxPooling1D())

model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# compila
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# encaixa no modelo
model.fit(X_train_pad, y_train, batch_size=30, epochs=20, validation_data=(X_test_pad, y_test), verbose=2)

"""## Testando acurácia"""

loss, accuracy = model.evaluate(X_test_pad, y_test, batch_size=128)
print('Probabilidade (Accuracy): %f' % (accuracy*100))

"""## Realizando testes manuais"""

frase_nao_depressiva = "hoje eu acordei e sai para tomar sorverte"
frase_depressiva = "hoje estou triste e querendo ficar sozinha, por que nao tomei sorverte"

test_samples = [frase_nao_depressiva, frase_depressiva]

test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=100)

model.predict(x=test_samples_tokens_pad)
