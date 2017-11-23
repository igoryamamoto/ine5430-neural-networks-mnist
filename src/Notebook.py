
# coding: utf-8

# # Importação de módulos

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from keras import backend as K
import psutil
import os 
from keras.models import Sequential
from keras.layers import Dense, Activation
#from keras.layers import Dropout
from keras.utils import np_utils
#from keras.layers import Flatten
#from keras.layers.convolutional import Convolution2D
#from keras.layers.convolutional import MaxPooling2D
#from keras import backend as K
K.set_image_dim_ordering('th')
import time

num_cores = psutil.cpu_count()
GPU= True

if GPU:
    num_GPU = 1
    num_CPU = 1
    print('Usando a GPU')
else:
    num_CPU = 1
    num_GPU = 0
    print('Usando apenas o CPU')

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


# # Tratamento de Dados

# ### Carregamento de dados

# In[3]:


data = np.loadtxt('exdata.csv', delimiter=',')

features = data[:-1].T
target = data[-1]

target[target == 10] = 0


# ### Normalização de dados

# 
# Para normalizar os dados foi utilizada a classe StandardScaler que permite guardar os parâmetros utilizados para pode reverter a normalização dos dados ou utilizar a mesma normalização em outros dados. Para normalizar é calculado a média e desvio padrão para criar os parâmetros. O método fit_transform realiza o cálculo e a normalização.
#     
# A normalização da saída é feita utilizando OneHotEncoder, que transforma aquela saída única em um vetor colocando 0 paras as outras possíveis saídas da rede e 1 para a saída correta.

# In[4]:


data_scaler = StandardScaler()
data_normalized = data_scaler.fit_transform(features)

target_scaler = OneHotEncoder()
target_normalized = target_scaler.fit_transform(target.reshape((-1, 1))).todense()


# ### Separação de dados treinano/teste

# 
# Foi separado 15% dos dados para formarem o conjunto de teste.

# In[5]:


data_train, data_test, target_train, target_test = train_test_split(
    data_normalized,
    target_normalized,
    train_size=(85/100),
)


# ## Definição da arquitetura

# 
# A rede é configurada da seguinte forma: a camada de entrada com 400 neurônios, a hidden layer com 205 neurônios e a camada de saída com 10 neurônios.

# In[73]:


def mlp():
    model = Sequential()
    model.add(Dense(205, input_dim=400, kernel_initializer="normal"))
    model.add(Activation('relu'))
    model.add(Dense(10, kernel_initializer="normal"))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ## Treinamento da rede

# In[74]:


model = mlp()

start_time = time.time()

treinar = True
if treinar:
    print("Treinando a rede")
    model.fit(data_train, target_train, validation_data=(data_test, target_test), epochs=100, batch_size=150, verbose=0)
    model.save_weights("mlp.h5")
    print("Pesos salvos")
else:
    model.load_weights("mlp.h5")
    print("Pesos recuperados do disco")

scores = model.evaluate(data_test, target_test, verbose=0)

print("Baseline Error: %.2f%%" % (100-scores[1]*100))
print("Rede treinada/buscada em %.2f segundos" % (time.time() - start_time))


# ## Resultados

# #### Teste Utilizando Todos os Dados

# In[75]:


predictions_all = model.predict(data_normalized)
predictions_all = np.argmax(predictions_all, axis=1)
print("Test set accuracy: {:.2%}".format(
    metrics.accuracy_score(target, predictions_all)))


# 
# Matriz de confusão para todo o conjunto de dados.

# In[76]:


confusion_matrix = metrics.confusion_matrix(target, predictions_all)
sn.heatmap(confusion_matrix, annot=True, fmt='d')


# In[77]:


print(metrics.classification_report(target, predictions_all))


# #### Teste Utilizando Apenas o Conjunto de Teste

# In[78]:


predictions = model.predict(data_test)
predictions = np.argmax(predictions, axis=1)
target_test_classes = np.asarray(target_test.argmax(axis=1)).reshape(-1)
print("Test set accuracy: {:.2%}".format(
    metrics.accuracy_score(target_test_classes, predictions)))


# 
# Matriz de confusão para o conjunto teste.

# In[79]:


confusion_matrix = metrics.confusion_matrix(target_test_classes, predictions)
sn.heatmap(confusion_matrix, annot=True, fmt='d')


# In[80]:


print(metrics.classification_report(target_test_classes, predictions))

