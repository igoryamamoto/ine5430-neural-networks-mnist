
# coding: utf-8

# # Importação de módulos

# In[194]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
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

# In[223]:


mnist = np.loadtxt('exdata.csv', delimiter=',')

data = mnist[:-1].T
data = np.array(list(map(lambda d: d.reshape((20,20)).T.flatten(), data)))
target = mnist[-1]

target[target == 10] = 0


# In[224]:


a = data[4800]
plt.imshow(a.reshape((20,20)), cmap=plt.get_cmap('gray'))


# ### Normalização de dados

# A normalização da saída é feita utilizando OneHotEncoder, que transforma aquela saída única em um vetor colocando 0 paras as outras possíveis saídas da rede e 1 para a saída correta.

# In[225]:


data_normalized = data

target_scaler = OneHotEncoder()
target_normalized = target_scaler.fit_transform(target.reshape((-1, 1))).todense()


# ### Separação de dados treinano/teste

# 
# Foi separado 15% dos dados para formarem o conjunto de teste.

# In[226]:


data_train, data_test, target_train, target_test = train_test_split(
    data_normalized,
    target_normalized,
    train_size=(85/100),
)


# ## Definição da arquitetura

# 
# A rede é configurada da seguinte forma: a camada de entrada com 400 neurônios, a hidden layer com 205 neurônios e a camada de saída com 10 neurônios.

# In[227]:


def mlp():
    model = Sequential()
    model.add(Dense(205, input_dim=400, kernel_initializer="normal"))
    model.add(Activation('relu'))
    model.add(Dense(10, kernel_initializer="normal"))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ## Treinamento da rede

# In[228]:


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

# In[229]:


predictions_all = model.predict(data_normalized)
predictions_all = np.argmax(predictions_all, axis=1)
print("Test set accuracy: {:.2%}".format(
    metrics.accuracy_score(target, predictions_all)))


# 
# Matriz de confusão para todo o conjunto de dados.

# In[230]:


confusion_matrix = metrics.confusion_matrix(target, predictions_all)
sn.heatmap(confusion_matrix, annot=True, fmt='d')


# In[231]:


print(metrics.classification_report(target, predictions_all))


# #### Teste Utilizando Apenas o Conjunto de Teste

# In[232]:


predictions = model.predict(data_test)
predictions = np.argmax(predictions, axis=1)
target_test_classes = np.asarray(target_test.argmax(axis=1)).reshape(-1)
print("Test set accuracy: {:.2%}".format(
    metrics.accuracy_score(target_test_classes, predictions)))


# 
# Matriz de confusão para o conjunto teste.

# In[233]:


confusion_matrix = metrics.confusion_matrix(target_test_classes, predictions)
sn.heatmap(confusion_matrix, annot=True, fmt='d')


# In[234]:


print(metrics.classification_report(target_test_classes, predictions))


# In[251]:


from PIL import Image
import matplotlib.image as mpimg
from resizeimage import resizeimage


with open('5.png', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [20, 20]).convert('L')
        cover.save('5inho.png', image.format)
img = mpimg.imread('5inho.png').flatten()
img.shape
img2 = np.vstack((data_scaler.fit_transform(img),data_scaler.fit_transform(img)))
y = model.predict(img2)
output = y.argmax(axis=1)
print(y[0][output])
print(output)


# In[239]:


plt.imshow(img.reshape((20,20)), cmap=plt.get_cmap('gray'))

