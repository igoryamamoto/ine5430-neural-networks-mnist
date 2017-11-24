
# coding: utf-8

# # Importação de módulos

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import random
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
from keras.utils import np_utils
from keras import optimizers
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

# Os dados são treinados são carregados, separando-se entre os dados contendo features e labels. Os dados de imagens são rotacionados para ficarem na posição correta.

# In[2]:


mnist = np.loadtxt('../data/exdata.csv', delimiter=',')

data = mnist[:-1].T
data = np.array(list(map(lambda d: d.reshape((20,20)).T.flatten(), data)))
target = mnist[-1]

target[target == 10] = 0


# ### Exemplos de imagens do conjunto de dados

# In[3]:


rows = 2
cols = 5
f, axarr = plt.subplots(rows, cols)
for r in range(rows):
    for c in range(cols):
        rand_data = random.choice(data)
        axarr[r, c].imshow(rand_data.reshape((20,20)), cmap=plt.get_cmap('gray'))


# ### Normalização de dados

# A saída é normalizada através do OneHotEncoder, que transforma o valor do target em um vetor de zeros e valor 1 no índicie correspondente ao valor do target

# In[4]:


target_scaler = OneHotEncoder()
target_normalized = target_scaler.fit_transform(target.reshape((-1, 1))).todense()


# ### Separação de dados treinano/teste

# 15% dos dados serão separados para teste

# In[5]:


data_train, data_test, target_train, target_test = train_test_split(
    data,
    target_normalized,
    train_size=(85/100),
)


# ## Definição da arquitetura

# Camada de entrada: 400 neurônios; Hidden layer: 205 neurônios; saída: 10 neurônios
# Função de Ativação: Relu
# Otimizador SGD

# In[14]:


def mlp():
    model = Sequential()
    model.add(Dense(205, input_dim=400, kernel_initializer="normal"))
    model.add(Activation('relu'))
    model.add(Dense(10, kernel_initializer="normal"))
    model.add(Activation('softmax'))
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='mean_squared_error', optimizer=sgd)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ## Treinamento da rede

# In[18]:


model = mlp()

start_time = time.time()

treinar = True
if treinar:
    print("Treinando a rede")
    model.fit(data_train, 
              target_train, 
              validation_data=(data_test, target_test), 
              epochs=1000, 
              batch_size=150, 
              verbose=0
             )
    model.save_weights("mlp.h5")
    print("Pesos salvos")
else:
    model.load_weights("mlp.h5")
    print("Pesos recuperados do disco")

scores = model.evaluate(data_test, target_test, verbose=0)

print("Rede treinada/buscada em %.2f segundos" % (time.time() - start_time))


# ## Resultados

# #### Teste Utilizando Todos os Dados

# In[19]:


predictions_all = model.predict(data)
predictions_all = np.argmax(predictions_all, axis=1)
print("Test set accuracy: {:.2%}".format(
    metrics.accuracy_score(target, predictions_all)))


# 
# Matriz de confusão para todo o conjunto de dados.

# In[24]:


confusion_matrix = metrics.confusion_matrix(target, predictions_all)
sn.heatmap(confusion_matrix, annot=True, fmt='d')


# In[11]:


print(metrics.classification_report(target, predictions_all))


# #### Teste Utilizando Apenas o Conjunto de Teste

# In[20]:


predictions = model.predict(data_test)
predictions = np.argmax(predictions, axis=1)
target_test_classes = np.asarray(target_test.argmax(axis=1)).reshape(-1)
print("Test set accuracy: {:.2%}".format(
    metrics.accuracy_score(target_test_classes, predictions)))


# 
# Matriz de confusão para o conjunto teste.

# In[21]:


confusion_matrix = metrics.confusion_matrix(target_test_classes, predictions)
sn.heatmap(confusion_matrix, annot=True, fmt='d')


# In[22]:


print(metrics.classification_report(target_test_classes, predictions))


# ### Alguns erros de classificação

# In[23]:


wrong_class = [i for i in range(predictions.size) if predictions[i]!=target_test_classes[i]]
rows = 2
cols = 4
f, axarr = plt.subplots(rows, cols)
for r in range(rows):
    for c in range(cols):
        rand_i = random.choice(wrong_class)
        axarr[r, c].imshow(data_test[rand_i].reshape((20,20)), cmap=plt.get_cmap('gray'))
        axarr[r, c].set_title('Pred:{}, Real:{}'.format(predictions[rand_i], 
                                                        target_test_classes[rand_i]))


# # Classificação de novos dados

# In[16]:


from PIL import Image
import matplotlib.image as mpimg
from resizeimage import resizeimage


with open('5.png', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [20, 20]).convert('L')
        cover.save('5inho.png', image.format)
with open('7.png', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [20, 20]).convert('L')
        cover.save('7inho.png', image.format)
img5 = mpimg.imread('5inho.png').flatten()
img7 = mpimg.imread('7inho.png').flatten()
img = np.vstack((img5,img7))
y = model.predict(img)
output = y.argmax(axis=1)
print(y[0][output])
print(output)
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(img5.reshape((20,20)), cmap=plt.get_cmap('gray'))
axarr[1].imshow(img7.reshape((20,20)), cmap=plt.get_cmap('gray'))

