
# coding: utf-8

# # Importação de módulos

# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot
import seaborn as sn
import pandas as pd
import numpy as np
import random

from neupy import environment, algorithms, layers, plots
from neupy.exceptions import StopTraining

from sklearn import datasets, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import theano


theano.config.floatX = 'float32'

environment.reproducible()


# # Tratamento de Dados

# ### Carregamento de dados

# In[32]:


data = np.loadtxt('exdata.csv', delimiter=',')

features = data[:-1].T
target = data[-1]

target[target == 10] = 0


# ### Normalização de dados

# In[33]:


data_scaler = StandardScaler()
data_normalized = data_scaler.fit_transform(features)

target_scaler = OneHotEncoder()
target_normalized = target_scaler.fit_transform(target.reshape((-1, 1))).todense()


# 
# Para normalizar os dados foi utilizada a classe StandardScaler que permite guardar os parâmetros utilizados para pode reverter a normalização dos dados ou utilizar a mesma normalização em outros dados. Para normalizar é calculado a média e desvio padrão para criar os parâmetros. O método fit_transform realiza o cálculo e a normalização.
#     
# A normalização da saída é feita utilizando OneHotEncoder, que transforma aquela saída única em um vetor colocando 0 paras as outras possíveis saídas da rede e 1 para a saída correta.

# ### Separação de dados treinano/teste

# In[34]:


data_train, data_test, target_train, target_test = train_test_split(
    data_normalized,
    target_normalized,
    train_size=(85/100),
)


# 
# Foi separado 15% dos dados para formarem o conjunto de teste.

# # Treinamento da Rede Neural

# ### Definição da arquitetura

# 
# A rede é configurada da seguinte forma: a camada de entrada com 400 neurônios, a hidden layer com 35 neurônios e a camada de saída com 10 neurônios, ambas com a função tangente.
# 
# O algoritmo de treinamento utilizado foi o MinibatchGradientDescent que é um gradiente descentente divido em batchs para melhoro desempenho. Também foi configurado um objetivo para tentar alcançar um erro de 0.01 e uma taxa de aprendizado de 0.1.

# In[42]:


def check_goal(goal):
    def callback(net):
        if net.errors.last() < goal:
            raise StopTraining("Goal reached")

    return callback

net = algorithms.MinibatchGradientDescent(
    [
        layers.Input(400),
        layers.Tanh(30),
        layers.Tanh(10),
    ],
    verbose=True,
    show_epoch=1000,
#     nesterov=True,
    epoch_end_signal=check_goal(0.001),
)


# In[43]:


net.architecture()


# ### Treinamento da rede

# 
# Ao treinar a rede foi estabelecido os sinais de parada de 10000 épocas ou atingir o objetivo para o erro já estabelecido acima de 0.01.

# In[44]:


net.train(data_train, target_train, data_test, target_test, epochs=10000)


# In[45]:


plots.error_plot(net)


# ### Teste da rede

# #### Teste Utilizando Todos os Dados

# In[46]:


predicted = net.predict(data_normalized)
predicted_classes = np.asarray(predicted.argmax(axis=1)).reshape(-1)


# 
# Matriz de confusão para todo o conjunto de dados.

# In[57]:


confusion_matrix = metrics.confusion_matrix(target, predicted_classes)
sn.heatmap(confusion_matrix, annot=True, fmt='d')


# In[41]:


print(metrics.classification_report(target, predicted_classes))


# #### Teste Utilizando Apenas o Conjunto de Teste

# In[49]:


predicted_test = net.predict(data_test)
predicted_test_classes = np.asarray(predicted_test.argmax(axis=1)).reshape(-1)
target_test_classes = np.asarray(target_test.argmax(axis=1)).reshape(-1)


# 
# Matriz de confusão para o conjunto teste.

# In[53]:


confusion_matrix = metrics.confusion_matrix(target_test_classes, predicted_test_classes)
sn.heatmap(confusion_matrix, annot=True)


# In[51]:


print(metrics.classification_report(target_test_classes, predicted_test_classes))


# ### Precisão Final

# In[52]:


print("Dataset accuracy: {:.2%}".format(
    metrics.accuracy_score(target, predicted_classes)))
print("Test set accuracy: {:.2%}".format(
    metrics.accuracy_score(target_test_classes, predicted_test_classes)))

