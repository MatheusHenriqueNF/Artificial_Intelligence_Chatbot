import sys
print(sys.executable)

#@title Responda Aqui

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("filme.csv", sep=",")

dataset.head()

## variaveis numericas - id, ano, nota, tempo em min
## categorica - genero

#@title Responda Aqui   

dataset2 = dataset['nota'].describe() ## 1. exibe a média, 2. Mediana, 3. Mínimo, 4. Quartil (25, 50, 75), 5. Máximo
print(dataset2)

print("Desvio Padrão:",dataset['nota'].std()) ## calculando o desvio padrão

## gráficos
sns.boxplot(dataset['nota'], orient='h')
print(dataset.dtypes)

import matplotlib.pyplot as plt

# Gerando o gráfico de dispersão entre ano e nota
plt.scatter(dataset['tempo_minutos'], dataset['nota'], color='blue', label="Ano vs Nota")

# Ajustando o gráfico
plt.xlabel('Tempo')
plt.ylabel('Nota')
plt.title('Relação entre Tempo e Nota dos Filmes')
plt.legend()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Selecionando apenas as colunas numéricas
dataset_numerico = dataset.select_dtypes(include=['float64', 'int64'])

# Calculando a correlação entre as variáveis numéricas
correlacao = dataset_numerico.corr()

# Gerando o mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)

# Ajustando o gráfico
plt.title('Mapa de Calor - Correlação entre as Variáveis')
plt.show()

#@title Responda Aqui
from sklearn.model_selection import train_test_split

dataset[['titulo','generos']]

## colunas preditivas
x = dataset.drop(columns = ['titulo', 'generos'])
y = dataset['generos']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Criando o objeto do classificador
lda_model = LinearDiscriminantAnalysis()  # Mudamos o nome para lda_model

# Treinando o classificador com os dados de treinamento
lda_model.fit(x_train, y_train)

# Você pode agora fazer previsões usando lda_model
y_pred = lda_model.predict(x_test)

y_predicoes = lda_model.predict(x_test)

## matriz de confusão
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

dataset['generos'].value_counts()

matriz_confusao = confusion_matrix(
    y_true=y_test,
    y_pred=y_predicoes,
    labels=unique_genres 
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz_confusao, 
    display_labels=unique_genres 
)
disp.plot(values_format='d')
plt.show()

accuracy_score(y_test, y_predicoes)

lda_model.coef_[0] #c1, c2
lda_model.coef_[0,1]
lda_model.intercept_/lda_model.coef_[0,1]
lda_model.coef_[0,0]/lda_model.coef_[0,1]
x1 = np.arange(0,10)
x2 = -2.635*x1 + 18.51
x_train

import numpy as np

# Get the coefficients of the trained ML model
coefficients = lda_model.coef_[0]

# Assign the first two coefficients to c1 and c2
c1, c2 = coefficients[:2]  # Use slicing to get the first two values

# Get the intercept for the first class (index 0)
b = lda_model.intercept_[0]  # Assuming you want to visualize the first class

# Calculate the equation of the LDA line for the first class
x1 = np.arange(0, 10)
x2 = -(c1 / c2) * x1 - b / c2

plt.plot(x2,x1)
plt.scatter(x_train['ano'],x_train['nota'])
plt.xlabel('Ano') 
plt.ylabel('Nota')