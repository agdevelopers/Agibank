import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
from scipy import stats

import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

# Carregar o dataset
df = pd.read_csv('df.csv')

#Fatiando
x = df.drop('Conta Bancária', axis=1)
y = df['Conta Bancária']

#Separação de Treino e Teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

#Instanciamento
model = LogisticRegression(max_iter=2000)

#Treino
model.fit(x_train, y_train)

#Avaliação
ylogisticRegression = model.predict(x_test)

#Resultados
previsao = x_test[['País',
                    'Local',
                    'Celular',
                    'Tamanho da Família',
                    'Idade',
                    'Gênero',
                    'Relação com Chefe de Família',
                    'Estado Civil',
                    'Nível Educacional',
                    'Tipo de Trabalho']]
previsao['Conta Bancária'] = y_test.values
previsao['Logistic Regression'] = ylogisticRegression

#Dashboard
st.title('Banck Account')
st.sidebar.image('./logo.png')
st.sidebar.subheader('Teste Analista de Ciência de Dados')
st.header('SIMULADOR')
st.subheader('Modelo de Classificação')


#variáveis
idade = st.sidebar.number_input('Informe a Idade: ', value=26)

familia = st.sidebar.number_input('tamanho da Família: ', value=3)

local = st.sidebar.selectbox('Selecione o Local: ', ('-','Urbano', 'Rural'))
local = 1 if local == 'Urbano' else 0

genero = st.sidebar.selectbox('Selecione o Gênero: ', ('-','Masculino', 'Feminino'))
genero = 1 if genero == 'Masculino' else 0

celular = st.sidebar.selectbox('Acesso à Celular: ', ('-','Sim', 'Não'))
celular = 1 if celular == 'Sim' else 0

pais = st.sidebar.selectbox('Selecione o País: ', ('-','Kenya', 'Rwanda', 'Tanzania', 'Uganda'))
if pais == 'Rwanda':
    pais = 1
elif pais == 'Tanzania':
    pais = 2
elif pais == 'Kenya':
    pais = 3
else:
    pais = 4

estado_civil = st.sidebar.selectbox('Selecione o Estado Civil: ', ('-','Casados / Morando Juntos', 'Solteiro(a)', 'Viúvo(a)', 'Divorciados / Separados', 'Não Sabe'))
if estado_civil == 'Casados / Morando Juntos':
    estado_civil = 1
elif estado_civil == 'Solteiro(a)':
    estado_civil = 2
elif estado_civil == 'Viúvo(a)':
    estado_civil = 3
elif estado_civil == 'Divorciados / Separados':
    estado_civil = 4
else:
    estado_civil = 5

educacao = st.sidebar.selectbox('Selecione o Nível de Educação: ', ('-','Sem Educação Formal', 'Ensino Primário', 'Ensino Médio', 'Ensino Superior', 'Treinamento Vocacional / Especializado', 'Outros'))
if educacao == 'Ensino Primário':
    educacao = 1
elif educacao == 'Sem Educação Formal':
    educacao = 2
elif educacao == 'Ensino Médio':
    educacao = 3
elif educacao == 'Ensino Superior':
    educacao = 4
elif educacao == 'Treinamento Vocacional / Especializado':
    educacao = 5
else:
    educacao = 6

trabalho = st.sidebar.selectbox('Selecione o Tipo de Trabalho: ', ('-','Trabalhador Informal', 'Autônomo', 'Emprego Formal', 'Funcionário Público', 'Pensionista', 'Agricultura e Pesca', 'Dependente de Remessa', 'Outros Rendimentos', 'Sem Renda', 'Não Sabe'))
if trabalho == 'Autônomo':
    trabalho = 1
elif trabalho == 'Trabalhador Informal':
    trabalho = 2
elif trabalho == 'Agricultura e Pesca':
    trabalho = 3
elif trabalho == 'Dependente de Remessa':
    trabalho = 4
elif trabalho == 'Outros Rendimentos':
    trabalho = 5
if trabalho == 'Emprego Formal':
    trabalho = 6
if trabalho == 'Sem Renda':
    trabalho = 7
if trabalho == 'Funcionário Público':
    trabalho = 8
if trabalho == 'Pensionista':
    trabalho = 9
else:
    trabalho = 6

chefe = st.sidebar.selectbox('Relação com o Chefe da Família: ', ('-','Chefe de Família', 'Conjugê', 'Filho(a)', 'Pais', 'Outro Relacionamento', 'Sem Relancionamento'))
if chefe == 'Chefe de Família':
    chefe = 1
elif chefe == 'Conjugê':
    chefe = 2
elif chefe == 'Filho(a)':
    chefe = 3
elif chefe == 'Pais':
    chefe = 4
elif chefe == 'Outro Relacionamento':
    chefe = 5
else:
    chefe = 6

# Botão
btn = st.sidebar.button('Realizar Predição')
if btn == True:
    result = model.predict([[pais,local,celular,familia,idade,genero,chefe,estado_civil,educacao,trabalho]])
    resultado = str(result[0])
    st.write(f'O Entrevistado possui Conta Bancária?  **{resultado}.**')
    #Acurácia    
    acuracia = model.score(x_test, y_test)
    st.write(f'*O Modelo* acertou {(acuracia * 100).round(decimals=2)}% das previsões.')
