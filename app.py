import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

def main():
    st.title('Você sobreviveria ao Titanic?')
    st.write('Modelo de classificação com RandomForest para prever sobrevivência ou morte de passageiros no Titanic')
    st.subheader('Autor')
    st.write('hwww.linkedin.com/in/lucaszonin')
    st.write('')
    st.subheader('Agradecimentos')
    st.write('Felipe Maia Polo que me deu algumas dicas:')
    st.write('https://www.linkedin.com/in/felipemaiapolo/')
    st.write('')

    titanic_v1 = pd.read_csv('datasets/train.csv')
    del titanic_v1['Cabin']
    del titanic_v1['PassengerId']
    del titanic_v1['Ticket']
    del titanic_v1['SibSp']
    del titanic_v1['Parch']
    titanic_v1['Age'] = titanic_v1['Age'].fillna(np.mean(titanic_v1['Age']))
    titanic_v1['Age'] = titanic_v1['Age'].astype('int64')
    titanic_v1 = titanic_v1.dropna()

    titanic_v1.loc[titanic_v1['Sex'] == 'male', 'Sex'] = 0
    titanic_v1.loc[titanic_v1['Sex'] == 'female', 'Sex'] = 1
    titanic_v1['Sex'] = titanic_v1['Sex'].astype(int)

    titanic_v1.loc[titanic_v1['Embarked'] == 'C', 'Embarked'] = 0
    titanic_v1.loc[titanic_v1['Embarked'] == 'Q', 'Embarked'] = 1
    titanic_v1.loc[titanic_v1['Embarked'] == 'S', 'Embarked'] = 2
    titanic_v1['Embarked'] = titanic_v1['Embarked'].astype(int)


    #PUXAR SEXO
    sexo = st.radio(
        label='Sexo do passageiro',
        options=('Feminino', 'Masculino')
    )


    #PUXAR IDADE
    idade_passenger = st.slider(
        label='Idade do passageiro',
        min_value=1,
        max_value=max(titanic_v1['Age'])
    )

    #PUXAR EMBARCACAO
    embarked = st.radio(
        label='Cidade onde embarcou',
        options=('Cherbourg', 'Queenstown', 'Southampton')
    )


    #PUXAR VALOR DA PASSAGEM
    valor_pago = st.slider(
        label='Valor pago pela passagem',
        min_value=1,
        max_value=600
    )

    #PUXAR CLASSE
    classe = st.radio(
        label='Classe do passageiro',
        options=('Primeira', 'Segunda', 'Terceira')
    )

    if sexo == 'Feminino':
        
        sexo_modelo = 1

    else:

        sexo_modelo = 0

    if embarked == 'Cherbourg':

        embarked_modelo = 0


    elif embarked == 'Queenstown':

        embarked_modelo = 1
        

    elif embarked == 'Southampton':

        embarked_modelo = 2

    if classe == 'Primeira':

        classe_modelo = 1


    elif classe == 'Segunda':

        classe_modelo = 2
        

    elif classe == 'Terceira':

        classe_modelo = 3

    titanic_modelo = titanic_v1

    y = titanic_modelo['Survived']
    x = titanic_modelo[['Pclass','Sex','Age','Fare','Embarked']]
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=30)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    model_randomForest = model.predict_proba(X_test)
    #st.write(accuracy_score(y_test,model_randomForest))

    if st.button(label="Prever"):

        st.title('Dados do passageiro:')
        st.write('Sexo :', sexo)
        st.write('Idade :', idade_passenger)
        st.write('Cidade onde embarcou :', embarked)
        st.write('Valor da passagem : US$', valor_pago)
        st.write('Classe da passagem :', classe)

        x_input = pd.DataFrame({'Pclass':classe_modelo,'Sex':sexo_modelo,'Age':idade_passenger,'Fare':valor_pago,'Embarked':embarked_modelo}, index=[0])
        new_model = RandomForestClassifier()
        new_model.fit(X_train, y_train)
        pred = new_model.predict_proba(x_input)

        st.title('Previsão:')
        st.write('')
        'Probabilidade de morrer:', pred[0,0] * 100
        'Probabilidade de sobreviver:', pred[0,1] * 100


if __name__ == '__main__':
    main()