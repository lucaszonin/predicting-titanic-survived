import streamlit as st
import webbrowser
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def main():
    st.title('Você sobreviveria ao Titanic?')
    st.write('Modelo de classificação com RandomForest para prever sobrevivência ou morte de passageiros no Titanic')
    st.write('')
    if st.button('Linkedin - Lucas Zonin Soares'):
        webbrowser.open_new_tab('https://www.linkedin.com/in/lucas-z-86a319160/')
    st.write('')

if __name__ == '__main__':
    main()