import streamlit as st

st.title("Application de détection des faux billets")

from joblib import load

res2_load = load('lr_saved.joblib')
XtrainBis = load('XtrainBis.joblib')

import pandas as pd
import numpy as np
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from statsmodels.api import Logit
# Importation de l'outil
from statsmodels.tools import add_constant


tab1,tab2 = st.tabs(["Données entrées manuellement","Données à partir du fichier"])

with tab1:
    col1,col2,col3 = st.columns(3)
    with col1:
        st.subheader("Mesures du billet 1")
        margin_up1 = st.number_input("margin_up billet 1")
        margin_low1 = st.number_input("margin_low billet 1")
        length1 = st.number_input("length billet 1")
        height_right1 = st.number_input("height_right billet 1")

        st.subheader("Mesures du billet 4")
        margin_up4 = st.number_input("margin_up billet 4")
        margin_low4 = st.number_input("margin_low billet 4")
        length4 = st.number_input("length billet 4")
        height_right4 = st.number_input("height_right billet 4")

    with col2:
        st.subheader("Mesures du billet 2")
        margin_up2 = st.number_input("margin_up billet 2")
        margin_low2 = st.number_input("margin_low billet 2")
        length2 = st.number_input("length billet 2")
        height_right2 = st.number_input("height_right billet 2")

        st.subheader("Mesures du billet 5")
        margin_up5 = st.number_input("margin_up billet 5")
        margin_low5 = st.number_input("margin_low billet 5")
        length5 = st.number_input("length billet 5")
        height_right5 = st.number_input("height_right billet 5")

    with col3:
        st.subheader("Mesures du billet 3")
        margin_up3 = st.number_input("margin_up billet 3")
        margin_low3 = st.number_input("margin_low billet 3")
        length3 = st.number_input("length billet 3")
        height_right3 = st.number_input("height_right billet 3")



    df = pd.DataFrame({'margin_up':[margin_up1,margin_up2,margin_up3,margin_up4,margin_up5],'margin_low':[margin_low1,margin_low2,margin_low3,margin_low4,margin_low5],'length':[length1,length2,length3,length4,length5],'height_right':[height_right1,height_right2,height_right3,height_right4,height_right5]})
    st.subheader("Tableau des données entrées ")
    st.write(df)

    if st.button("Calculer à partir des données entrées"):
        df = sm.tools.add_constant(df)
        result = res2_load.predict(df)
        if result[0]>0.5:
            st.text("Le billet 1 est vrai")
        elif result[0]<0.5:
            st.text("Le billet 1 est faux")
        if result[1]>0.5:
            st.text("Le billet 2 est vrai")
        elif result[1]<0.5:
            st.text("Le billet 2 est faux")
        if result[2]>0.5:
            st.text("Le billet 3 est vrai")
        elif result[2]<0.5:
            st.text("Le billet 3 est faux")
        if result[3]>0.5:
            st.text("Le billet 4 est vrai")
        elif result[3]<0.5:
            st.text("Le billet 4 est faux")
        if result[4]>0.5:
            st.text("Le billet 5 est vrai")
        elif result[4]<0.5:
            st.text("Le billet 5 est faux")


with tab2:

    billets_production = pd.read_csv("billets_production.csv")
    id = billets_production['id']

    st.subheader("Tableau des données du fichier test")
    st.markdown("Fichier billets_production.csv :")
    st.table(billets_production)


    # Entrer le chemin du fichier entre les guillemets
    fichier = "billets_production.csv"

    test_algo = pd.read_csv(fichier,sep=None,engine='python') #Lecture du fichier

    test_algo = test_algo.fillna(XtrainBis.mean().round(2)) #Attribution de la moyenne de la variable pour les valeurs manquantes
    # (moyenne sur les données d'entraînement)
    st.text('Dataframe sans valeurs manquantes :')
    st.table(test_algo)

    st.text('On garde les quatre variables explicatives : ')
    test_algo = test_algo[['margin_up','margin_low','length','height_right']] # Les 4 variables explicatives
    st.table(test_algo)

    test_algo = sm.tools.add_constant(test_algo) # Ajout de la constante
    st.text('On ajoute la constante :')
    st.table(test_algo)

    predProbaSm = res2_load.predict(test_algo) # Prédiction

    predSm = np.where(predProbaSm > 0.5, True, False)
        
    st.markdown("Prédiction sur les billets du fichier billets_production.csv : ")
    prediction = pd.DataFrame({"identifiant":id,"Probabilité":predProbaSm,"Prédiction":predSm})
    st.table(prediction)

    st.write("Nombre de vrais billets : ",prediction['Probabilité'].loc[prediction['Probabilité']>0.5].count())
    st.write("Nombre de faux billets : ",prediction['Probabilité'].loc[prediction['Probabilité']<0.5].count())


    st.subheader("Télécharger le fichier avec les billets à évaluer")
    from io import StringIO

    # Entrer le chemin du fichier entre les guillemets
    fichier = st.file_uploader("Choisissez un fichier")
    
    st.info("Télécharger un fichier contenant les mesures de vos billets afin de procéder au calcul. Prenez le fichier test pour modèle.")


    if st.button("Calculer à partir du fichier téléchargé"):

        if fichier is not None:
            bytes_data = fichier.getvalue()

            stringio = StringIO(fichier.getvalue().decode("utf-8"))

            string_data = stringio.read()

            dataframe = pd.read_csv(fichier)
            st.write(dataframe)
            test_algo = dataframe

            st.text('Dataframe sans valeurs manquantes :')
            st.table(test_algo)

            st.text('On garde les quatre variables explicatives : ')
            test_algo = test_algo[['margin_up','margin_low','length','height_right']] # Les 4 variables explicatives
            st.table(test_algo)

            test_algo = sm.tools.add_constant(test_algo) # Ajout de la constante
            st.text('On ajoute la constante :')
            st.table(test_algo)

            predProbaSm = res2_load.predict(test_algo) # Prédiction

            predSm = np.where(predProbaSm > 0.5, True, False)

            st.markdown("Prédiction sur les billets du fichier billets_production.csv : ")
            prediction = pd.DataFrame({"identifiant":id,"Probabilité":predProbaSm,"Prédiction":predSm})
            st.table(prediction)

            st.write("Nombre de vrais billets : ",prediction['Probabilité'].loc[prediction['Probabilité']>0.5].count())
            st.write("Nombre de faux billets : ",prediction['Probabilité'].loc[prediction['Probabilité']<0.5].count())
        
        else:
            st.error("Télécharger un fichier pour le calcul")
