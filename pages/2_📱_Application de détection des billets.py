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


        predSm = np.where(result > 0.5, True, False)

        st.markdown("Prédiction sur les billets du fichier : ")
        pred = pd.DataFrame({"Probabilité":result,"Prédiction":predSm})
        st.table(pred)

        st.write("Nombre de vrais billets : ",pred['Probabilité'].loc[pred['Probabilité']>0.5].count())
        st.write("Nombre de faux billets : ",pred['Probabilité'].loc[pred['Probabilité']<0.5].count())


with tab2:

    billets_production = pd.read_csv("billets_production.csv")
    id = billets_production['id']

    st.subheader("Tableau des données du fichier test")
    st.markdown("Fichier billets_production.csv :")
    st.table(billets_production)

    df = pd.read_csv("billets_production.csv")

    @st.experimental_memo
    def convert_df(df):
       return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(df)

    st.download_button(
       "Télécharger le fichier billets_production",
       csv,
       "billets_production.csv",
       "text/csv",
       key='download-csv'
    )

    st.info("Vous pouvez télécharger le fichier test pour avoir un aperçu des résultats du traitement")

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


            #On garde les variables explicatives
            test_algo = test_algo[['margin_up','margin_low','length','height_right']] # Les 4 variables explicatives

            test_algo = sm.tools.add_constant(test_algo) # Ajout de la constante

            predProbaSm = res2_load.predict(test_algo) # Prédiction

            predSm = np.where(predProbaSm > 0.5, True, False)

            st.markdown("Prédiction sur les billets du fichier : ")
            prediction = pd.DataFrame({"identifiant":id,"Probabilité":predProbaSm,"Prédiction":predSm})
            st.table(prediction)

            st.write("Nombre de vrais billets : ",prediction['Probabilité'].loc[prediction['Probabilité']>0.5].count())
            st.write("Nombre de faux billets : ",prediction['Probabilité'].loc[prediction['Probabilité']<0.5].count())
        
        else:
            st.error("Télécharger un fichier pour le calcul")