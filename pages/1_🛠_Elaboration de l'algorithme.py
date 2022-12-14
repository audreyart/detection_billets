import streamlit as st

st.title("Elaboration de l'algorithme - régression logistique")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing

st.subheader("1.Lecture du fichier")
billets = pd.read_csv("billets.csv", sep=';')
billets # Affichage du dataframe

# Changement des booléens "True" et "False" en valeurs numériques 1 et 0
billets['is_genuine'] = billets['is_genuine'].replace([True], 1)
billets['is_genuine'] = billets['is_genuine'].replace([False], 0)

is_genuine = billets[['is_genuine']]

# Affichage des valeurs manquantes
st.subheader("2. Des valeurs sont manquantes dans le dataframe billets")
billets_nulls = billets.loc[billets['margin_low'].isnull()]
billets_nulls

# Retrait de la colonne is_genuine
billets = billets[['diagonal','height_left','height_right','margin_low','margin_up','length']]

# Les valeurs manquantes sont remplacées en utilisant une régression linéaire
st.subheader("3. Remplacement des valeurs manquantes grâce à une régression linéaire")

billets2 = billets.dropna()

X = billets2[['diagonal','height_left','height_right','margin_up','length']]

y = billets2['margin_low']

from sklearn.model_selection import train_test_split
# Définition du set d'entraînement et du set de test
billetsTrain, billetsTest = train_test_split(billets2, train_size = 0.8, random_state=0)

y_test = billetsTest['margin_low']

reg_simple_Train1 = smf.ols('margin_low~length', data=billetsTrain).fit()  # Une seule variable utilisée
reg_simple_Train1.summary() # Résultats sous forme textuelle

# Prédiction des valeurs pour les billets aux "margin_low" nuls
margin_low_pred = reg_simple_Train1.predict(billets_nulls).round(2)
st.table(margin_low_pred)
# Valeurs manquantes remplacées par les valeurs prédites
billets['margin_low'].loc[billets['margin_low'].isnull()] = margin_low_pred

# Régression logistique
st.subheader("4. Régression logistique")

# Importation de l'outil
from statsmodels.tools import add_constant

# Importation de la classe de calcul
from statsmodels.api import Logit

billets['is_genuine'] = is_genuine
y = billets['is_genuine']
X = billets[['margin_up','margin_low','length','height_right']]
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=1)

ytrain.value_counts()  # Comptage du nombre vrais billets (1) et de faux billets(0)
st.markdown("Taille des différents sets")
st.write("xtrain.shape :",xtrain.shape)
st.write("ytrain.shape :",ytrain.shape)
st.write("xtest.shape :",xtest.shape)
st.write("ytest.shape :",ytest.shape)

# Données X avec la constante
st.markdown("Le set d'entraînement après l'ajout de la constante")
XtrainBis = sm.tools.add_constant(xtrain)
XtrainBis

# Régression logistique (variable cible et variables explicatives)
lr = Logit(endog=ytrain,exog=XtrainBis)

# Calcul
res2 = lr.fit()


# Evaluation du modèle
st.subheader("5. Evaluation du modèle défini")
st.markdown("#### Test de significativité des coefficients")
st.text("H0 : les coefficients sont nulls.")
# Test : rapport de vraisemblance
deviance_modele = (-2) * res2.llf
st.write("La déviance du modèle : %.4f" %(deviance_modele))

deviance_trivial_modele = (-2) * res2.llnull
st.write("La déviance du modèle : %.4f" %(deviance_trivial_modele))

# Statistique du rapport de vraisemblance = deviance_trivial_modele - deviance_modele
st.write("Statistique du rapport de vraisemblance : %.4f" %(res2.llr))

st.write("Nombre de coefficients estimés (sauf constante) :", res2.df_model)
st.write("La pvalue :", res2.llr_pvalue)
st.text("Au seuil de 5%, on rejette l'hypothèse H0 de nullité des coefficients.")
st.table(res2.params)
st.markdown("Test de Wald")
# Test de Wald
m = [[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
res2.wald_test(m)
st.write(res2.wald_test_terms())
st.text("Au niveau de test 5%, on rejette ici aussi l'hypothèse de nullité des coefficients.")
st.text("Les coefficients sont donc bien significatifs.")
# Cross validation
# Evaluation du modèle de régression logistique avec Repeated k-fold 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
# Préparation de la procédure de validation croisée
cv = RepeatedKFold(n_splits=5, n_repeats= 100, max_iter=1000, random_state=1)
model = LogisticRegression()
precision = cross_val_score(model, X, y, scoring='precision', cv=cv, n_jobs=-1)
# Taux d'erreur
scores_error = 1 - cross_val_score(model, X, y, scoring='precision', cv=cv, n_jobs=-1)

st.subheader("Taux d'erreur : ")
fig,ax = plt.subplots()
ax.hist(scores_error)
st.pyplot(fig)


# Sauvegarde du modèle
from joblib import dump, load

dump(res2, 'lr_saved.joblib')

res2_load = load('lr_saved.joblib')


dump(XtrainBis,'XtrainBis.joblib')

XtrainBis = load('XtrainBis.joblib')
