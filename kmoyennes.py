# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: kmoyennes.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions pour les k-moyennes

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt

import math
import random

# ---------------------------
# Dans ce qui suit, remplacer la ligne "raise.." par les instructions Python
# demandées.
# ---------------------------

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
    
def normalisation(dataframe):
    mini, maxi = dataframe.min(), dataframe.max()
    normalized_df=(dataframe-mini)/(maxi -mini)
    return normalized_df

def dist_vect(vecteur1, vecteur2):
    return math.sqrt(sum((vecteur1 - vecteur2)**2))

def centroide(matrice):
    return pd.DataFrame(matrice.mean(axis=0),list(matrice)).transpose()

def inertie_cluster(data):
    c = centroide(data).iloc[0]
    distance = lambda ligne : dist_vect(ligne, c)**2
    return sum(data.apply(distance, axis=1))
    
def initialisation(K, data):
    dataF = data.sample(K)
    dataF.index = range(K)
    return dataF

def plus_proche(exemple, dataframe_centroide):
    distance_a_point = lambda x: dist_vect(x, exemple)
    return dataframe_centroide.apply(distance_a_point, axis=1).idxmin()

def affecte_cluster(base, ensemble_centroides):
    matrice =  {}
    for i in range(ensemble_centroides.shape[0]):
        matrice[i] =[]
    for i in range(base.shape[0]):
        k = plus_proche(base.iloc[i],ensemble_centroides)
        matrice[k].append(i)
    return matrice


def nouveaux_centroides(df, matrice):
    new_C = centroide(df.iloc[matrice[list(matrice.keys())[0]]])
    for i in range(1, len(matrice)):
        new_C = new_C.append(centroide(df.iloc[matrice[list(matrice.keys())[i]]]))
    new_C.index = range(len(matrice))
    return new_C



def inertie_globale(df, matrice):
    s = 0
    for i in matrice:
        s +=inertie_cluster(df.iloc[matrice[i]])
    return s



def kmoyennes(k, df, epsilon, iter_max):
    #initialisation
    centroides = initialisation(k,df)
    matrice = affecte_cluster(df, centroides)
    m2 = affecte_cluster(df, nouveaux_centroides(df, matrice))
    i = 1
    while i < iter_max and inertie_globale(df, matrice) -inertie_globale(df, m2) > epsilon:
        print("iteration",  i , " Inertie : ", inertie_globale(df, matrice) ,  "Difference:" , inertie_globale(df, matrice) -inertie_globale(df, m2))
        matrice = m2
        centroides = nouveaux_centroides(df, matrice)
        m2 = affecte_cluster(df, centroides)
        i+=1
    return centroides, m2

def impact_K(df, K):
    a, b =[], []
    for i in range(1,K):
        les_centres, l_affectation = kmoyennes(i, df, 0.01, 100)
        b.append(inertie_globale(df, l_affectation))
        a.append(i)
    plt.plot(a, b, "b")
    plt.show()

def affiche_resultat(Df,les_centres,l_affectation, O=3):
    #plt.scatter(Df['X'],Df['Y'],color='b')
    #plt.scatter(les_centres['X'],les_centres['Y'],color='r',marker='x')
    colors=["green", "black", "blue", "yellow", "orange", "pink", "brown", "violet", "indigo", "red"]
    for k in range(O):
        plt.scatter(les_centres.iloc[k]['X'],les_centres.iloc[k]['Y'],color=colors[k],marker='x')
        plt.scatter(Df.iloc[l_affectation[k]]['X'],Df.iloc[l_affectation[k]]['Y'],color=colors[k])
# -------

def dist_intracluster(df):
    maxi = 0
    for i, row in df.iterrows():
        distance = lambda ligne : dist_vect(ligne, row)
        maxi_tmp = max(df.apply(distance, axis=1))
        if(maxi_tmp > maxi):
            maxi= maxi_tmp
        
    
    return maxi

def global_intraclusters(df, affec):
    maxi = 0
    for i in range(len(affec)):
        maxi_tmp = dist_intracluster(df.query('index in ' + str(affec[i])))
        if maxi < maxi_tmp:
            maxi = maxi_tmp
    return maxi


def sep_clusters(centroides):
    mini = float("inf")
    for i, c in centroides.iterrows():
        distance = lambda ligne : dist_vect(ligne, c) if dist_vect(ligne, c) != 0 else float("inf") 
        mini_tmp = min(centroides.apply(distance, axis=1))
        if(mini_tmp < mini):
            mini= mini_tmp
    return mini

def evaluation(nom, df, centroides, affec):
    if(nom == "Dunn"):
        return global_intraclusters(df, affec) / sep_clusters(centroides)
    else:
        return inertie_globale(df, affec) / sep_clusters(centroides)
    
    
def test(df):
    a, b , c=[], [] , []
    for i in range(2,11):
        les_centres, l_affectation = kmoyennes(i, df, 0.05, 100)
        b.append(evaluation("Dunn",df,les_centres,l_affectation))
        c.append(evaluation("XB",df,les_centres,l_affectation))
        a.append(i)
    plt.plot(a, b, "b", a , c, "r")
    plt.legend(["dunn", "Xie & Beni"])
    plt.show()


