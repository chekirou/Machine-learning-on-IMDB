# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
from iads import LabeledSet as ls
import random
import math
# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
 
     #TODO: A Compléter

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
        raise NotImplementedError("Please Implement this method")
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        per=0
        for i in range(dataset.size()):
            if self.predict(dataset.getX(i)) == dataset.getY(i):
                per+=1
        return float(per)/dataset.size()
        
# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    #TODO: A Compléter
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.w = 2*np.random.rand(input_dimension)-1
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        s = 0
        for i in range(len(x)):
            s += x[i]*self.w[i]
        return (1 if s>=0 else -1)

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        print(" no trainning for this model")
    
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    #TODO: A Compléter
 
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k= k
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        #construit la matrice des distance de x
        tab = []
        for i in range(self.trainingSet.size()):
            #calcul de la distance entre x et tainningSet[i]
            tab.append(np.linalg.norm(x - self.trainingSet.getX(i)))
            
        
        tab = np.argsort(tab)
        s = 0
        for i in range(self.k):
            s+= self.trainingSet.getY(tab[i])
        return(1 if s>=0 else -1)

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        self.trainingSet = labeledSet

# ---------------------------
class ClassifierPerceptronRandom(Classifier):
    def __init__(self, input_dimension):
        """ Argument:
                - input_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        v = np.random.rand(input_dimension)     # vecteur aléatoire à input_dimension dimensions
        self.w = (2* v - 1) / np.linalg.norm(v) # on normalise par la norme de v

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        return z
        
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        print("No training needed")



class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        ##TODO
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        #w i j  == poids entre le noeud d'entréej et neurone j 
        
        self.w = (2* np.random.rand(self.input_dimension))-1
        
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        ##TODO
        return 1 if np.dot(self.w, x)>0 else -1

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        ##TODO
        self.out = 0
        self.trainingSet = labeledSet
        r = list(range(self.trainingSet.size()))
        np.random.shuffle(r)
        for i in r:
            out = self.predict(self.trainingSet.getX(i))
            if out * self.trainingSet.getY(i) <0:
                self.w = self.w + self.learning_rate *self.trainingSet.getY(i) *self.trainingSet.getX(i)
                

                
class ClassifierPerceptronRandom(Classifier):
    def __init__(self, input_dimension):
        """ Argument:
                - input_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        v = np.random.rand(input_dimension)     # vecteur aléatoire à input_dimension dimensions
        self.w = (2* v - 1) / np.linalg.norm(v) # on normalise par la norme de v

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        return z
        
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        print("No training needed")

        
        
        
        
        
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        ##TODO
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        #w i j  == poids entre le noeud d'entréej et neurone j 
        
        self.w = (2* np.random.rand(self.input_dimension))-1
        
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        ##TODO
        return 1 if np.dot(self.w, x)>0 else -1

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        ##TODO
        self.out = 0
        self.trainingSet = labeledSet
        r = list(range(self.trainingSet.size()))
        np.random.shuffle(r)
        for i in r:
            out = self.predict(self.trainingSet.getX(i))
            if out * self.trainingSet.getY(i) <0:
                self.w = self.w + self.learning_rate *self.trainingSet.getY(i) *self.trainingSet.getX(i)
                


class KernelBias:
    def transform(self,x):
        y=np.asarray([x[0],x[1],1])
        return y

class ClassifierPerceptronKernel(Classifier):
    def __init__(self,dimension_kernel,learning_rate,kernel):
        
        self.input_dimension = dimension_kernel
        self.learning_rate = learning_rate
        self.w = (2* np.random.rand(dimension_kernel))-1
        self.k = kernel
    def predict(self,x):
        return 1 if np.dot(self.w, self.k.transform(x))>0 else -1

    
    def train(self,labeledSet):
        self.out = 0
        self.trainingSet = labeledSet
        for i in range(self.trainingSet.size()):
            out = self.predict(self.trainingSet.getX(i))
            if out * self.trainingSet.getY(i) <0:
                self.w = self.w + self.learning_rate *self.trainingSet.getY(i) *self.k.transform(self.trainingSet.getX(i))

                
class KernelPoly:
    def transform(self,x):
       ##TODO
        y=np.asarray([1, x[0],x[1],x[0]*x[0], x[1]*x[1], x[0]*x[1]])
        return y

def classe_majoritaire(the_set):
    a = [0,0]
    for i in range(the_set.size()):
        if the_set.getY(i) == 1:
            a[0]+=1
        else:
            a[1]+=1
    p = np.argmax(a)
    return 1 if p== 0 else -1


def entropie(L):
    #calcul des distributions de probas
    a = np.zeros(2)
    for i in range(L.size()):
        if L.getY(i) == 1:
            a[0]+=1
        else:
            a[1]+=1
    prob = a/L.size()
    return shannon(prob)

def discretise(LSet, col):
    """ LabelledSet * int -> tuple[float, float]
        Hypothèse: LSet.size() >= 2
        col est le numéro de colonne sur X à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation:
    min_entropie = 1.1  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0     
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)
    
    # calcul des distributions des classes pour E1 et E2:
    inf_plus  = 0               # nombre de +1 dans E1
    inf_moins = 0               # nombre de -1 dans E1
    sup_plus  = 0               # nombre de +1 dans E2
    sup_moins = 0               # nombre de -1 dans E2       
    # remarque: au départ on considère que E1 est vide et donc E2 correspond à E. 
    # Ainsi inf_plus et inf_moins valent 0. Il reste à calculer sup_plus et sup_moins 
    # dans E.
    for j in range(0,LSet.size()):
        if (LSet.getY(j) == -1):
            sup_moins += 1
        else:
            sup_plus += 1
    nb_total = (sup_plus + sup_moins) # nombre d'exemples total dans E
    
    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   # vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0;
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if LSet.getY(ind[i][col])[0] == -1:
            inf_moins += 1
            sup_moins -= 1
        else:
            inf_plus += 1
            sup_plus -= 1
        # calcul de la distribution des classes de chaque côté du seuil:
        nb_inf = (inf_moins + inf_plus)*1.0     # rem: on en fait un float pour éviter
        nb_sup = (sup_moins + sup_plus)*1.0     # que ce soit une division entière.
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon([inf_moins / nb_inf, inf_plus  / nb_inf])
        val_entropie_sup = shannon([sup_moins / nb_sup, sup_plus  / nb_sup])
        val_entropie = (nb_inf / nb_total) * val_entropie_inf \
                       + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)


def divise(LSet, att, seuil):
    l1, l2 = ls.LabeledSet(2), ls.LabeledSet(2)
    for i in range(LSet.size()):
        x= LSet.getX(i)
        if  x[att]<= seuil:
            l1.addExample(x, LSet.getY(i))
        else:
            l2.addExample(x, LSet.getY(i))
    return l1, l2


import graphviz as gv
# Eventuellement, il peut être nécessaire d'installer graphviz sur votre compte:
# pip install --user --install-option="--prefix=" -U graphviz


class ArbreBinaire:
    def __init__(self):
        self.attribut = None   # numéro de l'attribut
        self.seuil = None
        self.inferieur = None # ArbreBinaire Gauche (valeurs <= au seuil)
        self.superieur = None # ArbreBinaire Gauche (valeurs > au seuil)
        self.classe = None # Classe si c'est une feuille: -1 ou +1
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille """
        return self.seuil == None
    
    def ajoute_fils(self,ABinf,ABsup,att,seuil):
        """ ABinf, ABsup: 2 arbres binaires
            att: numéro d'attribut
            seuil: valeur de seuil
        """
        self.attribut = att
        self.seuil = seuil
        self.inferieur = ABinf
        self.superieur = ABsup
    
    def ajoute_feuille(self,classe):
        """ classe: -1 ou + 1
        """
        self.classe = classe
        
    def classifie(self,exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple: +1 ou -1
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.inferieur.classifie(exemple)
        return self.superieur.classifie(exemple)
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir
            l'afficher
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            self.inferieur.to_graph(g,prefixe+"g")
            self.superieur.to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))
        
        return g
    
    

    
def construit_AD(LSet,epsilon):
    """ LSet : LabeledSet
        epsilon : seuil d'entropie pour le critère d'arrêt 
    """
    if(entropie(LSet) <= epsilon):
        un_arbre= ArbreBinaire()
        un_arbre.ajoute_feuille(classe_majoritaire(LSet))
            
    else:
        un_arbre= ArbreBinaire()
        entropies = []
        s  = []
        #pour chaque att cons
        for i in range(len(LSet.getX(0))):
            #calculer les entropies
            seuil, e = discretise(LSet,i)
            entropies.append(e)
            s.append(seuil)
        #meilleur choix qui minimise l'entropie
        best_choice_attribut = np.argmin(entropies)
        best_choice_seuil = s[best_choice_attribut]
        inferieur, superieur = divise(LSet,best_choice_attribut,best_choice_seuil)
            
        #on cree deux fils 
        inferieur, superieur = divise(LSet,best_choice_attribut,best_choice_seuil)
        
        if(inferieur.size() > 0 and superieur.size() > 0 and gain(LSet, inferieur, superieur) >= epsilon):
            un_arbre.ajoute_fils(construit_AD( inferieur, epsilon),construit_AD( superieur, epsilon),best_choice_attribut , best_choice_seuil)
            
        else:
            un_arbre.ajoute_feuille(classe_majoritaire(LSet))   
    return un_arbre

class ArbreDecision(Classifier):
    # Constructeur
    def __init__(self,epsilon):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.racine = None
    
    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        classe = self.racine.classifie(x)
        if (classe == 1):
            return(1)
        else:
            return(-1)
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision 
        self.set=set
        self.racine = construit_AD(set,self.epsilon)

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)
    
def shannon(P):
    if(len(P) >1):
        somme = 0
        for i in range(0, len(P)):
            if P[i] == 0:
                somme += 0  

            else:
                somme += P[i]* math.log(P[i], len(P))

        return -1*somme; 
    else:
        return 0.0
def gain(L, I, S):
    ret = entropie(L) - (entropie(I) * (float(I.size())/L.size() )+entropie(S) *  (float(S.size()) /L.size()))
    return ret
def construit_AD(LSet,epsilon):
    """ LSet : LabeledSet
        epsilon : seuil d'entropie pour le critère d'arrêt 
    """
    if(entropie(LSet) <= epsilon):
        un_arbre= ArbreBinaire()
        un_arbre.ajoute_feuille(classe_majoritaire(LSet))
            
    else:
        un_arbre= ArbreBinaire()
        entropies = []
        s  = []
        #pour chaque att cons
        for i in range(len(LSet.getX(0))):
            #calculer les entropies
            seuil, e = discretise(LSet,i)
            entropies.append(e)
            s.append(seuil)
        #meilleur choix qui minimise l'entropie
        best_choice_attribut = np.argmin(entropies)
        best_choice_seuil = s[best_choice_attribut]
        inferieur, superieur = divise(LSet,best_choice_attribut,best_choice_seuil)
            
        #on cree deux fils 
        inferieur, superieur = divise(LSet,best_choice_attribut,best_choice_seuil)
        
        if(inferieur.size() > 0 and superieur.size() > 0 and gain(LSet, inferieur, superieur) >= epsilon):
            un_arbre.ajoute_fils(construit_AD( inferieur, epsilon),construit_AD( superieur, epsilon),best_choice_attribut , best_choice_seuil)
            
        else:
            un_arbre.ajoute_feuille(classe_majoritaire(LSet))   
    return un_arbre

def tirage(VX, m, r):
    if not r:
        return random.sample(VX,m)
    else:
        vecteur = []
        for i in range(m):
            vecteur.append(random.choice(VX))
        return vecteur
def echantillonLs(Set, m, r):
    retSet, autre = ls.LabeledSet(Set.getInputDimension()), ls.LabeledSet(Set.getInputDimension())
    inputs = list(range(Set.size()))
    index = tirage(inputs, m, r)
    for i in index:
        retSet.addExample(Set.getX(i), Set.getY(i))
    for i in range(Set.size()):
        if not i  in index: 
            autre.addExample(Set.getX(i), Set.getY(i))
    
    return retSet, autre
    

class ClassifierBaggingTree(Classifier):
    def __init__(self, B, m, epsilon, r):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.ensemble = []
        self.B = B
        self.m , self.r= m, r
    
    def predict(self, x):
        somme = 0
        for i in range(len(self.ensemble)):
            classe = self.ensemble[i].classifie(x)
            if (classe == 1):
                somme += 1
            else:
                somme += -1
        return 1 if somme >= 0 else -1
    
    def train(self, Set):
        for i in range(self.B):
            sample = echantillonLs(Set, int(self.m * Set.size()), self.r)[0]
            self.ensemble.append(construit_AD(sample,self.epsilon))
    
class ClassifierBaggingTreeOOB(Classifier):
    def __init__(self, B, m, epsilon, r):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.ensemble = []
        self.B = B
        self.m , self.r= m, r
        self.tests, self.others = [], []
    def predict(self, x):
        somme = 0
        for i in range(len(self.ensemble)):
            classe = self.ensemble[i].classifie(x)
            if (classe == 1):
                somme += 1
            else:
                somme += -1
        return 1 if somme >= 0 else -1
    
    def train(self, Set):
        somme = 0
        for i in range(self.B):
            sample, autre = echantillonLs(Set, int(self.m * Set.size()), self.r)
            arbre =  ArbreDecision(self.epsilon)
            arbre.train(sample)
            self.ensemble.append(arbre)
            self.others.append(autre)
            self.tests.append(sample)
    def accuracy(self, Set):
        somme = 0
        for j in range(self.B):
            somme += self.ensemble[j].accuracy(self.tests[j])
        return float(somme)/self.B
    
def construit_AD_aleatoire(LSet,epsilon, nbatt):
    """ LSet : LabeledSet
        epsilon : seuil d'entropie pour le critère d'arrêt 
    """
    if(entropie(LSet) <= epsilon):
        un_arbre= ArbreBinaire()
        un_arbre.ajoute_feuille(classe_majoritaire(LSet))
            
    else:
        un_arbre= ArbreBinaire()
        entropies = []
        s  = []
        #pour chaque att cons
        listeAtt = range(len(LSet.getX(0)))
        l = tirage(listeAtt, nbatt, False)
        for i in l:
            
            #calculer les entropies
            seuil, e = discretise(LSet,i)
            entropies.append(e)
            s.append(seuil)
        #meilleur choix qui minimise l'entropie
        best_choice_attribut = np.argmin(entropies)
        best_choice_seuil = s[best_choice_attribut]
        inferieur, superieur = divise(LSet,best_choice_attribut,best_choice_seuil)
            
        #on cree deux fils 
        inferieur, superieur = divise(LSet,best_choice_attribut,best_choice_seuil)
        
        if(inferieur.size() > 0 and superieur.size() > 0 and gain(LSet, inferieur, superieur) >= epsilon):
            un_arbre.ajoute_fils(construit_AD( inferieur, epsilon),construit_AD( superieur, epsilon),best_choice_attribut , best_choice_seuil)
            
        else:
            un_arbre.ajoute_feuille(classe_majoritaire(LSet))   
    return un_arbre

class ArbreDecisionAleatoire(ArbreDecision):
    # Constructeur
    def __init__(self,epsilon, nbatt):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon = epsilon
        self.att = nbatt
    
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,Set):
        # construction de l'arbre de décision 
        self.set=set
        self.racine = construit_AD_aleatoire(set,self.epsilon, self.att)

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)
class ClassifierRandomForest(Classifier):
    def __init__(self, B, m, epsilon, r, nbAtt):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.ensemble = []
        self.B = B
        self.m , self.r= m, r
        self.nbAtt = nbAtt
    
    def predict(self, x):
        somme = 0
        for i in range(len(self.ensemble)):
            classe = self.ensemble[i].classifie(x)
            if (classe == 1):
                somme += 1
            else:
                somme += -1
        return 1 if somme >= 0 else -1
    
    def train(self, Set):
        for i in range(self.B):
            sample = echantillonLs(Set, int(self.m * Set.size()), self.r)[0]
            a = construit_AD_aleatoire(sample, self.epsilon, self.nbAtt)
            self.ensemble.append(a)

class ClassifierPerceptron_regression(Classifier):
        """ Perceptron de Rosenblatt
        """
        def __init__(self,input_dimension,learning_rate):
            """ Argument:
                    - intput_dimension (int) : dimension d'entrée des exemples
                    - learning_rate :
                Hypothèse : input_dimension > 0
            """
            ##TODO
            self.input_dimension = input_dimension
            self.learning_rate = learning_rate
            #w i j  == poids entre le noeud d'entréej et neurone j 
            self.w = (2* np.random.rand(self.input_dimension))-1

        def predict(self,x):
            """ rend w * x
            """
            ##TODO
            return np.dot(self.w, x)


        def train(self,labeledSet):
            """ Permet d'entrainer le modele sur l'ensemble donné
            """
            ##TODO
            self.out = 0
            self.trainingSet = labeledSet
            r = list(range(self.trainingSet.size()))
            np.random.shuffle(r)
            for i in r:
                out = self.predict(self.trainingSet.getX(i))
                if out !=  self.trainingSet.getY(i):
                    self.w = self.w + self.learning_rate *self.trainingSet.getY(i) *self.trainingSet.getX(i)
        def loss(self, Set):
            s = 0
            for i in range(Set.size()):
                s+= abs(Set.getY(i)- np.dot(self.w, Set.getX(i))) 
                     
            return s 

def test_perceptron_regression(train, test, epsilon, taille, nbTrain):                 
    a=[]
    #accuracy =[]
    loss= []
    p1 = ClassifierPerceptron_regression(taille, epsilon)


    for i in range(nbTrain):
        p1.train(train)
        a.append(i)
        #accuracy.append(p1.accuracy(train)*100)

        loss.append(p1.loss(test))
    return a, loss




