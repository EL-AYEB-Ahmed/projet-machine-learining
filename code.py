### TD initiation à  l'IA ###


##IMPORTANT!!!!!

#Bien mettre à jour sklearn et matplotlib depuis Anaconda 3/Anaconda Prompt
#pip install -U scikit-learn
#pip install -U matplotlib

## Importation des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Q1 Lecture et importation des données du dataset
dataset = ...

#### Extraction des données de salaire et d'age du dataset d'entraà®nement
##Xt = dataset[[...]].to_numpy(dtype=float)
##yt = dataset[[...]].to_numpy(dtype=float).flatten()
##
#### Q3 Représentation graphique des points du dataset
##fig, ax = plt.subplots()
##plt.title("Points du dataset")
##scatter = ax.scatter(...,..., c=yt,cmap = ListedColormap(['r' , 'b' ]))
##legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
##ax.add_artist(legend1)
##...
##...
##plt.show()
##
#### Q4 Conditionnement des données d'entrée en valeurs centrées réduites
##mu=...
##sigma=...
##
##
###Normalisation des données de salaire et d'age
##Xtn=Xt.copy()
##Xtn[:,0]=...
##Xtn[:,1]=...
###Xt a maintenant des valeurs centrées réduites
##
##print("Espérance salaires centrés réduits=",Xtn[:,0].mean())
##print("Ecart type salaires centrés réduits=",Xtn[:,0].std())
##
#### Q5
##def sigmoide(z):
##    return ... # valeurs comprises entre 0 et 1
##
##def h(X,W):
##    return ... # valeurs de y=h(X,W) comprises entre 0 et 1
##
##def J(X,y,W):
##    N=X.shape[0]
##    s=...
##    return -1/N*s
##
##def gradient(X,y,W):
##    N=X.shape[0]
##    return ...
##
##def  Regression_Logistique(X,y,nb_iter,alpha):
##    N,P=X.shape
##    X=...     # rajout d'une colonne de 1 à la première colonne de Xt pour tenir compte du biais
##    W=np.ones((P+1,))*0.5 # initialisation arbitraire des poids à  0.5
##    L_W=[W]
##    hist_cout=[J(X,y,W)]
##    for i in range(...):
##        W=...
##        L_W.append(W)
##        hist_cout.append(J(X,y,W))
##    return L_W,hist_cout
##
##L_W,hist_cout=Regression_Logistique(Xtn,yt,200,1)
##
#### Q6 Tracé de l'historique du cout en fonction du nombre d'itérations
##plt.plot(np.arange(0,len(hist_cout),1), hist_cout)
##plt.ylim(0,0.7)
##plt.xlabel("Nombre d'itérations")
##plt.ylabel("cout J(X,y,W)")
##plt.show()
##
#### Q7 Prédiction du modèle de classification
##
##def prediction(X,W):
##    N=X.shape[0]
##    X=np.hstack((np.ones((N,1)),X))
##    y=X.dot(W)
##    ...
##
##
##    
##    return y
##
#### Q8 Tracé des prédiction du modèle de régression logistique linéaire
##resolution=100  
##
### Calcul des bornes des axes
##x0lim =(min(Xtn[:,0])-0.5,max(Xtn[:,0])+0.5)
##x1lim =(min(Xtn[:,1])-0.5,max(Xtn[:,1])+0.5)
##
### Meshgrid
##x0 = np. linspace ( x0lim [ 0 ], x0lim [ 1 ], resolution )
##x1 = np. linspace ( x1lim [ 0 ], x1lim [ 1 ], resolution )
##X0,X1 = np.meshgrid(x0, x1)
##
### Assemblage des 2 variables
##XX = np.vstack((X0.ravel (), X1.ravel ())).T
##
### Prédictions du modèle de régression logistique linéaire
##Z = prediction (XX, L_W[-1]) # prédiction pour chaque point du maillage
##Z = Z.reshape(( resolution , resolution ))
##yl = prediction(Xtn,L_W[-1]) # prédiction pour les points du dataset normalisés
##
### Tracé des points du dataset et de la prédiction du modèle de régression logistique
##fig, ax = plt.subplots()
##plt.title("Points du dataset et prédiction du modèle de régression logistique\nModèle linéaire")
##scatter = ax.scatter(Xtn[:,0],Xtn[:,1], c=yt,cmap = ListedColormap(['r' , 'b' ]))
##legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
##ax.add_artist(legend1)
##plt.xlabel("Salaires centrés réduits")
##plt.ylabel("Ages centrés réduits")
##ax.pcolormesh(X0, X1, Z, alpha=0.2,cmap = ListedColormap(['r' , 'b' ])) # coloration des deux demi-plans. alpha : transparence
##ax.contour(X0, X1, Z, colors ='k') # tracé de la frontière entre les deux demi-plans
##plt.show()
##
#### Q9 Matrice de confusion et analyse des résultats
##def analyse_resultats(yt,y):
##    [VN,VP,FN,FP]=[0]*4
##    for i in range(len(yt)):
##        if yt[i]==y[i]==0:
##            VN+=1
##        elif yt[i]==y[i]==1:
##            VP+=1
##        elif yt[i]==1 and y[i]==0:
##            FN+=1
##        else:
##            FP+=1
##    print("Matrice de confusion:")
##    C=...
##    print(C)
##    sensibilite=...
##    print("Sensibilité=",sensibilite)
##    precision=...
##    print("Précision=",precision)
##    justesse=...
##    print("Justesse=",justesse)
##    print("Score-F=",...)
##
##print("Résultats avec régression logistique linaire")
##analyse_resultats(yt,yl)
##
##
#### Q10 régression logistique avec sklearn et données non normalisées
##from sklearn import linear_model
##
##Modele_RL = linear_model.LogisticRegression(penalty='none')
##Modele_RL.fit(..., ...) # entraînement du modèle
##
##resolution=100  
##
### Calcul des bornes des axes
##x0lim =(min(Xt[:,0])-0.5,max(Xt[:,0])+0.5)
##x1lim =(min(Xt[:,1])-0.5,max(Xt[:,1])+0.5)
##
### Meshgrid
##x0 = np. linspace ( x0lim [ 0 ], x0lim [ 1 ], resolution )
##x1 = np. linspace ( x1lim [ 0 ], x1lim [ 1 ], resolution )
##X0,X1 = np.meshgrid(x0, x1)
##
### Assemblage des 2 variables
##XX = np.vstack((X0.ravel (), X1.ravel ())).T
##
### Prédictions du modèle de régression logistique
##Z=Modele_RL.predict(XX) # prédiction pour chaque point du maillage
##Z = Z.reshape(( resolution , resolution ))
##yls=Modele_RL.predict(Xt) # prédiction pour les points du dataset non normalisé
##
### Tracé des points du dataset et de la prédiction du modèle de régression logistique
##fig, ax = plt.subplots()
##plt.title("Points du dataset et prédiction du modèle de régression logistique\nModèle linéaire Sklearn")
##scatter = ax.scatter(Xtn[:,0],Xtn[:,1], c=yt,cmap = ListedColormap(['r' , 'b' ]))
##legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
##ax.add_artist(legend1)
##plt.xlabel("Salaires")
##plt.ylabel("Ages")
##ax.pcolormesh(X0, X1, Z, alpha=0.2,cmap = ListedColormap(['r' , 'b' ])) # coloration des deux demi-plans. alpha : transparence
##ax.contour(X0, X1, Z, colors ='k') # tracé de la frontière entre les deux demi-plans
##plt.show()
##
##
#### Q11 analyse des résultats obtenus par régression logistique Sklearn
##print("Résultats avec régression logistique linéaire Sklearn")
##analyse_resultats(yt,yls)
##
##print("Résultats du module metrics de sklearn")
##from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
##print("Matrice de confusion")
##cm = confusion_matrix(yt, yls, labels=Modele_RL.classes_)
##disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Modele_RL.classes_)
##disp.plot()
##
##justesse=accuracy_score(yt, yls, normalize=True, sample_weight=None)
##print("Justesse=",justesse)
##
##    
#### Q12  modèle de régression logistique polynomial d'ordre n
##N=Xtn.shape[0]
##n=3 # degré du modèle polynomial n>=1
##
##Xtnp=Xtn[:,:]
##if n>=2:
##    for i in range(n-1):
##        Xtnp=np.hstack((Xtnp,Xtn**(i+2)))
##
##L_Wp,hist_coutp=Regression_Logistique(Xtnp,yt,500,1)
##
##
##resolution=100  
##
### Calcul des bornes des axes
##x0lim =(min(Xtn[:,0])-0.5,max(Xtn[:,0])+0.5)
##x1lim =(min(Xtn[:,1])-0.5,max(Xtn[:,1])+0.5)
##
### Meshgrid
##x0 = np. linspace ( x0lim [ 0 ], x0lim [ 1 ], resolution )
##x1 = np. linspace ( x1lim [ 0 ], x1lim [ 1 ], resolution )
##X0,X1 = np.meshgrid(x0, x1)
##
### Assemblage des 2n variables
##XX = np.vstack(((X0.ravel ()).reshape((1,resolution**2)), (X1.ravel ()).reshape((1,resolution**2)))).T
##XXp=XX[:,:]
##if n>=2:
##    for i in range(n-1):
##        XXp=np.hstack((XXp,XX**(i+2)))
##
### Prédictions du modèle de régression logistique
##Z = prediction (XXp, L_Wp[-1]) # prédiction pour chaque point du maillage
##Z = Z.reshape(( resolution , resolution ))
##yq=prediction(Xtnp,L_Wp[-1]) # prédiction pour les points du dataset normalisés
##
### Tracé des points du dataset et de la prédiction du modèle de régression logistique
##fig, ax = plt.subplots()
##plt.title(f"Points du dataset et prédiction du modèle de régression logistique\nModèle polynomial de degré {n}")
##scatter = ax.scatter(Xtn[:,0],Xtn[:,1], c=yt,cmap = ListedColormap(['r' , 'b' ]))
##legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
##ax.add_artist(legend1)
##plt.xlabel("Salaire centré réduit")
##plt.ylabel("Age centré réduit")
##ax.pcolormesh(X0, X1, Z, alpha=0.2,cmap = ListedColormap(['r' , 'b' ])) # coloration des deux demi-plans. alpha : transparence
##ax.contour(X0, X1, Z, colors ='k') # tracé de la frontière entre les deux demi-plans
##plt.show()
##
### Analyse des résultats
##print(f"Résultats avec régression logistique polynomial de degré {n}")
##analyse_resultats(yt,yq)
