# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:58:46 2022

@author: baris
"""
# male - 0  female - 1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from yellowbrick.features import rank2d
from yellowbrick.target.feature_correlation import feature_correlation
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import partial_dependence,PartialDependenceDisplay,permutation_importance

df = pd.read_csv('lung-cancer.csv')


features = ["cinsiyet","yas","sigara","parmakSariligi","anksiyete","akranBaskisi","kronikHastalik","bitkinlik",
      "alerji","hirilti","alkolTuketimi","oksurme","nefesDarligi","yutmaZorlugu","gogusAgrisi"]
X=df[["cinsiyet","yas","sigara","parmakSariligi","anksiyete","akranBaskisi","kronikHastalik","bitkinlik",
      "alerji","hirilti","alkolTuketimi","oksurme","nefesDarligi","yutmaZorlugu","gogusAgrisi"]].values
Y=df[["akcigerKanseri"]].values

Y = np.ravel(Y)


rank2d(df, algorithm='pearson');

corr = df.corr() 
corr.style.background_gradient(cmap='coolwarm')
feature_correlation(X, Y, method='mutual_info-classification',labels=features)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=True)

modelKNN = KNeighborsClassifier(n_neighbors=3,metric="euclidean")
modelKNN.fit(x_train, y_train)
modelNB = GaussianNB()
modelNB.fit(x_train,y_train)
modelDT = tree.DecisionTreeClassifier()
modelDT.fit(x_train,y_train) 

testsayisi = len(x_test)
print("Test veri sayısı: %d" %testsayisi)

#tahmin yap
tahminKNN = modelKNN.predict(x_test)
tahminNB = modelNB.predict(x_test)
tahminDT = modelDT.predict(x_test)

#kaç doğru kaç yanlış tahmin var
dogruKNN = (tahminKNN == y_test).sum()
yanlisKNN = (tahminKNN != y_test).sum()
dogruNB = (tahminNB == y_test).sum()
yanlisNB = (tahminNB != y_test).sum()
dogruDT = (tahminDT == y_test).sum()
yanlisDT = (tahminDT != y_test).sum()

print("KNN sınıflandırması için doğru tahmin sayısı:%d" %dogruKNN)
print("KNN sınıflandırması için yanlış tahmin sayısı:%d" %yanlisKNN)
print("Naive Bayes sınıflandırması için doğru tahmin sayısı:%d" %dogruNB)
print("Naive Bayes sınıflandırması için yanlış tahmin sayısı:%d" %yanlisNB)
print("Karar ağacı sınıflandırması için doğru tahmin sayısı:%d" %dogruDT)
print("Karar ağacı sınıflandırması için yanlış tahmin sayısı:%d" %yanlisDT)
print("KNN algoritması")
accKNN = (100*dogruKNN)/testsayisi
print("Doğruluk : %f" %accKNN)
print(classification_report(y_test,tahminKNN))
print("*"*50)
print("Naive Bayes algoritması")
accNB = (100*dogruNB)/testsayisi
print("Doğruluk : %f" %accNB)
print(classification_report(y_test,tahminNB))
print("*"*50)
print("Karar Ağacı algoritması")
accDT = (100*dogruDT)/testsayisi
print("Doğruluk : %f" %accDT)
print(classification_report(y_test,tahminDT))
# a=tree.plot_tree(modelDT,impurity=(False),filled=(True),fontsize=(8),feature_names=features,class_names=(True))
# plt.show(a)
print(tree.export_text(modelDT,feature_names=features))
print("*"*50)

cmKNN = confusion_matrix(tahminKNN, y_test)
cmNB = confusion_matrix(tahminNB, y_test)
cmDT = confusion_matrix(tahminDT, y_test)

y_scoreKNN = modelKNN.predict_proba(x_test)
fprKNN, tprKNN, thresholdKNN = roc_curve(y_test, y_scoreKNN[:, 1])
roc_aucKNN = auc(fprKNN, tprKNN)

y_scoreNB = modelNB.predict_proba(x_test)
fprNB, tprNB, thresholdNB = roc_curve(y_test, y_scoreNB[:, 1])
roc_aucNB = auc(fprNB, tprNB)

y_scoreDT = modelDT.predict_proba(x_test)
fprDT, tprDT, thresholdDT = roc_curve(y_test, y_scoreDT[:, 1])
roc_aucDT = auc(fprDT, tprDT)

plt.title('Receiver Operating Characteristic')
plt.plot(fprKNN, tprKNN,'b', label = 'KNN için AUC = %0.2f' % roc_aucKNN)
plt.plot(fprNB, tprNB, 'g', label = 'Naive Bayes için AUC = %0.2f' % roc_aucNB)
plt.plot(fprDT, tprDT, 'k', label = 'Karar ağacı için AUC = %0.2f' % roc_aucDT)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()

#("cinsiyet","yas","sigara","parmakSariligi","anksiyete","akranBaskisi","kronikHastalik","bitkinlik",
#      "alerji","hirilti","alkolTuketimi","oksurme","nefesDarligi","yutmaZorlugu","gogusAgrisi")
# male - 0  female - 1

input_data= (0,22,0,0,1,0,0,0,0,0,1,0,0,0,0)
input_data_numpy_array = np.asarray(input_data)

input_data_reshape = input_data_numpy_array.reshape(1, -1)

predictionKNN = modelKNN.predict(input_data_reshape)
print("KNN modelinin tahmini : %d" %predictionKNN)

if (predictionKNN[0]== 0):
    print("Akciğer kanseri değilsin sağlığına dikkat etmeye devam!")
else:
        print("Akciğer kanseri olabilirsin en kısa sürede hastaneye git!")
        
predictionNB = modelNB.predict(input_data_reshape)
print("Naive Bayes modelinin tahmini : %d" %predictionNB)

if (predictionNB[0]== 0):
    print("Akciğer kanseri değilsin sağlığına dikkat etmeye devam!")
else:
        print("Akciğer kanseri olabilirsin en kısa sürede hastaneye git!")        

predictionDT = modelDT.predict(input_data_reshape)
print("Karar ağacı modelinin tahmini : %d" %predictionDT)

if (predictionDT[0]== 0):
    print("Akciğer kanseri değilsin sağlığına dikkat etmeye devam!")
else:
        print("Akciğer kanseri olabilirsin en kısa sürede hastaneye git!")

print("*"*50)