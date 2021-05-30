# -*- coding: utf-8 -*-
"""
Created on Sat May 29 15:49:00 2021

@author: camun
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from warnings import simplefilter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

url ='weatherAUS.csv'
data = pd.read_csv(url)

data.drop(['poutcome','contact','previous','month','day','duration','pdays'], axis = 1 ,inplace=True)
data['marital'].replace(['married','single','divorced'],[0,1,2], inplace=True)


x=np.array(data.drop(['y'],1))
y= np.array(data.y)

data_train = data[:850]
data_test = data[850:]

x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix,);
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_test, pred_y))
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
simplefilter(action='ignore', category=FutureWarning)
def metricas_entrenamiento(model, x_train, x_test, y_train, y_test):
    kfold = KFold(n_splits=10)
    cvscores = []
    for train, test in kfold.split(x_train, y_train):
        model.fit(x_train[train], y_train[train])
        scores = model.score(x_train[test], y_train[test])
        cvscores.append(scores)
    y_pred = model.predict(x_test)
    accuracy_validation = np.mean(cvscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    return model, accuracy_validation, accuracy_test, y_pred
def matriz_confusion_auc(model, x_test, y_test, y_pred):
    matriz_confusion = confusion_matrix(y_test, y_pred)
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    AUC = roc_auc_score(y_test, probs)
    return matriz_confusion, AUC, fpr, tpr
def show_roc_curve_matrix(fpr, tpr, matriz_confusion):
    sns.heatmap(matriz_confusion)
    plt.show()
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
def show_metrics(str_model, AUC, acc_validation, acc_test, y_test, y_pred):
    print('-' * 50 + '\n')
    print(str.upper(str_model))
    print('\n')
    print(f'Accuracy de validación: {acc_validation} ')
    print(f'Accuracy de test: {acc_test} ')
    print(classification_report(y_test, y_pred))
    print(f'AUC: {AUC} ')

#1.Arboles de decision
print('Arboles de decision')

arbol= DecisionTreeClassifier(max_depth=2, random_state=42) 
arbol.fit(x_train, y_train)
print(f'Accuracy de Entrenamiento: {arbol.score(x_train, y_train)}')
print(f'Accuracy de Test: {arbol.score(x_test, y_test)}')
y_pred = arbol.predict(x_test_out);
mostrar_resultados(y_test_out, y_pred)
probs = arbol.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)

print('-'*50)



#2.Vecinos mas cercanos
print('vecinos mas cercanos')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
print(f'Accuracy de Entrenamiento: {knn.score(x_train, y_train)}')
print(f'Accuracy de Test: {knn.score(x_test, y_test)}')
y_pred = knn.predict(x_test_out);
mostrar_resultados(y_test_out, y_pred)
probs = knn.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)
print('-'*50)



#3.Regresion logistica
print('Regresion logistica')
logreg = LogisticRegression(solver='lbfgs', max_iter=7600)
logreg.fit(x_train, y_train)
print(f'Accuracy de Entrenamiento: {logreg.score(x_train, y_train)}')
print(f'Accuracy de Test: {logreg.score(x_test, y_test)}')
y_pred = logreg.predict(x_test_out);
print(f'Accuracy de Validación: {accuracy_score(y_pred, y_test_out)}')
print('Matriz de confusión')
print(confusion_matrix(y_test_out, y_pred))
mostrar_resultados(y_test_out, y_pred)
probs = logreg.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)
print('-'*50)

#4.Regresion Logistica con validacion cruzada
print('Regresión Logística validación Cruzada')
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)
kfold = KFold(n_splits=10)
cvscores = []
for train, test in kfold.split(x_train, y_train):
    logreg.fit(x_train[train],y_train[train])
    scores = logreg.score(x_train[test], y_train[test])
    cvscores.append(scores)
print(f'Accuracy de Entrenamiento: {logreg.score(x_train, y_train)}')
print(f'Accuracy de Test: {logreg.score(x_test, y_test)}')
print(f'Accuracy de Validación: {accuracy_score(y_pred, y_test_out)}')
print('Matriz de confusión')
print(confusion_matrix(y_test_out, y_pred))
y_pred = logreg.predict(x_test_out);
mostrar_resultados(y_test_out, y_pred)
probs = logreg.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)


#5. Maquinas de soporte vectorial

print('Maquinas de soporte vectorial')
svc = SVC(gamma= 'auto')
svc.fit(x_train, y_train)
print(f'Accuracy de Entrenamiento: {svc.score(x_train, y_train)}')
print(f'Accuracy de Test: {svc.score(x_test, y_test)}')
y_pred = svc.predict(x_test_out);
mostrar_resultados(y_test_out, y_pred)
probs = svc.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)
print('*'* 50)


