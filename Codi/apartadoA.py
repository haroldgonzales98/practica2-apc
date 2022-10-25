import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc

#Creamos una función para poder leer todas las base de datos disponibles
def load_dataset(path, path1, path2, path3):

    list = []
    list.append(pd.read_csv(path, header = None))
    list.append(pd.read_csv(path1, header = None))
    list.append(pd.read_csv(path2, header = None))
    list.append(pd.read_csv(path3, header = None))

    dataset = pd.concat(list)
    return dataset

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

print("Cargamos la base de datos...")
dataset = load_dataset("../BBDD/0.csv","../BBDD/1.csv","../BBDD/2.csv","../BBDD/3.csv")
dataValues = dataset.values
dataTitles = dataset.columns.values

##########################      PRUEBAS DE SKLEARN
# wine = load_wine()
# X = wine.data
#
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# y = wine.target
# nom_atributs = wine.feature_names
# nom_classes = wine.target_names.reshape(-1)
##################################################

n_classes = 1
x = dataValues[:, :64]
X = standarize(x)
y = dataValues[:, 64]
#plt.figure()
# fig, sub = plt.subplots(1, 2, figsize=(16,6))
# sub[0].scatter(X[:, 0], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
# sub[1].scatter(X[:, 1], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
#plt.savefig("../Graficas/Resultados de Apartado A/Scatter")
#plt.show()
particions = [0.5, 0.7, 0.8]

for part in particions:
    x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=part)

    # Creem el regresor logístic
    logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)

    # l'entrenem
    logireg.fit(x_t, y_t)

    print("Correct classification Logistic ", part, "% of the data: ", logireg.score(x_v, y_v))

    # Creem el regresor logístic
    svc = svm.SVC(C=10.0, kernel='rbf', gamma=0.9, probability=True)

    # l'entrenem
    svc.fit(x_t, y_t)
    probs = svc.predict_proba(x_v)
    print("Correct classification SVM      ", part, "% of the data: ", svc.score(x_v, y_v))

    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i])
        average_precision[i] = average_precision_score(y_v == i, probs[:, i])
        plt.figure()
        plt.plot(recall[i], precision[i],
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="upper right")
        plt.show()


