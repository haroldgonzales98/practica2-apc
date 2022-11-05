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

dataset = load_dataset("../BBDD/0.csv","../BBDD/1.csv","../BBDD/2.csv","../BBDD/3.csv")
dataValues = dataset.values
dataTitles = dataset.columns.values

#  ESTE APARTADO DE CODIGO ES PARA VISUALIZAR LOS SCATTER (2) DE NUESTRA BBDD
n_classes = 4
x = dataValues[:, :63]
X = standarize(x)
y = dataValues[:, 64]
# plt.figure()
# fig, sub = plt.subplots(1, 2, figsize=(16,6))
# sub[0].scatter(X[:, 0], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
# sub[1].scatter(X[:, 1], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
# plt.savefig("../Graficas/Resultados de Apartado A/Scatter")
# plt.show()
#####################################################################################

# CALCULAMOS VALORES DE PRECISIÓN DE LOGISTIC REGRESSION Y SVM (NO GRÁFICAS)

# particions = [0.5, 0.7, 0.8]
# for part in particions:
#     x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=part)
#
#     # Creem el regresor logístic
#     logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
#
#     # l'entrenem
#     logireg.fit(x_t, y_t)
#
#     print("Correct classification Logistic ", part, "% of the data: ", logireg.score(x_v, y_v))
#
#     # Creem el regresor logístic
#     svc = svm.SVC(C=10.0, kernel='rbf', gamma=0.9, probability=True)
#
#     # l'entrenem
#     svc.fit(x_t, y_t)
#     probs = svc.predict_proba(x_v)
#     print("Correct classification SVM      ", part, "% of the data: ", svc.score(x_v, y_v))
##########################################################################################################

# USAMOS LOS MODELOS Y LOS ENTRENAMOS PARA PODER VISUALIZAR LAS GRÁFICAS
# lista_modelos = [LogisticRegression(), svm.SVC(probability=True)]
# text_modelos = ["Logistic Regression", "SVM"]
# j = 0
# aux = 0
# for a, model in enumerate(lista_modelos):
#     x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=0.8)
#     model.fit(x_t, y_t)
#     probs = model.predict_proba(x_v)
#
#     precision = {}
#     recall = {}
#     average_precision = {}
#     plt.figure()
#     for i in range(n_classes):
#         precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i])
#         average_precision[i] = average_precision_score(y_v == i, probs[:, i])
#         #plt.figure()
#         plt.plot(recall[i], precision[i],
#                  label='Precision-recall curve of class {0} (area = {1:0.2f})'
#                        ''.format(i, average_precision[i]))
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.legend(loc="upper right")
#     plt.savefig("../Graficas/Resultados de Apartado A/Precission Recall Curve " + text_modelos[j] + " " + str(aux))
#
#     plt.show()
#
#     fpr = {}
#     tpr = {}
#     roc_auc = {}
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_v == i, probs[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # Compute micro-average ROC curve and ROC area
#     # Plot ROC curve
#     plt.figure()
#     for i in range(n_classes):
#         plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
#     plt.savefig("../Graficas/Resultados de Apartado A/ROC " + text_modelos[j] + " " + str(aux))
#     j = j + 1
#     aux = aux + 1
#     plt.legend()
#     plt.show()
##########################################################################################################

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def show_C_effect(X, y, C=1.0, gamma=0.7, degree=3):

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    #C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C, max_iter=1000000),
              svm.SVC(kernel='rbf', gamma=gamma, C=C),
              svm.SVC(kernel='poly', degree=degree, gamma='auto', C=C))
    models = (clf.fit(X, y) for clf in models)

    plt.close('all')
    fig, sub = plt.subplots(2, 2, figsize=(14,9))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)


    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()

show_C_effect(X[:, :2], y, C=0.1)