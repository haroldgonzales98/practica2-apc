import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold

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

#COMPROBACIONES VARIAS
csv_0=pd.read_csv('../BBDD/0.csv', header=None)
csv_1=pd.read_csv('../BBDD/1.csv', header=None)
csv_2=pd.read_csv('../BBDD/2.csv', header=None)
csv_3=pd.read_csv('../BBDD/3.csv', header=None)

print("######################################################################################")
print("Primeros 5 elementos de todas las BBDD")
print(csv_0.head(5))
print(csv_1.head(5))
print(csv_2.head(5))
print(csv_3.head(5))
print("######################################################################################")
print("Tamaño de 0.csv: ", csv_0.shape)
print("Tamaño de 1.csv: ", csv_1.shape)
print("Tamaño de 2.csv: ", csv_2.shape)
print("Tamaño de 3.csv: ", csv_3.shape)
print("######################################################################################")
print('Numero de valores nulos en 0.csv: ',csv_0.isna().sum().sum())
print('Numero de valores nulos en 1.csv: ',csv_1.isna().sum().sum())
print('Numero de valores nulos en 2.csv: ',csv_2.isna().sum().sum())
print('Numero de valores nulos en 3.csv: ',csv_3.isna().sum().sum())
print("######################################################################################")


print("Cargamos la base de datos...")
dataset = load_dataset("../BBDD/0.csv","../BBDD/1.csv","../BBDD/2.csv","../BBDD/3.csv")
titles = dataset.columns.values
print("Tamaño de nuestra BBDD: ", dataset.shape)

dataset = load_dataset("../BBDD/0.csv","../BBDD/1.csv","../BBDD/2.csv","../BBDD/3.csv")
dataValues = dataset.values
dataTitles = dataset.columns.values

#  ESTE APARTADO DE CODIGO ES PARA VISUALIZAR LOS SCATTER (2) DE NUESTRA BBDD
n_classes = 4
x = dataValues[:, :64]
X = standarize(x)
y = dataValues[:, 64]

#PCA (intento)
# pca_pipe = make_pipeline(StandardScaler(), PCA())
# pca_pipe.fit(dataset)
# modelo_pca = pca_pipe.named_steps['pca']
# pd.DataFrame(
#     data    = modelo_pca.components_,
#     columns = dataset.columns
# )
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
# componentes = modelo_pca.components_
# plt.imshow(componentes.T, cmap='viridis', aspect='auto')
# plt.yticks(range(len(dataset.columns)), dataset.columns)
# plt.xticks(range(len(dataset.columns)), np.arange(modelo_pca.n_components_) + 1)
# plt.grid(False)
# plt.colorbar()
# plt.savefig("../Graficas/Resultados de Apartado B/PCA")
# plt.show()

#Crossvalidation

param_svm = {'C': [0.1,1], 'gamma': [1,0.1],'kernel': ['rbf', 'poly', 'sigmoid']}
param_knn = { 'n_neighbors' : [5,7,9,11],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
param_decisiontree = {
    'max_depth': [2, 3, 5],
    'min_samples_leaf': [5, 10, 20],
    'criterion': ["gini", "entropy"]
}

param_logireg = {'C': [0.01, 0.1, 1] }



param_grid = [param_svm, param_knn, param_decisiontree, param_logireg]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
#
# clf = svm.SVC().fit(X_train, y_train)
# print("Score SVC rbf: " + str(clf.score(X_test, y_test)))
# scores = cross_val_score(clf, X, y, cv=5)
# print("Scores splitting: " + str(scores))
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# scores2 = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
# print("Scores with scoring parameter: " + str(scores2))

models = [svm.SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), LogisticRegression()]
nom_models = ["Support Vector Machines", "KNN", "Decision Tree", "Logistic Regression"]

# for i,model in enumerate(models):
#     '''Busqueda exhaustiva de los mejores parametros'''
#     print("BUSQUEDA EXHAUSTIVA DE PARAMETROS")
#     grid = GridSearchCV(model, param_grid[i], verbose=3, n_jobs=-1)
#     grid.fit(X,y)
#     print("Els millors parametres: ",grid.best_params_)
#     print("El millor score: ", grid.best_score_)
#     print("")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# clf = svm.SVC()
# clf.fit(X_train, y_train)
# score = clf.score(X_train, y_train)
# print("Metrica del modelo", score)


conjunto = [2,3,4,5,6,7,8,9,10]
list_scores = []

# for value in conjunto:
#     print("K CON VALOR DE " + str(value))
#     kf = KFold(n_splits=value)
#     scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring="accuracy")
#     print("Metricas cross_validation", scores)
#     print("Media de cross_validation", scores.mean())
#     preds = clf.predict(X_test)
#     score_pred = metrics.accuracy_score(y_test, preds)
#     print("Metrica en Test", score_pred)
#     list_scores.append(scores)

# scores2 = LeaveOneOut()
# llista = []
# model = svm.SVC()
# aux = 0
# for train_index, test_index in scores2.split(X,y):
#     print("Index: " + str(aux))
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     model.fit(X_train, y_train)
#     llista.append(model.score(X_test, y_test))
#     aux = aux + 1
#     if aux == 100:
#         break
# print("Score mitja del Leave One Out: ", np.array(llista).mean())
# print("")

print("METRICAS DE EVALUACIÓN")
print("")
models = [svm.SVC(probability=True), KNeighborsClassifier(), DecisionTreeClassifier(), LogisticRegression()]
nom_models = ["Support Vector Machines", "KNN", "Decision Tree", "Logistic Regression"]
nom_classes = ["Class 0", "Class 1","Class 2","Class 3"]
for o, model in enumerate(models):
    print("---- ",nom_models[o], " ----")
    print("")
    x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=0.8)
    model.fit(x_t,y_t)
    prediccions = model.predict(x_v)
    print("Accuracy score: ",accuracy_score(y_v, prediccions))
    print("F1 score: ",f1_score(y_v, prediccions, average='weighted'))
    print("Precision score: ",precision_score(y_v, prediccions, average='weighted'))
    print("Recall score: ",recall_score(y_v, prediccions, average='weighted'))
    print(classification_report(y_v, prediccions, target_names=nom_classes))