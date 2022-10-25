import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn as svm

#Creamos una función para poder leer todas las base de datos disponibles
def load_dataset(path, path1, path2, path3):

    list = []
    list.append(pd.read_csv(path, header = None))
    list.append(pd.read_csv(path1, header = None))
    list.append(pd.read_csv(path2, header = None))
    list.append(pd.read_csv(path3, header = None))

    dataset = pd.concat(list)
    return dataset

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

#Dispersion



print("final")