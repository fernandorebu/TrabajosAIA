#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import numpy as np

from carga_datos import *


def particion_entr_prueba(X, y, test=0.20):
    clases = np.unique(y)
    indices_entr = []
    indices_prueba = []

    for clase in clases:
        idx_clase = np.where(y == clase)[0]
        n_prueba = round(len(idx_clase) * test)
        idx_mezclados = idx_clase.copy()
        np.random.shuffle(idx_mezclados)
        indices_prueba.extend(idx_mezclados[:n_prueba])
        indices_entr.extend(idx_mezclados[n_prueba:])

    indices_entr = sorted(indices_entr)
    indices_prueba = sorted(indices_prueba)

    return X[indices_entr], X[indices_prueba], y[indices_entr], y[indices_prueba]


Xe_votos, Xp_votos, ye_votos, yp_votos = particion_entr_prueba(X_votos, y_votos, test=1/3)
Xev_cancer, Xp_cancer, yev_cancer, yp_cancer = particion_entr_prueba(X_cancer, y_cancer, test=0.2)
X_train_iris, X_test_iris, y_train_iris, y_test_iris = particion_entr_prueba(X_iris, y_iris, test=0.2)


class Nodo:
    def __init__(self, atributo=None, umbral=None, izq=None, der=None, distr=None, *, clase=None):
        self.atributo = atributo
        self.umbral = umbral
        self.izq = izq
        self.der = der
        self.distr = distr
        self.clase = clase

    def es_hoja(self):
        return self.clase is not None


class ClasificadorNoEntrenado(Exception): pass


def _entropia(y):
    n = len(y)
    if n == 0:
        return 0.0
    _, cuentas = np.unique(y, return_counts=True)
    proporciones = cuentas / n
    proporciones = proporciones[proporciones > 0]
    return -np.sum(proporciones * np.log2(proporciones))


def _ganancia_informacion(y, y_izq, y_der):
    n = len(y)
    n_izq = len(y_izq)
    n_der = len(y_der)
    if n_izq == 0 or n_der == 0:
        return 0.0
    return (_entropia(y)
            - (n_izq / n) * _entropia(y_izq)
            - (n_der / n) * _entropia(y_der))


def _umbrales_candidatos(valores, clases):
    orden = np.argsort(valores)
    vals_ord = valores[orden]
    cls_ord = clases[orden]
    candidatos = []
    for i in range(len(vals_ord) - 1):
        if cls_ord[i] != cls_ord[i + 1] and vals_ord[i] != vals_ord[i + 1]:
            candidatos.append((vals_ord[i] + vals_ord[i + 1]) / 2.0)
    return candidatos


def _mejor_particion(X, y, atributos, prop_umbral):
    mejor_ganancia = -1
    mejor_atributo = None
    mejor_umbral = None
    n = len(y)

    for atr in atributos:
        n_muestra = max(2, int(n * prop_umbral))
        if n_muestra < n:
            idx_muestra = np.random.choice(n, n_muestra, replace=False)
        else:
            idx_muestra = np.arange(n)

        vals_muestra = X[idx_muestra, atr]
        cls_muestra = y[idx_muestra]
        candidatos = _umbrales_candidatos(vals_muestra, cls_muestra)

        for umbral in candidatos:
            mask_izq = X[:, atr] <= umbral
            mask_der = ~mask_izq
            if mask_izq.sum() == 0 or mask_der.sum() == 0:
                continue
            ganancia = _ganancia_informacion(y, y[mask_izq], y[mask_der])
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_atributo = atr
                mejor_umbral = umbral

    return mejor_atributo, mejor_umbral, mejor_ganancia


def _construye_arbol(X, y, atributos, min_ejemplos, max_prof, prop_umbral, prof=0):
    clases, cuentas = np.unique(y, return_counts=True)
    distr = dict(zip(clases, cuentas))
    clase_mayoritaria = clases[np.argmax(cuentas)]

    if prof >= max_prof or len(y) < min_ejemplos or len(clases) == 1:
        return Nodo(distr=distr, clase=clase_mayoritaria)

    mejor_atr, mejor_umbral, mejor_ganancia = _mejor_particion(X, y, atributos, prop_umbral)

    if mejor_atr is None or mejor_ganancia <= 0:
        return Nodo(distr=distr, clase=clase_mayoritaria)

    mask_izq = X[:, mejor_atr] <= mejor_umbral
    mask_der = ~mask_izq

    hijo_izq = _construye_arbol(X[mask_izq], y[mask_izq], atributos, min_ejemplos, max_prof, prop_umbral, prof + 1)
    hijo_der = _construye_arbol(X[mask_der], y[mask_der], atributos, min_ejemplos, max_prof, prop_umbral, prof + 1)

    return Nodo(atributo=mejor_atr, umbral=mejor_umbral, izq=hijo_izq, der=hijo_der, distr=distr)


def _clasifica_ejemplo(nodo, x):
    if nodo.es_hoja():
        return nodo.clase
    if x[nodo.atributo] <= nodo.umbral:
        return _clasifica_ejemplo(nodo.izq, x)
    else:
        return _clasifica_ejemplo(nodo.der, x)


def _clasifica_prob_ejemplo(nodo, x):
    if nodo.es_hoja():
        total = sum(nodo.distr.values())
        return {cls: cnt / total for cls, cnt in nodo.distr.items()}
    if x[nodo.atributo] <= nodo.umbral:
        return _clasifica_prob_ejemplo(nodo.izq, x)
    else:
        return _clasifica_prob_ejemplo(nodo.der, x)


def _imprime_nodo(nodo, nombre_atrs, nombre_clase, prefijo=""):
    if nodo.es_hoja():
        print(f"{prefijo}{nombre_clase}: {nodo.clase} -- {nodo.distr}")
    else:
        nombre_atr = nombre_atrs[nodo.atributo]
        umbral_str = f"{nodo.umbral:.3f}"
        print(f"{prefijo}{nombre_atr} <= {umbral_str}")
        _imprime_nodo(nodo.izq, nombre_atrs, nombre_clase, prefijo + "     ")
        print(f"{prefijo}{nombre_atr} > {umbral_str}")
        _imprime_nodo(nodo.der, nombre_atrs, nombre_clase, prefijo + "     ")


class ArbolDecision:
    def __init__(self, min_ejemplos_nodo_interior=5, max_prof=10, n_atrs=10, prop_umbral=1.0):
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof = max_prof
        self.n_atrs = n_atrs
        self.prop_umbral = prop_umbral
        self.raiz = None

    def entrena(self, X, y):
        n_total_atrs = X.shape[1]
        n_usar = min(self.n_atrs, n_total_atrs)
        atributos = np.random.choice(n_total_atrs, n_usar, replace=False)
        self.raiz = _construye_arbol(X, y, atributos, self.min_ejemplos_nodo_interior,
                                     self.max_prof, self.prop_umbral)

    def clasifica(self, X):
        if self.raiz is None:
            raise ClasificadorNoEntrenado("Hay que llamar a entrena() antes de clasificar.")
        return np.array([_clasifica_ejemplo(self.raiz, x) for x in X])

    def clasifica_prob(self, x):
        if self.raiz is None:
            raise ClasificadorNoEntrenado("Hay que llamar a entrena() antes de clasificar.")
        return _clasifica_prob_ejemplo(self.raiz, x)

    def imprime_arbol(self, nombre_atrs, nombre_clase):
        if self.raiz is None:
            raise ClasificadorNoEntrenado("Hay que llamar a entrena() antes de imprimir.")
        _imprime_nodo(self.raiz, nombre_atrs, nombre_clase)


def rendimiento(clasif, X, y):
    return sum(clasif.clasifica(X) == y) / X.shape[0]


class RandomForest:
    def __init__(self, n_arboles=5, prop_muestras=1.0,
                 min_ejemplos_nodo_interior=5, max_prof=10, n_atrs=10, prop_umbral=1.0):
        self.n_arboles = n_arboles
        self.prop_muestras = prop_muestras
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof = max_prof
        self.n_atrs = n_atrs
        self.prop_umbral = prop_umbral
        self.arboles = []

    def entrena(self, X, y):
        self.arboles = []
        n = X.shape[0]
        n_muestras = max(1, int(n * self.prop_muestras))
        for _ in range(self.n_arboles):
            idx = np.random.choice(n, n_muestras, replace=True)
            arbol = ArbolDecision(
                min_ejemplos_nodo_interior=self.min_ejemplos_nodo_interior,
                max_prof=self.max_prof,
                n_atrs=self.n_atrs,
                prop_umbral=self.prop_umbral
            )
            arbol.entrena(X[idx], y[idx])
            self.arboles.append(arbol)

    def clasifica(self, X):
        if not self.arboles:
            raise ClasificadorNoEntrenado("Hay que llamar a entrena() antes de clasificar.")
        votos = np.array([arbol.clasifica(X) for arbol in self.arboles])
        resultado = []
        for j in range(X.shape[0]):
            clases_j, cuentas_j = np.unique(votos[:, j], return_counts=True)
            resultado.append(clases_j[np.argmax(cuentas_j)])
        return np.array(resultado)


from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import os

df_credito = pd.read_csv("datos/credito.csv")
X_credito_raw = df_credito.iloc[:, :-1].values
y_credito_raw = df_credito.iloc[:, -1].values
enc_credito = OrdinalEncoder()
X_credito_enc = enc_credito.fit_transform(X_credito_raw)
X_train_credito, X_test_credito, y_train_credito, y_test_credito = \
    particion_entr_prueba(X_credito_enc, y_credito_raw, test=0.2)

df_adult = pd.read_csv("datos/adultDataset.csv")
X_adult_raw = df_adult.iloc[:, :-1].values
y_adult_raw = df_adult.iloc[:, -1].values
enc_adult = OrdinalEncoder()
X_adult_enc = X_adult_raw.copy()
col_categoricas = []
for j in range(X_adult_raw.shape[1]):
    try:
        X_adult_raw[:, j].astype(float)
    except ValueError:
        col_categoricas.append(j)
if col_categoricas:
    X_adult_enc[:, col_categoricas] = enc_adult.fit_transform(X_adult_raw[:, col_categoricas])
X_adult_enc = X_adult_enc.astype(float)
X_train_adult, X_test_adult, y_train_adult, y_test_adult = \
    particion_entr_prueba(X_adult_enc, y_adult_raw, test=0.2)


def carga_imagenes(ruta_imagenes):
    imagenes = []
    imagen_actual = []
    with open(ruta_imagenes, 'r') as f:
        for linea in f:
            linea = linea.rstrip('\n')
            linea = linea.ljust(28)[:28]
            fila = [0 if c == ' ' else 1 for c in linea]
            imagen_actual.append(fila)
            if len(imagen_actual) == 28:
                imagenes.append(imagen_actual)
                imagen_actual = []
    return np.array(imagenes).reshape(len(imagenes), -1)


def carga_etiquetas(ruta_etiquetas):
    etiquetas = []
    with open(ruta_etiquetas, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                etiquetas.append(int(linea))
    return np.array(etiquetas)


ruta_base = "datos/digitdata"
if os.path.exists(ruta_base):
    X_train_dg = carga_imagenes(os.path.join(ruta_base, "trainingimages"))
    y_train_dg = carga_etiquetas(os.path.join(ruta_base, "traininglabels"))
    X_valid_dg = carga_imagenes(os.path.join(ruta_base, "validationimages"))
    y_valid_dg = carga_etiquetas(os.path.join(ruta_base, "validationlabels"))
    X_test_dg = carga_imagenes(os.path.join(ruta_base, "testimages"))
    y_test_dg = carga_etiquetas(os.path.join(ruta_base, "testlabels"))


print("************ PRUEBAS EJERCICIO 1:")
print("**********************************\n")
Xe_votos, Xp_votos, ye_votos, yp_votos = particion_entr_prueba(X_votos, y_votos, test=1/3)
print("Partición votos: ", y_votos.shape[0], ye_votos.shape[0], yp_votos.shape[0])
print("Proporción original en votos: ", np.unique(y_votos, return_counts=True))
print("Estratificación entrenamiento en votos: ", np.unique(ye_votos, return_counts=True))
print("Estratificación prueba en votos: ", np.unique(yp_votos, return_counts=True))
print("\n")

Xev_cancer, Xp_cancer, yev_cancer, yp_cancer = particion_entr_prueba(X_cancer, y_cancer, test=0.2)
print("Proporción original en cáncer: ", np.unique(y_cancer, return_counts=True))
print("Estratificación entr-val en cáncer: ", np.unique(yev_cancer, return_counts=True))
print("Estratificación prueba en cáncer: ", np.unique(yp_cancer, return_counts=True))
Xe_cancer, Xv_cancer, ye_cancer, yv_cancer = particion_entr_prueba(Xev_cancer, yev_cancer, test=0.2)
print("Estratificación entrenamiento cáncer: ", np.unique(ye_cancer, return_counts=True))
print("Estratificación validación cáncer: ", np.unique(yv_cancer, return_counts=True))
print("\n")

Xe_credito, Xp_credito, ye_credito, yp_credito = particion_entr_prueba(X_credito_enc, y_credito_raw, test=0.4)
print("Estratificación entrenamiento crédito: ", np.unique(ye_credito, return_counts=True))
print("Estratificación prueba crédito: ", np.unique(yp_credito, return_counts=True))
print("\n\n\n")


print("************ PRUEBAS EJERCICIO 2:")
print("**********************************\n")

clf_titanic = ArbolDecision(max_prof=3, min_ejemplos_nodo_interior=5, n_atrs=3)
clf_titanic.entrena(X_train_titanic, y_train_titanic)
clf_titanic.imprime_arbol(["Pclass", "Mujer", "Edad"], "Partido")
rend_train_titanic = rendimiento(clf_titanic, X_train_titanic, y_train_titanic)
rend_test_titanic = rendimiento(clf_titanic, X_test_titanic, y_test_titanic)
print(f"****** Rendimiento DT titanic train: {rend_train_titanic}")
print(f"****** Rendimiento DT titanic test: {rend_test_titanic}\n\n\n\n ")

clf_votos = ArbolDecision(min_ejemplos_nodo_interior=3, max_prof=5, n_atrs=16)
clf_votos.entrena(Xe_votos, ye_votos)
nombre_atrs_votos = [f"Votación {i}" for i in range(1, 17)]
clf_votos.imprime_arbol(nombre_atrs_votos, "Partido")
rend_train_votos = rendimiento(clf_votos, Xe_votos, ye_votos)
rend_test_votos = rendimiento(clf_votos, Xp_votos, yp_votos)
print(f"****** Rendimiento DT votos en train: {rend_train_votos}")
print(f"****** Rendimiento DT votos en test:  {rend_test_votos}\n\n\n\n")

clf_iris = ArbolDecision(max_prof=3, n_atrs=4)
clf_iris.entrena(X_train_iris, y_train_iris)
clf_iris.imprime_arbol(["Long. Sépalo", "Anch. Sépalo", "Long. Pétalo", "Anch. Pétalo"], "Clase")
rend_train_iris = rendimiento(clf_iris, X_train_iris, y_train_iris)
rend_test_iris = rendimiento(clf_iris, X_test_iris, y_test_iris)
print(f"********************* Rendimiento DT iris train: {rend_train_iris}")
print(f"********************* Rendimiento DT iris test: {rend_test_iris}\n\n\n\n ")

clf_cancer = ArbolDecision(min_ejemplos_nodo_interior=3, max_prof=10, n_atrs=15)
clf_cancer.entrena(Xev_cancer, yev_cancer)
nombre_atrs_cancer = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error',
        'fractal dimension error', 'worst radius', 'worst texture',
        'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points',
        'worst symmetry', 'worst fractal dimension']
clf_cancer.imprime_arbol(nombre_atrs_cancer, "Es benigno")
rend_train_cancer = rendimiento(clf_cancer, Xev_cancer, yev_cancer)
rend_test_cancer = rendimiento(clf_cancer, Xp_cancer, yp_cancer)
print(f"***** Rendimiento DT cancer en train: {rend_train_cancer}")
print(f"***** Rendimiento DT cancer en test: {rend_test_cancer}\n\n\n")


print("************ RENDIMIENTOS FINALES RANDOM FOREST")
print("************************************************\n")

print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE IMDB:")
RF_IMDB = RandomForest(n_arboles=15, max_prof=15, n_atrs=50, prop_umbral=0.7,
                       min_ejemplos_nodo_interior=5, prop_muestras=1.0)
RF_IMDB.entrena(X_train_imdb, y_train_imdb)
print("Rendimiento RF entrenamiento sobre imdb: ", rendimiento(RF_IMDB, X_train_imdb, y_train_imdb))
print("Rendimiento RF test sobre imdb: ", rendimiento(RF_IMDB, X_test_imdb, y_test_imdb))
print("\n")

print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE CRÉDITO:")
RF_CREDITO = RandomForest(n_arboles=20, max_prof=8, n_atrs=6, prop_umbral=1.0,
                          min_ejemplos_nodo_interior=3, prop_muestras=1.0)
RF_CREDITO.entrena(X_train_credito, y_train_credito)
print("Rendimiento RF entrenamiento sobre crédito: ", rendimiento(RF_CREDITO, X_train_credito, y_train_credito))
print("Rendimiento RF test sobre crédito: ", rendimiento(RF_CREDITO, X_test_credito, y_test_credito))
print("\n")

print("==== MEJOR RENDIMIENTO RF SOBRE ADULT:")
RF_ADULT = RandomForest(n_arboles=15, max_prof=12, n_atrs=7, prop_umbral=0.8,
                        min_ejemplos_nodo_interior=5, prop_muestras=1.0)
RF_ADULT.entrena(X_train_adult, y_train_adult)
print("Rendimiento RF entrenamiento sobre adult: ", rendimiento(RF_ADULT, X_train_adult, y_train_adult))
print("Rendimiento RF test sobre adult: ", rendimiento(RF_ADULT, X_test_adult, y_test_adult))
print("\n")

print("==== MEJOR RENDIMIENTO RF SOBRE DÍGITOS:")
RF_DG = RandomForest(n_arboles=20, max_prof=20, n_atrs=50, prop_umbral=0.7,
                     min_ejemplos_nodo_interior=3, prop_muestras=1.0)
RF_DG.entrena(X_train_dg, y_train_dg)
print("Rendimiento RF entrenamiento sobre dígitos: ", rendimiento(RF_DG, X_train_dg, y_train_dg))
print("Rendimiento RF validación sobre dígitos: ", rendimiento(RF_DG, X_valid_dg, y_valid_dg))
print("Rendimiento RF test sobre dígitos: ", rendimiento(RF_DG, X_test_dg, y_test_dg))
