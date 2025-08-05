# -*- coding: utf-8 -*-
"""
Generador de gráficos para README - Proyecto Análisis Estadístico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer

# ===============================
# 1. CREAR CARPETA DE IMÁGENES
# ===============================
os.makedirs("img", exist_ok=True)

# ===============================
# 2. CARGAR Y LIMPIAR DATOS
# ===============================
dataset = pd.read_csv("dataset_estadistica.csv")

# Eliminar duplicados
dataset = dataset.drop_duplicates()

# Imputación de valores nulos
numeric_cols = ['Edad', 'Ingreso_Mensual', 'Horas_Estudio_Semanal']
categorical_cols = ['Nivel_Educativo', 'Genero']

numeric_imputer = SimpleImputer(strategy="mean")
dataset[numeric_cols] = numeric_imputer.fit_transform(dataset[numeric_cols])

categorical_imputer = SimpleImputer(strategy="most_frequent")
dataset[categorical_cols] = categorical_imputer.fit_transform(dataset[categorical_cols])

# ===============================
# 3. GENERAR GRÁFICOS
# ===============================

# 🎂 Distribución de Género
gender_counts = dataset['Genero'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    gender_counts,
    labels=gender_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette('pastel')
)
plt.title("Distribución de Género")
plt.axis('equal')
plt.savefig("img/grafico_genero.png", bbox_inches="tight")
plt.close()

# 🎓 Distribución de Nivel Educativo
nivel_educativo_percentages = dataset['Nivel_Educativo'].value_counts(normalize=True) * 100
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x=nivel_educativo_percentages.index,
    y=nivel_educativo_percentages.values,
    palette='viridis'
)
plt.title("Distribución de Nivel Educativo")
plt.xlabel("Nivel Educativo")
plt.ylabel("Porcentaje (%)")
plt.xticks(rotation=45, ha='right')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')
plt.savefig("img/grafico_nivel_educativo.png", bbox_inches="tight")
plt.close()

# 📈 Histogramas y 📦 Boxplots
for col in numeric_cols + ['Satisfaccion_Vida']:
    # Histograma
    plt.figure(figsize=(8, 4))
    sns.histplot(dataset[col], kde=True, color='skyblue')
    plt.title(f"Distribución de {col}")
    plt.savefig(f"img/histograma_{col}.png", bbox_inches="tight")
    plt.close()

    # Boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=dataset[col], color='lightgreen')
    plt.title(f"Boxplot de {col}")
    plt.savefig(f"img/boxplot_{col}.png", bbox_inches="tight")
    plt.close()

# 🧊 Matriz de Correlación
plt.figure(figsize=(8, 6))
sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.savefig("img/matriz_correlacion.png", bbox_inches="tight")
plt.close()

print("✅ Todas las imágenes han sido generadas en la carpeta 'img/'")
