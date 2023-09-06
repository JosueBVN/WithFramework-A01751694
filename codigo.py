# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

# Definir una variable global para random_state
random_seed = 42

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Crear un DataFrame de Pandas para visualización (opcional)
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Dividir el conjunto de datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

# Crear y entrenar el modelo de Árbol de Decisiones
clf = DecisionTreeClassifier(random_state=random_seed)
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=class_names)

# Imprimir resultados
print(f'Exactitud del modelo: {accuracy:.2f}')
print('\nMatriz de Confusión:')
print(confusion)
print('\nInforme de Clasificación:')
print(classification_rep)

# Visualizar el árbol de decisiones (opcional)
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=list(class_names), rounded=True)
plt.title('Árbol de Decisiones')
plt.show()

# Visualizar la matriz de confusión (opcional)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()
