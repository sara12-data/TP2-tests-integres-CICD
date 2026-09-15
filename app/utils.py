import numpy as np
from sklearn.linear_model import LinearRegression

# Données utilisées pour entraîner le modèle
X_train = np.array([[1], [2], [3], [4]])
y_train = np.array([3, 6, 9, 12])  # y = 3 * x

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

def predict(features):

 # Transformer les valeurs reçues dans le format attendu par le modèle
    features = np.array(features).reshape(-1, 1)

# Faire la prédiction et retourner le résultat sous forme de liste
    return model.predict(features).tolist()
