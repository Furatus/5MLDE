from prefect import task
from keras.models import Sequential
from keras.layers import Input,Dense

@task
def create_model(input_shape):
  model = Sequential()

  # Entrée du modèle
  model.add(Input(shape=input_shape))

  # Couches cachées
  model.add(Dense(20, activation="relu"))
  model.add(Dense(30, activation="relu"))
  model.add(Dense(20, activation="relu"))
  model.add(Dense(10, activation="relu"))

  # Couche de sortie
  model.add(Dense(1, activation="sigmoid"))

  # Compilation du modèle
  model.compile(optimizer="adam", loss="mae", metrics=["mae", "mse"])

  return model