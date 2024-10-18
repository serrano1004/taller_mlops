# src/train.py
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import mlflow
import mlflow.tensorflow

# Establecer la URI de tracking de MLflow para S3
mlflow.set_tracking_uri('s3://mlopstest123456')

# Iniciar un experimento en MLflow
mlflow.set_experiment('cancer_detection_experiment')

# Cargar los datos
def load_data(data_path):
    data = np.load(data_path)  # Suponiendo que los datos están en formato .npz o similar
    X, y = data['images'], data['labels']
    return X, y

# Definir el modelo CNN
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Entrenar el modelo
def train(data_path, epochs, batch_size):
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model(X_train.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Iniciar tracking de MLflow
    mlflow.start_run()
    
    # Registrar hiperparámetros en MLflow
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    
    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    # Evaluar el modelo en el conjunto de test
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Loggear métricas en MLflow
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # Guardar el modelo y registrarlo en MLflow
    model_path = 'models/cancer_detection_model.h5'
    model.save(model_path)
    mlflow.keras.log_model(model, "model")
    
    # Finalizar tracking de MLflow
    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Ruta a los datos de entrenamiento')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del lote')
    
    args = parser.parse_args()
    train(args.data_path, args.epochs, args.batch_size)
