import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
import argparse
import mlflow
import mlflow.tensorflow
import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cargar los parámetros desde params.yaml
with open('params.yaml') as f:
    params = yaml.safe_load(f)

epochs = params['model']['epochs']
batch_size = params['model']['batch_size']
learning_rate = params['model']['learning_rate']
data_path_default = params['data']['path']  # Cargar el valor por defecto de data_path

# Establecer la URI de tracking de MLflow para S3
mlflow.set_tracking_uri('${{ secrets.AWS_S3_BUCKET_URL }}')

# Iniciar un experimento en MLflow
mlflow.set_experiment('cancer_detection_experiment')

# Definir una función para cargar datos usando ImageDataGenerator
def load_data(data_path, batch_size):
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # Escalar los valores de píxeles

    # Generadores para entrenamiento y validación
    train_generator = datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_path, 'test'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, validation_generator, test_generator

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
        layers.Dense(1, activation='sigmoid')  # Salida para clasificación binaria
    ])
    return model

# Entrenar el modelo
def train(data_path, epochs, batch_size):
    train_generator, validation_generator, test_generator = load_data(data_path, batch_size)

    model = create_model((224, 224, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Iniciar tracking de MLflow
    mlflow.start_run()

    # Registrar hiperparámetros en MLflow
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    # Entrenar el modelo
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    # Evaluar el modelo en el conjunto de test
    test_loss, test_accuracy = model.evaluate(test_generator)

    # Loggear métricas en MLflow
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Guardar el modelo y registrarlo en MLflow
    input_example = {"input": np.array([[...]], dtype=np.float32)} 
    model_path = 'models/cancer_detection_model.keras'
    model.save(model_path)
    mlflow.keras.log_model(model, "model", signature=mlflow.models.infer_signature(inputs=input_example))

    # Finalizar tracking de MLflow
    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Usar el valor de data_path de params.yaml como valor por defecto
    parser.add_argument('--data_path', type=str, default=data_path_default, help='Ruta a los datos de entrenamiento')

    # Usar los valores de epochs y batch_size de params.yaml como valores por defecto
    parser.add_argument('--epochs', type=int, default=epochs, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Tamaño del lote')

    args = parser.parse_args()

    # Entrenar el modelo
    train(args.data_path, args.epochs, args.batch_size)
