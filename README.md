# Prueba de Concepto (PoC) de MLOps

Esta Prueba de Concepto (PoC) demuestra un flujo de trabajo de MLOps utilizando **GitHub Actions**, **MLflow** y **DVC** para entrenar un modelo de Machine Learning que detecta cáncer a partir de imágenes y datos.

## Contenido

- [Introducción](#introducción)
- [Requisitos](#requisitos)
- [Pasos de Implementación](#pasos-de-implementación)
  - [1. Configuración de MLflow](#1-configuración-de-mlflow)
  - [2. Configuración y gestión de datos con DVC](#2-configuración-y-gestión-de-datos-con-dvc)
  - [3. Definición del Workflow de GitHub Actions](#3-definición-del-workflow-de-github-actions)
  - [4. Monitoreo y verificación de resultados](#4-monitoreo-y-verificación-de-resultados)
  - [5. Presentación de la PoC](#5-presentación-de-la-poc)

## Introducción

El objetivo de esta PoC es crear un flujo de trabajo automatizado para el entrenamiento de modelos de Machine Learning, asegurando el versionado de datos y la trazabilidad de experimentos mediante MLflow. 

## Requisitos

- Python 3.8 o superior
- TensorFlow
- MLflow
- DVC
- GitHub

## Pasos de Implementación

### 1. Configuración de MLflow
   1. **Instalación de MLflow**:
        ```bash
        pip install mlflow
        ```
   2. **Iniciar el servidor de MLflow**:
        ```bash
        mlflow ui
        ```
   3. **Configurar el URI de tracking en S3**:
        En el archivo src/train.py, establece el URI de tracking:
        ```python
        mlflow.set_tracking_uri('s3://nombre_del_bucket')
        ```
   4. Iniciar un experimento:
        ```python
        mlflow.set_experiment('cancer_detection_experiment')
        ```

### 2. Configuración y gestión de datos con DVC
   1. **Inicializar DVC**:
        ```bash
        dvc init
        git add .dvc/
        git commit -m "Initialize DVC"
        ```
   2. **Agregar el dataset**:
        ```bash
        dvc add data/chest_xray
        git add data/chest_xray.dvc
        git commit -m "Add dataset to DVC"
        ```
   3. **Configurar almacenamiento remoto**:
        ```bash
        dvc remote add -d myremote s3://nombre_del_bucket/dvc-storage
        ```
   4. **Subir datos a DVC**:
        ```bash
        dvc push
        ```

### 3. Definición del Workflow de GitHub Actions
   1. **Crear el archivo de workflow**:
      Crea .github/workflows/mlops.yml con el siguiente contenido:

      ```yaml
      name: MLOps Workflow

      on:
        push:
          branches:
            - main

        pull_request:

        workflow_dispatch:

      jobs:
        train_model:
          runs-on: ubuntu-latest
          environment: mlops_test

          steps:
          - name: Checkout repository
            uses: actions/checkout@v4

          - name: Set up Python
            uses: actions/setup-python@v2
            with:
              python-version: 3.8

          - name: Install dependencies
            run: |
              python -m pip install --upgrade pip
              pip install -r requirements.txt
          
          - name: Config aws credentials
            run: |
              dvc remote add -d myremote ${{ secrets.AWS_S3_BUCKET_URL }} --force
              dvc remote modify myremote access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
              dvc remote modify myremote secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}

          - name: Repro data
            run: |
              dvc repro --force
              
          - name: Pull data from DVC storage
            run: |
              dvc pull

          - name: Train model
            run: |
              python3 src/train.py --data_path data/chest_xray --epochs ${model.epochs} --batch_size ${model.batch_size}

          - name: Push trained model to DVC
            run: |
              dvc add models/cancer_detection_model.h5
              git add models/cancer_detection_model.h5.dvc
              git commit -m "Add trained model"
              dvc push
      ```

### 4. Monitoreo y verificación de resultados
   1. **Acceder a la interfaz de MLflow**:
      Revisa los experimentos en http://localhost:5000 si lo ejecutas localmente.
   2. **Verificar resultados en GitHub Actions**:
      Accede a la pestaña "Actions" en tu repositorio para verificar el estado del flujo de trabajo.

### 5. Presentación de la PoC
   En la presentación, destaca:
   - Automatización del flujo de trabajo usando GitHub Actions.
   - Versionado de datos y modelos con DVC.
   - Monitoreo de experimentos y métricas en MLflow.
   - Escalabilidad del flujo de trabajo para nuevos datasets y modelos.

## Conclusión:
Esta PoC demuestra un flujo de trabajo completo de MLOps que permite entrenar, versionar y monitorear modelos de Machine Learning de manera eficiente. Se puede expandir fácilmente para incorporar nuevos modelos y datasets en el futuro.