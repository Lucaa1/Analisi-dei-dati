import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import Prediction as pred
import joblib
import time
import matplotlib.pyplot as plt

'''
In questo script ci sono le funzioni usate per la parte sulla complessità
'''


df = pd.read_csv('archive/New_dataset.csv')
features_df=df.drop(columns='Stage')
target_df=df.pop('Stage')

def save_and_measure_prediction_times(features_df, target_df, Model='RF', model_filename='modello di prova'):
    if Model == 'RF':
        model = RandomForestClassifier(n_estimators=120, max_depth=11, criterion='gini')
    elif Model == 'SVM':
        model = SVC(kernel='rbf', decision_function_shape='ovr', gamma='scale', C=1)
    elif Model == 'NN':
        model = pred.create_model(features_df.shape[1],np.max(target_df)+1, hidden_layers=[32,16], fun='relu')
    else:
        raise ValueError("Modello non valido. Utilizzare 'RF', 'SVM' o 'NN'.")

    # Tempo di addestramento
    start_time = time.time()
    model.fit(features_df, target_df)
    training_time = time.time() - start_time

    # Tempo di caricamento
    joblib.dump(model, model_filename)
    start_time = time.time()
    for k in range(100):
        loaded_model = joblib.load(model_filename)
    loading_time = (time.time() - start_time)/100

    # Tempo di predizione
    start_time = time.time()
    prediction = loaded_model.predict(features_df)
    prediction_time = (time.time() - start_time)/features_df.shape[0]

    return training_time, loading_time, prediction_time



if True:

    # Definisci le liste vuote per i tempi di addestramento, caricamento e previsione
    models = ['RF', 'SVM', 'NN']
    training_times = []
    loading_times = []
    prediction_times = []

    # Esegui la funzione per i diversi modelli
    for model in models:
        training_time, loading_time, prediction_time = save_and_measure_prediction_times( features_df, target_df, model)
        training_times.append(training_time)
        loading_times.append(loading_time)
        prediction_times.append(prediction_time)

    # Crea il primo grafico a barre (training time e loading time)
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    index = np.arange(len(models))

    plt.bar(index, training_times, width=bar_width, label='Addestramento')
    plt.bar(index + bar_width, loading_times, width=bar_width, label='Caricamento')

    plt.xlabel('Modello')
    plt.ylabel('Tempo (secondi)')
    plt.title('Tempi di Addestramento e Caricamento per ogni modello')
    plt.xticks(index + bar_width / 2, models)
    plt.legend(loc='upper right')

    plt.tight_layout()

    # Crea il secondo grafico a barre (prediction time)
    plt.figure(figsize=(8, 5))

    plt.bar(index, prediction_times)

    plt.xlabel('Modello')
    plt.ylabel('Tempo (secondi)')
    plt.title('Tempo di Previsione per ogni modello')
    plt.xticks(index, models)

    plt.tight_layout()
    plt.show()