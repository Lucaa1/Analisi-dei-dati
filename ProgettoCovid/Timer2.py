import numpy as np
from sklearn.svm import SVC
import Neural_network2 as NN
import joblib
import time
import matplotlib.pyplot as plt
import Read2 as Rd

'''
Funzionie per ottenere i grafici sulla complessità
'''


def save_and_measure_prediction_times(X_train, y_train, X_test, y_test, Model='RF', model_filename='modello di prova'):

    #Tempo di addestramento
    if Model == 'SVM':

        model=SVC(kernel='poly', decision_function_shape='ovr', gamma='scale', C=1)
        start_time = time.time()
        X_train = [np.reshape(x, (1,-1)).squeeze() for x in X_train]
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

    elif Model == 'NN':

        model=NN.create_model(hidden_layers=[32, 64, 128], fun='relu')
        start_time = time.time()
        model = NN.Neural_network(X_train, y_train, model)
        training_time = time.time() - start_time

    else:
        raise ValueError("Modello non valido. Utilizzare 'SVM' o 'NN'.")

    #Tempo di caricamento
    joblib.dump(model, model_filename)
    start_time = time.time()
    loaded_model = joblib.load(model_filename)
    loading_time = (time.time() - start_time)

    #Tempo di predizione
    if Model == 'NN':
        X_test, _=NN.Preparare_dati(X_test, y_test)
        start_time = time.time()
        prediction = loaded_model.predict(X_test)
        prediction_time = (time.time() - start_time)/X_test.shape[0]
    elif Model == 'SVM':
        X_test = [np.reshape(x, (1,-1)).squeeze() for x in X_test]
        start_time = time.time()
        prediction = loaded_model.predict(X_test)
        prediction_time = (time.time() - start_time)/len(X_test)

    return training_time, loading_time, prediction_time


X_train_NN, y_train_NN = Rd.Read_dataset('Covid19-dataset/train_ridim')
X_train_SVM, y_train_SVM = Rd.Read_dataset('Covid19-dataset/train_ridim_512')

X_test_NN, y_test_NN = Rd.Read_dataset('Covid19-dataset/test_ridim')
X_test_SVM, y_test_SVM = Rd.Read_dataset('Covid19-dataset/test_ridim_512')

models=['NN', 'SVM']
training_times = []
loading_times = []
prediction_times = []

training_time, loading_time, prediction_time = save_and_measure_prediction_times( X_train_NN, y_train_NN, X_test_NN, y_test_NN, 'NN')
training_times.append(training_time)
loading_times.append(loading_time)
prediction_times.append(prediction_time)

training_time, loading_time, prediction_time = save_and_measure_prediction_times( X_train_SVM, y_train_SVM, X_test_SVM, y_test_SVM, 'SVM')
training_times.append(training_time)
loading_times.append(loading_time)
prediction_times.append(prediction_time)


# Creo il primo grafico a barre (training time e loading time)
plt.figure(figsize=(8, 6))
bar_width = 0.2
index = np.arange(len(models))

plt.bar(index, training_times, width=bar_width, label='Addestramento')
plt.bar(index + bar_width, loading_times, width=bar_width, label='Caricamento')

plt.xlabel('Modello')
plt.ylabel('Tempo (secondi)')
plt.title('Tempi di addestramento e caricamento di ogni modello')
plt.xticks(index + bar_width / 2, models)
plt.legend(loc='upper right')

plt.tight_layout()

# Creo il secondo grafico a barre (prediction time)
plt.figure(figsize=(4, 6))
plt.bar(index, prediction_times, width=bar_width)
plt.xlabel('Modello')
plt.ylabel('Tempo (secondi)')
plt.title('Tempo di previsione di ogni modello')
plt.xticks(index, models)

plt.tight_layout()
plt.show()