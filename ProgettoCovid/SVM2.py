import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import Errore2 as Er
from sklearn.model_selection import train_test_split
import Read2 as Rd


'''
Script in cui ci sono le funzioni per creare la SVM e confrontare gli iperparametri
'''


def Support_vector_machine(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model
# Support_vector_machine(features, target, c=1, kernel='rbf')

def Iperparametri_SVM(X_train, y_train, X_valid, y_valid):
    C=[0.1, 1, 10, 100, 1000]
    Kernel=['poly', 'rbf', 'sigmoid']
    Result=np.zeros((len(Kernel), len(C)))
    best_acc=0
    for i,c in enumerate(C):
        for j, k in enumerate(Kernel):
            SVC_=SVC(kernel=k, decision_function_shape='ovr', gamma='scale', C=c)
            SVC_=Support_vector_machine(X_train, y_train, SVC_)

            y_pred = SVC_.predict(X_valid)
            result=Er.Errore(y_pred, y_valid)

            Result[j][i]=result[0][0]   #prendo la micro accuracy
            if best_acc < result[0][0]:
                model=SVC_
                best_acc=result[0][0]
                best_i = j
                best_j = i

    print('Con Kernel=', Kernel[best_i], 'e con c=', C[best_j],'ho un\'accuratezza del', best_acc)

    plt.figure(figsize=(10, 6))

    # Traccia le linee per ciascun kernel
    for i, kernel in enumerate(Kernel):
        plt.plot(C, Result[i, :], marker='o', label=kernel)

    # Impostazioni dell'asse x e delle etichette
    plt.xscale('log')
    plt.xlabel('Valori di C')
    plt.ylabel('Accuratezza')
    plt.title('Accuratezza al variare di C per diversi kernel')
    plt.xticks(C,C)
    plt.grid()
    plt.legend()
    
    # plt.show()
    return model

