import Read2 as Rd
import Neural_network2 as NN
import Errore2 as Er
import numpy as np
import Grafici2 as Gr
from sklearn.svm import SVC
import SVM2 as Sv
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot  as plt

'''
Funzione finale in cui faccio la ricerca dei parametri e valuto i modelli sul validation set e test set e mi faccio stampare i grafici
'''


X_train_NN, y_train_NN = Rd.Read_dataset('Covid19-dataset/train_ridim')
X_test_NN, y_test_NN = Rd.Read_dataset('Covid19-dataset/test_ridim')

if False:    # Ricerca iperparametri
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_NN, y_train_NN, test_size=0.2, random_state=42)
    Arc=[[32, 32, 64], [32, 64, 128], [64, 64, 128]]
    Act=['sigmoid', 'relu', 'tanh']
    Modello = NN.Iperparametri_NN(X_train, y_train, X_valid, y_valid, Arc, Act)

    X_valid,_=NN.Preparare_dati(X_valid, y_valid)
    y_pred = Modello.predict(X_valid, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)   
    result_NN=Er.Errore(y_pred, y_valid)

    X_train_SVM, y_train_SVM = Rd.Read_dataset('Covid19-dataset/train_ridim_512')
    X_test_SVM, y_test_SVM = Rd.Read_dataset('Covid19-dataset/test_ridim_512')

    X_train_SVM = [np.reshape(x, (1,-1)).squeeze() for x in X_train_SVM]
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_SVM, y_train_SVM, test_size=0.2, random_state=42)

    SVC_= Sv.Iperparametri_SVM(X_train, y_train, X_valid, y_valid)
    SVC_.fit(X_train_SVM, y_train_SVM)
    y_pred = SVC_.predict(X_valid)
    result_SVM=Er.Errore(y_pred, y_valid)
else:       # Fase di Test
    model = NN.Neural_network(X_train_NN, y_train_NN, NN.create_model(hidden_layers=[64, 64, 128], fun='relu'))
    X_test_NN,_=NN.Preparare_dati(X_test_NN, y_test_NN)
    y_pred = model.predict(X_test_NN, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)   
    result_NN=Er.Errore(y_pred, y_test_NN)

    X_train_SVM, y_train_SVM = Rd.Read_dataset('Covid19-dataset/train_ridim_512')
    X_test_SVM, y_test_SVM = Rd.Read_dataset('Covid19-dataset/test_ridim_512')

    X_train_SVM = [np.reshape(x, (1,-1)).squeeze() for x in X_train_SVM]
    X_test_SVM = [np.reshape(x, (1,-1)).squeeze() for x in X_test_SVM]

    SVC_=SVC(kernel='poly', decision_function_shape='ovr', gamma='scale', C=1)
    SVC_.fit(X_train_SVM, y_train_SVM)
    y_pred = SVC_.predict(X_test_SVM)
    result_SVM=Er.Errore(y_pred, y_test_SVM)


#Plot confusion Matrix
if True:
    Gr.plot_confusion_matrix(result_NN[2], title='Confusion matrix NN')
    Gr.plot_confusion_matrix(result_SVM[2], title='Confusion matrix SVM')


# Plot del grafico a barre raggruppato
if True:
    avg_vectors = [np.concatenate((result_SVM[0],result_SVM[1]), axis=0), np.concatenate((result_NN[0],result_NN[1]), axis=0)]
    method_labels = ['SVM', 'Neural Netword']
    Gr.plot_grouped_bar_chart(avg_vectors, method_labels, 'validation set', metrics=['Accuracy', 'Balanced Accuracy', 'Specificity', 'Weighted accuracy', 'Weighted precision', 'Weighted specificity', 'Macro F1-Score'])
    plt.show()