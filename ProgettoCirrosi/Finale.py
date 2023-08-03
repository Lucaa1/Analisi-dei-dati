import Prediction as pred
import Grafici as Gf
import pandas as pd
import numpy as np
import  matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
import Timer as tm


'''
Script in cui per ogni modello faccio la ricerca dei parmetri e confronto i modelli sul test set creando tutti i grafici
'''


df = pd.read_csv('archive/New_dataset.csv')
features_df=df.drop(columns='Stage')
target_df=df.pop('Stage')
target_df=target_df.astype(int)

features_df, features_test, target_df, target_test = train_test_split(features_df, target_df, test_size=0.1, random_state=1)


#Random forest
n,d, Risultati_RF=pred.Iperparametri_randomforest(features_df, target_df, 'gini')
# Risultati_RF = pred.Random_forest_from_k(features_df, target_df, 11, 120, 'gini')
Pred_RF, m_avg_RF, M_avg_RF,Conf_matrix_RF=Risultati_RF

#SVM
Risultati_SVM=pred.Iperparametri_SVM(features_df, target_df)
# Risultati_SVM=pred.Support_vector_machine(features_df, target_df, c=1, kernel='rbf')
Pred_SVM, m_avg_SVM, M_avg_SVM, Conf_matrix_SVM=Risultati_SVM

#NN
Arch=[[32,16], [8,8], [16, 16], [16, 8]]
Risultati_NN=pred.Train_NN(features_df, target_df,  Architecture=Arch, fun=['sigmoid', 'relu', 'tanh'])
# Risultati_NN=pred.Neural_network(features_df, target_df, pred.create_model(features_df.shape[1],np.max(target_df)+1, hidden_layers=[32,16], fun='relu'))
Pred_NN, m_avg_NN, M_avg_NN, Conf_matrix_NN=Risultati_NN


# Plot del grafico a barre raggruppato
if True:
    avg_vectors = [np.concatenate((m_avg_RF, M_avg_RF), axis=0), np.concatenate((m_avg_SVM, M_avg_SVM), axis=0), np.concatenate((m_avg_NN, M_avg_NN), axis=0)]
    method_labels = ['Random Forest', 'SVM', 'Neural Network']
    Gf.plot_grouped_bar_chart(avg_vectors, method_labels, 'train set', metrics=['Accuracy', 'Specificity', 'Balanced Accuracy', 'Macro precision', 'Macro specificity', 'Macro F1-Score'])


#Calcolo tempi
if False:
    tempi_RF=tm.save_and_measure_prediction_times(Pred_RF, features_df, 'RF_model')
    tempi_SVM=tm.save_and_measure_prediction_times(Pred_SVM, features_df, 'SVM_model')
    tempi_NN=tm.save_and_measure_prediction_times(Pred_NN, features_df, 'NN_model')

    print(tempi_RF, tempi_NN, tempi_SVM)

#Test set
if True:
    y_pred_SVM = Pred_SVM.predict(features_test)
    y_pred_RF = Pred_RF.predict(features_test)
    y_pred_NN = Pred_NN.predict(features_test, verbose=0)
    y_pred_NN = np.argmax(y_pred_NN, axis=1)   

    m_avg_RF, M_avg_RF,Conf_matrix_RF=pred.Errore(y_pred_RF, target_test)
    m_avg_SVM, M_avg_SVM,Conf_matrix_SVM=pred.Errore(y_pred_SVM, target_test)
    m_avg_NN, M_avg_NN,Conf_matrix_NN=pred.Errore(y_pred_NN, target_test)

    # Plot delle confusion matrix
    if True:
        conf_matrices = [Conf_matrix_SVM, Conf_matrix_RF, Conf_matrix_NN]
        labels =  [1, 2, 3, 4]
        Gf.plot_confusion_matrix1(Conf_matrix_SVM, labels, 'SVM Confusion Matrix su test set')
        Gf.plot_confusion_matrix1(Conf_matrix_RF, labels, 'RF Confusion Matrix su test set')
        Gf.plot_confusion_matrix1(Conf_matrix_NN, labels, 'NN Confusion Matrix su test set')

    # Plot del grafico a barre raggruppato
    if True:
        avg_vectors = [np.concatenate((m_avg_RF, M_avg_RF), axis=0), np.concatenate((m_avg_SVM, M_avg_SVM), axis=0), np.concatenate((m_avg_NN, M_avg_NN), axis=0)]
        method_labels = ['Random Forest', 'SVM', 'Neural Network']
        Gf.plot_grouped_bar_chart(avg_vectors, method_labels, 'test set', metrics=['Accuracy', 'Specificity', 'Balanced Accuracy', 'Macro precision', 'Macro specificity', 'Macro F1-Score'])

    plt.show()

