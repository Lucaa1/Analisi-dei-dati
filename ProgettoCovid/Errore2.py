import numpy as np
from sklearn.metrics import confusion_matrix
import Read2 as Rd

'''
Funzione che calcola gli errori
'''

_, labels=Rd.Read_dataset(folder='Covid19-dataset/train_ridim')
weights=np.bincount(labels)
weights=[1/x for x in weights]

def Errore(y_pred, y_true):
    conf_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    VP = conf_matrix.diagonal()
    VN = np.sum(conf_matrix) - np.sum(conf_matrix, axis=0) - np.sum(conf_matrix, axis=1) + VP
    FP = np.sum(conf_matrix, axis=0) - VP
    FN = np.sum(conf_matrix, axis=1) - VP
        
    # Calcolo le metriche
    m_avg_accuracy = np.sum(VP) / np.sum(conf_matrix)
    m_avg_specificity = np.sum(VN) / np.sum(VN + FP)
    m_avg_balanced_acc=np.mean(conf_matrix.diagonal()/np.sum(conf_matrix, axis=1))

    w_avg_accuracy = np.average(VP / np.sum(conf_matrix, axis=1), weights=weights)
    w_avg_specificity = np.average(VN / (VN + FP), weights=weights)
    w_avg_precision = np.average(VP / (VP + FP), weights=weights) 
    M_avg_f1 = np.mean((2 * VP) / (2 * VP + FP + FN))

    return [m_avg_accuracy, m_avg_balanced_acc, m_avg_specificity], [w_avg_accuracy,  w_avg_precision, w_avg_specificity, M_avg_f1], conf_matrix
