from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Prediction as pred
import Grafici as Gf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''
Script che fa la forward selection e i grafici
'''


def backward_features_selection(Model, features_df, target_df):
    features_train, features_test, target_train, target_test = train_test_split(features_df, target_df, test_size=0.1, random_state=1)

    selected_features = list(features_df.columns)
    best_acc = 0.0
    Best_feature = []
    best_err_train=[]
    Best_err_train=[]

    #While per guardare tutte le features
    while len(selected_features) > 1:
        worst_feature = None
        best_score = 0.0

        #For sulle features che sono ancora selezionate
        for feature in selected_features:
            subset_features = [f for f in selected_features if f != feature]

            _, Err_train = Addestra(Model, features_train[subset_features], target_train)
            score = Err_train[1][0]

            if score > best_score:
                worst_feature = feature
                best_score = score
                best_err_train=Err_train

        #Rimuovo la caratteristica peggiore 
        selected_features.remove(worst_feature)

        #Aggiorno la migliore prestazione 
        if best_score > best_acc:
            best_acc = best_score
            Best_feature = selected_features.copy()
            Best_err_train=best_err_train

    optimal_feature_subset = features_train[Best_feature]
    model, _ = Addestra(Model, optimal_feature_subset, target_train)

    #Predizione sul set di test utilizzando le caratteristiche selezionate
    Err_test = Predizioni(model, Model, features_test[Best_feature], target_test)
    return Err_test, Best_err_train, Best_feature

def Addestra(Model, features_df, target_df):
    if Model == 'RF':
        model, m_avg, M_avg, Conf_matrix = pred.Random_forest_from_k(features_df, target_df, 11, 120, 'gini')
        Err=m_avg, M_avg, Conf_matrix
        return model,Err 
    elif Model == 'SVM':
        model,m_avg, M_avg, Conf_matrix=pred.Support_vector_machine(features_df, target_df, c=1, kernel='rbf')
        Err=m_avg, M_avg, Conf_matrix
        return model, Err 
    elif Model == 'NN':
        model,m_avg, M_avg, Conf_matrix=pred.Neural_network(features_df, target_df, pred.create_model(features_df.shape[1],np.max(target_df)+1, hidden_layers=[32,16], fun='relu'))
        Err=m_avg, M_avg, Conf_matrix
        return model, Err
    else:
        raise ValueError("Modello non valido. Utilizzare 'RF', 'SVM' o 'NN'.")

def Predizioni(model, Model, features_test, target_test):
    if Model == 'RF':
        y_pred = model.predict(features_test)
    elif Model == 'SVM':
        y_pred = model.predict(features_test)
    elif Model == 'NN':
        y_pred = model.predict(features_test, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)   
    else:
        raise ValueError("Modello non valido. Utilizzare 'RF', 'SVM' o 'NN'.")
    return pred.Errore(y_pred, target_test)

df = pd.read_csv('archive/New_dataset.csv')
features_df=df.drop(columns='Stage')
target_df=df.pop('Stage')

Err_test_RF, Err_train_RF, Best_feature_RF=backward_features_selection('RF', features_df, target_df)
m_avg_test_RF, M_avg_test_RF, Conf_matrix_test_RF=Err_test_RF
m_avg_train_RF, M_avg_train_RF, Conf_matrix_train_RF=Err_train_RF

Err_test_SVM, Err_train_SVM, Best_feature_SVM=backward_features_selection('SVM', features_df, target_df)
m_avg_test_SVM, M_avg_test_SVM, Conf_matrix_test_SVM=Err_test_SVM
m_avg_train_SVM, M_avg_train_SVM, Conf_matrix_train_SVM=Err_train_SVM

#Per stampare le feaures
if True:
    print('Migliori features per backward Selection per SCM  e RF')
    print(len(Best_feature_SVM), len(Best_feature_RF))
    print(Best_feature_SVM, '\n' ,Best_feature_RF)

#Per fare i grafici
avg_vectors = [np.concatenate((m_avg_test_RF, M_avg_test_RF), axis=0), np.concatenate((m_avg_test_SVM, M_avg_test_SVM), axis=0)]
method_labels = ['Random Forest', 'SVM']
Gf.plot_grouped_bar_chart(avg_vectors, method_labels, 'test set', metrics=['Accuracy', 'Specificity', 'Balanced Accuracy', 'Macro precision', 'Macro specificity', 'Macro F1-Score'])

avg_vectors = [np.concatenate((m_avg_train_RF, M_avg_train_RF), axis=0), np.concatenate((m_avg_train_SVM, M_avg_train_SVM), axis=0)]
method_labels = ['Random Forest', 'SVM']
Gf.plot_grouped_bar_chart(avg_vectors, method_labels, 'train set', metrics=['Accuracy', 'Specificity', 'Balanced Accuracy', 'Macro precision', 'Macro specificity', 'Macro F1-Score'])

plt.show()
