from sklearn.model_selection import train_test_split
import Prediction as pred
import numpy as np
import pandas as pd
import Grafici as Gf
import matplotlib.pyplot as plt

'''
Script che fa la forward selection e i grafici
'''

df = pd.read_csv('archive/New_dataset.csv')
features_df=df.drop(columns='Stage')
target_df=df.pop('Stage')


def forward_features_selection(Model, features_df, target_df,):
    features_df, features_test, target_df, target_test = train_test_split(features_df, target_df, test_size=0.1, random_state=1)

    selected_features = []
    best_acc = 0.0
    Best_feature=[]
    best_err_train=[]
    Best_err_train=[]

    #while per considerare ogni feature
    while len(selected_features) < len(features_df.columns):
        best_feature = None
        best_score = 0.0

        #For su ogni feature che ancora non ho preso
        for feature in features_df.columns:
            if feature not in selected_features:

                if selected_features == 0:
                    _, Err_train=Addestra(Model, features_df[feature], target_df)
                else:
                    _, Err_train=Addestra(Model, features_df[selected_features + [feature]], target_df)
                score = Err_train[1][0]

                if score > best_score:
                    best_feature = feature
                    best_score = score
                    best_err_train=Err_train

        #Aggiungo la caratteristica migliore tra quelle non ancora selezionate
        selected_features.append(best_feature)

        #Aggiorno la migliore prestazione finora
        if best_score > best_acc:
            best_acc = best_score
            Best_feature=selected_features.copy()
            Best_err_train=best_err_train

    optimal_feature_subset = features_df[Best_feature]
    model,_=Addestra(Model, optimal_feature_subset, target_df)

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

Err_test_RF, Err_train_RF, Best_feature_RF=forward_features_selection('RF', features_df, target_df)
m_avg_test_RF, M_avg_test_RF, Conf_matrix_test_RF=Err_test_RF
m_avg_train_RF, M_avg_train_RF, Conf_matrix_train_RF=Err_train_RF

Err_test_SVM, Err_train_SVM, Best_feature_SVM=forward_features_selection('SVM', features_df, target_df)
m_avg_test_SVM, M_avg_test_SVM, Conf_matrix_test_SVM=Err_test_SVM
m_avg_train_SVM, M_avg_train_SVM, Conf_matrix_train_SVM=Err_train_SVM

#Per stampare le feaures
if True:
    print('Migliori features per Forward Selection per SCM e RF')
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