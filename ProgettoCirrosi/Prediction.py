import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import os


'''
In questo script ci sono le funzioni usate per creare i modelli e confrontare gli iperparametri
'''


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.seterr(divide='ignore')
np.seterr(invalid='ignore')

Print=False
Plot=False

# features_df, features_test, target_df, target_test = train_test_split(features, target, test_size=0.1, random_state=42)


def Errore(y_pred, y_true):
    conf_matrix=confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])
    VP = conf_matrix.diagonal()
    VN = np.sum(conf_matrix) - np.sum(conf_matrix, axis=0) - np.sum(conf_matrix, axis=1) + VP
    FP = np.sum(conf_matrix, axis=0) - VP
    FN = np.sum(conf_matrix, axis=1) - VP

    # Calcolo le varie metriche
    m_avg_accuracy = np.sum(VP) / np.sum(conf_matrix)
    m_avg_specificity = np.sum(VN) / np.sum(VN + FP)
    balanced_acc=np.mean(conf_matrix.diagonal()/np.sum(conf_matrix, axis=1))
     
    balanced_acc=np.mean(conf_matrix.diagonal()/np.sum(conf_matrix, axis=1))
    M_avg_precision = np.nanmean(VP/ (VP + FP))
    M_avg_specificity = np.mean(VN / (VN + FP))
    M_avg_f1 = np.mean((2 * VP) / (2 * VP + FP + FN))

    return [m_avg_accuracy, m_avg_specificity], [balanced_acc,  M_avg_precision, M_avg_specificity, M_avg_f1], conf_matrix


############################################################################################################################
def Random_forest_from_k(features_df, target_df, n=50, d=6, criterion='gini'):

    random_forest = RandomForestClassifier(n_estimators=n, max_depth=d, criterion='log_loss', random_state=42)
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    Conf_matrix=[]
    M_avg=[]
    m_avg=[]
    for train_index, test_index in skf.split(features_df, target_df):
        X_train, X_test = features_df.iloc[train_index], features_df.iloc[test_index]
        y_train, y_test = target_df.iloc[train_index].values.ravel(), target_df.iloc[test_index].values.ravel()

        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        
        # Valuto l'accuratezza delle predizioni
        v, w, C=Errore(y_pred, y_test)
        m_avg.append(v)
        M_avg.append(w)
        Conf_matrix.append(C)

    Conf_matrix=np.mean(Conf_matrix, axis=0)
    m_avg=np.mean(m_avg, axis=0)
    M_avg=np.mean(M_avg, axis=0)

    if Print:
        print('Random Forest accuracy is', M_avg[0])
    if Plot:
        disp=ConfusionMatrixDisplay(Conf_matrix)
        disp.plot()
    return random_forest, m_avg, M_avg, Conf_matrix
# _,_,_,c=Random_forest_from_k(features, target)
# plt.show()


def Iperparametri_randomforest(features_df, target_df, criterio='gini'):

    fig, axes=plt.subplots(1,2,figsize=(12, 6))
    D= np.arange(5, 25, 2)
    N=np.arange(5, 150, 5)
    result=[]

    # Guardo le profondità
    for d in D:
        _,_,v,_=Random_forest_from_k(features_df, target_df, 100, d, criterio)
        result.append(v[0])
    axes[0].plot(D, result, 'o-', label=criterio)
    ind=np.argmax(result)
    axes[0].set_xlabel('Profondità massima')
    axes[0].set_ylabel('Balanced accuracy')

    result=[]
    temp_max=0

    #Guardo il numero di alberi
    for n in N:
        Pred=Random_forest_from_k(features_df, target_df, n, D[ind], criterion=criterio)
        v=Pred[2]
        result.append(v[0])
        if temp_max < v[0]:
            temp_max=v[0]
            Pred_best = Pred
    
    #Plotto il grafico
    axes[1].plot(N, result, 'o-',label=criterio)
    ind1=np.argmax(result)
    axes[1].set_xlabel('Numero di Alberi')
    axes[1].set_ylabel('Balanced accuracy')

    print('Con la profondità massima =', D[ind], 'e con', N[np.argmax(result)],'alberi ho un\'accuratezza del', result[ind1])
    
    plt.tight_layout()
    return N[np.argmax(result)], D[ind], Pred_best
# Iperparametri_randomforest(features_df, target_df, 'gini')




#############################################################################################################################
#Support Vector Machine

def Support_vector_machine(features_df, target_df, c=1, kernel='rbf'):
    SVC_model=SVC(kernel=kernel, decision_function_shape='ovr', gamma='scale', C=c)
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    Conf_matrix=[]
    M_avg=[]
    m_avg=[]
    for train_index, test_index in skf.split(features_df, target_df):
        X_train, X_test = features_df.iloc[train_index], features_df.iloc[test_index]
        y_train, y_test = target_df.iloc[train_index].values.ravel(), target_df.iloc[test_index].values.ravel()

        SVC_model.fit(X_train, y_train)
        y_pred = SVC_model.predict(X_test)

        # Valuto l'accuratezza delle predizioni
        v, w, C=Errore(y_pred, y_test)
        m_avg.append(v)
        M_avg.append(w)
        Conf_matrix.append(C)

    Conf_matrix=np.mean(Conf_matrix, axis=0)
    m_avg=np.mean(m_avg, axis=0)
    M_avg=np.mean(M_avg, axis=0)

    if Print:
        print('SVM accuracy is', M_avg[0])
    if Plot:
        disp=ConfusionMatrixDisplay(Conf_matrix)
        disp.plot(cmap=plt.cm.Blues)
    return SVC_model, m_avg, M_avg, Conf_matrix
# Support_vector_machine(features, target, c=1, kernel='rbf')

def Iperparametri_SVM(features_df, target_df):
    C=[0.1, 1, 10, 100, 1000]
    Kernel=['poly', 'rbf', 'sigmoid']
    Result=np.zeros([4,5])
    best_acc=0

    #Trovo gli iperparametri migliori
    for i,c in enumerate(C):
        for j, k in enumerate(Kernel):
            SVC_=Support_vector_machine(features_df, target_df, c, k)
            Result[j][i]=SVC_[2][0]
            if best_acc<SVC_[2][0]:
                best_acc=SVC_[2][0]
                best_i = j
                best_j = i
                model=SVC_

    print('Con Kernel=', Kernel[best_i], 'e con c=', C[best_j],'ho un\'accuratezza del', best_acc)

    #Faccio il grafico
    plt.figure(figsize=(10, 6))

    for i, kernel in enumerate(Kernel):
        plt.plot(C, Result[i, :], marker='o', label=kernel)

    plt.xscale('log')
    plt.xlabel('Valori di C')
    plt.ylabel('Balanced accuracy')
    plt.title('Accuratezza al variare di C per diversi kernel')
    plt.xticks(C,C)
    plt.grid()
    plt.legend()
    
    plt.show()
    return model
# Iperparametri_SVM(features, target)


################################################################################
#Neural Network

def create_model(input_dim, output_dim, hidden_layers=[64,32], fun='sigmoid'):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(hidden_layers[0], activation=fun, input_dim=input_dim))
    for num_neurons in hidden_layers[1:]:
        model.add(tf.keras.layers.Dense(num_neurons, activation=fun))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def Neural_network(features, target, model):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    Conf_matrix=[]
    M_avg=[]
    m_avg=[]

    # Stratified K-Fold cross-validation
    for train_index, test_index in skf.split(features, target):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index].values.ravel(), target.iloc[test_index].values.ravel()

        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        y_pred = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)              

        # Valuta l'accuratezza delle predizioni
        v, w, C=Errore(y_pred, y_test)
        m_avg.append(v)
        M_avg.append(w)
        Conf_matrix.append(C)

    Conf_matrix=np.mean(Conf_matrix, axis=0)
    m_avg=np.mean(m_avg, axis=0)
    M_avg=np.mean(M_avg, axis=0)

    if Print:
        print("Average accuracy:", M_avg[0])
    if Plot:
        plt.figure(figsize=(10,10))
        sns.heatmap(Conf_matrix, annot=True, cmap='coolwarm')
        plt.title("Matrice di correlazione")
        plt.show()

    return model, m_avg, M_avg, Conf_matrix


def Train_NN(features_df, target_df, Architecture, fun):
    input=features_df.shape[1]
    output=np.max(target_df)+1
    #Serve +1 perchè la funzione di loss loss='sparse_categorical_crossentropy' se gli passi n come output prende valori in [0,n)

    best_accuracy = 0
    best_model = None
    best_architecture = None
    best_activation = None
    all_accuracies = np.zeros([len(Architecture), len(fun)])


    for i, arch in enumerate(Architecture):
        for j, activation in enumerate(fun):
            model = create_model(input, output, arch, activation)
            NN_ = Neural_network(features_df, target_df, model)

            # Salva l'accuratezza media per questa combinazione
            all_accuracies[i][j]=NN_[2][0]

            # Aggiorna il miglior modello se si trova un'accuratezza migliore
            if NN_[2][0] > best_accuracy:
                best_accuracy = NN_[2][0]
                best_model = NN_
                best_architecture = arch
                best_activation = activation

    print("Best model:")
    print("Architecture:", best_architecture)
    print("Activation function:", best_activation)
    print("Average accuracy:", best_accuracy)

    # Faccio il grafico
    x = np.arange(np.shape(all_accuracies)[0])
    width = 0.3

    fig, ax = plt.subplots()
    n = np.shape(all_accuracies)[1]
    for i, activation in enumerate(fun):
        ax.bar(x - width * (n - 1) / 2 + i * width, all_accuracies[:, i], width, label=activation)

    ax.set_xlabel('Architecture')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Accuracy for Different Architectures and Activations')
    ax.set_xticks(x)
    ax.set_xticklabels(Architecture)
    ax.legend()

    fig.tight_layout()
    plt.show()

    return best_model


