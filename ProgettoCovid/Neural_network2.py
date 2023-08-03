import tensorflow as tf
import Read2 as Rd
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import Errore2 as Er

'''
Script in cui ci sono le funzioni per creare la rete neurale e confrontare gli iperparametri
'''


def create_model(input_dim=(256,256,1), output_dim=3, hidden_layers=[64,32], fun='sigmoid'):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=fun, input_shape=input_dim))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    for num_neurons in hidden_layers[1:]:
        model.add(tf.keras.layers.Conv2D(num_neurons, kernel_size=(3, 3), activation=fun))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))    
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=fun))
    model.add(tf.keras.layers.Dense(output_dim, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def Neural_network(X_train, y_train, model):
    X_train, y_train=Preparare_dati(X_train, y_train)
    model.fit(X_train, y_train, batch_size=32, epochs=10)
    return model

def Preparare_dati(X,y):
    X=np.array(X)/255
    y=np.array(y)
    X=np.expand_dims(X, axis=-1)
    y=to_categorical(y, num_classes=3)
    return X, y

def Iperparametri_NN(X_train, y_train, X_valid, y_valid, Arc, Act):
    X_valid,_=Preparare_dati(X_valid, y_valid)

    Best_accuracy=0
    Best_arc=[]
    Best_fun=''
    Best_model=None
    Result=np.zeros((len(Act), len(Arc)))
    for i, fun in enumerate(Act):
        for j, a in enumerate(Arc):
            model=create_model(input_dim = np.shape(X_valid[0]), output_dim=3, hidden_layers=a, fun=fun)
            model=Neural_network(X_train, y_train, model)

            y_pred = model.predict(X_valid, verbose=0)
            y_pred = np.argmax(y_pred, axis=1)   
            result=Er.Errore(y_pred, y_valid)

            Result[i][j]=result[0][0]   #prendo la micro accuracy
            if Best_accuracy < result[0][0]:
                Best_model=model
                Best_accuracy=result[0][0]
                Best_arc=a
                Best_fun=fun

    print("Best model:")
    print("Architecture:", Best_arc)
    print("Activation function:", Best_fun)
    print("Average accuracy:", Best_accuracy)

    fig, ax = plt.subplots()

    # Imposta le posizioni dei gruppi di  barre sul grafico
    bar_width = 0.2
    x_pos = np.arange(len(Arc))

    # Per ogni funzione Act, crea un gruppo di barre con colori diversi
    for i, fun in enumerate(Act):
        result_values = Result[i]
        ax.bar(x_pos + i * bar_width, result_values, bar_width, label=fun)

    # Etichette sugli assi e legenda
    ax.set_xlabel('Arc')
    ax.set_ylabel('Micro Accuracy')
    ax.set_title('Risultati in base agli Iperparametri')
    ax.set_xticks(x_pos + bar_width * len(Act) / 2)
    ax.set_xticklabels(Arc)
    ax.legend()

    # plt.show()
    return Best_model









