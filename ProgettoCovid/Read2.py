import os
import cv2
from PIL import Image
import numpy as np


'''
Script in cui ci sono le funzioni per leggere le immagini, ridimensionare le foto e creare delle nuove cartelle con le immagini ridimensionate
'''

def Read_dataset(folder='Covid19-dataset/train_ridim'):
    # Definisco il dizionario di mappatura dei label
    class_labels = {
        'Covid': 2,
        'Viral Pneumonia': 1,
        'Normal': 0
    }

    data = []
    labels=[]

    # Loop per vedere tutte le foto
    for class_folder, label in class_labels.items():
        folder_path = os.path.join(folder, class_folder)
        
        for image_filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_filename)
    
            image = Image.open(image_path).convert("L")
            image = np.array(image)
            
            data.append(image)
            labels.append(label)
    return data, labels


def Ridimensiona_immagini(partenza, arrivo, width=1024, height=1024):
    folder_path = partenza

    # Loop per le immagini nella cartella
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (width, height))
        cv2.imwrite(arrivo +'/'+ image_filename, resized_image)

def Ridimensiona(width=512, height=512):
    Ridimensiona_immagini('Covid19-dataset/train/Covid', 'Covid19-dataset/train_ridim_512/Covid', width, height)
    Ridimensiona_immagini('Covid19-dataset/train/Normal', 'Covid19-dataset/train_ridim_512/Normal', width, height)
    Ridimensiona_immagini('Covid19-dataset/train/Viral Pneumonia', 'Covid19-dataset/train_ridim_512/Viral Pneumonia', width, height)

    Ridimensiona_immagini('Covid19-dataset/test/Covid', 'Covid19-dataset/test_ridim_512/Covid', width, height)
    Ridimensiona_immagini('Covid19-dataset/test/Normal', 'Covid19-dataset/test_ridim_512/Normal', width, height)
    Ridimensiona_immagini('Covid19-dataset/test/Viral Pneumonia', 'Covid19-dataset/test_ridim_512/Viral Pneumonia', width, height)

# Ridimensiona( )
