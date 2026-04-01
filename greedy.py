import os
import pandas as pd
import numpy as np
import math

# Leggo i dati in formatoo int e lascio una precisione di due cifre decimali
np.set_printoptions(suppress=True, precision=2)

fold = input("Inserisci il nome della cartella(n25/50/100): ")
file_name = input("Inserisci il nome del file(es.C101.txt): ")

# Percorso del file 
path_base = r'C:\Users\mgand\OneDrive\Desktop\Ottimizzazzione_sr\Progetto\Istanze'

path = os.path.join(path_base, fold, file_name)

# Lettura dell'header del file
try:
    veichle_info = np.genfromtxt(path, skip_header=4, max_rows=1)

    veichle_quantity = veichle_info[0]
    veichle_capacity = veichle_info[1]
    print("Veichle info")
    print("Veichle quantity: ", veichle_quantity)
    print("Veichle capacity: ", veichle_capacity)

    data = np.genfromtxt(path, skip_header=9)

    cols = {'ID': 0, 'X': 1, 'Y': 2, 'DEM': 3, 'READY': 4, 'DUE': 5, 'SERV': 6}

    print(data[:2])

    # Matrice delle distanze

    distances = np.zeros((data.shape[0], data.shape[0]))

    # print(distances)
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i == j:
                distances[i][j] = 0
            else:
                distances[i][j] =  math.sqrt((data[i][1]-data[j][1])**2+(data[i][2]-data[j][2])**2)

    #print(distances[:1])

    # Algoritmo 1:

    # Ciclo su ogni veicolo per assegnare il path
    for i in range(veichle_quantity): 
        y = 0   # Tempo di inizio del percorso
        capacity = veichle_capacity # Merce iniale di ogni veicolo
        actual_node = 0 # Parto dal magazzino

        # Scrittura del path
        while capacity >= 0:
            
            # Calcolo delle distanze minime dal nodo attuale
            index = np.argmin(distances[actual_node])
            



except FileNotFoundError:
    print(f"ERRORE: Il file '{file_name}' non esiste nella cartella '{fold}'.")
    print("Controlla di aver scritto correttamente i nomi.")
except Exception as e:
    print(f"ERRORE imprevisto: {e}")