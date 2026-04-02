import os
import pandas as pd
import numpy as np
import math

# Leggo i dati in formatoo int e lascio una precisione di due cifre decimali
np.set_printoptions(suppress=True, precision=2)

fold = input("Inserisci il nome della cartella(n25/50/100): ")
file_name = input("Inserisci il nome del file(es.C101.txt): ")

# Percorso del file 
path_base = r'C:\Users\safet\OneDrive\Desktop\Progetto\Istanze'

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
    demand = data[:, 3]
    print(demand)
    #print(data[:2])

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
    print("Algoritmo 1")
    n_nodi = len(distances)
    visitati = np.zeros(n_nodi, dtype=bool)
    visitati[0] = True  # Il magazzino è già "visitato" (punto di partenza)
    percorsi_totali = []
    # Ciclo su ogni veicolo per assegnare il path
    for i in range(int(veichle_quantity)): 
        if np.all(visitati):break #se tutti i clienti sono serviti ci fermiamo
        y = 0   # Tempo di inizio del percorso
        percorso_attuale = [0]
        capacity = veichle_capacity # Merce iniale di ogni veicolo
        actual_node = 0 # Parto dal magazzino

        # Scrittura del path
        while True:
            # Creiamo una copia temporanea delle distanze dal nodo attuale
            distanze_temporanee = np.array(distances[actual_node],dtype=float)
            # Escludiamo i nodi già visitati o troppo pesanti impostando la loro distanza a infinito
            for j in range(n_nodi):
                if visitati[j] or demand[j] > capacity:
                    distanze_temporanee[j] = np.inf
            prossimo_nodo = int(np.argmin(distanze_temporanee))
            # Se np.argmin restituisce un nodo con distanza infinita, significa che:
        # 1. Non ci sono più nodi da visitare
        # 2. La capacità residua non permette di servire altri nodi
            if distanze_temporanee[prossimo_nodo] == np.inf:
                percorso_attuale.append(0) # Torna al magazzino
                break
            # Aggiornamento stato
            visitati[prossimo_nodo] = True
            capacity -= demand[prossimo_nodo]
            percorso_attuale.append(prossimo_nodo)
            actual_node = prossimo_nodo
        percorsi_totali.append(percorso_attuale)
    for idx, p in enumerate(percorsi_totali):
        print(f"Veicolo {idx+1}: {p}")
except FileNotFoundError:
    print(f"ERRORE: Il file '{file_name}' non esiste nella cartella '{fold}'.")
    print("Controlla di aver scritto correttamente i nomi.")
except Exception as e:
    print(f"ERRORE imprevisto: {e}")