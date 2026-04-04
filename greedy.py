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
    
    veichle_quantity = int(veichle_info[0])
    veichle_capacity = veichle_info[1]
    print("Veichle info")
    print("Veichle quantity: ", veichle_quantity)
    print("Veichle capacity: ", veichle_capacity)

    data = np.genfromtxt(path, skip_header=9)
    
    cols = {'ID': 0, 'X': 1, 'Y': 2, 'DEM': 3, 'READY': 4, 'DUE': 5, 'SERV': 6}
    demand = data[:, 3]
    n_clienti = len(data)-1
    print(n_clienti)
    
    #print(data[:2])
    
    # Matrice delle distanze

    #distances = np.zeros((data.shape[0], data.shape[0]))

    # Calcolo dei costi
    def cost(pos_i, pos_j):
        # Distanza euclidea
        e_ij = math.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
    # Troncamento alla prima cifra decimale
        return math.floor(10 * e_ij) / 10
    #Matrice delle distanze in cui teniamo conto dei costi
    def matrice_distanze(data_array):
        n = data_array.shape[0]
    # Inizializziamo una matrice n x n con zeri
        matrice = np.zeros((n, n))
    
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrice[i][j] = 0
                else:
                # Calcolo distanza Euclidea [cite: 90]
                    dist_euclidea = math.sqrt((data_array[i,1] - data_array[j,1])**2 + (data_array[i,2] - data_array[j,2])**2)
                
                # Troncamento alla prima cifra decimale 
                    matrice[i, j] = math.floor(10 * dist_euclidea) / 10
        return matrice
    # print(distances)
    """
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i == j:
                distances[i][j] = 0
            else:
                distances[i][j] =  math.sqrt((data[i][1]-data[j][1])**2+(data[i][2]-data[j][2])**2)
    """
    #print(distances[:1])
    dist_matrix = matrice_distanze(data)
    # Algoritmo 1:
    print("Algoritmo 1")
    def solve_vrptw(n_clienti, v_cap, dati_nodi, costi):
        visitati = np.zeros(n_clienti, dtype=bool)
        visitati[0] = True # Il deposito è il punto di partenza [cite: 27, 31]
        
        percorsi_totali = []
        costo_totale_global = 0
        clienti_serviti = 0

        # Ciclo sui veicoli disponibili [cite: 34]
        while clienti_serviti < n_clienti and len(percorsi_totali) < veichle_quantity:
            percorso_attuale = [0]
            nodo_corrente = 0 # Indice intero
            capacita_residua = v_cap
            tempo_attuale = 0 # y_i^k [cite: 42, 71]

            while True:
                miglior_prossimo = None
                distanza_minima = float('inf')
                orario_inizio_prossimo = 0

                for i in range(1, n_clienti):
                    if not visitati[i]:
                        # Il costo c_ij è uguale al tempo di viaggio t_ij [cite: 93]
                        t_ij = costi[nodo_corrente, i]
                        
                        # Calcolo orario di inizio servizio [cite: 17, 18, 76]
                        # tempo_attuale + service_time_corrente + tempo_viaggio
                        arrivo = max(tempo_attuale + dati_nodi[nodo_corrente, 6] + t_ij, 
                                     dati_nodi[i, 4])
                        
                        # Verifica vincoli: Capacità (1f) e Time Window (1h) [cite: 57, 61, 70, 72]
                        if capacita_residua >= dati_nodi[i, 3] and arrivo <= dati_nodi[i, 5]:
                            if t_ij < distanza_minima:
                                distanza_minima = t_ij
                                miglior_prossimo = i
                                orario_inizio_prossimo = arrivo

                if miglior_prossimo is None:
                    # Ritorno al deposito [cite: 27, 32]
                    costo_ritorno = costi[nodo_corrente, 0]
                    costo_totale_global += costo_ritorno
                    percorso_attuale.append(0)
                    break
                
                # Aggiornamento stato
                visitati[miglior_prossimo] = True
                clienti_serviti += 1
                capacita_residua -= dati_nodi[miglior_prossimo, 3]
                tempo_attuale = orario_inizio_prossimo
                costo_totale_global += distanza_minima
                percorso_attuale.append(miglior_prossimo)
                nodo_corrente = miglior_prossimo # Mantiene l'indice intero

            percorsi_totali.append(percorso_attuale)
            
        return percorsi_totali, costo_totale_global

    # Esecuzione
    percorsi, costo_tot = solve_vrptw(n_clienti, veichle_capacity, data, dist_matrix)
    for idx, p in enumerate(percorsi):
        print(f"Veicolo {idx+1}: {p}")
    print(f"Costo Totale della Soluzione: {costo_tot:.1f}")
except FileNotFoundError:
    print(f"ERRORE: Il file '{file_name}' non esiste nella cartella '{fold}'.")
    print("Controlla di aver scritto correttamente i nomi.")
except Exception as e:
    print(f"ERRORE imprevisto: {e}")