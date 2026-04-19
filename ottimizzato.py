import os
import pandas as pd
import numpy as np
import random as rd
import math
import copy

# Funzione per il calcolo della matrice dei costi del dataset passato come parametro
def matrice_distanze(data_array):
    n = data_array.shape[0]
    # Inizializziamo una matrice n x n con zeri
    matrice = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrice[i][j] = 0
            else:
            # Calcolo distanza Euclidea
                dist_euclidea = math.sqrt((data_array[i,1] - data_array[j,1])**2 + (data_array[i,2] - data_array[j,2])**2)
            
            # Troncamento alla prima cifra decimale 
                matrice[i, j] = math.floor(10 * dist_euclidea) / 10
    return matrice

# Funzione utilizzata per validare la feasibility di una singola rotta
def valida_rotta(percorso, veichle_capacity, data, dist_matrix):
    carico = 0
    tempo = 0
    costo = 0

    # len -1 poichè torniamo sempre nel magazzino alla fine
    for k in range(len(percorso) - 1):
        u, v = percorso[k], percorso[k+1]
        t_uv = dist_matrix[u, v] # Il tempo è uguale al costo [cite: 93]
        
        carico += data[v, 3] # Domanda del cliente [cite: 9, 28]
        if carico > veichle_capacity: return False, 0
        
        # Calcolo tempo di arrivo e inizio servizio [cite: 17, 58, 76]
        arrivo = tempo + data[u, 6] + t_uv # tempo + service + travel
        inizio_servizio = max(arrivo, data[v, 4]) # max(arrivo, ready_time) [cite: 17, 61]
        
        if inizio_servizio > data[v, 5]: return False, 0 # [cite: 16, 61]
        
        tempo = inizio_servizio
        costo += t_uv
    return True, costo

def greedy_1(n_clienti, veichle_quantity, v_cap, dati_nodi, costi):

    visitati = np.zeros(n_clienti+1, dtype=bool)
    visitati[0] = True # Il deposito è il punto di partenza
    
    # Definisco la lista dei percorsi
    percorsi_totali = []
    # Inizializzo i costi
    costo_totale_global = 0
    # Controllo i clienti serviti
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

            for i in range(1, n_clienti+1):
                if not visitati[i]:
                    # Il costo c_ij è uguale al tempo di viaggio t_ij
                    t_ij = costi[nodo_corrente, i]
                    
                    # Calcolo orario di inizio servizio
                    # tempo_attuale + service_time_corrente + tempo_viaggio
                    arrivo = max(tempo_attuale + dati_nodi[nodo_corrente, 6] + t_ij, 
                                    dati_nodi[i, 4])
                    
                    # Verifica vincoli: Capacità (1f) e Time Window (1h)
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
    if len(percorsi_totali) < veichle_quantity:
        while len(percorsi_totali) < veichle_quantity:
            percorsi_totali.append([0, 0])
        
    return percorsi_totali, costo_totale_global

# Secondo algoritmo greedy: un veicolo dedicato per ogni nodo (se possibile) + cheapest insertion
def greedy_2(n_clienti, veichle_quantity, v_cap, dati_nodi, costi):
    percorsi_totali = []
    costo_totale_global = 0.0
    clienti_in_attesa = []
    
    # FASE 1: rotte "singolette" finché ci sono veicoli
    for i in range(1, n_clienti + 1):
        
        # 1. Recupero dati del cliente
        domanda_cliente = dati_nodi[i, 3]  # q_i
        fine_finestra = dati_nodi[i, 5]    # b_i
        
        # 2. Calcolo costi di trasporto
        costo_andata = costi[0, i]  # c_0i
        costo_ritorno = costi[i, 0] # c_i,n+1 (torna al deposito)
        
        # 3. Verifica ammissibilità temporale
        # Il veicolo arriva e inizia il servizio non prima di a_i
        tempo_arrivo = max(costo_andata, dati_nodi[i, 4]) 
        
        # Controllo dei vincoli
        # Verifica capacità (1f)  e tempo massimo (1h)
        if domanda_cliente > v_cap or tempo_arrivo > fine_finestra:
            # Clienti i infeasible anche in rotta singola
            continue
        if len(percorsi_totali) < veichle_quantity:
            # Creazione rotta elementare: Deposito -> Cliente -> Deposito
            percorso_attuale = [0, i, 0] 
            percorsi_totali.append(percorso_attuale)
            costo_totale_global += (costo_andata + costo_ritorno)
        else:
            # Veicoli esauriti: metto il cliente in attesa per la fase 2
            clienti_in_attesa.append(i)
    # FASE 2: Completo le rotte tramite cheapest insertion per inserire tutti i clienti
    for cliente in clienti_in_attesa:
        miglior_costo_extra = float('inf')
        miglior_rotta_idx = None
        miglior_pos = None
        for r_idx, rotta in enumerate(percorsi_totali):
            # Provo ogni posizione di inserimento nella rotta
            for pos in range(1, len(rotta)):
                nuova_rotta = rotta[:pos] + [cliente] + rotta[pos:]
                check, nuovo_costo = valida_rotta(nuova_rotta, v_cap, dati_nodi, costi)
                if not check: continue

                _, costo_attuale = valida_rotta(rotta, v_cap, dati_nodi, costi)
                costo_extra = nuovo_costo - costo_attuale

                if costo_extra < miglior_costo_extra:
                    miglior_costo_extra = costo_extra
                    miglior_rotta_idx = r_idx
                    miglior_pos = pos
        if miglior_rotta_idx is not None:
            # Inserimento nella rotta migliore trovata
            percorsi_totali[miglior_rotta_idx].insert(miglior_pos, cliente)
            costo_totale_global += miglior_costo_extra
        else:
            print(f"Cliente {int(dati_nodi[cliente, 0])} non inseribile in nessuna rotta esistente.")

    return percorsi_totali, round(costo_totale_global, 1)

# MAIN PROGRAM
def main():
    # Leggo i dati in formatoo int e lascio una precisione di due cifre decimali
    np.set_printoptions(suppress=True, precision=2)

    fold = input("Inserisci il nome della cartella(n25/50/100): ")
    file_name = input("Inserisci il nome del file(es.C101.txt): ")

    # Percorso del file 
    #path_base = r'C:\Users\safet\OneDrive\Desktop\Progetto\Progetto\Istanze'
    path_base = r'C:\Users\mgand\OneDrive\Desktop\Ottimizzazzione_sr\Progetto\Istanze'

    path = os.path.join(path_base, fold, file_name)

    # Lettura dell'header del file
    try:

        # Leggo le informazioni testuali sui veicoli
        veichle_info = np.genfromtxt(path, skip_header=4, max_rows=1)

        # Ricavo le informazioni sui veicoli
        veichle_quantity = int(veichle_info[0])
        veichle_capacity = veichle_info[1]
        print("Veichle info")
        print("Veichle quantity: ", veichle_quantity)
        print("Veichle capacity: ", veichle_capacity)

        # Leggo le restanti informazioni sui clienti
        data = np.genfromtxt(path, skip_header=9)
        # Specifico le colonne
        cols = {'ID': 0, 'X': 1, 'Y': 2, 'DEM': 3, 'READY': 4, 'DUE': 5, 'SERV': 6}
        demand = data[:, 3]
        n_clienti = len(data)-1
        print("Number of clients: ", n_clienti)

        # Matrice contente le distanze e i costi per ogni coppia di cliente
        dist_matrix = matrice_distanze(data)

        # Approccio Greedy numero 1: Nearest Neighborhood
        print("Algoritmo 1")
        percorsi, costo_tot = greedy_1(n_clienti, veichle_quantity, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_tot:.1f}")

        # Approccio Greedy numero 2: "rivial solution o singleton solution"
        percorsi_2, costo_tot_2 = greedy_2(n_clienti, veichle_quantity, veichle_capacity,data, dist_matrix)
        for idx, p in enumerate(percorsi_2):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_tot_2:.1f}")



    except FileNotFoundError:
        print(f"ERRORE: Il file '{file_name}' non esiste nella cartella '{fold}'.")
        print("Controlla di aver scritto correttamente i nomi.")
    except Exception as e:
        print(f"ERRORE imprevisto: {e}")


if __name__ == "__main__":
    main()