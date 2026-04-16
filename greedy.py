import os
import pandas as pd
import numpy as np
from numpy import random
import math

# Leggo i dati in formatoo int e lascio una precisione di due cifre decimali
np.set_printoptions(suppress=True, precision=2)

fold = input("Inserisci il nome della cartella(n25/50/100): ")
file_name = input("Inserisci il nome del file(es.C101.txt): ")

# Percorso del file 
path_base = r'C:\Users\safet\OneDrive\Desktop\Progetto\Progetto\Istanze'
#path_base = r'C:\Users\mgand\OneDrive\Desktop\Ottimizzazzione_sr\Progetto\Istanze'

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
        
    # Matrice delle distanze

    #distances = np.zeros((data.shape[0], data.shape[0]))

    # Funzione di controllo dell'ammisibilità della soluzione
    def valida_rotta(percorso):
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
    def greedy_1(n_clienti, v_cap, dati_nodi, costi):
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
    # Secondo algoritmo greedy
    
    # Secondo algoritmo greedy: un veicolo dedicato per ogni nodo (se possibile)
    def greedy_2(n_clienti, v_cap, dati_nodi, costi):
        percorsi_totali = []
        costo_totale_global = 0.0
        
        # Iteriamo su tutti i clienti (da 1 a n_clienti)
        # dati_nodi[i, 0] è l'ID del cliente
        for i in range(1, n_clienti + 1):
            
            # 1. Recupero dati del cliente dai metadati [cite: 94]
            domanda_cliente = dati_nodi[i, 3]  # q_i [cite: 9, 28]
            fine_finestra = dati_nodi[i, 5]    # b_i [cite: 14, 28]
            
            # 2. Calcolo costi di trasporto [cite: 20, 31, 87]
            costo_andata = costi[0, i]  # c_0i
            costo_ritorno = costi[i, 0] # c_i,n+1 (torna al deposito)
            
            # 3. Verifica ammissibilità temporale
            # Il veicolo arriva e inizia il servizio non prima di a_i [cite: 17]
            tempo_arrivo = max(costo_andata, dati_nodi[i, 4]) 
            
            # --- CONTROLLO VINCOLI (1f e 1h) ---
            # Verifica capacità (1f)  e tempo massimo (1h) [cite: 61, 72]
            if domanda_cliente <= v_cap and tempo_arrivo <= fine_finestra:
                
                # Creazione rotta elementare: Deposito -> Cliente -> Deposito [cite: 32]
                percorso_attuale = [0, i, 0] 
                percorsi_totali.append(percorso_attuale)
                
                # L'obiettivo è minimizzare il costo totale (1a) [cite: 44, 67]
                costo_totale_global += (costo_andata + costo_ritorno)
                
            else:
                # Segnalazione se il cliente viola i vincoli fondamentali [cite: 13, 16]
                motivo = "Capacità" if domanda_cliente > v_cap else "Tempo"
                print(f"Cliente {int(dati_nodi[i, 0])} scartato. Motivo: {motivo}")

        return percorsi_totali, round(costo_totale_global, 1)

    #1 Neighborhood: Insertion --> provo a togliere un cliente da un path e lo inserisco in un'altra con FIRST IMPROVMENT
    def neigh_1(path, costo_tot):
        print("\n Neighborhood 1: Insertion")
        
        costo_attuale = costo_tot
        miglioramento = True

        while miglioramento:
            miglioramento = False

            # Itero su tutte le rotte
            for r1_idx in range(len(path)):
                rotta_src = path[r1_idx]

                # Controllo sulla lunghezza della rotta
                if len(rotta_src) <= 2:
                    continue

                # Itero sui clienti della rotta R1
                for idx_pos in range(1, len(rotta_src)-1):
                    cliente = rotta_src[idx_pos] # ID REALE DEL CLIENTE
                    # Seleziono una rotta di destinazione in cui inserire il cliente
                    for r2_idx in range(len(path)):

                        rotta_dest = path[r2_idx]
                        
                        # Definisco le posizioni in cui può essere inserito il cliente
                        for pos in range(1, len(rotta_dest)):
                            # Evito un reinserimento nella stessa posizione dello stesso path
                            if r1_idx == r2_idx and (pos == idx_pos or pos == idx_pos +1):
                                continue
                            
                            # Provo lo spostamento

                            nuova_rotta_src = rotta_src[:]
                            nuova_rotta_src.pop(idx_pos)

                            nuova_rotta_dest = rotta_dest[:] if r1_idx != r2_idx else nuova_rotta_src[:]
                            # Se src e dest sono la stessa, lavoriamo sulla stessa lista già modificata
                            if r1_idx == r2_idx:
                                # Se abbiamo rimosso un elemento prima della posizione di inserimento, l'indice scala
                                adj_pos = pos - 1 if pos > cliente else pos
                                nuova_rotta_dest.insert(adj_pos, cliente)
                            else:
                                nuova_rotta_dest.insert(pos, cliente)

                            ok_dest, costo_dest = valida_rotta(nuova_rotta_dest) 
                            if not ok_dest: continue

                            ok_src, costo_src = valida_rotta(nuova_rotta_src)
                            if not ok_src: continue

                            # Calcolo del nuovo costo totale
                            nuovo_costo_tot = costo_attuale
                            # Sottraiamo i vecchi costi delle due rotte e aggiungiamo i nuovi
                            # (Nota: serve salvare i costi vecchi delle rotte prima del ciclo o ricalcolarli)
                            _, vecchio_costo_src = valida_rotta(rotta_src)
                            _, vecchio_costo_dest = valida_rotta(rotta_dest)
                            
                            nuovo_costo_tot = costo_attuale - (vecchio_costo_src + vecchio_costo_dest) + (costo_src + costo_dest)
                            
                            if nuovo_costo_tot < costo_attuale - 0.01: # Tolleranza decimale
                                path[r1_idx] = nuova_rotta_src
                                path[r2_idx] = nuova_rotta_dest
                                costo_attuale = nuovo_costo_tot
                                miglioramento = True
                                break # Esci dai cicli interni e ricomincia la ricerca
                        if miglioramento: break
                    if miglioramento: break
                if miglioramento: break
            if miglioramento: break # esce dal ciclo delle rotte sorgente
        return path, costo_attuale
    
    def Sim_Annealing(path, costo_tot):

        # Definizione dei parmetri
        T_init = 0.8 * costo_tot
        T_end = 0.1
        alpha = random.uniform(0.8, 0.99)

        # Passo la soluzione iniziale in s per non modificarla subito
        s = path
        while T_init > T_end:
            # Randomizzare la scelta del neighborhood
            # Scelgo la rotta da cui prendere solo tra quelle "attive"
            rotte_attive = [idx for idx, rotta in enumerate(s) if len(rotta) > 2]
            idx_partenza = random.choice(rotte_attive)
            rotta_sorgente = s[idx_partenza]
            idx_cliente = random.randint(1, len(rotta_sorgente)-2)
            cliente = rotta_sorgente.pop(idx_cliente)

            # Scelgo la destinazione 
            
            # Scelta del parametro u

            u = random.uniform()

            # Controllo se accetto la mossa

            # In caso sia feasible e migliori la aggiungo a quella iniziale

            T_init = alpha * T_init
        return path
    # Esecuzione
    percorsi, costo_tot = greedy_1(n_clienti, veichle_capacity, data, dist_matrix)
    for idx, p in enumerate(percorsi):
        print(f"Veicolo {idx+1}: {p}")
    print(f"Costo Totale della Soluzione: {costo_tot:.1f}")
    percorsi_2, costo_tot_2 = greedy_2(n_clienti,veichle_capacity,data, dist_matrix)
    for idx, p in enumerate(percorsi_2):
        print(f"Veicolo {idx+1}: {p}")
    print(f"Costo Totale della Soluzione: {costo_tot_2:.1f}")
    percorsi, costo_tot = neigh_1(percorsi, costo_tot)

    print("-- Dopo local search 1 --")
    for idx, p in enumerate(percorsi):
        print(f"Veicolo {idx+1}: {p}")
    print(f"Costo Totale della Soluzione: {costo_tot:.1f}")

    # Implementazione simulated annealing



except FileNotFoundError:
    print(f"ERRORE: Il file '{file_name}' non esiste nella cartella '{fold}'.")
    print("Controlla di aver scritto correttamente i nomi.")
except Exception as e:
    print(f"ERRORE imprevisto: {e}")