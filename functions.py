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

# Funzione che permette di trovare i clienti vicini
def calcola_vicini(dist_matrix, k=10):
    n = dist_matrix.shape[0]
    vicini = {}
    for i in range(n):
        # Ordina per distanza, escludi il nodo stesso
        distanze = [(j, dist_matrix[i,j]) for j in range(n) if j != i]
        distanze.sort(key=lambda x: x[1])
        vicini[i] = [j for j, _ in distanze[:k]]
    return vicini

def valida_rotta_senza_vincoli(percorso, dist_matrix, data):
    # Questa funzione calcola il costo di UN SINGOLO furgone
    # Assume che 'dist_matrix' e 'data' siano variabili globali
    costo = 0
    tempo = 0

    for k in range(len(percorso) - 1):
        # Convertiamo in int per evitare problemi di indexing con NumPy
        u, v = int(percorso[k]), int(percorso[k+1])
        
        t_uv = dist_matrix[u, v] 
        
        # Calcolo tempo: tempo precedente + service time di u + viaggio uv
        arrivo = tempo + data[u, 6] + t_uv 
        inizio_servizio = max(arrivo, data[v, 4]) 
        
        tempo = inizio_servizio
        costo += t_uv
            
    return costo

def controllo_costo(path, veichle_capacity, data, dist_matrix):
    check_costo = 0
    for idx, rotta in enumerate(path):
        _, c = valida_rotta(rotta, veichle_capacity, data, dist_matrix)
        check_costo += c
    
    return check_costo

def greedy_1(n_clienti, veichle_quantity, v_cap, dati_nodi, costi):

    visitati = np.zeros(n_clienti+1, dtype=bool)
    visitati[0] = True # Il deposito è il punto di partenza
    
    # Definisco la lista dei percorsi
    percorsi_totali = []
    # Inizializzo i costi
    costo_totale_global = 0
    # Controllo i clienti serviti
    clienti_serviti = 0

    # Ciclo sui veicoli disponibili
    while clienti_serviti < n_clienti and len(percorsi_totali) < veichle_quantity:
        percorso_attuale = [0]
        nodo_corrente = 0 # Indice intero
        capacita_residua = v_cap
        tempo_attuale = 0 

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
    costo_reale = sum(valida_rotta(r, v_cap, dati_nodi, costi)[1] 
                  for r in percorsi_totali)
    return percorsi_totali, costo_reale

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
    costo_reale = sum(valida_rotta(r, v_cap, dati_nodi, costi)[1] 
                  for r in percorsi_totali)
    return percorsi_totali, round(costo_reale, 1)

#1 Neighborhood: Insertion --> provo a togliere un cliente da un path e lo inserisco in un'altra con FIRST IMPROVMENT
def neigh_1(path, veichle_capacity, data, dist_matrix, costo_tot):
    #print("\n Neighborhood 1: Insertion")
    iterazione = 0
    costo_attuale = sum(valida_rotta(r, veichle_capacity, data, dist_matrix)[1] for r in path)
    miglioramento = True
    #print(f"Costo ingresso: {costo_tot:.1f}, costo ricalcolato: {costo_attuale:.1f}")
    # Per limitare i costi conmputazionali limito il neighborhood ai clienti vicini
    vicini = calcola_vicini(dist_matrix, k=10)

    client_to_route = {cliente: r_idx 
                   for r_idx, rotta in enumerate(path) 
                   for cliente in rotta}
    
    capacita_rotte = {idx: sum(data[c, 3] for c in rotta if c != 0)
                  for idx, rotta in enumerate(path)}
    
    while miglioramento:
        iterazione += 1
        #print(f"Iterazione {iterazione}, costo: {costo_attuale:.1f}, rotte attive: {sum(1 for r in path if len(r) > 2)}")
        costo_reale = sum(valida_rotta(r, veichle_capacity, data, dist_matrix)[1] for r in path)
        assert abs(costo_attuale - costo_reale) < 0.1, \
        f"DIVERGENZA: costo_attuale={costo_attuale:.1f}, costo_reale={costo_reale:.1f}"
        miglioramento = False

        # Precalcolo i costi di tutte le rotte prima di entrare nei cicli
        costi_rotte = {}
        for idx, rotta in enumerate(path):
            _, c = valida_rotta(rotta, veichle_capacity, data, dist_matrix)
            costi_rotte[idx] = c

        # Itero su tutte le rotte
        for r1_idx in range(len(path)):
            rotta_src = path[r1_idx]

            # Controllo sulla lunghezza della rotta
            if len(rotta_src) <= 2:
                continue

            # Itero sui clienti della rotta R1
            for idx_pos in range(1, len(rotta_src)-1):
                cliente = rotta_src[idx_pos] # ID REALE DEL CLIENTE

                nuova_rotta_src = rotta_src[:]
                nuova_rotta_src.pop(idx_pos)
                ok_src, costo_src = valida_rotta(nuova_rotta_src, veichle_capacity, data, dist_matrix)
                if not ok_src: continue  # se src non è feasible senza questo cliente, salta subito
                # Costruisci insieme delle rotte candidate
                rotte_candidate = set()
                for vicino in vicini[cliente]:
                    if vicino in client_to_route:         
                        rotte_candidate.add(client_to_route[vicino])
                rotte_candidate.add(r1_idx)
                # Seleziono una rotta di destinazione in cui inserire il cliente
                for r2_idx in rotte_candidate:

                    rotta_dest = path[r2_idx]
                    domanda_cliente = data[cliente, 3]
                    # Pre-check capacità: se fallisce salta TUTTA la rotta, non solo una posizione
                    if r1_idx != r2_idx:
                        if capacita_rotte[r2_idx] + domanda_cliente > veichle_capacity:
                            continue  # salta direttamente al prossimo r2_idx
                    # Definisco le posizioni in cui può essere inserito il cliente
                    # Costruita una volta sola per tutti i pos di questa rotta
                    for pos in range(1, len(rotta_dest)):
                        # Evito un reinserimento nella stessa posizione dello stesso path
                        if r1_idx == r2_idx and (pos == idx_pos or pos == idx_pos +1):
                            continue

                        # Se src e dest sono la stessa, lavoriamo sulla stessa lista già modificata
                        if r1_idx == r2_idx:
                            # Se abbiamo rimosso un elemento prima della posizione di inserimento, l'indice scala
                            nuova_rotta_dest = nuova_rotta_src[:]
                            adj_pos = pos - 1 if pos > idx_pos else pos
                            nuova_rotta_dest.insert(adj_pos, cliente)
                        else:
                            nuova_rotta_dest = rotta_dest[:]
                            nuova_rotta_dest.insert(pos, cliente)

                        ok_dest, costo_dest = valida_rotta(nuova_rotta_dest, veichle_capacity, data, dist_matrix) 
                        if not ok_dest: continue
                        
                        if r1_idx == r2_idx:
                        # Una sola rotta coinvolta, costo_src è intermedio e non serve
                            nuovo_costo_tot = costo_attuale - costi_rotte[r1_idx] + costo_dest
                        # Calcolo del nuovo costo totale
                        else:
                            nuovo_costo_tot = (costo_attuale - costi_rotte[r1_idx] - costi_rotte[r2_idx] + costo_src + costo_dest)
                        
                        if nuovo_costo_tot < costo_attuale - 0.01: # Tolleranza decimale
                            path[r1_idx] = nuova_rotta_src
                            path[r2_idx] = nuova_rotta_dest
                            costo_attuale = nuovo_costo_tot
                             # Aggiorna solo quando la soluzione cambia
                            client_to_route = {c: r_idx 
                                                for r_idx, rotta in enumerate(path) 
                                                for c in rotta}
                            capacita_rotte = {idx: sum(data[c, 3] for c in rotta if c != 0)
                                                for idx, rotta in enumerate(path)}
                            miglioramento = True
                            break # Esci dai cicli interni e ricomincia la ricerca
                    if miglioramento: break
                if miglioramento: break
            if miglioramento: break
        
    return path, costo_attuale

# Or-opt-2: Scambi di segmenti intrarotta
def neigh_2(path, veichle_capacity, data, dist_matrix, costo_tot):
    miglioramento = True
    
    while miglioramento:
        miglioramento = False

        # Itero sulle rotte (solo intrarotta)
        for idx_rotta in range(len(path)):
            rotta_src = path[idx_rotta]

            # Serve una rotta con almeno 5 nodi per spostare 2 clienti: 
            # Deposito_Inizio + Cliente1 + Cliente2 + Cliente_Altro + Deposito_Fine
            if len(rotta_src) < 5: 
                continue

            # i è l'indice di inizio del blocco di 2 clienti da spostare
            for i in range(1, len(rotta_src) - 2):
                c1 = rotta_src[i]
                c2 = rotta_src[i + 1]
                
                # Nodi vicini al blocco nella rotta originale
                prev_node = rotta_src[i-1]
                next_node = rotta_src[i+2]

                # 1. Calcolo del risparmio togliendo c1-c2 dalla posizione attuale
                # Togliamo: (prev->c1), (c1->c2), (c2->next)
                # Aggiungiamo il ponte: (prev->next)
                '''
                rimosso = (dist_matrix[prev_node][c1] + 
                           dist_matrix[c1][c2] + 
                           dist_matrix[c2][next_node])
                '''
                #ponte_estrazione = dist_matrix[prev_node][next_node]
                
                # Creiamo la rotta senza i due clienti
                rotta_ridotta = rotta_src[:i] + rotta_src[i+2:]
                
                # 2. Proviamo a inserire il blocco [c1, c2] in ogni posizione k della rotta ridotta
                for k in range(1, len(rotta_ridotta)):
                    n_in_prev = rotta_ridotta[k-1]
                    n_in_next = rotta_ridotta[k]

                    # Costo dell'inserimento tra n_in_prev e n_in_next
                    '''
                    ponte_rotto = dist_matrix[n_in_prev][n_in_next]
                    nuovo_inserimento = (dist_matrix[n_in_prev][c1] + 
                                        dist_matrix[c1][c2] + 
                                        dist_matrix[c2][n_in_next])
                    '''
                                        
                    nuova_rotta = rotta_ridotta[:k] + [c1, c2] + rotta_ridotta[k:]
                    #delta = (ponte_estrazione - rimosso) + (nuovo_inserimento - ponte_rotto)
                    _, costo_iniziale = valida_rotta(rotta_src, veichle_capacity, data, dist_matrix)
                    feasible, costo_nuovo = valida_rotta(nuova_rotta, veichle_capacity, data, dist_matrix)

                    if not feasible: continue

                    delta = costo_nuovo - costo_iniziale

                    if delta < -1e-6:
                        # Applichiamo il miglioramento
                        #nuova_rotta = rotta_ridotta[:k] + [c1, c2] + rotta_ridotta[k:]
                        #feasible, _ = valida_rotta(nuova_rotta, veichle_capacity, data, dist_matrix)
                        #if feasible:
                        path[idx_rotta] = nuova_rotta
                        costo_tot += delta
                        miglioramento = True
                        break 
                if miglioramento: break
            if miglioramento: break
            
    return path, costo_tot

#Swap neighbourhood 
def neigh_3(path, veichle_capacity, data, dist_matrix, costo_tot):

    miglioramento = True
    
    while miglioramento:

        miglioramento = False

        # Itero sulle rotte scambiando veicoli tra rotte diverse

        for r_idx in range(len(path)):
            
            rotta_1 = path[r_idx]

            # Controllo sulla lunghezza della rotta
            if len(rotta_1) <= 2:
                continue
            
            # Precalcolo costo originale fuori dai loop su i e j
            _, costo_originale = valida_rotta(rotta_1, veichle_capacity, data, dist_matrix)

            for i in range(1, len(rotta_1) - 1):
                for j in range(i+1, len(rotta_1)-1):
                    cliente_1 = rotta_1[i]
                    cliente_2 = rotta_1[j]

                    # Nodi adiacenti a cliente 1
                    p1, n1 = rotta_1[i-1], rotta_1[i+1]
                    # Nodi adiacenti a cliente 2
                    p2, n2 = rotta_1[j-1], rotta_1[j+1]

                    '''
                    if j-i >1:
                        costo_attuale = (dist_matrix[p1][cliente_1] + dist_matrix[cliente_1][n1] + 
                                         dist_matrix[p2][cliente_2] + dist_matrix[cliente_2][n2])
                        
                        costo_nuovo = (dist_matrix[p1][cliente_2] + dist_matrix[cliente_2][n1] + 
                                       dist_matrix[p2][cliente_1] + dist_matrix[cliente_2][n2])
                    
                    # Caso B: I nodi sono adiacenti (es: [0, C1, C2, 0])
                    else:
                        costo_attuale = (dist_matrix[p1][cliente_1] + dist_matrix[cliente_1][cliente_2] + dist_matrix[cliente_2][n2])
                        costo_nuovo = (dist_matrix[p1][cliente_2] + dist_matrix[cliente_2][cliente_1] + dist_matrix[cliente_1][n2])
                    '''
                    # Creiamo la rotta potenziale scambiando i nodi
                    nuova_rotta_test = list(rotta_1) # Copia veloce
                    nuova_rotta_test[i], nuova_rotta_test[j] = nuova_rotta_test[j], nuova_rotta_test[i]
                    ammissibile, costo_nuovo = valida_rotta(nuova_rotta_test, veichle_capacity, data, dist_matrix)
                    
                    if not ammissibile: continue

                    delta = costo_nuovo - costo_originale

                    # Se lo scambio riduce la distanza
                    if delta < -1e-6:
                        # Creiamo la rotta potenziale scambiando i nodi
                        #nuova_rotta_test = list(rotta_1) # Copia veloce
                        #nuova_rotta_test[i], nuova_rotta_test[j] = nuova_rotta_test[j], nuova_rotta_test[i]
                        
                        # Applichiamo il miglioramento alla rotta reale
                        path[r_idx] = nuova_rotta_test
                        costo_tot += delta
                        miglioramento = True
                        break # Esci dal ciclo j
                if miglioramento: break # Esci dal ciclo i
            if miglioramento: break # Esci dal ciclo idx_rotta
            
    return path, costo_tot

# Simulated annealing
def Sim_Annealing(path, costo_tot, veichle_capacity, dist_matrix, data):

    # Definizione dei parmetri
    T_init = 1000
    T_end = 0.1
    alpha = 0.99
    #alpha = rd.uniform(0.8, 0.99)

    # Passo la soluzione iniziale in s per non modificarla subito
    s = copy.deepcopy(path)
    costo_current = costo_tot

    # Inizializzo la soluzione migliore
    s_best = copy.deepcopy(path)
    costo_best = costo_tot

    # Controllo 
    contatore = 0
    mosse_accettate = 0
    mosse_feasible = 0
    miglioramenti = 0

    # Inizio ciclo
    while T_init > T_end:
        contatore +=1
        s_new = copy.deepcopy(s)
    
        # Scelgo la rotta da cui prendere solo tra quelle "attive" [solo veicoli utilizzati]
        rotte_attive = [idx for idx, rotta in enumerate(s) if len(rotta) > 2]
        if not rotte_attive:
            # Se non c'è nulla da spostare, raffredda e ricomincia il ciclo
            T_init *= alpha
            continue

        # Randomizzo la scelta del neighborhood [in questo caso solo insertion]
        # Rotta da cui prendo il cliente
        idx_partenza = rd.choice(rotte_attive)
        rotta_sorgente = s_new[idx_partenza]
        max_idx = len(rotta_sorgente)-2
        if max_idx < 1:
            T_init *= alpha
            continue

        # cliente da spostare
        idx_cliente = rd.randint(1, max_idx)
        cliente = rotta_sorgente.pop(idx_cliente)

        # Scelgo la destinazione 
        #if len(s_new) == 0:
        #   s_new.append([0, 0])

        idx_rotta_dest = rd.randint(0,len(s_new)-1)
        rotta_dest = s_new[idx_rotta_dest]
        #pos_ins = random.randint(1, len(rotta_dest)-1)
        max_ins = len(rotta_dest) - 1
        if max_ins < 1:
            idx_inserimento = 1 # Forza l'inserimento se la lista è troppo corta
        else:
            idx_inserimento = rd.randint(1, max_ins)

        # Inseriamo il cliente in una posizione a caso tra i depositi
        rotta_dest.insert(idx_inserimento, cliente)
        #Calcolo il costo della nuova rotta ma senza i vincoli
        costo_nuovo = sum(valida_rotta_senza_vincoli(r, dist_matrix, data) for r in s_new)
        #Calcolo del delta
        s_new, costo_nuovo = neigh_1(s_new, veichle_capacity, data, dist_matrix, costo_nuovo)
        delta = costo_nuovo - costo_current
        # Scelta del parametro u

        u = rd.random()

        # Controllo se accetto la mossa
        if delta<0 or math.exp(-delta/T_init)>u:
            #Se viene rispettata questa condizione accetto la mossa
            s = s_new
            costo_current = costo_nuovo
            mosse_accettate += 1
            
            # Se miglioro ed è feasible salvo la nuova rotta
            if all(valida_rotta(r, veichle_capacity, data, dist_matrix)[0] for r in s_new):
                costo_feasible = sum(valida_rotta(r, veichle_capacity, data, dist_matrix)[1] for r in s_new)
                mosse_feasible += 1
                if costo_feasible < costo_best:
                    s_best = copy.deepcopy(s_new)
                    costo_best = costo_feasible
                    miglioramenti += 1
            
        
        T_init = alpha * T_init

    print(f"Iterazioni totali:     {contatore}")
    print(f"Mosse accettate:       {mosse_accettate} ({100*mosse_accettate/contatore:.1f}%)")
    print(f"Di cui feasible:       {mosse_feasible} ({100*mosse_feasible/max(1,mosse_accettate):.1f}%)")
    print(f"Miglioramenti best:    {miglioramenti}")
    print(f"Costo iniziale:        {costo_tot:.1f}")
    print(f"Miglior costo feasible trovato: {costo_best:.1f}\n")
    return s_best, costo_best

# Grasp
def grasp1(path,costo_tot, veichle_capacity, veichle_quantity, dist_matrix, data, n_clienti):
    I_max = 1000
    alpha = 0.3
    # Passo la soluzione iniziale in s per non modificarla subito
    s = copy.deepcopy(path)
    costo_current = costo_tot

    # Inizializzo la soluzione migliore
    s_best = copy.deepcopy(path)
    costo_best = costo_tot

    for iter in range(I_max):
        visitati = np.zeros(n_clienti+1, dtype=bool)
        visitati[0] = True # Il deposito è il punto di partenza
    
        # Definisco la lista dei percorsi
        percorsi_totali = []
        # Inizializzo i costi
        costo_totale_global = 0
        # Controllo i clienti serviti
        clienti_serviti = 0
        while clienti_serviti < n_clienti and len(percorsi_totali) < veichle_quantity :

            percorso_attuale = [0]
            nodo_corrente = 0 # Indice intero
            capacita_residua = veichle_capacity
            tempo_attuale = 0 

            while True:
                clienti_feasible = []
                dist_minima = float('inf')
                orario_inizio_prossimo = 0
                for i in range(1, n_clienti+1):
                    if not visitati[i]:

                        tij = dist_matrix[nodo_corrente, i]
                         # Calcolo orario di inizio servizio
                        # tempo_attuale + service_time_corrente + tempo_viaggio
                        arrivo = max(tempo_attuale + data[nodo_corrente, 6] + tij, 
                                    data[i, 4])
                        if capacita_residua >= data[i, 3] and arrivo <= data[i, 5]:
                            clienti_feasible.append((i, tij, arrivo))
                    
                if not clienti_feasible:
                    percorso_attuale.append(0)
                    break
                    
                dmin = min(c[1] for c in clienti_feasible)
                dmax = max(c[1] for c in clienti_feasible)
                soglia = dmin + alpha * (dmax - dmin)
                best_vicini = [c for c in clienti_feasible if c[1] <= soglia]
                scelta = rd.choice(best_vicini)
                miglior_prossimo, distanza_scelta, arrivo_scelto = scelta
            
                # Aggiornamento stato
                visitati[miglior_prossimo] = True
                clienti_serviti += 1
                capacita_residua -= data[miglior_prossimo, 3]
                tempo_attuale = arrivo_scelto
                costo_totale_global += distanza_scelta
                percorso_attuale.append(miglior_prossimo)
                nodo_corrente = miglior_prossimo # Mantiene l'indice intero

            percorsi_totali.append(percorso_attuale)

        # Padding rotte vuote
        while len(percorsi_totali) < veichle_quantity:
            percorsi_totali.append([0, 0])
        
        costo_reale = sum(valida_rotta(r, veichle_capacity, data, dist_matrix)[1] 
                  for r in percorsi_totali)

        # Randomicamente seleziono un neighborhood da esplorare
        neigh = rd.randint(1,3)
        
        if(neigh == 1):
            s_prime, costo_nuovo = neigh_1(percorsi_totali, veichle_capacity, data, dist_matrix, costo_reale)
        elif(neigh == 2):
            s_prime, costo_nuovo = neigh_2(percorsi_totali, veichle_capacity, data, dist_matrix, costo_reale)
        elif(neigh==3):
            s_prime, costo_nuovo = neigh_3(percorsi_totali, veichle_capacity, data, dist_matrix, costo_reale)

        if costo_nuovo <= costo_best:
            s_best = copy.deepcopy(s_prime)
            costo_best = costo_nuovo
            
    return s_best, costo_best

# VNS
def vns(path, costo_tot, veichle_capacity, veichle_quantity, dist_matrix, data):

    # Soluzione iniziale da manipolare
    s = copy.deepcopy(path)
    costo_attuale = costo_tot
    
    # Soluzione migliore iniziale
    s_best = copy.deepcopy(s)
    costo_best = costo_attuale

    # Numero di neighborhood
    p = 3 

    # Massimo numero di iterazioni senza miglioramento
    max_no_improve = 100
    no_improve = 0

    while no_improve < max_no_improve:

        # Neighborhood da cui prendo la mossa
        k=1
        while k <= p:

            s_new = copy.deepcopy(s)
            rotte_attive = [idx for idx, rotta in enumerate(s_new) if len(rotta)>2]

            if not rotte_attive:
                k+=1
                continue

            # Mossa dal primo neighborhood
            if k == 1:
                
                idx_src = rd.choice(rotte_attive)
                rotta_src = s_new[idx_src]
                idx_cliente = rd.randint(1, len(rotta_src)-2)
                cliente = rotta_src.pop(idx_cliente)
                idx_dest = rd.randint(0, len(s_new)-1)
                rotta_dest = s_new[idx_dest]
                pos_ins = rd.randint(1, len(rotta_dest)-1)
                rotta_dest.insert(pos_ins, cliente)
                costo_nuovo = sum(valida_rotta_senza_vincoli(r, dist_matrix, data) for r in s_new)

            # Mossa dal seconda neighborhood
            if k == 2:
                
                rotte_valide = [idx for idx, rotta in enumerate(s_new) if len(rotta)>4]
                if not rotte_valide:
                    k+=1
                    continue
                idx_src = rd.choice(rotte_valide)
                rotta_src = s_new[idx_src]
                idx_seg = rd.randint(1, len(rotta_src) - 3)
                c1 = rotta_src.pop(idx_seg)
                c2 = rotta_src.pop(idx_seg)
                idx_dest = rd.randint(0, len(s_new) - 1)
                rotta_dest = s_new[idx_dest]
                idx_ins = rd.randint(1, max(1, len(rotta_dest) - 1))
                rotta_dest.insert(idx_ins, c1)
                rotta_dest.insert(idx_ins + 1, c2)
                costo_nuovo = sum(valida_rotta_senza_vincoli(r, dist_matrix, data) for r in s_new)

            # Mossa dal terzo neighborhood
            if k == 3:
                rotte_valide = [idx for idx in rotte_attive if len(s_new[idx]) >= 4]
                if not rotte_valide:
                    k += 1
                    continue
                idx_r = rd.choice(rotte_valide)
                rotta = s_new[idx_r]
                i = rd.randint(1, len(rotta) - 2)
                j = rd.randint(1, len(rotta) - 2)
                while j == i:
                    j = rd.randint(1, len(rotta) - 2)
                rotta[i], rotta[j] = rotta[j], rotta[i]
                costo_nuovo = sum(valida_rotta_senza_vincoli(r, dist_matrix, data) for r in s_new)

            # Local Search con k-esimo neigh
            if k == 1:
                s_local, costo_new = neigh_1(s_new, veichle_capacity, data, dist_matrix, costo_nuovo)
            # Altri casi
            if k == 2:
                s_local, costo_new = neigh_2(s_new, veichle_capacity, data, dist_matrix, costo_nuovo)
            if k == 3:
                s_local, costo_new = neigh_3(s_new, veichle_capacity, data, dist_matrix, costo_nuovo)


            if costo_new < costo_attuale:
                s = copy.deepcopy(s_local)
                costo_attuale = costo_new
                if all(valida_rotta(r, veichle_capacity, data, dist_matrix)[0] for r in s):
                    if costo_new < costo_best:
                        s_best = copy.deepcopy(s)
                        costo_best = costo_new
                    no_improve = 0
            else:
                k+=1
                no_improve +=1

    return s_best, costo_best

# Tabu Search
def Tabu_Search(path, costo_iniziale, veichle_capacity, data, dist_matrix):
    # --- 1. Parametri ---
    I_max = 3000
    d = 15
    tabu_list = {}

    s = copy.deepcopy(path)
    s_best = copy.deepcopy(s)
    costo_best = costo_iniziale
    costo_current = costo_iniziale

    for iterazione in range(I_max):
        best_delta = float('inf')
        best_move = None

        # Esplorazione intorno di 's'
        for r_src_idx, route_src in enumerate(s):
            if len(route_src) <= 2: continue 

            for i in range(1, len(route_src) - 1):
                cliente = route_src[i]
                
                for r_dest_idx, route_dest in enumerate(s):
                    # Definiamo il range di inserimento
                    range_j = len(route_dest)
                    
                    for j in range(1, range_j + 1):
                        if r_src_idx == r_dest_idx and (j == i or j == i + 1):
                            continue

                        # --- SIMULAZIONE ---
                        new_src = route_src[:i] + route_src[i+1:]
                        new_dest = route_dest[:j] + [cliente] + route_dest[j:]

                        # --- VALIDAZIONE (Ordine: rotta, data, matrice, capacità) ---
                        f1, c1 = valida_rotta(new_src, veichle_capacity, data, dist_matrix)
                        f2, c2 = valida_rotta(new_dest, veichle_capacity, data, dist_matrix)

                        # ENTRA QUI SOLO SE LA MOSSA È FEASIBLE
                        if f1 and f2:
                            # Calcoliamo i costi originali (sempre stesso ordine parametri!)
                            _, old_c1 = valida_rotta(route_src,veichle_capacity, data, dist_matrix )
                            _, old_c2 = valida_rotta(route_dest, veichle_capacity, data, dist_matrix)
                            
                            # ORA delta è sicuramente assegnato
                            delta = (c1 + c2) - (old_c1 + old_c2)
                            
                            mossa_id = (cliente, r_dest_idx)
                            is_tabu = tabu_list.get(mossa_id, 0) > iterazione
                            
                            # --- LOGICA DI SCELTA (DENTRO L'IF) ---
                            # Aspirazione: se batte il record assoluto, ignora il Tabu
                            if not is_tabu or (costo_current + delta < costo_best - 1e-9):
                                if delta < best_delta:
                                    best_delta = delta
                                    best_move = (r_src_idx, i, r_dest_idx, j, mossa_id)

        # --- ESECUZIONE DELLA MOSSA ---
        if best_move:
            r_s, pos_i, r_d, pos_j, m_id = best_move
            
            c_estratto = s[r_s].pop(pos_i)
            actual_j = pos_j if (r_s != r_d or pos_i > pos_j) else pos_j - 1
            s[r_d].insert(actual_j, c_estratto)
            
            costo_current += best_delta
            tabu_list[m_id] = iterazione + d
            
            if costo_current < costo_best - 1e-9:
                costo_best = costo_current
                s_best = copy.deepcopy(s)
               

    return s_best, costo_best

# Algoritmi genetici
def costo_soluzione(path, veichle_capacity, data, dist_matrix):
    return sum(valida_rotta(r, veichle_capacity, data, dist_matrix)[1] for r in path)

def verifica_completezza(path, n_clienti):
    clienti_presenti = set()
    for r in path:
        clienti_presenti.update(r[1:-1])
    return len(clienti_presenti) == n_clienti

def selezione_torneo(popolazione, k=3):
    torneo = rd.sample(popolazione, k)
    torneo.sort(key=lambda x: x[1])
    return torneo[0]

def costruzione_semi_greedy(n_clienti, veichle_quantity, v_cap, dati_nodi, costi, alpha=0.3):
    visitati = np.zeros(n_clienti + 1, dtype=bool)
    visitati[0] = True
    percorsi_totali = []
    clienti_serviti = 0

    while clienti_serviti < n_clienti and len(percorsi_totali) < veichle_quantity:
        percorso_attuale = [0]
        nodo_corrente = 0
        capacita_residua = v_cap
        tempo_attuale = 0

        while True:
            clienti_feasible = []
            for i in range(1, n_clienti + 1):
                if not visitati[i]:
                    tij = costi[nodo_corrente, i]
                    arrivo = max(tempo_attuale + dati_nodi[nodo_corrente, 6] + tij,
                                 dati_nodi[i, 4])
                    if capacita_residua >= dati_nodi[i, 3] and arrivo <= dati_nodi[i, 5]:
                        clienti_feasible.append((i, tij, arrivo))

            if not clienti_feasible:
                percorso_attuale.append(0)
                break

            dmin = min(c[1] for c in clienti_feasible)
            dmax = max(c[1] for c in clienti_feasible)
            soglia = dmin + alpha * (dmax - dmin)
            rcl = [c for c in clienti_feasible if c[1] <= soglia]
            scelto = rd.choice(rcl)
            miglior_prossimo, dist_scelta, arrivo_scelto = scelto

            visitati[miglior_prossimo] = True
            clienti_serviti += 1
            capacita_residua -= dati_nodi[miglior_prossimo, 3]
            tempo_attuale = arrivo_scelto
            percorso_attuale.append(miglior_prossimo)
            nodo_corrente = miglior_prossimo

        percorsi_totali.append(percorso_attuale)

    while len(percorsi_totali) < veichle_quantity:
        percorsi_totali.append([0, 0])

    if clienti_serviti < n_clienti:
        return None  # non è stato possibile servire tutti i clienti con questi vincoli

    costo = costo_soluzione(percorsi_totali, v_cap, dati_nodi, costi)
    return percorsi_totali, costo

def crea_popolazione_iniziale(pop_size, n_clienti, veichle_quantity, v_cap, data, dist_matrix):
    popolazione = []

    p_g1, c_g1 = greedy_1(n_clienti, veichle_quantity, v_cap, data, dist_matrix)
    p_g2, c_g2 = greedy_2(n_clienti, veichle_quantity, v_cap, data, dist_matrix)
    popolazione.append((p_g1, c_g1))
    popolazione.append((p_g2, c_g2))

    tentativi = 0
    max_tentativi = (pop_size - 2) * 20

    while len(popolazione) < pop_size and tentativi < max_tentativi:
        tentativi += 1
        alpha = rd.uniform(0.1, 0.5)
        risultato = costruzione_semi_greedy(n_clienti, veichle_quantity, v_cap,
                                             data, dist_matrix, alpha)
        if risultato is not None:
            popolazione.append(risultato)

    while len(popolazione) < pop_size:
        seme = rd.choice([(p_g1, c_g1), (p_g2, c_g2)])
        popolazione.append(copy.deepcopy(seme))

    return popolazione

def crossover_twopoints(parent1, parent2, v_cap, data, dist_matrix):
    p1 = copy.deepcopy(parent1)
    p2 = copy.deepcopy(parent2)
    
    n_rotte = len(p1)
    if n_rotte < 3:
        # Troppo poche rotte per avere due punti di taglio distinti e significativi
        costo = costo_soluzione(p1, v_cap, data, dist_matrix)
        return p1, costo
    
    # 1. Scelgo due punti di taglio distinti sull'indice delle rotte
    punto1, punto2 = sorted(rd.sample(range(1, n_rotte), 2))

    # 2. Costruisco il figlio
    figlio = copy.deepcopy(p1[: punto1]) + copy.deepcopy(p2[punto1:punto2]) + copy.deepcopy(p1[punto2:])

    # 3. Rimuovo i duplicati: un cliente che compare sia nel segmento "esterno" (da p1)
    #    sia nel segmento "centrale" (da p2) va tolto da uno dei due -> lo tolgo dagli esterni,
    #    dando priorità al segmento centrale appena innestato
    clienti_centrale = set()
    for r in figlio[punto1:punto2]:      
        clienti_centrale.update(r[1:-1])
    for idx in list(range(0, punto1)) + list(range(punto2, len(figlio))):
        figlio[idx] = [nodo for nodo in figlio[idx] if nodo == 0 or nodo not in clienti_centrale]

    # 4. Trovo i clienti mancanti 
    tutti_i_clienti = set()
    for r in p1:
        tutti_i_clienti.update(r[1:-1])

    clienti_presenti = set()
    for r in figlio:
        clienti_presenti.update(r[1:-1])

    clienti_mancanti = tutti_i_clienti - clienti_presenti

    # 5. Reinserisco i mancanti con cheapest insertion (stessa logica di greedy_2 fase 2)
    for cliente in clienti_mancanti:
        miglior_costo_extra = float('inf')
        miglior_rotta_idx = None
        miglior_pos = None

        for r_idx, rotta in enumerate(figlio):
            _, costo_attuale = valida_rotta(rotta, v_cap, data, dist_matrix)
            for pos in range(1, len(rotta)):
                nuova = rotta[:pos] + [cliente] + rotta[pos:]
                ok, nuovo_costo = valida_rotta(nuova, v_cap, data, dist_matrix)
                if not ok:
                    continue
                extra = nuovo_costo - costo_attuale
                if extra < miglior_costo_extra:
                    miglior_costo_extra = extra
                    miglior_rotta_idx = r_idx
                    miglior_pos = pos

        if miglior_rotta_idx is not None:
            figlio[miglior_rotta_idx].insert(miglior_pos, cliente)
        # se non è inseribile da nessuna parte, resta fuori:
        # la verifica di completezza nel ciclo principale gestirà il fallback

    costo_figlio = costo_soluzione(figlio, v_cap, data, dist_matrix)
    return figlio, costo_figlio

def mutazione(path, v_cap, data, dist_matrix, prob=0.15):
    if rd.random() > prob:
        return path

    figlio = copy.deepcopy(path)
    rotte_attive = [i for i, r in enumerate(figlio) if len(r) > 2]
    if not rotte_attive:
        return figlio

    tipo = rd.choice(['relocate', 'swap'])

    if tipo == 'relocate':
        idx_src = rd.choice(rotte_attive)
        rotta_src = figlio[idx_src]
        if len(rotta_src) <= 2:
            return figlio
        pos = rd.randint(1, len(rotta_src) - 2)
        cliente = rotta_src.pop(pos)
        idx_dest = rd.randint(0, len(figlio) - 1)
        rotta_dest = figlio[idx_dest]
        pos_ins = rd.randint(1, len(rotta_dest) - 1)
        rotta_dest.insert(pos_ins, cliente)

    else:  # swap
        idx_r1 = rd.choice(rotte_attive)
        idx_r2 = rd.choice(rotte_attive)
        r1 = figlio[idx_r1]
        r2 = figlio[idx_r2]
        if len(r1) <= 2 or len(r2) <= 2:
            return figlio
        pos1 = rd.randint(1, len(r1) - 2)
        pos2 = rd.randint(1, len(r2) - 2)
        r1[pos1], r2[pos2] = r2[pos2], r1[pos1]

    return figlio

def Memetic_Algorithm(n_clienti, veichle_quantity, v_cap, data, dist_matrix,
                       pop_size=30, generazioni=200, prob_mutazione=0.15,
                       tasso_local_search=1.0):

    popolazione = crea_popolazione_iniziale(pop_size, n_clienti, veichle_quantity,
                                             v_cap, data, dist_matrix)

    # local search anche sulla popolazione iniziale
    for idx in range(len(popolazione)):
        path, costo = popolazione[idx]
        neigh_scelto = rd.choice([neigh_1, neigh_2, neigh_3])
        path_ls, costo_ls = neigh_scelto(path, v_cap, data, dist_matrix, costo)
        popolazione[idx] = (path_ls, costo_ls)

    popolazione.sort(key=lambda x: x[1])
    miglior_soluzione = copy.deepcopy(popolazione[0][0])
    miglior_costo = popolazione[0][1]

    for gen in range(generazioni):
        nuova_popolazione = [copy.deepcopy(popolazione[0])]  

        while len(nuova_popolazione) < pop_size:
            p1, _ = selezione_torneo(popolazione, k=3)
            p2, _ = selezione_torneo(popolazione, k=3)

            figlio, costo_figlio = crossover_twopoints(p1, p2, v_cap, data, dist_matrix)

            # Fallback se il crossover non è riuscito a servire tutti i clienti
            if not verifica_completezza(figlio, n_clienti):
                figlio = copy.deepcopy(p1)
                costo_figlio = costo_soluzione(figlio, v_cap, data, dist_matrix)

            figlio_mut = mutazione(figlio, v_cap, data, dist_matrix, prob=prob_mutazione)
            if all(valida_rotta(r, v_cap, data, dist_matrix)[0] for r in figlio_mut):
                figlio = figlio_mut
                costo_figlio = costo_soluzione(figlio, v_cap, data, dist_matrix)

            if rd.random() < tasso_local_search:
                neigh_scelto = rd.choice([neigh_1, neigh_2, neigh_3])
                figlio, costo_figlio = neigh_scelto(figlio, v_cap, data, dist_matrix, costo_figlio)

            nuova_popolazione.append((figlio, costo_figlio))

        popolazione = nuova_popolazione
        popolazione.sort(key=lambda x: x[1])

        if popolazione[0][1] < miglior_costo:
            miglior_costo = popolazione[0][1]
            miglior_soluzione = copy.deepcopy(popolazione[0][0])

        if gen % 10 == 0:
            print(f"Generazione {gen}: miglior costo trovato = {miglior_costo:.1f}")

    return miglior_soluzione, miglior_costo