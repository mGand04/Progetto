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

def Tabu_Search(path, costo_iniziale, veichle_capacity, data, dist_matrix):

    # Parametri tabu search
    I_max = 1500
    d = 10

    # Conterrà le mosse tabu: Dizionario (cliente, mossa) | iterazione di scadenza
    tabu_list = {}

    # Inizializzazioni
    s_best = copy.deepcopy(path)
    s = s_best
    costo_best = costo_iniziale

    for iterazione in range(I_max):

        # Variabili per tracciare la miglior mossa non tabù
        best_delta = float('inf')
        best_move = None  #
        



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
        print("Greedy 1")
        percorsi, costo_tot = greedy_1(n_clienti, veichle_quantity, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_tot:.1f}")

        # Insertion applicato a greedy 1
        percorsi_local, costo_tot_local = neigh_1(copy.deepcopy(percorsi), veichle_capacity, data, dist_matrix, costo_tot)
        # Controllo costo totale nuove rotte
        check_costo = controllo_costo(percorsi_local, veichle_capacity, data, dist_matrix)
        print("\nLocal search 1: Eventuale cambio di rotta per singoli clienti")
        for idx, p in enumerate(percorsi_local):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_tot_local:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")

        # Or-opt-2 applicato a greedy 1
        percorsi_local_or, costo_tot_local_or = neigh_2(copy.deepcopy(percorsi),veichle_capacity,data,dist_matrix,costo_tot)
        # Controllo costo totale nuove rotte
        check_costo = controllo_costo(percorsi_local_or, veichle_capacity, data, dist_matrix)
        print("\nLocal search 2: Or-opt-2 scambio intra-rotta di blocchi di due clienti")
        for idx, p in enumerate(percorsi_local_or):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_tot_local_or:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")

        # Swap applicato a greedy 1
        percorsi_local3, costo_tot_local3 = neigh_3(copy.deepcopy(percorsi),veichle_capacity,data,dist_matrix,costo_tot)
        check_costo = controllo_costo(percorsi_local3, veichle_capacity, data, dist_matrix)
        print("\nLocal search 3: Swap di due clienti iter-rotta")
        for idx, p in enumerate(percorsi_local3):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_tot_local3:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")

        # Simulated annealing: Applico alla solzuione già migliorata con local search
        #1st neighborhood
        print("\nSimulated annealing applicato alla soluzione del primo neighborhood: ")
        percorsi_sim, costo_sim = Sim_Annealing(copy.deepcopy(percorsi_local), costo_tot_local, veichle_capacity, dist_matrix, data)
        check_costo = controllo_costo(percorsi_sim, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_sim):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_sim:.1f}")
        print(f"Costo Totale della solzuione controllato in seguito: {check_costo:.1f}")
        print("\n")
        # 2nd neighborhood
        print("\nSimulated annealing applicato alla soluzione del secondo neighborhood: ")
        percorsi_sim2, costo_sim2 = Sim_Annealing(copy.deepcopy(percorsi_local_or), costo_tot_local_or, veichle_capacity, dist_matrix, data)
        check_costo = controllo_costo(percorsi_sim2, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_sim2):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_sim2:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")
        print("\n")
        # 3rd neighborhood
        print("\nSimulated annealing applicato alla soluzione del secondo neighborhood: ")
        percorsi_sim3, costo_sim3 = Sim_Annealing(copy.deepcopy(percorsi_local3), costo_tot_local3, veichle_capacity, dist_matrix, data)
        check_costo = controllo_costo(percorsi_sim3, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_sim3):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_sim3:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f} ")

       # Approccio Greedy numero 2: "rivial solution o singleton solution"    
        print('\nGreedy 2: ') 
        percorsi_2, costo_tot_2 = greedy_2(n_clienti, veichle_quantity, veichle_capacity,data, dist_matrix)
        check_costo = controllo_costo(percorsi_2, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_2):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_tot_2:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")

        # Insertion applicato a greedy 2
        print("\nLocal search 1: Eventuale cambio di rotta per singoli clienti")
        percorsi_local2, costo_tot_local2 = neigh_1(copy.deepcopy(percorsi_2), veichle_capacity, data, dist_matrix, costo_tot_2)
        check_costo = controllo_costo(percorsi_local2, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_local2):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_tot_local2:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")
        
        # Or-opt-2 applicato a greedy 2
        print('\nLocal search 2: Or-opt-2 scambio intra-rotta di blocchi di due clienti')
        percorsi_local2or, costo_tot_local2or = neigh_2(copy.deepcopy(percorsi_2), veichle_capacity, data, dist_matrix, costo_tot_2)
        check_costo = controllo_costo(percorsi_local2or, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_local2or):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_tot_local2or:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")

        # Swap applicato a greedy 2
        print('\nLocal search 3: Swap di due clienti iter-rotta')
        percorsi_local_greedy2_3, costo_tot_local_greedy2_3 = neigh_3(copy.deepcopy(percorsi_2), veichle_capacity, data, dist_matrix, costo_tot_2)
        check_costo = controllo_costo(percorsi_local_greedy2_3, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_local_greedy2_3):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_tot_local_greedy2_3:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")

        # Simulated annealing: Applico alla solzuione già migliorata con local search
        #1st neighborhood
        print('\nSimulated annealing applicato alla soluzione del primo neighborhood: ')
        percorsi_sim2, costo_sim2 = Sim_Annealing(copy.deepcopy(percorsi_local2), costo_tot_local2, veichle_capacity, dist_matrix, data)
        check_costo = controllo_costo(percorsi_sim2, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_sim2):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_sim2:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")
        # 2nd neighborhood
        print('\nSimulated annealing applicato alla soluzione del secondo neighborhood: ')
        percorsi_sim2or, costo_sim2or = Sim_Annealing(copy.deepcopy(percorsi_local2or), costo_tot_local2or, veichle_capacity, dist_matrix, data)
        check_costo = controllo_costo(percorsi_sim2or, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_sim2or):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_sim2or:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")
        # 3rd neighborhood
        print('\nSimulated annealing applicato alla soluzione del primo neighborhood: ')
        percorsi_sim3, costo_sim3 = Sim_Annealing(copy.deepcopy(percorsi_local_greedy2_3), costo_tot_local_greedy2_3, veichle_capacity, dist_matrix, data)
        check_costo = controllo_costo(percorsi_sim3, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_sim3):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_sim3:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")

    except FileNotFoundError:
        print(f"ERRORE: Il file '{file_name}' non esiste nella cartella '{fold}'.")
        print("Controlla di aver scritto correttamente i nomi.")
    except Exception as e:
        print(f"ERRORE imprevisto: {e}")


if __name__ == "__main__":
    main()