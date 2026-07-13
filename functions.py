import os
import pandas as pd
import numpy as np
import random as rd
import math
import copy
import time

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

# Funzione utilizzata per validare la feasibility di una singola rotta: valida una rotta e ne calcola il costo reale.
def valida_rotta(percorso, veichle_capacity, data, dist_matrix):

    if len(percorso) < 2:
        return False, 0.0

    costo_totale = 0.0
    carico_attuale = 0.0
    tempo_attuale = 0.0
    
    COL_DEMAND  = 3
    COL_READY   = 4
    COL_DUE     = 5
    COL_SERVICE = 6

    # Ciclo su tutti gli archi della rotta
    for k in range(len(percorso) - 1):
        u = int(percorso[k])
        v = int(percorso[k+1])
        
        # 1. Calcolo distanza e aggiornamento costo
        distanza_uv = dist_matrix[u, v]
        costo_totale += distanza_uv
        
        # 2. Controllo Capacità (solo se v non è il deposito)
        if v != 0:
            carico_attuale += data[v, COL_DEMAND]
            if carico_attuale > veichle_capacity:
                return False, 0.0 
        
        # 3. Calcolo Arrivo e Inizio Servizio
        arrivo = tempo_attuale + data[u, COL_SERVICE] + distanza_uv
        inizio_servizio = max(arrivo, data[v, COL_READY])
        
        # 4. Controllo Finestra Temporale per il nodo v (cliente o deposito finale)
        if inizio_servizio > data[v, COL_DUE]:
            return False, 0.0 
            
        # Aggiorniamo il tempo per il prossimo arco
        tempo_attuale = inizio_servizio
        
    # Rotta feasible
    return True, round(costo_totale, 2)

# Funzione che permette di trovare i clienti vicini
def calcola_vicini(dist_matrix, k=10):
    n = dist_matrix.shape[0]  #Numero totale di nodi
    vicini = {}
    for i in range(n):
        # Ordina per distanza, escludi il nodo stesso
        distanze = [(j, dist_matrix[i,j]) for j in range(n) if j != i]
        distanze.sort(key=lambda x: x[1]) #Ordina dal valore più grande
        vicini[i] = [j for j, _ in distanze[:k]] #Estraggo i primi k più vicini
    return vicini

# Funzione utilizzata per validate rotte senza vincoli
def valida_rotta_senza_vincoli(percorso, dist_matrix, data):
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

# Funzione che controlla il costo totale di una soluzione
def controllo_costo(path, veichle_capacity, data, dist_matrix):
    check_costo = 0
    for idx, rotta in enumerate(path):
        _, c = valida_rotta(rotta, veichle_capacity, data, dist_matrix)
        check_costo += c
    
    return check_costo

# Funzione ausiliaria che controlla tutti i clienti siano serviti in una rotta
def soluzione_completa(percorsi, n_clienti):
    visti = [c for rotta in percorsi for c in rotta if c!=0]
    serviti = set(visti)
    mancanti = sorted(set(range(1, n_clienti + 1)) - serviti)
    duplicati = sorted({c for c in visti if visti.count(c) > 1})
    completa = (not mancanti and not duplicati)
    return completa, mancanti, duplicati

# Funzione ausiliaria che verifica quale inseremento sia il migliore se feasible
def _miglior_inserimento(rotta, cliente, cap, data, dist):
    _, costo_orig = valida_rotta(rotta, cap, data, dist)
    best_rotta, best_delta = None, float('inf')
    for pos in range(1, len(rotta)):
        cand = rotta[:pos] + [cliente] + rotta[pos:]
        ok, costo = valida_rotta(cand, cap, data, dist)
        if ok and (costo - costo_orig) < best_delta:
            best_delta, best_rotta = costo - costo_orig, cand
    return best_rotta, best_delta

# Funzione che ricolloca elementi delle rotte per ottenere una soluzione feasible completa
def _inserisci_con_ejection(rotte, c, cap, data, dist):
    
    for i, rotta in enumerate(rotte):
        for xi in range(1, len(rotta) - 1):
            x = rotta[xi] #Seleziono un cliente x dalla rotta corrente
            senza_x = rotta[:xi] + rotta[xi+1:] #Creo una versione senza di questo cliente x
            cand_c, _ = _miglior_inserimento(senza_x, c, cap, data, dist)
            if cand_c is None:
                continue
            for j in range(len(rotte)):
                base = cand_c if j == i else rotte[j]
                cand_x, _ = _miglior_inserimento(base, x, cap, data, dist)
                if cand_x is not None:
                    if j == i:
                        rotte[i] = cand_x
                    else:
                        rotte[i], rotte[j] = cand_c, cand_x
                    return True
    return False

# Funzione che elimina una singola rotte ridistribuendo i clienti nelle rotte rimanenti
def _elimina_rotta(percorsi, r_idx, cap, data, dist):
    clienti = [c for c in percorsi[r_idx] if c != 0]
    altre = [list(r) for j, r in enumerate(percorsi) if j != r_idx] #Copia di tutte le rotte rimanenti su cui verranno tentati i reinserimenti
    for c in clienti:
        best_j, best_rotta, best_delta = None, None, float('inf')
        for j, rotta in enumerate(altre):
            cand, delta = _miglior_inserimento(rotta, c, cap, data, dist)
            if cand is not None and delta < best_delta:
                best_j, best_rotta, best_delta = j, cand, delta
        if best_j is not None:
            altre[best_j] = best_rotta
        elif not _inserisci_con_ejection(altre, c, cap, data, dist):
            return False
    percorsi[:] = altre
    return True

# Primo costruttivo Greedy: Nearest neighborhood senza vincoli di veicoli + compattazione per mantenere la feasibility se possibile
def greedy_1(n_clienti, veichle_quantity, v_cap, dati_nodi, costi):
    visitati = np.zeros(n_clienti + 1, dtype=bool)
    visitati[0] = True
    percorsi_totali = []
    clienti_serviti = 0

    # COSTRUZIONE NEAREST-NEIGHBOR
    while clienti_serviti < n_clienti:
        percorso_attuale = [0]
        nodo_corrente = 0
        capacita_residua = v_cap
        tempo_attuale = 0

        while True:
            miglior_prossimo, distanza_minima, orario_inizio = None, float('inf'), 0
            for i in range(1, n_clienti + 1):
                if not visitati[i]:
                    t_ij = costi[nodo_corrente, i]
                    arrivo = max(tempo_attuale + dati_nodi[nodo_corrente, 6] + t_ij,
                                 dati_nodi[i, 4])
                    rientro = arrivo + dati_nodi[i, 6] + costi[i, 0]
                    if (capacita_residua >= dati_nodi[i, 3] and
                            arrivo <= dati_nodi[i, 5] and
                            rientro <= dati_nodi[0, 5]):
                        if t_ij < distanza_minima:
                            distanza_minima, miglior_prossimo, orario_inizio = t_ij, i, arrivo

            if miglior_prossimo is None:
                percorso_attuale.append(0) #Aggiungo zero al percorso
                break

            visitati[miglior_prossimo] = True
            clienti_serviti += 1
            capacita_residua -= dati_nodi[miglior_prossimo, 3]
            tempo_attuale = orario_inizio
            percorso_attuale.append(miglior_prossimo)
            nodo_corrente = miglior_prossimo

        if percorso_attuale == [0, 0]:
            break
        percorsi_totali.append(percorso_attuale)

    # COMPATTAZIONE: rientro nella flotta eliminando le rotte piu' piccole
    n_rotte_pre = len(percorsi_totali)
    while len(percorsi_totali) > veichle_quantity:
        ordinate = sorted(range(len(percorsi_totali)),
                          key=lambda i: len([c for c in percorsi_totali[i] if c != 0]))
        eliminata = False
        for idx in ordinate:
            if _elimina_rotta(percorsi_totali, idx, v_cap, dati_nodi, costi):
                eliminata = True
                break
        if not eliminata:
            break

    # Log di controllo
    serviti = {c for r in percorsi_totali for c in r if c != 0}
    non_serviti = sorted(set(range(1, n_clienti + 1)) - serviti)
    if non_serviti:
        print(f"[greedy_1] INCOMPLETA: fuori {len(non_serviti)}: {non_serviti}")
    elif len(percorsi_totali) > veichle_quantity:
        print(f"[greedy_1] tutti serviti ma {len(percorsi_totali)} rotte > {veichle_quantity} veicoli")
    else:
        print(f"[greedy_1] OK: {n_clienti} clienti in {len(percorsi_totali)} rotte "
              f"(NN ne aveva aperte {n_rotte_pre})")

    while len(percorsi_totali) < veichle_quantity:
        percorsi_totali.append([0, 0])# Se uso meno veicoli aggiungo rotte vuote per coerenza dimensionale

    costo_reale = sum(valida_rotta(r, v_cap, dati_nodi, costi)[1] for r in percorsi_totali)

    # controllo di sicurezza: ferma se una singola istanza sbaglia
    #assert len(serviti) == n_clienti and len([r for r in percorsi_totali if len(r) > 2]) <= veichle_quantity, \
    #    f"greedy_1 non valida: mancano {non_serviti}, rotte={len([r for r in percorsi_totali if len(r)>2])}"

    return percorsi_totali, costo_reale

# Secondo algoritmo greedy: un veicolo dedicato per ogni nodo (se possibile) + regret insertion
def greedy_2(n_clienti, veichle_quantity, v_cap, dati_nodi, costi):
    percorsi_totali = []
    costo_totale_global = 0.0
    clienti_in_attesa = []
    
    # FASE 1: rotte "singolette" finché ci sono veicoli
    candidati_seed = []
    for i in range(1, n_clienti + 1):
        domanda_cliente = dati_nodi[i, 3] 
        fine_finestra = dati_nodi[i, 5]    
        costo_andata = costi[0, i]  
        costo_ritorno = costi[i, 0] 
        
        tempo_arrivo = max(costo_andata, dati_nodi[i, 4]) 
        tempo_rientro_deposito = tempo_arrivo + dati_nodi[i, 6] + costo_ritorno
        
        if (domanda_cliente > v_cap or tempo_arrivo > fine_finestra or tempo_rientro_deposito > dati_nodi[0, 5]):
            continue

        candidati_seed.append(i)

    seed_indices = []
    if candidati_seed:
        primo = max(candidati_seed, key=lambda c: costi[0,c])# Scelgo il cliente più lontano dal deposito
        seed_indices.append(primo)

        while len(seed_indices) < veichle_quantity and len(seed_indices) < len(candidati_seed):
            migliore = max((c for c in candidati_seed if c not in seed_indices), key=lambda c: min(costi[c,s] for s in seed_indices))
            seed_indices.append(migliore)
    
    seed_set = set(seed_indices)
    # For che crea le rotte che partono dal deposito, servono e rientrano
    for i in seed_indices:
        costo_andata = costi[0,i]
        costo_ritorno = costi[i, 0]
        percorso_attuale = [0,i,0]
        percorsi_totali.append(percorso_attuale)
        costo_totale_global += (costo_andata + costo_ritorno)
    # Clienti che devono essere ancora inseriti
    for i in range(1, n_clienti + 1):
        if i not in seed_set:
            clienti_in_attesa.append(i)
    
    # Completo le rotte vuote se non ho abbastanza veicoli occupati dai seed
    while len(percorsi_totali) < veichle_quantity:
        percorsi_totali.append([0, 0])

    # FASE 2: Completo le rotte
    clienti_rimanenti = list(clienti_in_attesa)

    while clienti_rimanenti:
        miglior_regret = -float('inf')
        miglior_scelta = None

        for cliente in clienti_rimanenti:
            opzioni_inserimento = [] # Lista di tuple: (costo_extra, rotta_idx, pos)

            # Valutiamo tutte le possibili rotte e posizioni per il cliente attuale
            for r_idx, rotta in enumerate(percorsi_totali):
                costo_attuale = valida_rotta(rotta, v_cap, dati_nodi, costi)[1]
                for pos in range(1, len(rotta)):
                    nuova_rotta = rotta[:pos] + [cliente] + rotta[pos:]
                    check, nuovo_costo = valida_rotta(nuova_rotta, v_cap, dati_nodi, costi)
                    if check:
                        opzioni_inserimento.append((nuovo_costo - costo_attuale, r_idx, pos))

            if not opzioni_inserimento:
                # Il cliente non può entrare in nessuna rotta esistente
                continue

            # Ordiniamo le opzioni per questo cliente dalla più economica alla più costosa
            opzioni_inserimento.sort(key=lambda x: x[0])

            # Calcolo del REGRET: Differenza di costo tra la 2° migliore opzione e la 1° migliore, se ho solo un'opzione di inserimento inserisco subito quel cliente 
            if len(opzioni_inserimento) >= 2:
                regret = opzioni_inserimento[1][0] - opzioni_inserimento[0][0]
            else:
                # Regret infinito se ho solo un'opzione valida
                regret = float('inf')

            # Selezioniamo il cliente che ha il REGRET MASSIMO
            if regret > miglior_regret:
                miglior_regret = regret
                miglior_scelta = (cliente, opzioni_inserimento[0][1], opzioni_inserimento[0][2], opzioni_inserimento[0][0])

        if miglior_scelta is not None:
            cliente, r_idx, pos, costo_extra = miglior_scelta
            percorsi_totali[r_idx].insert(pos, cliente)
            costo_totale_global += costo_extra
            clienti_rimanenti.remove(cliente)
        else:
            for c in clienti_rimanenti:
                print(f"Cliente {int(dati_nodi[c, 0])} non inseribile in nessuna rotta esistente.")
            break

    costo_reale = sum(valida_rotta(r, v_cap, dati_nodi, costi)[1] 
                  for r in percorsi_totali)
    return percorsi_totali, round(costo_reale, 1)

#1 Neighborhood: Insertion --> provo a togliere un cliente da un path e lo inserisco in un'altra con FIRST IMPROVMENT
def neigh_1(path, veichle_capacity, data, dist_matrix, costo_tot, vicini=None):
    costo_attuale = sum(valida_rotta(r, veichle_capacity, data, dist_matrix)[1] for r in path)
    miglioramento = True

    # Calcolo i vicini dei vari clienti, questo rende l'esecuzione più leggera 
    if vicini is None:
        vicini = calcola_vicini(dist_matrix, k=10)
    # Questi mi dicono dove si trova ogni cliente e quanto è carico ogni camion
    client_to_route = {cliente: r_idx
                        for r_idx, rotta in enumerate(path)
                        for cliente in rotta}

    capacita_rotte = {idx: sum(data[c, 3] for c in rotta if c != 0)
                       for idx, rotta in enumerate(path)}

    while miglioramento:
        miglioramento = False

        costi_rotte = {}
        for idx, rotta in enumerate(path):
            _, c = valida_rotta(rotta, veichle_capacity, data, dist_matrix)
            costi_rotte[idx] = c

        costo_reale = sum(costi_rotte.values())

        # Controllo di sicurezza rispetto al costo passato e quello rialcolato 
        assert abs(costo_attuale - costo_reale) < 0.1, \
            f"DIVERGENZA: costo_attuale={costo_attuale:.1f}, costo_reale={costo_reale:.1f}"

        for r1_idx in range(len(path)):
            rotta_src = path[r1_idx]

            if len(rotta_src) <= 2:
                continue

            for idx_pos in range(1, len(rotta_src) - 1):
                cliente = rotta_src[idx_pos]

                nuova_rotta_src = rotta_src[:]
                nuova_rotta_src.pop(idx_pos)
                #Controllo se senza quel cliente la rotta è ancora valida
                ok_src, costo_src = valida_rotta(nuova_rotta_src, veichle_capacity, data, dist_matrix)
                if not ok_src:
                    continue
                #Allora cerco una nuova rotta di destinazione tra quelle dei suoi vicini
                rotte_candidate = set()
                for vicino in vicini[cliente]:
                    if vicino in client_to_route:
                        rotte_candidate.add(client_to_route[vicino])
                rotte_candidate.add(r1_idx)

                for r2_idx in rotte_candidate:
                    rotta_dest = path[r2_idx]
                    domanda_cliente = data[cliente, 3]
                    #Rotta diversa da quella sorgente
                    if r1_idx != r2_idx:
                        if capacita_rotte[r2_idx] + domanda_cliente > veichle_capacity:
                            continue

                    for pos in range(1, len(rotta_dest)):
                        if r1_idx == r2_idx and (pos == idx_pos or pos == idx_pos + 1):
                            continue

                        if r1_idx == r2_idx:
                            #Nuova rotta src ha un elemento in meno e tutti gli indici 
                            # dopo sono spostati di uno rispetto alla rotta originale
                            nuova_rotta_dest = nuova_rotta_src[:]
                            adj_pos = pos - 1 if pos > idx_pos else pos
                            nuova_rotta_dest.insert(adj_pos, cliente)
                        else:
                            nuova_rotta_dest = rotta_dest[:]
                            nuova_rotta_dest.insert(pos, cliente)

                        ok_dest, costo_dest = valida_rotta(nuova_rotta_dest, veichle_capacity, data, dist_matrix)
                        if not ok_dest:
                            continue

                        if r1_idx == r2_idx:
                            nuovo_costo_tot = costo_attuale - costi_rotte[r1_idx] + costo_dest
                        else:
                            nuovo_costo_tot = (costo_attuale - costi_rotte[r1_idx] - costi_rotte[r2_idx]
                                                + costo_src + costo_dest)
                        #Se migliora il costo anche di 0.01 (First Improvement)
                        if nuovo_costo_tot < costo_attuale - 0.01:
                            path[r1_idx] = nuova_rotta_src
                            path[r2_idx] = nuova_rotta_dest
                            costo_attuale = nuovo_costo_tot
                            client_to_route = {c: r_idx
                                                for r_idx, rotta in enumerate(path)
                                                for c in rotta}
                            capacita_rotte = {idx: sum(data[c, 3] for c in rotta if c != 0)
                                               for idx, rotta in enumerate(path)}
                            miglioramento = True
                            break
                    if miglioramento:
                        break
                if miglioramento:
                    break
            if miglioramento:
                break

    return path, costo_attuale

# 2 Neighborhood: Or-opt-2, scambi di segmenti intrarotta
def neigh_2(path, veichle_capacity, data, dist_matrix, costo_tot):
    miglioramento = True

    while miglioramento:
        miglioramento = False
        # Iteriamo sulle rotte (solo intrarotta)
        for idx_rotta in range(len(path)):
            rotta_src = path[idx_rotta]
            # Serve una rotta con almeno 5 nodi per spostare 2 clienti:
            if len(rotta_src) < 5:
                continue

            # Calcoliamo il costo iniziale solo una volta all'inizio
            _, costo_iniziale = valida_rotta(rotta_src, veichle_capacity, data, dist_matrix)

            # i è l'indice di inizio del blocco di 2 clienti da spostare
            for i in range(1, len(rotta_src) - 2):
                c1 = rotta_src[i]
                c2 = rotta_src[i + 1]

                # Creiamo la rotta senza i due clienti
                rotta_ridotta = rotta_src[:i] + rotta_src[i + 2:]

                # Proviamo a inserire il blocco [c1, c2] in ogni posizione k della rotta ridotta
                for k in range(1, len(rotta_ridotta)):
                    nuova_rotta = rotta_ridotta[:k] + [c1, c2] + rotta_ridotta[k:]

                    feasible, costo_nuovo = valida_rotta(nuova_rotta, veichle_capacity, data, dist_matrix)
                    if not feasible:
                        continue

                    delta = costo_nuovo - costo_iniziale
                    if delta < -1e-6:
                        path[idx_rotta] = nuova_rotta
                        costo_tot += delta
                        miglioramento = True
                        break

                if miglioramento:
                    break
            if miglioramento:
                break

    return path, costo_tot

# 3 Neighborhood: Swap di due clienti intrarotta
def neigh_3(path, veichle_capacity, data, dist_matrix, costo_tot):

    miglioramento = True
    
    while miglioramento:

        miglioramento = False

        # Itero sulle rotte scambiando veicoli nella stessa rotta
        for r_idx in range(len(path)):
            
            rotta_1 = path[r_idx]

            # Controllo sulla lunghezza della rotta
            if len(rotta_1) <= 2:
                continue
            
            # Precalcolo costo originale fuori dai loop su i e j
            _, costo_originale = valida_rotta(rotta_1, veichle_capacity, data, dist_matrix)
            #Ciclo su tutte le coppie di posizioni interne alla rotta
            for i in range(1, len(rotta_1) - 1):
                for j in range(i+1, len(rotta_1)-1):
                    #Clienti candidati allo scambio
                    cliente_1 = rotta_1[i]
                    cliente_2 = rotta_1[j]

                    # Nodi adiacenti a cliente 1
                    p1, n1 = rotta_1[i-1], rotta_1[i+1]
                    # Nodi adiacenti a cliente 2
                    p2, n2 = rotta_1[j-1], rotta_1[j+1]

                    # Creiamo la rotta potenziale scambiando i nodi
                    nuova_rotta_test = list(rotta_1) # Copia veloce
                    nuova_rotta_test[i], nuova_rotta_test[j] = nuova_rotta_test[j], nuova_rotta_test[i]
                    ammissibile, costo_nuovo = valida_rotta(nuova_rotta_test, veichle_capacity, data, dist_matrix)
                    
                    if not ammissibile: continue

                    delta = costo_nuovo - costo_originale

                    # Se lo scambio riduce la distanza
                    if delta < -1e-6:   
                        # Applichiamo il miglioramento alla rotta reale
                        path[r_idx] = nuova_rotta_test
                        costo_tot += delta
                        miglioramento = True
                        break 
                if miglioramento: break 
            if miglioramento: break
            
    return path, costo_tot

# Simulated annealing
def Sim_Annealing(path, costo_tot, veichle_capacity, dist_matrix, data):

    # Definizione dei parmetri
    T_init = 1000
    T_end = 0.1
    alpha = 0.99

    # Passiamo la soluzione iniziale in s
    s = copy.deepcopy(path)
    costo_current = costo_tot

    # Inizializzo la soluzione migliore (quella attuale)
    s_best = copy.deepcopy(path)
    costo_best = costo_tot

    # Variabili di controllo 
    contatore = 0
    mosse_accettate = 0
    mosse_feasible = 0
    miglioramenti = 0

    # Inizio ciclo
    while T_init > T_end:
        contatore +=1
        s_new = copy.deepcopy(s)
    
        # Scelgo la rotta da cui prendere solo tra quelle attive
        rotte_attive = [idx for idx, rotta in enumerate(s) if len(rotta) > 2]
        if not rotte_attive:
            # Se non c'è nulla da spostare ricominciamo il ciclo
            T_init *= alpha
            continue

        # Randomizziamo la scelta dell'insertion
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

        # Scegliamo la destinazione sempre ranomd
        idx_rotta_dest = rd.randint(0,len(s_new)-1)
        rotta_dest = s_new[idx_rotta_dest]
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

    # Log di controllo
    print(f"Iterazioni totali:     {contatore}")
    print(f"Mosse accettate:       {mosse_accettate} ({100*mosse_accettate/contatore:.1f}%)")
    print(f"Di cui feasible:       {mosse_feasible} ({100*mosse_feasible/max(1,mosse_accettate):.1f}%)")
    print(f"Miglioramenti best:    {miglioramenti}")
    print(f"Costo iniziale:        {costo_tot:.1f}")
    print(f"Miglior costo feasible trovato: {costo_best:.1f}\n")

    return s_best, costo_best

# GRASP          
def grasp1(path, costo_tot, veichle_capacity, veichle_quantity, dist_matrix, data, n_clienti):
    # Parametri
    I_max = 1000
    alpha = 0.3
    tempo_limite = 60
    max_no_improve = 150

    # Soluzioni best passata inizialmente
    s_best = copy.deepcopy(path)
    costo_best = costo_tot
    tempo_inizio = time.time()
    no_improve = 0
    iterazione = 0

    # Inizio ciclo
    while iterazione < I_max and no_improve < max_no_improve:
        if time.time() - tempo_inizio > tempo_limite:
            break

        # Costruzione semi-greedy della nuova soluzione 
        visitati = np.zeros(n_clienti + 1, dtype=bool)
        visitati[0] = True
        percorsi_totali = []
        clienti_serviti = 0
        #Finchè non terminano i clienti oppure i veicoli disponibili
        while clienti_serviti < n_clienti and len(percorsi_totali) < veichle_quantity:
            percorso_attuale = [0]
            nodo_corrente = 0
            capacita_residua = veichle_capacity
            tempo_attuale = 0

            while True:
                clienti_feasible = []
                for i in range(1, n_clienti + 1):
                    if not visitati[i]:
                        tij = dist_matrix[nodo_corrente, i]
                        arrivo = max(tempo_attuale + data[nodo_corrente, 6] + tij,
                                     data[i, 4])
                        #Verifica che si può rientrare al deposito entro la chiusura globale
                        tempo_rientro_deposito = arrivo + data[i, 6] + dist_matrix[i, 0]

                        # controllo sul rientro al deposito
                        #Evito di costruire rotte che si bloccano
                        if (capacita_residua >= data[i, 3] and
                                arrivo <= data[i, 5] and
                                tempo_rientro_deposito <= data[0, 5]):
                            clienti_feasible.append((i, tij, arrivo))

                if not clienti_feasible:
                    percorso_attuale.append(0)
                    break
                
                # Scelta dell'inserimento
                #Creo una lista di clienti candidati la cui distanza è entro una soglia
                #prop all'intervallo [dmin,dmax]
                dmin = min(c[1] for c in clienti_feasible)
                dmax = max(c[1] for c in clienti_feasible)
                soglia = dmin + alpha * (dmax - dmin)
                best_vicini = [c for c in clienti_feasible if c[1] <= soglia]
                #Scelgo casualmente tra i candidati
                scelta = rd.choice(best_vicini)
                miglior_prossimo, distanza_scelta, arrivo_scelto = scelta

                visitati[miglior_prossimo] = True
                clienti_serviti += 1
                capacita_residua -= data[miglior_prossimo, 3]
                tempo_attuale = arrivo_scelto
                percorso_attuale.append(miglior_prossimo)
                nodo_corrente = miglior_prossimo

            if percorso_attuale == [0, 0]:
                break

            percorsi_totali.append(percorso_attuale)

        # FASE DI REPAIR: reinserisco i clienti rimasti fuori
        clienti_non_serviti = [i for i in range(1, n_clienti + 1) if not visitati[i]]

        for cliente in clienti_non_serviti:
            migliore_delta = float('inf')
            migliore_idx = None
            migliore_rotta = None

            for idx, rotta in enumerate(percorsi_totali):
                _, costo_originale = valida_rotta(rotta, veichle_capacity, data, dist_matrix)
                for k in range(1, len(rotta)):
                    rotta_candidata = rotta[:k] + [cliente] + rotta[k:]
                    feasible, costo_candidato = valida_rotta(
                        rotta_candidata, veichle_capacity, data, dist_matrix
                    )
                    if not feasible:
                        continue
                    delta = costo_candidato - costo_originale
                    if delta < migliore_delta:
                        migliore_delta = delta
                        migliore_idx = idx
                        migliore_rotta = rotta_candidata

            if migliore_idx is None and len(percorsi_totali) < veichle_quantity:
                rotta_nuova = [0, cliente, 0]
                feasible, _ = valida_rotta(rotta_nuova, veichle_capacity, data, dist_matrix)
                if feasible:
                    percorsi_totali.append(rotta_nuova)
                    visitati[cliente] = True
                    clienti_serviti += 1
                continue

            if migliore_idx is not None:
                percorsi_totali[migliore_idx] = migliore_rotta
                visitati[cliente] = True
                clienti_serviti += 1

        while len(percorsi_totali) < veichle_quantity:
            percorsi_totali.append([0, 0])

        costo_reale = sum(valida_rotta(r, veichle_capacity, data, dist_matrix)[1]
                           for r in percorsi_totali)

        # Local search finale solo su neigh_1
        s_prime, costo_nuovo = neigh_1(percorsi_totali, veichle_capacity, data, dist_matrix, costo_reale)
        completa, mancanti, duplicati = soluzione_completa(s_prime, n_clienti)

        # Evitiamo di ritornarne soluzione incomplete se non risuciamo a inserire tutti i mancanti
        if not completa:
            # Log di controllo
            print(f"[GRASP iter {iterazione}] scartata - mancanti={mancanti} duplicati={duplicati}")
            no_improve += 1
            iterazione += 1
            continue   

        # Aggiornamento
        if costo_nuovo < costo_best - 1e-9:
            s_best = copy.deepcopy(s_prime)
            costo_best = costo_nuovo
            no_improve = 0
        else:
            no_improve += 1

        iterazione += 1

    return s_best, costo_best

# VNS
def vns(path, costo_tot, veichle_capacity, veichle_quantity, dist_matrix, data):

    # Soluzione iniziale da manipolare
    s = copy.deepcopy(path)
    costo_attuale = costo_tot
    
    # Soluzione migliore iniziale
    s_best = copy.deepcopy(s)
    costo_best = costo_attuale  
    p = 3     # Numero di neighborhood
    max_no_improve = 100  # Massimo numero di iterazioni senza miglioramento
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
            #Prendo un cliente a caso da una rotta e lo metto in un'altra a caso
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
                #Richiede rotte con almeno 5 elementi
                rotte_valide = [idx for idx, rotta in enumerate(s_new) if len(rotta)>4]
                if not rotte_valide:
                    k+=1
                    continue
                #Estrae una coppia di clienti consecutivi e poi li reinserisce altrove
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
            #Sceglie due posizioni diverse nella stessa rotta e scambia i clienti
            # Sempre randomicamente
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
    # Parametri
    I_max=3000
    tempo_limite=60
    d = 15                 # numero di mosse vietate
    coeff_div = 0.5        # intensità della diversificazione long-term
    tabu_list = {}         # short term memory
    frequenza = {}         # long-term memory

    s = copy.deepcopy(path)
    s_best = copy.deepcopy(s)
    costo_best = costo_iniziale
    costo_current = costo_iniziale

    # Precalcolo dei vicini spaziali (dipende solo da dist_matrix, invariante durante l'esecuzione)
    vicini = calcola_vicini(dist_matrix, k=10)

    # Scala della penalità: costo medio per cliente
    n_clienti = sum(len(r) - 2 for r in s)
    scala = costo_iniziale / max(n_clienti, 1)

    tempo_inizio = time.time()

    for iterazione in range(I_max):
        # Budget di tempo: interrompe l'esecuzione se supera il limite
        if time.time() - tempo_inizio > tempo_limite:
            break

        best_delta = float('inf')
        best_delta_valutato = float('inf')
        best_move = None

        # Mappa cliente -> indice rotta, ricalcolata una volta per iterazione
        # (serve per restringere le rotte candidate ai vicini spaziali)
        client_to_route = {c: idx for idx, r in enumerate(s) for c in r[1:-1]}

        # Esplorazione intorno di 's'
        #Considero solo le rotte che contengono i suoi vicini spaziali
        for r_src_idx, route_src in enumerate(s):
            if len(route_src) <= 2:
                continue

            _, old_c1 = valida_rotta(route_src, veichle_capacity, data, dist_matrix)

            for i in range(1, len(route_src) - 1):
                cliente = route_src[i]

                rotte_candidate = {client_to_route[v] for v in vicini[cliente] if v in client_to_route}
                rotte_candidate.add(r_src_idx)  # includo sempre la rotta di origine (riposizionamento interno)

                for r_dest_idx in rotte_candidate:
                    route_dest = s[r_dest_idx]

                    _, old_c2 = valida_rotta(route_dest, veichle_capacity, data, dist_matrix)
                    range_j = len(route_dest)

                    for j in range(1, range_j):
                        #se la posizione è uguale continua
                        if r_src_idx == r_dest_idx and (j == i or j == i + 1):
                            continue

                        # SIMULAZIONE DELLA MOSSA
                        new_src = route_src[:i] + route_src[i+1:]
                        new_dest = route_dest[:j] + [cliente] + route_dest[j:]

                        # VALIDAZIONE
                        f1, c1 = valida_rotta(new_src, veichle_capacity, data, dist_matrix)
                        f2, c2 = valida_rotta(new_dest, veichle_capacity, data, dist_matrix)

                        if f1 and f2:
                            delta = (c1 + c2) - (old_c1 + old_c2)
                            #Coppia tabu
                            mossa_id = (cliente, r_dest_idx)
                            #se maggiore di iterazione allora la mossa è ancora vietata
                            is_tabu = tabu_list.get(mossa_id, 0) > iterazione

                            # LONG-TERM: penalizza solo le mosse NON migliorative proporzionalmente a quante volte sono già state eseguite
                            if delta >= 0:
                                penalita = coeff_div * frequenza.get(mossa_id, 0) * scala
                            else:
                                penalita = 0.0
                            delta_valutato = delta + penalita

                            # Aspirazione: se batte il record assoluto, ignora il Tabu
                            if not is_tabu or (costo_current + delta < costo_best - 1e-9):
                                if delta_valutato < best_delta_valutato:
                                    best_delta_valutato = delta_valutato
                                    best_delta = delta
                                    best_move = (r_src_idx, i, r_dest_idx, j, mossa_id)

        # Esecuzione della mossa
        if best_move:
            #Estrazione mossa migliore
            r_s, pos_i, r_d, pos_j, m_id = best_move
            #rimuovo il cliente dalla rotta sorgente
            c_estratto = s[r_s].pop(pos_i)
            actual_j = pos_j if (r_s != r_d or pos_i > pos_j) else pos_j - 1
            s[r_d].insert(actual_j, c_estratto)

            costo_current += best_delta
            tabu_list[m_id] = iterazione + d
            frequenza[m_id] = frequenza.get(m_id, 0) + 1 

            if costo_current < costo_best - 1e-9:
                costo_best = costo_current
                s_best = copy.deepcopy(s)

    return s_best, costo_best

# Algoritmo genetico
# Costo totale del path
def costo_soluzione(path, veichle_capacity, data, dist_matrix):
    tot = 0.0
    for r in path:
        ok, c = valida_rotta(r, veichle_capacity, data, dist_matrix)
        if not ok:
            return float('inf')     
        tot += c
    return tot

# Funzione asuliaria che verifica la completezza della soluzione
#Ovvero verifica che tutti i clienti siano presenti una sola volta
def verifica_completezza(path, n_clienti):
    clienti_presenti = set()
    for r in path:
        clienti_presenti.update(r[1:-1])
    return len(clienti_presenti) == n_clienti

# Funzione ausiliaria per la selezione dei parent che vincono il torneo
# Vengono ordinati per costo, ritorna il migliore dei tre
def selezione_torneo(popolazione, k=3):
    torneo = rd.sample(popolazione, k)
    torneo.sort(key=lambda x: x[1])
    return torneo[0]

# Costruzione semi-greedy per popolare l'algoritmo
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
                    tempo_rientro = arrivo + dati_nodi[i, 6] + costi[i, 0]

                    if capacita_residua >= dati_nodi[i, 3] and arrivo <= dati_nodi[i, 5] and tempo_rientro <= dati_nodi[0,5]:
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

# Funzione che genera la popolazione iniziale
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

# Crossover per la generazione del filgio dai due parent scelti
def crossover_twopoints(parent1, parent2, v_cap, data, dist_matrix):
    p1 = copy.deepcopy(parent1)
    p2 = copy.deepcopy(parent2)
    
    n_rotte = len(p1)
    if n_rotte < 3:
        # Troppo poche rotte per avere due punti di taglio distinti e significativi
        costo = costo_soluzione(p1, v_cap, data, dist_matrix)
        return p1, costo
    
    # 1 Scelgo due punti di taglio distinti sull'indice delle rotte
    punto1, punto2 = sorted(rd.sample(range(1, n_rotte), 2))

    # 2. Costruisco il figlio
    figlio = copy.deepcopy(p1[: punto1]) + copy.deepcopy(p2[punto1:punto2]) + copy.deepcopy(p1[punto2:])

    # 3 Rimuovo i duplicati
    clienti_centrale = set()
    for r in figlio[punto1:punto2]:      
        clienti_centrale.update(r[1:-1])
    for idx in list(range(0, punto1)) + list(range(punto2, len(figlio))):
        figlio[idx] = [nodo for nodo in figlio[idx] if nodo == 0 or nodo not in clienti_centrale]

    # 4 Trovo i clienti mancanti 
    tutti_i_clienti = set()
    for r in p1:
        tutti_i_clienti.update(r[1:-1])

    clienti_presenti = set()
    for r in figlio:
        clienti_presenti.update(r[1:-1])

    clienti_mancanti = tutti_i_clienti - clienti_presenti

    # 5 Reinserisco i mancanti con cheapest insertion
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

# Mutazione del filgio 
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

# Struttura generale dell'algoritmo completo
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

        # Il migliore sopravvive intatto
        nuova_popolazione = [copy.deepcopy(popolazione[0])]  

        # Nuova popolazione
        while len(nuova_popolazione) < pop_size:
            # Selezione dei parent
            p1, _ = selezione_torneo(popolazione, k=3)
            p2, _ = selezione_torneo(popolazione, k=3)

            # Crossover
            figlio, costo_figlio = crossover_twopoints(p1, p2, v_cap, data, dist_matrix)

            # Fallback se il crossover non è riuscito a servire tutti i clienti
            if not verifica_completezza(figlio, n_clienti):
                figlio = copy.deepcopy(p1)
                costo_figlio = costo_soluzione(figlio, v_cap, data, dist_matrix)

            # Mutazione
            figlio_mut = mutazione(figlio, v_cap, data, dist_matrix, prob=prob_mutazione)
            # Controllo feasibility
            if all(valida_rotta(r, v_cap, data, dist_matrix)[0] for r in figlio_mut):
                figlio = figlio_mut
                costo_figlio = costo_soluzione(figlio, v_cap, data, dist_matrix)

            # Local search
            if rd.random() < tasso_local_search:
                neigh_scelto = rd.choice([neigh_1, neigh_2, neigh_3])
                figlio, costo_figlio = neigh_scelto(figlio, v_cap, data, dist_matrix, costo_figlio)

            nuova_popolazione.append((figlio, costo_figlio))

        popolazione = nuova_popolazione
        popolazione.sort(key=lambda x: x[1])

        if popolazione[0][1] < miglior_costo:
            miglior_costo = popolazione[0][1]
            miglior_soluzione = copy.deepcopy(popolazione[0][0])

        # Log di controllo
        if gen % 10 == 0:
            print(f"Generazione {gen}: miglior costo trovato = {miglior_costo:.1f}")

    return miglior_soluzione, miglior_costo