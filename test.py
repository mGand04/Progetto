import os
import time
import copy
import numpy as np
import pandas as pd
import random as rd

# Importazione delle tue funzioni esistenti
from functions import (
    matrice_distanze, valida_rotta, calcola_vicini, valida_rotta_senza_vincoli,
    controllo_costo, greedy_1, greedy_2, neigh_1, neigh_2, neigh_3,
    Sim_Annealing, Tabu_Search, grasp1, vns, Memetic_Algorithm
)
def run_benchmark():
    # Leggo i dati in formatoo int e lascio una precisione di due cifre decimali
    np.set_printoptions(suppress=True, precision=2)
    # Percorso del file 
    #path_base = r'C:\Users\safet\OneDrive\Desktop\Progetto\Progetto\Istanze'
    path_base = r'C:\Users\mgand\OneDrive\Desktop\Ottimizzazzione_sr\Progetto\Istanze'
    FOLDERS = ['n25', 'n50', 'n100'] 

    # Numero di run per ogni istanza
    n_run = input("Inserisci il numero di run per ogni istanza: ")

    # Liste per archiviare i dati
    raw_runs_data = []      # Dati generali della run (Tempi totali della run e best costi della run)
    raw_methods_data = []   # Dati specifici di ogni singolo metodo dentro le run

    print("Inizio del ciclo di test...\n")

    for fold in FOLDERS:
        folder_path = os.path.join(path_base, fold)
        if not os.path.exists(folder_path):
            print(f"Cartella non trovata: {folder_path}, salto.")
            continue

        # Prendo tutti i file di testo nella cartella
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            print(f"\nElaborazione: {fold} / {file_name}")

            try:
                veichle_info = np.genfromtxt(file_path, skip_header=4, max_rows=1)
                veichle_quantity = int(veichle_info[0])
                veichle_capacity = veichle_info[1]

                data = np.genfromtxt(file_path, skip_header=9)
                cols = {'ID': 0, 'X': 1, 'Y': 2, 'DEM': 3, 'READY': 4, 'DUE': 5, 'SERV': 6}
                demand = data[:, 3]
                n_clienti = len(data)-1
                dist_matrix = matrice_distanze(data)

                for run in range(1, n_run + 1):
                    print(f"  -> Run {run}/{n_run}")

                    # Dizionario contenete i risultati della run corrente
                    run_results = {}

                    start_run_time = time.time()

                    # Greedy 1
                    t0 = time.time()
                    p_g1, c_g1 = greedy_1(n_clienti, veichle_quantity, veichle_capacity, data, dist_matrix)
                    run_results['Greedy_1'] = (c_g1, time.time() - t0)

                    # Local search su Greedy 1
                    t0 = time.time()
                    p_l1, c_l1 = neigh_1(copy.deepcopy(p_g1), veichle_capacity, data, dist_matrix, c_g1)
                    run_results['G1_Neigh1_Insertion'] = (c_l1, time.time() - t0)

                    t0 = time.time()
                    p_l2, c_l2 = neigh_2(copy.deepcopy(p_g1), veichle_capacity, data, dist_matrix, c_g1)
                    run_results['G1_Neigh2_OrOpt'] = (c_l2, time.time() - t0)

                    t0 = time.time()
                    p_l3, c_l3 = neigh_3(copy.deepcopy(p_g1), veichle_capacity, data, dist_matrix, c_g1)
                    run_results['G1_Neigh3_Swap'] = (c_l3, time.time() - t0)

                    # Metaeuristiche su Greedy 1
                    t0 = time.time()
                    p_SA1l1, c_sa1 = Sim_Annealing(copy.deepcopy(p_l1), c_l1, veichle_capacity, dist_matrix, data)
                    run_results['SA1_on_Neigh1'] = (c_sa1, time.time() - t0)

                    t0 = time.time()
                    p_SA1l2, c_sa2 = Sim_Annealing(copy.deepcopy(p_l2), c_l2, veichle_capacity, dist_matrix, data)
                    run_results['SA1_on_Neigh2'] = (c_sa2, time.time() - t0)

                    t0 = time.time()
                    p_SA1l3, c_sa3 = Sim_Annealing(copy.deepcopy(p_l3), c_l3, veichle_capacity, dist_matrix, data)
                    run_results['SA1_on_Neigh3'] = (c_sa3, time.time() - t0)

                    t0 = time.time()
                    p_grasp1, c_grasp = grasp1(copy.deepcopy(p_l1), c_l1, veichle_capacity, veichle_quantity, dist_matrix, data, n_clienti)
                    run_results['GRASP_G1'] = (c_grasp, time.time() - t0)

                    t0 = time.time()
                    p_T1, c_tabu = Tabu_Search(copy.deepcopy(p_l1), c_l1, veichle_capacity, data, dist_matrix)
                    run_results['Tabu_G1'] = (c_tabu, time.time() - t0)

                    t0 = time.time()
                    p_vns1, c_vns = vns(copy.deepcopy(p_l1), c_l1, veichle_capacity, veichle_quantity, dist_matrix, data)
                    run_results['VNS_G1'] = (c_vns, time.time() - t0)

                    # Greedy 2
                    t0 = time.time()
                    p_g2, c_g2 = greedy_2(n_clienti, veichle_quantity, veichle_capacity, data, dist_matrix)
                    run_results['Greedy_2'] = (c_g2, time.time() - t0)

                    # Local search su Greedy 2
                    t0 = time.time()
                    p2_l1, c2_l1 = neigh_1(copy.deepcopy(p_g2), veichle_capacity, data, dist_matrix, c_g2)
                    run_results['G2_Neigh1_Insertion'] = (c2_l1, time.time() - t0)

                    t0 = time.time()
                    p2_l2, c2_l2 = neigh_2(copy.deepcopy(p_g2), veichle_capacity, data, dist_matrix, c_g2)
                    run_results['G2_Neigh2_OrOpt'] = (c2_l2, time.time() - t0)

                    t0 = time.time()
                    p2_l3, c2_l3 = neigh_3(copy.deepcopy(p_g2), veichle_capacity, data, dist_matrix, c_g2)
                    run_results['G2_Neigh3_Swap'] = (c2_l3, time.time() - t0)

                    # Metaeuristiche su greedy 2
                    t0 = time.time()
                    p_SA2l1, c_2sa1 = Sim_Annealing(copy.deepcopy(p2_l1), c2_l1, veichle_capacity, dist_matrix, data)
                    run_results['SA2_on_Neigh1'] = (c_2sa1, time.time() - t0)

                    t0 = time.time()
                    p_SA2l2, c_2sa2 = Sim_Annealing(copy.deepcopy(p2_l2), c2_l2, veichle_capacity, dist_matrix, data)
                    run_results['SA2_on_Neigh2'] = (c_2sa2, time.time() - t0)

                    t0 = time.time()
                    p_SA2l3, c_2sa3 = Sim_Annealing(copy.deepcopy(p2_l3), c2_l3, veichle_capacity, dist_matrix, data)
                    run_results['SA2_on_Neigh3'] = (c_2sa3, time.time() - t0)

                    t0 = time.time()
                    p_grasp2, c_grasp2 = grasp1(copy.deepcopy(p2_l1), c2_l1, veichle_capacity, veichle_quantity, dist_matrix, data, n_clienti)
                    run_results['GRASP_G2'] = (c_grasp2, time.time() - t0)

                    t0 = time.time()
                    p_T2, c_tabu2 = Tabu_Search(copy.deepcopy(p2_l1), c2_l1, veichle_capacity, data, dist_matrix)
                    run_results['Tabu_G2'] = (c_tabu2, time.time() - t0)

                    t0 = time.time()
                    p_vns2, c_vns2 = vns(copy.deepcopy(p2_l1), c2_l1, veichle_capacity, veichle_quantity, dist_matrix, data)
                    run_results['VNS_G2'] = (c_vns2, time.time() - t0)

                    # Memetico
                    t0 = time.time()
                    p_mem, costo_mem = Memetic_Algorithm(
                        n_clienti, veichle_quantity, veichle_capacity, data, dist_matrix,
                        pop_size=30, generazioni=200, prob_mutazione=0.15, tasso_local_search=1.0
                    )
                    run_results['Memetic_Algorithm'] = (costo_mem, time.time() - t0)

                    total_run_time = time.time() - start_run_time

                    # Analisi dei costi interni alla run corrente
                    best_metodo_run = min(run_results, key=lambda k: run_results[k][0])
                    best_costo_run = run_results[best_metodo_run][0]

                    # 1. dati totali per questa run
                    raw_runs_data.append({
                        'Cartella': fold,
                        'Istanza': file_name,
                        'Clienti': n_clienti,
                        'Run': run,
                        'Tempo_Totale_Run': total_run_time,
                        'Best_Costo_Della_Run': best_costo_run
                    })

                    # 2. dati in dettaglio dei singoli metodi per questa run
                    for metodo, (costo, tempo_metodo) in run_results.items():
                        raw_methods_data.append({
                            'Cartella': fold,
                            'Istanza': file_name,
                            'Run': run,
                            'Metodo': metodo,
                            'Costo': costo,
                            'Tempo_Metodo': tempo_metodo
                        })

            except Exception as e:
                print(f"Errore durante l'elaborazione del file {file_name}: {e}")

        if raw_runs_data:
            df_runs = pd.DataFrame(raw_runs_data)
            df_methods = pd.DataFrame(raw_methods_data)

            final_summary = []

            # Raggruppiamo i dati macro-run e i dati di dettaglio per Istanza
            grouped_runs = df_runs.groupby(['Cartella', 'Istanza', 'Clienti'])
            
            for (fold, istanza, clienti), run_group in grouped_runs:
                
                # Sub-dataset dei metodi relativo unicamente a QUESTA istanza
                method_sub_group = df_methods[(df_methods['Cartella'] == fold) & (df_methods['Istanza'] == istanza)]

                # Richiesta 1: Tempo MINIMO di esecuzione di un'intera RUN nelle n run
                tempo_run_min = run_group['Tempo_Totale_Run'].min()
                
                # Richiesta 2: Tempo MEDIO di esecuzione di un'intera RUN nelle n run
                tempo_run_medio = run_group['Tempo_Totale_Run'].mean()
                
                # Richiesta 4: Costo medio calcolato sui migliori risultati di ogni singola run
                costo_medio_dei_best = run_group['Best_Costo_Della_Run'].mean()

                # Richiesta 3: Best costo trovato in assoluto e da quale metodo
                idx_best_assoluto = method_sub_group['Costo'].idxmin()
                best_costo_assoluto = method_sub_group.loc[idx_best_assoluto, 'Costo']
                best_metodo_assoluto = method_sub_group.loc[idx_best_assoluto, 'Metodo']

                # Dizionario base della riga dell'istanza
                row_istanza = {
                    'Cartella': fold,
                    'Istanza': istanza,
                    'Clienti': clienti,
                    'Tempo_Min_Intera_Run': tempo_run_min,
                    'Tempo_Medio_Intera_Run': tempo_run_medio,
                    'Best_Costo_Assoluto': best_costo_assoluto,
                    'Metodo_Best_Costo': best_metodo_assoluto,
                    'Costo_Medio_dei_Best_delle_Run': costo_medio_dei_best
                }

                # Richiesta Extra: Tempi medi e Costi migliori per OGNI SINGOLO metodo sulle n run
                # Calcoliamo medie dei tempi e i minimi (migliori) dei costi per metodo
                stats_per_metodo = method_sub_group.groupby('Metodo').agg(
                    Tempo_Medio=('Tempo_Metodo', 'mean'),
                    Best_Costo=('Costo', 'min')
                )

                # Inseriamo i dati nel record in modo dinamico sotto forma di colonne dedicate
                for metodo_nome, stats in stats_per_metodo.iterrows():
                    row_istanza[f'TempoMedio_{metodo_nome}'] = stats['Tempo_Medio']
                    row_istanza[f'BestCosto_{metodo_nome}'] = stats['Best_Costo']

                final_summary.append(row_istanza)

            # Creazione del dataframe finale e salvataggio
            df_final = pd.DataFrame(final_summary)
            
            # Riordino le colonne per renderlo visivamente perfetto
            df_final.to_excel("report_strutturato_istanze.xlsx", index=False)
            df_final.to_csv("report_strutturato_istanze.csv", index=False, sep=";")
            
            print("\n[OK] Analisi conclusa con successo!")
            print(" -> Generato 'report_strutturato_istanze.xlsx' pronto per i grafici.")
    else:
        print("\n[ATTENZIONE] Nessun dato estratto. Controlla i percorsi delle istanze.")

if __name__ == "__main__":
    run_benchmark()