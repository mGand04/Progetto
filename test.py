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

    # Lista che conterrà tutti i record dei risultati
    results_records = []

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

                    # Dizionario per memorizzare i risultati di questa specifica run
                    current_record = {
                        'Cartella': fold,
                        'Istanza': file_name,
                        'Clienti': n_clienti,
                        'Run': run
                    }

                    t0 = time.time()

            except Exception as e:
                print(f"Errore durante l'elaborazione del file {file_name}: {e}")

if __name__ == "__main__":
    run_benchmark()