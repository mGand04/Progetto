import os
import pandas as pd
import numpy as np
import random as rd
import time
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from functions import matrice_distanze,valida_rotta,calcola_vicini,valida_rotta_senza_vincoli,controllo_costo,greedy_1,greedy_2,neigh_1,neigh_2,neigh_3,Sim_Annealing,Tabu_Search, grasp1, vns, Memetic_Algorithm

def main():
    # Leggo i dati in formatoo int e lascio una precisione di due cifre decimali
    np.set_printoptions(suppress=True, precision=2)
    # Percorso del file 
    #path_base = r'C:\Users\safet\OneDrive\Desktop\Progetto\Progetto\Istanze'
    path_base = r'C:\Users\mgand\OneDrive\Desktop\Ottimizzazzione_sr\Progetto\Istanze'
    FOLDERS = ['n25', 'n50', 'n100'] 
    OUTPUT_DIR = r'.\risultati'
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'risultati_run.csv')
 
    SAVE_ROUTES_JSON = False                
    RESUME = True                           # se True, salta le run gia' presenti nel CSV
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_ROUTES_JSON:
        os.makedirs(os.path.join(OUTPUT_DIR, 'routes'), exist_ok=True)

    n_run = input("Inserisci il numero di run per ogni istanza: ")

    try:
        for folder in FOLDERS:
            cartella = os.path.join(path_base, folder)
            if not os.path.isdir(cartella):
                print(f"Cartella non trovata: {cartella}, salto.")
                continue

            file_list = sorted(glob.glob(os.path.join(cartella, '*.txt')))
            print(f"\n=== Cartella {folder}: {len(file_list)} istanze trovate ===")
     
    except FileNotFoundError:
        print("Controlla di aver scritto correttamente i nomi.")
    except Exception as e:
        print(f"ERRORE imprevisto: {e}")