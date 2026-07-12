# Genetici
import os
import numpy as np
import time
import copy

from functions import matrice_distanze,valida_rotta,calcola_vicini,valida_rotta_senza_vincoli,controllo_costo,greedy_1,greedy_2,neigh_1,neigh_2,neigh_3,Sim_Annealing,Tabu_Search, grasp1, vns, Memetic_Algorithm

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
        start = time.time()
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
        print("Distanza 0-1:", dist_matrix[0, 1])
        print("Max valore matrice:", np.max(dist_matrix))

        print("SOMMA DATA:", np.sum(data))
        print("SOMMA MATRICE:", np.sum(dist_matrix)) 
        print("\nAlgoritmo Memetico: ")
        percorsi_mem, costo_mem = Memetic_Algorithm(n_clienti, veichle_quantity, veichle_capacity, data, dist_matrix,pop_size=30, generazioni=200, prob_mutazione=0.15,tasso_local_search=1.0)
        check_costo = controllo_costo(percorsi_mem, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_mem):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_mem:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f} ")
    except FileNotFoundError:
        print(f"ERRORE: Il file '{file_name}' non esiste nella cartella '{fold}'.")
        print("Controlla di aver scritto correttamente i nomi.")
    except Exception as e:
        print(f"ERRORE imprevisto: {e}")

if __name__ == "__main__":
    main()