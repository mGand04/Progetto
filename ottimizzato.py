import os
import pandas as pd
import numpy as np
import random as rd
import math
import copy
from functions import matrice_distanze,valida_rotta,calcola_vicini,valida_rotta_senza_vincoli,controllo_costo,greedy_1,greedy_2,neigh_1,neigh_2,neigh_3,Sim_Annealing,Tabu_Search, grasp1, vns

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

        # Grasp
        print("\nGrasp con local search 1: ")
        percorsi_grasp, costo_grasp = grasp1(copy.deepcopy(percorsi_local), costo_tot_local, veichle_capacity, veichle_quantity, dist_matrix, data, n_clienti)
        check_costo = controllo_costo(percorsi_grasp, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_grasp):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_grasp:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f} ")


        # Taboo Search con local search 1
        print("\nTaboo Search con il primo local search: ")
        percorsi_tab_search, costo_tab_search = Tabu_Search(copy.deepcopy(percorsi_local),costo_tot_local, veichle_capacity, data, dist_matrix)
        check_costo = controllo_costo(percorsi_tab_search, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_tab_search):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_tab_search:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f} ")

        # VNS
        print("\nVNS con local search 1: " )
        percorsi_vns, costo_vns = vns(copy.deepcopy(percorsi_local), costo_tot_local, veichle_capacity, veichle_quantity, dist_matrix, data)
        check_costo = controllo_costo(percorsi_vns, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi_vns):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo_vns:.1f}")
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

        # Simulated annealing: Applico alla soluzione già migliorata con local search
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

        # Grasp
        print("\nGrasp con local search 1: ")
        percorsi2_grasp, costo2_grasp = grasp1(copy.deepcopy(percorsi_local2), costo_tot_2 , veichle_capacity, veichle_quantity, dist_matrix, data, n_clienti)
        check_costo = controllo_costo(percorsi2_grasp, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi2_grasp):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo2_grasp:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f} ")


        # Taboo Search con local search 1
        print("\nTaboo Search con il primo local search: ")
        percorsi2_tab_search, costo2_tab_search = Tabu_Search(copy.deepcopy(percorsi_local2),costo_tot_2, veichle_capacity, data, dist_matrix)
        check_costo = controllo_costo(percorsi2_tab_search, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi2_tab_search):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo2_tab_search:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f} ")

        # VNS
        print("\nVNS con local search 1: " )
        percorsi2_vns, costo2_vns = vns(copy.deepcopy(percorsi_local2), costo_tot_2, veichle_capacity, veichle_quantity, dist_matrix, data)
        check_costo = controllo_costo(percorsi2_vns, veichle_capacity, data, dist_matrix)
        for idx, p in enumerate(percorsi2_vns):
            print(f"Veicolo {idx+1}: {p}")
        print(f"Costo Totale della Soluzione: {costo2_vns:.1f}")
        print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f} ")

    except FileNotFoundError:
        print(f"ERRORE: Il file '{file_name}' non esiste nella cartella '{fold}'.")
        print("Controlla di aver scritto correttamente i nomi.")
    except Exception as e:
        print(f"ERRORE imprevisto: {e}")

if __name__ == "__main__":
    main()