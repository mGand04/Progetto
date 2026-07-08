import os
import numpy as np
import sys
from functions import matrice_distanze,valida_rotta,calcola_vicini,valida_rotta_senza_vincoli,controllo_costo,greedy_2

def main():
     # Leggo i dati in formatoo int e lascio una precisione di due cifre decimali
    np.set_printoptions(suppress=True, precision=2)
    path_base = r'C:\Users\safet\OneDrive\Desktop\Progetto\Progetto\Istanze'
    #path_base = r'C:\Users\mgand\OneDrive\Desktop\Ottimizzazzione_sr\Progetto\Istanze'
    path_file = r'C:\Users\safet\OneDrive\Desktop\Progetto\Progetto'
    folders = ['n25', 'n50', 'n100']

    # Definisco il percorso del file .txt in cui salvare l'output completo
    output_file_path = os.path.join(path_file, "risultati_greedy2.txt")

    print("Inizio del ciclo di test per greedy2...\n")

    # Salviamo il terminale originale per poterci stampare alla fine
    terminale_originale = sys.stdout

    # Apro il file in modalità scrittura ('w')
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        sys.stdout = f_out
        for fold in folders:
            folder_path = os.path.join(path_base, fold)
            if not os.path.exists(folder_path):
                print(f"Cartella non trovata: {folder_path}, salto.")
                continue
            # Prendo tutti i file di testo nella cartella
            files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

            for file_name in files:
                file_path = os.path.join(folder_path, file_name)

                # Scrivo una riga di separazione nel file txt per renderlo leggibile
                print(f"\n{'='*50}\nElaborazione: {fold} / {file_name}\n{'='*50}")

                try:
                    veichle_info = np.genfromtxt(file_path, skip_header=4, max_rows=1)
                    veichle_quantity = int(veichle_info[0])
                    veichle_capacity = veichle_info[1]

                    data = np.genfromtxt(file_path, skip_header=9)
                    cols = {'ID': 0, 'X': 1, 'Y': 2, 'DEM': 3, 'READY': 4, 'DUE': 5, 'SERV': 6}
                    demand = data[:, 3]
                    n_clienti = len(data)-1
                    dist_matrix = matrice_distanze(data)

                    # Scrittura delle informazioni del veicolo sul file
                    print("Veichle info")
                    print(f"Veichle quantity: {veichle_quantity}")
                    print(f"Veichle capacity: {veichle_capacity}")
                    print(f"Number of clients: {n_clienti}")
                    print("\nGreedy 2")

                    percorsi_2, costo_tot_2 = greedy_2(n_clienti, veichle_quantity, veichle_capacity,data, dist_matrix)
                    check_costo = controllo_costo(percorsi_2, veichle_capacity, data, dist_matrix)
                    for idx, p in enumerate(percorsi_2):
                        print(f"Veicolo {idx+1}: {p}")
                    print(f"Costo Totale della Soluzione: {costo_tot_2:.1f}")
                    print(f"Costo Totale della Soluzione controllato in seguito: {check_costo:.1f}")


                except Exception as e:
                    print(f"Errore durante l'elaborazione del file {file_name}: {e}")

        sys.stdout = terminale_originale

    print("\nProcesso completato.")

if __name__ == "__main__":
    main()