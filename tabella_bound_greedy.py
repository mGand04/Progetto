import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. DIZIONARIO UPPER BOUND (Dati forniti)
upper_bounds = {
    (25, 'C101'): 191.3, (25, 'C104'): 186.9, (25, 'C108'): 191.3,
    (25, 'R101'): 617.1, (25, 'R104'): 416.9, (25, 'R108'): 397.3,
    (25, 'RC101'): 461.1, (25, 'RC104'): 306.6, (25, 'RC108'): 294.5,
    (25, 'C201'): 214.7, (25, 'C204'): 213.1, (25, 'C208'): 214.5,
    (25, 'R201'): 463.3, (25, 'R204'): 355.0, (25, 'R208'): 328.2,
    (25, 'RC201'): 360.2, (25, 'RC204'): 299.7, (25, 'RC208'): 269.1,
    
    (50, 'C101'): 362.4, (50, 'C104'): 358.0, (50, 'C108'): 362.4,
    (50, 'R101'): 1044.0, (50, 'R104'): 625.4, (50, 'R108'): 617.7,
    (50, 'RC101'): 944.0, (50, 'RC104'): 545.8, (50, 'RC108'): 598.1,
    (50, 'C201'): 360.2, (50, 'C204'): 350.1, (50, 'C208'): 350.5,
    (50, 'R201'): 791.9, (50, 'R204'): 506.4, (50, 'R208'): 487.7,
    (50, 'RC201'): 684.8, (50, 'RC204'): 444.2, (50, 'RC208'): 476.7,
    
    (100, 'C101'): 827.3, (100, 'C104'): 822.9, (100, 'C108'): 827.3,
    (100, 'R101'): 1637.7, (100, 'R104'): 971.5, (100, 'R108'): 932.1,
    (100, 'RC101'): 1619.8, (100, 'RC104'): 1132.3, (100, 'RC108'): 1114.2,
    (100, 'C201'): 589.1, (100, 'C204'): 588.1, (100, 'C208'): 585.8,
    (100, 'R201'): 1143.2, (100, 'R204'): 731.3, (100, 'R208'): 701.2,
    (100, 'RC201'): 1261.8, (100, 'RC204'): 783.5, (100, 'RC208'): 776.1
}

# 2. CARICAMENTO E PREPARAZIONE DATI
file_path = "report_strutturato_istanze.xlsx"

if not os.path.exists(file_path):
    print(f"[ERRORE] Il file non esiste in: {file_path}")
    exit()

df_bench = pd.read_excel(file_path)
df_bench['Istanza_Clean'] = df_bench['Istanza'].astype(str).str.replace('.txt', '', regex=False)
df_bench['Upper_Bound'] = df_bench.apply(lambda r: upper_bounds.get((int(r['Clienti']), r['Istanza_Clean']), None), axis=1)
df_bench = df_bench.dropna(subset=['Upper_Bound'])

# Calcolo del GAP %
df_bench['GAP_%'] = ((df_bench['BestCosto_Greedy_1'] - df_bench['Upper_Bound']) / df_bench['Upper_Bound']) * 100

# 3. AGGREGAZIONE DATI (Riepilogo solo dei risultati medi, min e max)
summary_data = []
for n_clienti in [25, 50, 100]:
    df_sub = df_bench[df_bench['Clienti'] == n_clienti]
    if not df_sub.empty:
        summary_data.append([
            f"{n_clienti} Clienti",
            f"{df_sub['GAP_%'].mean():.2f}%",
            f"{df_sub['GAP_%'].min():.2f}%",
            f"{df_sub['GAP_%'].max():.2f}%"
        ])

# 4. RENDERING GRAFICO DELLA TABELLA COMPATTA
headers = ["Dimensione Istanza", "GAP % Medio", "GAP % Minimo", "GAP % Massimo"]

fig, ax = plt.subplots(figsize=(8, 2.5)) # Dimensioni ridotte e compatte, perfette per il report
ax.axis('tight')
ax.axis('off')

matplotlib_table = ax.table(
    cellText=summary_data, 
    colLabels=headers, 
    cellLoc='center', 
    loc='center',
    colWidths=[0.30, 0.23, 0.23, 0.23]
)

# Impostazioni Font e Padding interno
matplotlib_table.auto_set_font_size(False)
matplotlib_table.set_fontsize(11)

# Stile dell'Intestazione (Header)
for col_idx in range(len(headers)):
    cell = matplotlib_table[0, col_idx]
    cell.set_text_props(fontweight='bold', color='white', fontsize=11)
    cell.set_facecolor('#2c3e50') # Blu notte professionale
    cell.set_height(0.22)

# Stile delle righe dei dati
for row_idx in range(1, len(summary_data) + 1):
    for col_idx in range(len(headers)):
        cell = matplotlib_table[row_idx, col_idx]
        cell.set_height(0.18) # Bel padding verticale largo
        
        # Evidenziamo la prima colonna in semibold
        if col_idx == 0:
            cell.set_text_props(fontweight='semibold', color='#34495e')
            
        # Alternanza colori righe (Zebra style discreto)
        if row_idx % 2 == 0:
            cell.set_facecolor('#f8f9fa')
        else:
            cell.set_facecolor('#ffffff')

plt.tight_layout()
plt.savefig("tabella_riepilogo_gap_medi.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n[OK] Tabella riassuntiva generata! Controlla 'tabella_riepilogo_gap_medi.png'")