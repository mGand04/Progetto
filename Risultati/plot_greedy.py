import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Impostiamo lo stile dei grafici
sns.set_theme(style="whitegrid")

# 1. CARICAMENTO DATI
file_path = "report_strutturato_istanze.xlsx"

if not os.path.exists(file_path):
    print(f"[ERRORE] Il file '{file_path}' non esiste in questa cartella.")
    exit()

df = pd.read_excel(file_path)
df = df.sort_values(by=['Clienti', 'Istanza'])
df['Istanza_Label'] = df['Istanza'].astype(str).str.replace('.txt', '', regex=False)

# ==================================================
# GRAFICO 1 MODIFICATO: ISTOGRAMMA CON COSTI ESPLICITI
# ==================================================
fig, ax = plt.subplots(figsize=(16, 7))  # Allargato leggermente per dare respiro ai numeri

# Generiamo il grafico a barre
sns.barplot(
    data=df, 
    x='Istanza_Label', 
    y='BestCosto_Greedy_1', 
    hue='Cartella', 
    palette='Set2',
    ax=ax
)

# --------------------------------------------------
# AGGIUNTA DEI VALORI ESPLICITI SOPRA LE BARRE
# --------------------------------------------------
# Iteriamo su ogni gruppo di barre (n25, n50, n100) generato dal hue
for container in ax.containers:
    ax.bar_label(
        container, 
        fmt='%.0f',          # Mostra il costo come intero senza decimali (es. 1934 invece di 1934.2)
        padding=4,           # Spazio in punti tra la cima della barra e il testo
        rotation=90,         # Ruota il testo in verticale per evitare sovrapposizioni
        fontsize=8.5,        # Dimensione del font leggermente ridotta per massima leggibilità
        fontweight='semibold',
        color='#444444'      # Grigio scuro per non appesantire la vista
    )

# Regoliamo i limiti dell'asse Y per fare spazio ai testi in verticale sopra le barre più alte
ax.set_ylim(0, df['BestCosto_Greedy_1'].max() * 1.15)

plt.title("Costo della Soluzione trovato da Greedy 1 per ogni Istanza", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Nome Istanza", fontsize=12, labelpad=10)
plt.ylabel("Costo Totale", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(title="Dimensione (Clienti)", loc='upper left')
plt.tight_layout()

# Salviamo il grafico modificato
plt.savefig("plot_greedy1_costi_con_valori.png", dpi=300)
plt.show()

# ==========================================
# GRAFICO 2: GRAFICO A PUNTI DEI TEMPI (GREEDY 1)
# ==========================================
plt.figure(figsize=(14, 6))

# Usiamo stripplot o scatterplot per mostrare i tempi di esecuzione come punti
sns.scatterplot(
    data=df, 
    x='Istanza_Label', 
    y='TempoMedio_Greedy_1', 
    hue='Cartella', 
    style='Cartella',
    s=100,                 # Dimensione dei punti
    palette='Dark2'
)

# Tracciamo una linea leggera che unisce i punti per vedere il trend di crescita computazionale
plt.plot(df['Istanza_Label'], df['TempoMedio_Greedy_1'], color='gray', linestyle='--', alpha=0.5, zorder=1)

plt.title("Tempo di Esecuzione di Greedy 1 per ogni Istanza", fontsize=14, fontweight='bold')
plt.xlabel("Nome Istanza", fontsize=12)
plt.ylabel("Tempo Medio (secondi)", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.legend(title="Dimensione (Clienti)")
plt.tight_layout()

# Salviamo il secondo grafico
plt.savefig("plot_greedy1_tempi.png", dpi=300)
plt.show()

print("\n[OK] Grafici di Greedy 1 generati con successo!")
print(" -> Salvato: 'plot_greedy1_costi.png'")
print(" -> Salvato: 'plot_greedy1_tempi.png'")