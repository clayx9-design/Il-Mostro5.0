import streamlit as st
import pandas as pd
import numpy as np
import os
import time

# --- Configurazione e File ---
FILE_EXCEL_PRINCIPALE = 'Il Mostro 5.0.xlsx'
TEAM_SHEET_NAMES = [
    "Atalanta", "Bologna", "Cagliari", "Como", "Cremonese", "Fiorentina", 
    "Genoa", "Hellas Verona", "Inter", "Juventus", "Lazio", "Lecce", 
    "Milan", "Napoli", "Parma", "Pisa", "Roma", "Sassuolo", "Torino", "Udinese"
]
# NOTA: Assicurati che il file 'optimized_prediction_model.py' sia presente 
# con la classe SuperAdvancedCardPredictionModel
# from optimized_prediction_model import SuperAdvancedCardPredictionModel 

# --- Funzioni di Inizializzazione e Caricamento Dati ---

# Imposta lo stato iniziale della sessione Streamlit
if 'elaborato' not in st.session_state:
    st.session_state.elaborato = False
if 'scrolled_exclusions' not in st.session_state:
    st.session_state.scrolled_exclusions = []

# Funzione per il caricamento dei dati da Excel con caching
@st.cache_data
def load_data_from_excel(sheet_name):
    """
    Legge i dati dal foglio specificato all'interno del file Excel principale.
    """
    if not os.path.exists(FILE_EXCEL_PRINCIPALE):
        # Questo errore viene visualizzato solo se il file principale non esiste
        st.error(f"File Excel principale non trovato: {FILE_EXCEL_PRINCIPALE}.")
        return pd.DataFrame()
        
    try:
        # Legge il foglio richiesto direttamente dal file XLSX
        df = pd.read_excel(FILE_EXCEL_PRINCIPALE, sheet_name=sheet_name)
        return df
    except ValueError:
        # Questo errore si verifica se il nome del foglio non è corretto
        st.error(f"Foglio '{sheet_name}' non trovato nel file '{FILE_EXCEL_PRINCIPALE}'.")
        return pd.DataFrame()
    except Exception as e:
        # Altri errori di lettura
        st.error(f"Errore nella lettura del foglio '{sheet_name}': {e}")
        return pd.DataFrame()

# --- Funzioni di Visualizzazione ---

def prognostico_giocatori_inter(df_inter): 
    """
    Visualizza i nomi dei giocatori dell'Inter.
    CORREZIONE: Usa il DataFrame Inter corretto e corregge l'etichetta.
    """
    st.markdown("---") # Separatore visivo

    # Filtraggio dei giocatori (assumendo che le colonne 'giocatore' e 'ruolo' esistano)
    # L'etichetta ora è "INTER"
    giocatori_filtrati = df_inter[
        (df_inter["ruolo"] == "POR")
        | (df_inter["ruolo"] == "DIF")
        | (df_inter["ruolo"] == "CC")
        | (df_inter["ruolo"] == "ATT")
    ]
    
    giocatori_stringa = ", ".join(
        [
            f"{row['giocatore']} ({row['ruolo']})"
            for index, row in giocatori_filtrati.iterrows()
        ]
    )
    
    st.write(
        f"**INTER**", 
        giocatori_stringa,
    )

    st.markdown("---")


# --- Funzione Principale Streamlit ---

def run(squadra):
    st.title("⚽ Il Mostro 5.0 - Sistema Avanzato Predizione Cartellini")
    st.sidebar.header(f"Analisi Partita")

    # --- 1. CARICAMENTO DATI ---

    # Carica i dati degli Arbitri dal foglio 'Arbitri'
    df_arbitri = load_data_from_excel("Arbitri")
    
    if df_arbitri.empty:
        st.warning("⚠️ Impossibile caricare i dati degli arbitri. Verifica file e foglio.")
        elenco_arbitri = ["Arbitro Sconosciuto"]
    else:
        try:
            # Estrae l'elenco dei nomi per il selettore
            elenco_arbitri = df_arbitri['Nome'].dropna().unique().tolist()
            elenco_arbitri.sort()
        except KeyError:
            st.error("Colonna 'Nome' mancante nel foglio 'Arbitri'.")
            elenco_arbitri = ["Errore Colonna Nome"]
            
    elenco_arbitri.insert(0, "") # Aggiunge l'opzione vuota
    
    # Carica i dati dei Giocatori (Unendo tutti i fogli squadra)
    df_list = []
    for sheet_name in TEAM_SHEET_NAMES:
        df_team = load_data_from_excel(sheet_name)
        if not df_team.empty:
            df_list.append(df_team)
    
    if df_list:
        df_all_players = pd.concat(df_list, ignore_index=True)
    else:
        st.error("Nessun dato dei giocatori caricato. Impossibile eseguire.")
        return

    # Estrae il DataFrame specifico dell'Inter per la correzione di visualizzazione
    df_inter = load_data_from_excel("Inter") 
    
    # --- 2. SELETTORI INTERATTIVI ---
    
    # Selettori Squadre
    elenco_squadre = [s for s in TEAM_SHEET_NAMES if s != squadra]
    elenco_squadre.insert(0, "")
    
    sq_casa = st.sidebar.selectbox("Squadra Casa:", elenco_squadre, index=0, key='sq_casa')
    sq_trasf = st.sidebar.selectbox("Squadra Trasferta:", elenco_squadre, index=0, key='sq_trasf')

    # Selettore Arbitro (ora usa l'elenco corretto estratto da Excel)
    arbitro_selezionato = st.sidebar.selectbox(
        "Seleziona l'Arbitro:", 
        elenco_arbitri, 
        index=0, 
        key='arbitro_selezionato'
    )
    
    # ... (Omissis: Il resto della logica di pre-elaborazione, verifica titolarità, 
    #      pulsante Elabora Pronostico, invocazione del modello, 
    #      e visualizzazione dei risultati del TOP 4...)

    # --- Esempio di Utilizzo della Funzione Corretta ---
    if st.session_state.elaborato:
        
        # ... (mostra i risultati del pronostico e l'analisi arbitro)
        
        # Esempio di visualizzazione dei giocatori dell'Inter
        if sq_casa == "Inter" or sq_trasf == "Inter":
            # Chiamata alla funzione di visualizzazione corretta
            prognostico_giocatori_inter(df_inter) 

# Esegui l'applicazione
run(None)