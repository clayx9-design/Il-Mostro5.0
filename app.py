# ... (import e config invariati)

# =========================================================================
# FUNZIONI DI CARICAMENTO DATI (MODIFICATO)
# =========================================================================
@st.cache_data
def load_excel_data():
    """Carica tutti i dati dal file Excel."""
    # ... (codice di caricamento invariato fino a data['players'] = pd.concat(...))

    data['referees'] = df_referees
    data = preprocess_data(data)  # Ora preprocess_data include il filtro globale
    
    return data

def preprocess_data(data):
    """Preprocessa i dati. NOVITÀ: Filtro globale per >=5 partite giocate."""
    df_players = data['players']
    
    # *** NUOVO: FILTRO GLOBALE PER ESCLUDERE GIOCATORI CON <5 PARTITE ***
    initial_count = len(df_players)
    df_players = df_players[df_players.get('90s Giocati Totali', 0) >= 5].copy()
    excluded_count = initial_count - len(df_players)
    if excluded_count > 0:
        st.sidebar.warning(f"🔧 Filtro applicato: {excluded_count} giocatori esclusi (meno di 5 partite totali).")
    
    # ... (resto del preprocessing invariato: normalizza colonne, nomi, posizioni, ecc.)
    
    # Assicuriamo che la colonna Ruolo esista per la FASE 1
    if 'Ruolo' not in df_players.columns:
        try:
            from optimized_prediction_model import get_player_role as get_role
            df_players['Ruolo'] = df_players['Posizione_Primaria'].apply(get_role)
        except ImportError:
            df_players['Ruolo'] = 'MF'
            
    # ... (resto invariato)
    
    data['players'] = df_players
    return data

# =========================================================================
# FUNZIONI GESTIONE TITOLARI (FASE 1) - MODIFICATA
# =========================================================================
def get_fouls_suffered_metric(df):
    """
    Estrae le metriche dei falli subiti. NOVITÀ: Applica filtro >=5 qui per sicurezza.
    """
    df = df.copy()
    
    # *** NUOVO: FILTRO LOCALE PER >=5 PARTITE (ridondante ma sicuro) ***
    df = df[df.get('90s Giocati Totali', 0) >= 5].copy()
    
    # ... (resto invariato: total_col, seasonal_col, calcoli Falli_Subiti_Used, ecc.)
    
    return df

def identify_high_risk_victims(home_df, away_df):
    """
    Identifica i giocatori che SUBISCONO molti falli.
    MODIFICA: Soglia da >=1 a >=5 per coerenza.
    """
    all_victims = []
    
    for df, team_type in [(home_df, 'Casa'), (away_df, 'Trasferta')]:
        df = get_fouls_suffered_metric(df)  # Ora include filtro >=5
        
        # *** MODIFICA: Cambiato da >=1 a >=5 ***
        df_valid = df[(df['Falli_Subiti_Used'] > 0) & (df['90s Giocati Totali'] >= 5)].copy()
        
        if df_valid.empty:
            st.sidebar.info(f"ℹ️ {team_type}: Nessun giocatore valido (>=5 partite) con falli subiti >0.")
            continue
            
        # ... (resto invariato: calcola Spread, soglie, victims_standard, ecc.)
        
        # Aggiungi log per trasparenza
        valid_count = len(df_valid)
        victims_count = len(victims_standard) + len(victims_forced_seasonal)
        st.sidebar.info(f"📊 {team_type}: {valid_count} validi (>=5 part.), {victims_count} ad alto rischio.")
        
        all_victims.extend(victims_standard.to_dict('records'))
        all_victims.extend(victims_forced_seasonal.to_dict('records'))
    
    return pd.DataFrame(all_victims).drop_duplicates(subset=['Player']).to_dict('records')

# ... (resto di app.py invariato: display_dynamic_top_4, main_prediction_interface, ecc.)

# Nel MAIN: Aggiungi info nel sidebar dopo caricamento
def main():
    # ... (invariato)
    
    if data is None:
        # ... (invariato)
    
    df_players = data['players']
    df_referees = data['referees']
    
    # *** NUOVO: Statistiche post-filtro nel sidebar ***
    total_players_after_filter = len(df_players)
    st.sidebar.success(f"✅ Dati caricati e filtrati: {total_players_after_filter} giocatori (>=5 part.), {len(df_referees)} arbitri")
    
    # ... (resto invariato)