import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
from typing import Dict, Any

warnings.filterwarnings('ignore')

# =========================================================================
# IMPORT MODELLO
# =========================================================================
MODEL_LOADED = False
try:
    # Ho sostituito SuperAdvancedCardPredictionModel con la classe fornita nel file
    from optimized_prediction_model import OptimizedCardPredictionModel as SuperAdvancedCardPredictionModel, get_field_zone, get_player_role_category
    MODEL_LOADED = True
except ImportError:
    st.error("⚠️ Modello ottimizzato non trovato! Assicurati che 'optimized_prediction_model.py' sia nella cartella.")

# =========================================================================
# CONFIGURAZIONE
# =========================================================================
EXCEL_FILE_NAME = 'Il Mostro 5.0.xlsx'
REFEREE_SHEET_NAME = 'Arbitri'
TEAM_SHEET_NAMES = [
    'Atalanta', 'Bologna', 'Cagliari', 'Como', 'Cremonese', 'Fiorentina', 
    'Genoa', 'Hellas Verona', 'Inter', 'Juventus', 'Lazio', 'Lecio', 
    'Milan', 'Napoli', 'Parma', 'Pisa', 'Roma', 'Sassuolo', 'Torino', 'Udinese'
]

st.set_page_config(
    page_title="⚽ Il Mostro 5.0 - Sistema Avanzato Predizione Cartellini",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# STILE PERSONALIZZATO (Invariato)
# =========================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: #333;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1, .main-header p {
        color: #333;
    }
    .metric-box {
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-box h4 {
        margin-top: 0;
        opacity: 0.8;
    }
    .referee-info {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .player-card {
        border-left: 6px solid;
        padding: 15px 15px;
        margin-bottom: 15px;
        border-radius: 10px;
        background-color: #f7f9fc;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: background-color 0.2s;
    }
    .player-card:hover {
        background-color: #f0f3f7;
    }
    .player-details h4 {
        margin-bottom: 3px !important;
        font-size: 1.1em;
        font-weight: 600;
    }
    .player-details p {
        font-size: 0.9em;
        color: #777;
    }
    .player-details .rank-number {
        font-size: 1.2em; 
        font-weight: 900; 
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# INIZIALIZZAZIONE SESSION STATE (Invariato)
# =========================================================================
def init_session_state():
    """Inizializza tutti i valori di session state."""
    defaults = {
        'full_df_players': None,
        'excluded_pre': [],
        'result': None,
        'recalculated_result': None,
        'scrolled_top_4': None,
        'scrolled_exclusions': [],
        'home_team': None,
        'away_team': None,
        'referee': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =========================================================================
# FUNZIONI DI CARICAMENTO DATI (Modificato: rimozione try/except su get_player_role)
# =========================================================================
@st.cache_data
def load_excel_data():
    """Carica tutti i dati dal file Excel."""
    try:
        xls = pd.ExcelFile(EXCEL_FILE_NAME)
    except Exception as e:
        st.error(f"Errore nel caricamento del file Excel: {e}")
        return None

    data = {}
    team_dataframes = []
    
    available_sheets = xls.sheet_names
    
    for sheet in TEAM_SHEET_NAMES:
        if sheet in available_sheets:
            try:
                df = pd.read_excel(xls, sheet)
                df.insert(1, 'Squadra', sheet)
                team_dataframes.append(df)
            except Exception as e:
                st.sidebar.warning(f"⚠️ Errore {sheet}: {e}")
                continue

    if not team_dataframes:
        st.error("Nessun foglio squadra trovato nel file Excel.")
        return None

    data['players'] = pd.concat(team_dataframes, ignore_index=True)
    
    # Caricamento arbitri
    default_referee_data = {
        'Nome': ['Doveri', 'Orsato', 'Mariani', 'Pairetto', 'Massa', 'Guida'],
        'Gialli a partita': [4.2, 3.8, 5.1, 4.5, 3.2, 4.8]
    }
    
    SERIE_A_AVG_CARDS = 4.2

    df_referees = None
    try:
        if REFEREE_SHEET_NAME in available_sheets:
            df_referees = pd.read_excel(xls, REFEREE_SHEET_NAME)
            
            st.sidebar.info(f"📋 Colonne trovate nel foglio Arbitri: {list(df_referees.columns)}")
            
            # Identifica colonna nome
            nome_col = None
            for col in df_referees.columns:
                col_lower = str(col).lower().strip()
                if any(keyword in col_lower for keyword in ['nome', 'arbitro', 'name', 'referee']):
                    nome_col = col
                    break
            
            # Identifica colonna gialli
            gialli_col = None
            for col in df_referees.columns:
                col_lower = str(col).lower().strip()
                if 'giall' in col_lower and 'partita' in col_lower:
                    gialli_col = col
                    break
                elif 'card' in col_lower or 'yellow' in col_lower:
                    gialli_col = col
                    break
            
            if nome_col is None and len(df_referees.columns) > 0:
                nome_col = df_referees.columns[0]
                st.sidebar.warning(f"⚠️ Colonna nome non trovata, uso: {nome_col}")
            
            if gialli_col is None and len(df_referees.columns) > 1:
                gialli_col = df_referees.columns[1]
                st.sidebar.warning(f"⚠️ Colonna gialli non trovata, uso: {gialli_col}")
            
            if nome_col and gialli_col:
                df_referees = df_referees[[nome_col, gialli_col]].copy()
                df_referees.columns = ['Nome', 'Gialli a partita']
                
                # Pulizia dati
                df_referees['Nome'] = df_referees['Nome'].astype(str).str.strip()
                df_referees = df_referees[
                    (df_referees['Nome'].notna()) & 
                    (df_referees['Nome'] != '') & 
                    (df_referees['Nome'] != 'nan') &
                    (df_referees['Nome'].str.lower() != 'unnamed')
                ]
                
                df_referees['Gialli a partita'] = pd.to_numeric(
                    df_referees['Gialli a partita'], 
                    errors='coerce'
                )
                df_referees['Gialli a partita'].fillna(SERIE_A_AVG_CARDS, inplace=True)
                df_referees.drop_duplicates(subset=['Nome'], keep='first', inplace=True)
                df_referees.reset_index(drop=True, inplace=True)
                
                st.sidebar.success(f"✅ Caricati {len(df_referees)} arbitri dal foglio Excel")
            else:
                st.sidebar.error("❌ Impossibile identificare le colonne Nome e Gialli")
                df_referees = pd.DataFrame(default_referee_data)
                
    except Exception as e:
        st.sidebar.error(f"⚠️ Errore caricamento arbitri: {e}")
        df_referees = None

    if df_referees is None or df_referees.empty:
        st.sidebar.warning("⚠️ Uso dati arbitri di default")
        df_referees = pd.DataFrame(default_referee_data)

    data['referees'] = df_referees
    data = preprocess_data(data)
    
    return data

def preprocess_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocessa i dati con filtro >=5 partite."""
    df_players = data['players']
    
    initial_count = len(df_players)
    # Il filtro viene gestito anche nel modello, ma lo manteniamo qui per coerenza.
    df_players = df_players[df_players.get('90s Giocati Totali', 0).fillna(0) >= 5].copy() 
    excluded_count = initial_count - len(df_players)
    if excluded_count > 0:
        st.sidebar.warning(f"🔧 Filtro applicato: {excluded_count} giocatori esclusi (meno di 5 partite totali).")
    
    # Normalizza colonne numeriche
    numeric_columns = [
        'Media Falli Subiti 90s Totale', 'Media Falli Fatti 90s Totale',
        'Cartellini Gialli Totali', 'Media Falli per Cartellino Totale',
        'Media 90s per Cartellino Totale', 'Ritardo Cartellino (Minuti)',
        'Minuti Giocati Totali', '90s Giocati Totali', 
        'Media Falli Subiti 90s Stagionale'
    ]
    
    for col in numeric_columns:
        if col in df_players.columns:
            df_players[col] = pd.to_numeric(df_players[col], errors='coerce').fillna(0)
    
    # Normalizza nome giocatore
    if 'Giocatore' in df_players.columns and 'Player' not in df_players.columns:
        df_players.rename(columns={'Giocatore': 'Player'}, inplace=True)
    elif 'Player' not in df_players.columns:
        df_players['Player'] = df_players.iloc[:, 0].astype(str)
    
    # Normalizza posizione
    if 'Posizione_Primaria' not in df_players.columns:
        if 'Pos' in df_players.columns:
            df_players['Posizione_Primaria'] = df_players['Pos']
        else:
            df_players['Posizione_Primaria'] = 'MF'
    
    # Assicura colonna Ruolo (usando la funzione get_player_role se disponibile)
    if 'Ruolo' not in df_players.columns:
        try:
            # Assumiamo che get_player_role sia definita nel modello
            from optimized_prediction_model import get_player_role 
            df_players['Ruolo'] = df_players['Posizione_Primaria'].apply(get_player_role)
        except (ImportError, AttributeError):
            df_players['Ruolo'] = df_players['Posizione_Primaria'].apply(
                lambda x: 'DIF' if 'D' in str(x).upper() else ('ATT' if 'A' in str(x).upper() else 'CEN')
            )
            
    df_players['Posizione_Primaria'] = df_players['Posizione_Primaria'].astype(str).str.strip().str.upper()
    
    if 'Heatmap' not in df_players.columns:
        df_players['Heatmap'] = 'Central activity'
    df_players['Heatmap'] = df_players['Heatmap'].astype(str)
    
    data['players'] = df_players
    st.session_state['full_df_players'] = df_players # Salva il DF completo per l'uso successivo
    return data

# =========================================================================
# FUNZIONI GESTIONE TITOLARI (FASE 1) (Invariate)
# =========================================================================
# ... (get_fouls_suffered_metric, identify_high_risk_victims, display_starter_verification)

# =========================================================================
# FUNZIONI BILANCIAMENTO TOP 4 (Invariate)
# =========================================================================
# ... (apply_balancing_logic, get_risk_color)

# =========================================================================
# FUNZIONE DI PREDIZIONE CENTRALE (NUOVA)
# =========================================================================
@st.cache_data(show_spinner="⏳ Calcolo delle predizioni in corso...", max_entries=1)
def run_prediction(
    full_df_players: pd.DataFrame, 
    df_referees: pd.DataFrame, 
    home_team: str, 
    away_team: str, 
    referee_name: str,
    excluded_pre_list: list,
    model_class: type
) -> Dict[str, Any]:
    """Esegue la predizione del modello e il bilanciamento del TOP 4."""
    if not MODEL_LOADED:
        return {'error': "Modello non caricato."}
    
    # 1. Filtra i DF di squadra e arbitro
    home_df = full_df_players[
        (full_df_players['Squadra'] == home_team) & 
        (~full_df_players['Player'].isin(excluded_pre_list))
    ].copy()
    
    away_df = full_df_players[
        (full_df_players['Squadra'] == away_team) & 
        (~full_df_players['Player'].isin(excluded_pre_list))
    ].copy()
    
    referee_df = df_referees[df_referees['Nome'] == referee_name].head(1).copy()
    
    if home_df.empty or away_df.empty or referee_df.empty:
        return {'error': "Dati squadra o arbitro non trovati/insufficienti per la predizione."}
    
    # 2. Inizializza il modello
    model = model_class()
    
    # 3. Esegue la predizione
    prediction_result = model.predict_match_cards(home_df, away_df, referee_df)
    
    if 'error' in prediction_result:
        return prediction_result

    all_predictions_df = prediction_result['all_predictions']
    
    # 4. Applica il bilanciamento 2-2 al TOP 4
    top_4_predictions = apply_balancing_logic(all_predictions_df, home_team, away_team)
    
    # 5. Aggiunge i dati bilanciati al risultato
    prediction_result['top_4_predictions'] = top_4_predictions
    
    return prediction_result

# =========================================================================
# DISPLAY TOP 4 DINAMICO (Invariato)
# =========================================================================
# ... (display_dynamic_top_4)

# =========================================================================
# DISPLAY ANALISI (Invariato, ma incompleto nel codice fornito, solo l'inizio)
# =========================================================================
# ... (display_match_analysis, display_referee_analysis)

# =========================================================================
# MAIN APP
# =========================================================================
def main():
    init_session_state()
    
    st.markdown("<div class='main-header'><h1>⚽ Il Mostro 5.0</h1><p>Sistema Avanzato Predizione Cartellini (v5.0)</p></div>", unsafe_allow_html=True)

    data = load_excel_data()
    
    if data is None:
        st.stop()
        
    df_players = data['players']
    df_referees = data['referees']
    
    team_list = sorted(df_players['Squadra'].unique().tolist())
    referee_list = sorted(df_referees['Nome'].unique().tolist())

    # --- Sidebar per Input ---
    st.sidebar.header("📝 Configura Partita")
    
    # Selezione Squadre
    st.session_state['home_team'] = st.sidebar.selectbox("🏠 Squadra Casa", team_list, index=team_list.index('Inter') if 'Inter' in team_list else 0)
    st.session_state['away_team'] = st.sidebar.selectbox("✈️ Squadra Trasferta", [t for t in team_list if t != st.session_state['home_team']], index=team_list.index('Juventus') if 'Juventus' in team_list and 'Juventus' != st.session_state['home_team'] else 0)
    
    # Selezione Arbitro
    st.session_state['referee'] = st.sidebar.selectbox("⚖️ Arbitro", referee_list, index=referee_list.index('Orsato') if 'Orsato' in referee_list else 0)
    
    # Pulsante per avviare la predizione
    run_button_key = 'run_prediction_button'
    if st.sidebar.button("🚀 Esegui Predizione", key=run_button_key, type='primary'):
        # 1. Resetta i risultati precedenti per un nuovo calcolo
        st.session_state['result'] = None
        st.session_state['excluded_pre'] = []
        st.session_state['scrolled_top_4'] = None
        st.session_state['scrolled_exclusions'] = []
        
        # 2. Avvia la predizione
        with st.spinner(f"Calcolo in corso: {st.session_state['home_team']} vs {st.session_state['away_team']}..."):
            
            # Qui eseguiamo la Fase 1: Identificazione delle vittime ad alto rischio
            home_df_filtered = df_players[df_players['Squadra'] == st.session_state['home_team']]
            away_df_filtered = df_players[df_players['Squadra'] == st.session_state['away_team']]
            high_risk_victims = identify_high_risk_victims(home_df_filtered, away_df_filtered)
            
            if high_risk_victims:
                st.session_state['need_phase_1_confirmation'] = True
                st.session_state['high_risk_victims'] = high_risk_victims
                # Si ferma qui per chiedere conferma e riesegue dopo la selezione
            else:
                # Se non ci sono vittime, salta la fase 1 e calcola direttamente
                st.session_state['need_phase_1_confirmation'] = False
                st.session_state['result'] = run_prediction(
                    df_players, 
                    df_referees, 
                    st.session_state['home_team'], 
                    st.session_state['away_team'], 
                    st.session_state['referee'],
                    [], # Nessuna esclusione iniziale
                    SuperAdvancedCardPredictionModel
                )
        
    
    # --- Gestione FASE 1: Verifica Titolarità ---
    if st.session_state.get('need_phase_1_confirmation', False) and st.session_state.get('high_risk_victims'):
        
        st.markdown("---")
        excluded_pre = display_starter_verification(st.session_state['high_risk_victims'])
        
        if st.button("✅ Conferma Titolari e Prosegui", key='confirm_phase_1', type='secondary'):
            st.session_state['excluded_pre'] = excluded_pre
            st.session_state['need_phase_1_confirmation'] = False # Completata la fase 1
            st.session_state['result'] = None # Forza ricalcolo
            st.rerun() # Ricarica per procedere con il calcolo del modello
            
        return # Ferma l'esecuzione finché l'utente non conferma

    # --- Esecuzione Modello e Visualizzazione Risultati ---
    
    # Se il risultato non è stato calcolato (es. dopo fase 1 o al primo avvio) e non è in attesa di fase 1
    if st.session_state['result'] is None and not st.session_state.get('need_phase_1_confirmation', False):
        if st.session_state['home_team'] and st.session_state['away_team'] and st.session_state['referee']:
            # Esegue la predizione (il filtro sulle esclusioni avviene dentro run_prediction)
            st.session_state['result'] = run_prediction(
                df_players, 
                df_referees, 
                st.session_state['home_team'], 
                st.session_state['away_team'], 
                st.session_state['referee'],
                st.session_state['excluded_pre'],
                SuperAdvancedCardPredictionModel
            )
            
    # Visualizzazione
    if st.session_state['result']:
        result = st.session_state['result']
        
        if 'error' in result:
            st.error(f"❌ Errore di Predizione: {result['error']}")
            return
            
        st.markdown("---")
        display_match_analysis(result)
        st.markdown("---")
        display_referee_analysis(result['referee_profile'])
        st.markdown("---")
        display_dynamic_top_4()
        
    else:
        st.info("Seleziona le squadre e l'arbitro nella barra laterale e clicca su 'Esegui Predizione' per iniziare.")

if __name__ == '__main__':
    main()