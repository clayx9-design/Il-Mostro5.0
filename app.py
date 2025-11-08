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
    # Importiamo tutte le dipendenze necessarie dal file modello
    from optimized_prediction_model import OptimizedCardPredictionModel as SuperAdvancedCardPredictionModel, get_field_zone, get_player_role_category, get_player_role, normalize_data
    MODEL_LOADED = True
except ImportError as e:
    st.error(f"⚠️ Modello ottimizzato non trovato o import fallita: {e}")
    MODEL_LOADED = False

# =========================================================================
# CONFIGURAZIONE
# =========================================================================
EXCEL_FILE_NAME = 'Il Mostro 5.0.xlsx'
REFEREE_SHEET_NAME = 'Arbitri'
TEAM_SHEET_NAMES = [
    'Atalanta', 'Bologna', 'Cagliari', 'Como', 'Cremonese', 'Fiorentina', 
    'Genoa', 'Hellas Verona', 'Inter', 'Juventus', 'Lazio', 'Lecce', 
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
# FUNZIONI DI UTILITY E BILANCIAMENTO (DEFINITE PRIMA DI run_prediction)
# =========================================================================

def get_risk_color(risk_score: float) -> str:
    """Restituisce colore basato sul rischio."""
    if risk_score >= 0.70:
        return "#f44336"
    elif risk_score >= 0.55:
        return "#ff9800"
    elif risk_score >= 0.40:
        return "#ffc107"
    else:
        return "#4caf50"

def apply_balancing_logic(all_predictions_df: pd.DataFrame, home_team: str, away_team: str) -> list:
    """Applica logica di bilanciamento 2-2 al TOP 4."""
    if all_predictions_df.empty:
        return []
    
    risk_col = 'Rischio_Finale' if 'Rischio_Finale' in all_predictions_df.columns else 'Rischio'
    
    home_risks = all_predictions_df[all_predictions_df['Squadra'] == home_team].sort_values(
        risk_col, ascending=False
    ).head(10)
    away_risks = all_predictions_df[all_predictions_df['Squadra'] == away_team].sort_values(
        risk_col, ascending=False
    ).head(10)
    
    combined_top = pd.concat([home_risks, away_risks]).sort_values(
        risk_col, ascending=False
    ).head(4)
    
    top_4_iniziale = combined_top.copy()
    count_home = len(top_4_iniziale[top_4_iniziale['Squadra'] == home_team])
    count_away = len(top_4_iniziale[top_4_iniziale['Squadra'] == away_team])
    
    RISK_DIFFERENCE_THRESHOLD = 0.10
    top_4_ottimizzato = []

    # Logica di bilanciamento 3-1 -> 2-2
    if count_home == 3 and count_away == 1:
        dominant_risks = top_4_iniziale[top_4_iniziale['Squadra'] == home_team]
        if len(dominant_risks) >= 3 and len(away_risks) > 1:
            risk_dominant_3rd = dominant_risks.iloc[2][risk_col]
            risk_minor_2nd = away_risks.iloc[1][risk_col]
            
            if risk_dominant_3rd > (risk_minor_2nd + RISK_DIFFERENCE_THRESHOLD):
                top_4_ottimizzato = top_4_iniziale.to_dict('records')
            else:
                top_4_ottimizzato.extend(home_risks.head(2).to_dict('records'))
                top_4_ottimizzato.extend(away_risks.head(2).to_dict('records'))
        else:
            top_4_ottimizzato = top_4_iniziale.to_dict('records')
    
    elif count_home == 1 and count_away == 3:
        dominant_risks = top_4_iniziale[top_4_iniziale['Squadra'] == away_team]
        if len(dominant_risks) >= 3 and len(home_risks) > 1:
            risk_dominant_3rd = dominant_risks.iloc[2][risk_col]
            risk_minor_2nd = home_risks.iloc[1][risk_col]
            
            if risk_dominant_3rd > (risk_minor_2nd + RISK_DIFFERENCE_THRESHOLD):
                top_4_ottimizzato = top_4_iniziale.to_dict('records')
            else:
                top_4_ottimizzato.extend(home_risks.head(2).to_dict('records'))
                top_4_ottimizzato.extend(away_risks.head(2).to_dict('records'))
        else:
            top_4_ottimizzato = top_4_iniziale.to_dict('records')

    elif count_home == 2 and count_away == 2:
        top_4_ottimizzato = top_4_iniziale.to_dict('records')
    
    else:
        # Copre i casi 4-0, 0-4 o i casi estremi (si prende il Top 2 per squadra se possibile)
        top_4_ottimizzato.extend(home_risks.head(min(2, len(home_risks))).to_dict('records'))
        top_4_ottimizzato.extend(away_risks.head(min(2, len(away_risks))).to_dict('records'))
        top_4_ottimizzato = sorted(top_4_ottimizzato, key=lambda x: x[risk_col], reverse=True)[:4]

    final_df = pd.DataFrame(top_4_ottimizzato).sort_values(risk_col, ascending=False)
    
    # Assicura che Ruolo esista (necessario per l'output)
    if 'Ruolo' not in final_df.columns:
        final_df = pd.merge(final_df, all_predictions_df[['Player', 'Ruolo']].drop_duplicates(), on='Player', how='left')

    return final_df[['Player', 'Squadra', risk_col, 'Ruolo']].to_dict('records')


# =========================================================================
# FUNZIONE DI PREDIZIONE CENTRALE
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
        return {'error': "Modello non caricato. Controlla il file optimized_prediction_model.py."}
    
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
        return {'error': "Dati squadra o arbitro non trovati/insufficienti per la predizione dopo il filtro titolari."}
    
    # 2. Inizializza il modello
    try:
        model = model_class()
    except Exception as e:
        return {'error': f"Errore nell'inizializzazione del modello: {e}"}
    
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
# INIZIALIZZAZIONE SESSION STATE
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
        'referee': None,
        'need_phase_1_confirmation': False,
        'high_risk_victims': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =========================================================================
# FUNZIONI DI CARICAMENTO DATI
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
    
    # Assicura colonna Ruolo
    if 'Ruolo' not in df_players.columns:
        try:
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
    st.session_state['full_df_players'] = df_players
    return data

# =========================================================================
# FUNZIONI GESTIONE TITOLARI (FASE 1)
# =========================================================================
def get_fouls_suffered_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola metriche falli subiti."""
    df = df.copy()
    df = df[df.get('90s Giocati Totali', 0).fillna(0) >= 5].copy()
    
    total_col = 'Media Falli Subiti 90s Totale'
    seasonal_col = 'Media Falli Subiti 90s Stagionale'
    
    has_total = total_col in df.columns
    has_seasonal = seasonal_col in df.columns
    
    df['Falli_Subiti_Totale'] = df.get(total_col, 0.0)
    df['Falli_Subiti_Stagionale'] = df.get(seasonal_col, 0.0)
    df['90s Giocati Totali'] = df.get('90s Giocati Totali', 0.0)
    
    if has_total:
        df['Falli_Subiti_Used'] = df['Falli_Subiti_Totale']
        if has_seasonal:
            mask_zero = (df['Falli_Subiti_Used'] == 0) | df['Falli_Subiti_Used'].isna()
            df.loc[mask_zero, 'Falli_Subiti_Used'] = df.loc[mask_zero, 'Falli_Subiti_Stagionale']
            df['Falli_Subiti_Source'] = df.apply(
                lambda row: 'Stagionale' if mask_zero.get(row.name, False) else 'Totale',
                axis=1
            )
        else:
            df['Falli_Subiti_Source'] = 'Totale'
    elif has_seasonal:
        df['Falli_Subiti_Used'] = df['Falli_Subiti_Stagionale']
        df['Falli_Subiti_Source'] = 'Stagionale'
    else:
        df['Falli_Subiti_Used'] = 0
        df['Falli_Subiti_Source'] = 'N/A'
    
    df['Falli_Subiti_Used'] = df['Falli_Subiti_Used'].fillna(0)
    return df

def identify_high_risk_victims(home_df: pd.DataFrame, away_df: pd.DataFrame) -> list:
    """Identifica giocatori ad alto rischio (falli subiti)."""
    all_victims = []
    
    for df, team_type in [(home_df, 'Casa'), (away_df, 'Trasferta')]:
        df = get_fouls_suffered_metric(df)
        df_valid = df[(df['Falli_Subiti_Used'] > 0) & (df['90s Giocati Totali'] >= 5)].copy()
        
        if df_valid.empty:
            continue
        
        df_valid['Stagional_Spread'] = np.where(
            (df_valid['Falli_Subiti_Stagionale'] > 0) & (df_valid['Falli_Subiti_Totale'] > 0),
            (df_valid['Falli_Subiti_Stagionale'] - df_valid['Falli_Subiti_Totale']),
            0
        )
        
        SPREAD_THRESHOLD_HIGH = 0.5
        MIN_FOULS_STANDARD = 2.0
        MIN_90S_ACTIVE = 3.0
        MIN_90S_TOP_PLAYER = 5.0
        MIN_SEASONAL_FOULS = 2.0
        
        victims_forced_seasonal = df_valid[
            (df_valid['Stagional_Spread'] >= SPREAD_THRESHOLD_HIGH) &
            (df_valid['Falli_Subiti_Stagionale'] >= MIN_SEASONAL_FOULS) &
            (df_valid['90s Giocati Totali'] >= MIN_90S_ACTIVE)
        ].copy()
        
        victims_standard = df_valid[
            (df_valid['Falli_Subiti_Used'] >= MIN_FOULS_STANDARD) &
            (df_valid['90s Giocati Totali'] >= 2.0)
        ].copy()
        
        victims_top_player = df_valid[
            (df_valid['90s Giocati Totali'] >= MIN_90S_TOP_PLAYER) &
            (df_valid['Falli_Subiti_Used'] >= 1.0)
        ].copy()
        
        combined_victims = pd.concat([
            victims_forced_seasonal,
            victims_standard,
            victims_top_player
        ], ignore_index=True).drop_duplicates(subset=['Player'])
        
        for _, victim in combined_victims.iterrows():
            all_victims.append({
                'Player': victim['Player'],
                'Squadra': victim['Squadra'],
                'Falli_Subiti': victim['Falli_Subiti_Used'],
                '90s': victim['90s Giocati Totali'],
                'Fonte': victim['Falli_Subiti_Source']
            })
    
    return all_victims

def display_starter_verification(high_risk_victims: list) -> list:
    """Visualizza verifica titolarità FASE 1."""
    st.markdown("### 🔍 FASE 1: Verifica Titolarità")
    st.info("⚙️ I seguenti giocatori sono stati identificati come ad alto rischio (falli subiti). Conferma chi è titolare:")
    
    excluded_pre = []
    
    for victim in high_risk_victims:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{victim['Player']}** ({victim['Squadra']}) - Falli subiti: {victim['Falli_Subiti']:.2f}/90'")
        with col2:
            key = f"exclude_pre_{victim['Player']}"
            default_state = victim['Player'] in st.session_state.get('excluded_pre', [])
            if st.checkbox("Non titolare", key=key, value=default_state):
                excluded_pre.append(victim['Player'])
    
    return excluded_pre


# =========================================================================
# DISPLAY TOP 4 DINAMICO
# =========================================================================
def display_dynamic_top_4():
    """Visualizza TOP 4 con pulsante Escludi."""
    if 'result' not in st.session_state or st.session_state['result'] is None:
        return

    result = st.session_state['result']
    all_predictions_df = result['all_predictions']
    
    if 'scrolled_top_4' in st.session_state and st.session_state['scrolled_top_4'] is not None:
        current_top_4 = st.session_state['scrolled_top_4']
    else:
        current_top_4 = result['top_4_predictions']
        
    st.markdown("## 🎯 TOP 4 PRONOSTICO CARTELLINI")
    st.markdown("Clicca sul pulsante '❌ Escludi' per rimuovere un giocatore non titolare.")

    if 'scrolled_exclusions' not in st.session_state:
        st.session_state['scrolled_exclusions'] = []
    
    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    
    cols = st.columns(2)
    
    # Determina quale campo di rischio usare
    if len(current_top_4) > 0:
        risk_field = 'Rischio_Finale' if 'Rischio_Finale' in current_top_4[0] else 'Rischio'
    else:
        risk_field = 'Rischio'
    
    for i, prediction in enumerate(current_top_4, 1):
        player_name = prediction.get('Player', 'Sconosciuto')
        squadra = prediction.get('Squadra', 'N/A')
        ruolo = prediction.get('Ruolo', 'N/A')
        rischio = prediction.get(risk_field, 0.0)
        
        card_color = get_risk_color(rischio)
        
        with cols[(i-1) % 2]:
            st.markdown(f"""
            <div class='player-card' style='border-left-color: {card_color};'>
                <div class='player-details'>
                    <div>
                        <h4 style='color: #2c3e50; margin-bottom: 3px !important;'>
                            <span class='rank-number' style='color: {card_color};'>#{i}</span>
                            {player_name}
                        </h4>
                        <p style='margin: 0;'>
                            {squadra} • Ruolo: {ruolo} • Rischio: {rischio:.1%}
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if player_name not in st.session_state['scrolled_exclusions']:
                key_hash = hashlib.md5(f"{player_name}_{squadra}_{i}".encode()).hexdigest()[:8]
                button_key = f"exclude_btn_{key_hash}"
                
                # Aggiungi un piccolo spazio prima del bottone se è il primo elemento nella colonna
                if i % 2 != 0:
                    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
                    
                if st.button(f"❌ Escludi", key=button_key, use_container_width=False):
                    st.session_state['scrolled_exclusions'].append(player_name)
                    
                    excluded_players = st.session_state['scrolled_exclusions']
                    new_top_predictions_df = all_predictions_df[
                        ~all_predictions_df['Player'].isin(excluded_players)
                    ].copy()
                    
                    # Ricalcola il TOP 4 con la nuova lista di giocatori disponibili
                    new_top_4 = apply_balancing_logic(
                        new_top_predictions_df, 
                        st.session_state['home_team'], 
                        st.session_state['away_team']
                    )
                    
                    st.session_state['scrolled_top_4'] = new_top_4
                    st.rerun()
            else:
                st.markdown("<span style='color: #f44336; font-weight: bold;'>ESCLUSO</span>", unsafe_allow_html=True)

    if st.session_state['scrolled_exclusions']:
        st.warning(f"⚠️ TOP 4 modificato. Esclusi: {', '.join(st.session_state['scrolled_exclusions'])}")
        if st.button("↩️ Ripristina TOP 4 Originale", key='reset_scrolling', type='primary'):
            st.session_state['scrolled_top_4'] = None
            st.session_state['scrolled_exclusions'] = []
            st.rerun()
    
    st.markdown("---")

# =========================================================================
# DISPLAY ANALISI
# =========================================================================
def display_match_analysis(result: Dict[str, Any]):
    """Visualizza analisi partita."""
    match_info = result['match_info']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h4>🏠 Squadra Casa</h4>
            <h3>{match_info['home_team']}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <h4>✈️ Squadra Trasferta</h4>
            <h3>{match_info['away_team']}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <h4>🟨 Cartellini Attesi</h4>
            <h3>{match_info['expected_total_cards']}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        confidence_color = "🟢" if match_info.get('algorithm_confidence', 'High') == 'High' else "🟡"
        st.markdown(f"""
        <div class='metric-box'>
            <h4>📊 Confidenza Algoritmo</h4>
            <h3>{confidence_color} {match_info.get('algorithm_confidence', 'High')}</h3>
        </div>
        """, unsafe_allow_html=True)

def display_referee_analysis(referee_profile: Dict[str, Any]):
    """Visualizza analisi arbitro."""
    severity_emoji = {
        'strict': '🚫',
        'medium': '⚖️',
        'permissive': '✅'
    }
    
    referee_name = referee_profile['Nome']
    severity = referee_profile['Severity']
    emoji = severity_emoji.get(severity, '❓')
    
    st.markdown(f"""
    <div class='referee-info'>
        <h3>{emoji} Analisi Arbitro: {referee_name}</h3>
        <p>
            Media Cartellini Gialli a Partita: <strong>{referee_profile['Gialli_a_partita']:.1f}</strong>
        </p>
        <p>
            Livello di Severità: <strong>{severity.upper()}</strong> ({referee_profile['Description']})
        </p>
    </div>
    """, unsafe_allow_html=True)
    
# =========================================================================
# MAIN APP
# =========================================================================
def main():
    init_session_state()
    
    st.markdown("<div class='main-header'><h1>⚽ Il Mostro 5.0</h1><p>Sistema Avanzato Predizione Cartellini (v5.0)</p></div>", unsafe_allow_html=True)

    data = load_excel_data()
    
    # ********* CORREZIONE QUI *********
    if data is None or st.session_state['full_df_players'] is None or st.session_state['full_df_players'].empty:
        st.error("❌ Impossibile caricare o processare i dati dei giocatori. Verifica che il file Excel esista, sia accessibile e contenga dati squadra validi.")
        st.stop()
        
    df_players = st.session_state['full_df_players']
    df_referees = data['referees']
    
    # Tentiamo di estrarre i dati in un blocco try/except in caso di colonne mancanti
    try:
        team_list = sorted(df_players['Squadra'].unique().tolist())
        referee_list = sorted(df_referees['Nome'].unique().tolist())
    except KeyError as e:
        st.error(f"❌ Errore nella lettura delle colonne necessarie: {e}. Controlla la struttura del tuo file Excel.")
        st.stop()
    # ********************************

    # --- Sidebar per Input ---
    st.sidebar.header("📝 Configura Partita")
    
    # Selezione Squadre
    default_home_index = team_list.index('Inter') if 'Inter' in team_list else 0
    default_away_index = team_list.index('Juventus') if 'Juventus' in team_list and 'Juventus' != team_list[default_home_index] else 0
    st.session_state['home_team'] = st.sidebar.selectbox("🏠 Squadra Casa", team_list, index=default_home_index)
    
    # Filtra away team per non essere uguale a home team
    filtered_away_list = [t for t in team_list if t != st.session_state['home_team']]
    if st.session_state['away_team'] not in filtered_away_list:
        if 'Juventus' in filtered_away_list:
            default_away_index = filtered_away_list.index('Juventus')
        elif len(filtered_away_list) > 0:
            default_away_index = 0
        else:
            default_away_index = 0
            
    st.session_state['away_team'] = st.sidebar.selectbox("✈️ Squadra Trasferta", filtered_away_list, index=default_away_index)
    
    # Selezione Arbitro
    default_referee_index = referee_list.index('Orsato') if 'Orsato' in referee_list else 0
    st.session_state['referee'] = st.sidebar.selectbox("⚖️ Arbitro", referee_list, index=default_referee_index)
    
    # Pulsante per avviare la predizione
    run_button_key = 'run_prediction_button'
    if st.sidebar.button("🚀 Esegui Predizione", key=run_button_key, type='primary'):
        # 1. Resetta i risultati precedenti per un nuovo calcolo
        st.session_state['result'] = None
        st.session_state['excluded_pre'] = []
        st.session_state['scrolled_top_4'] = None
        st.session_state['scrolled_exclusions'] = []
        
        # 2. Avvia la FASE 1: Identificazione delle vittime ad alto rischio
        with st.spinner("Preparazione e identificazione titolari ad alto rischio..."):
            home_df_filtered = df_players[df_players['Squadra'] == st.session_state['home_team']]
            away_df_filtered = df_players[df_players['Squadra'] == st.session_state['away_team']]
            high_risk_victims = identify_high_risk_victims(home_df_filtered, away_df_filtered)
            
            if high_risk_victims:
                st.session_state['need_phase_1_confirmation'] = True
                st.session_state['high_risk_victims'] = high_risk_victims
            else:
                # Se non ci sono vittime, salta la fase 1 e calcola direttamente
                st.session_state['need_phase_1_confirmation'] = False
                st.session_state['result'] = run_prediction(
                    df_players, 
                    df_referees, 
                    st.session_state['home_team'], 
                    st.session_state['away_team'], 
                    st.session_state['referee'],
                    [], 
                    SuperAdvancedCardPredictionModel
                )
        st.rerun()
        
    
    # --- Gestione FASE 1: Verifica Titolarità ---
    if st.session_state.get('need_phase_1_confirmation', False) and st.session_state.get('high_risk_victims'):
        
        st.markdown("---")
        excluded_pre = display_starter_verification(st.session_state['high_risk_victims'])
        
        if st.button("✅ Conferma Titolari e Prosegui con Predizione", key='confirm_phase_1', type='secondary'):
            st.session_state['excluded_pre'] = excluded_pre
            st.session_state['need_phase_1_confirmation'] = False 
            st.session_state['result'] = None 
            st.rerun() 
            
        return 

    # --- Esecuzione Modello e Visualizzazione Risultati ---
    
    if st.session_state['result'] is None and not st.session_state.get('need_phase_1_confirmation', False):
        if st.session_state['home_team'] and st.session_state['away_team'] and st.session_state['referee']:
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