import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib

warnings.filterwarnings('ignore')

# =========================================================================
# IMPORT MODELLO
# =========================================================================
try:
    from optimized_prediction_model import SuperAdvancedCardPredictionModel, get_field_zone, get_player_role_category
    MODEL_LOADED = True
except ImportError:
    st.error("⚠️ Modello ottimizzato non trovato! Assicurati che 'optimized_prediction_model.py' sia nella cartella.")
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
# STILE PERSONALIZZATO
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
        'referee': None
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

def preprocess_data(data):
    """Preprocessa i dati con filtro >=5 partite."""
    df_players = data['players']
    
    initial_count = len(df_players)
    df_players = df_players[df_players.get('90s Giocati Totali', 0) >= 5].copy()
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
            from optimized_prediction_model import get_player_role as get_role
            df_players['Ruolo'] = df_players['Posizione_Primaria'].apply(get_role)
        except ImportError:
            df_players['Ruolo'] = 'MF'
            
    df_players['Posizione_Primaria'] = df_players['Posizione_Primaria'].astype(str).str.strip().str.upper()
    
    if 'Heatmap' not in df_players.columns:
        df_players['Heatmap'] = 'Central activity'
    df_players['Heatmap'] = df_players['Heatmap'].astype(str)
    
    data['players'] = df_players
    return data

# =========================================================================
# FUNZIONI GESTIONE TITOLARI (FASE 1)
# =========================================================================
def get_fouls_suffered_metric(df):
    """Calcola metriche falli subiti."""
    df = df.copy()
    df = df[df.get('90s Giocati Totali', 0) >= 5].copy()
    
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

def identify_high_risk_victims(home_df, away_df):
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

def display_starter_verification(high_risk_victims):
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
            if st.checkbox("Non titolare", key=key):
                excluded_pre.append(victim['Player'])
    
    st.session_state['excluded_pre'] = excluded_pre
    return excluded_pre

# =========================================================================
# FUNZIONI BILANCIAMENTO TOP 4
# =========================================================================
def apply_balancing_logic(all_predictions_df, home_team, away_team):
    """Applica logica di bilanciamento 2-2 al TOP 4."""
    if all_predictions_df.empty:
        return []
    
    # FIX: Usa 'Rischio_Finale' invece di 'Rischio'
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
    
    if count_home == 3 and count_away == 1:
        dominant_risks = top_4_iniziale[top_4_iniziale['Squadra'] == home_team]
        minor_risks = top_4_iniziale[top_4_iniziale['Squadra'] == away_team]
        
        if len(dominant_risks) >= 3 and len(minor_risks) >= 1:
            risk_dominant_3rd = dominant_risks.iloc[2][risk_col]
            risk_minor_2nd = away_risks.iloc[1][risk_col] if len(away_risks) > 1 else 0
            
            if risk_dominant_3rd > (risk_minor_2nd + RISK_DIFFERENCE_THRESHOLD):
                top_4_ottimizzato = top_4_iniziale.to_dict('records')
            else:
                top_4_ottimizzato = []
                top_4_ottimizzato.extend(home_risks.head(2).to_dict('records'))
                top_4_ottimizzato.extend(away_risks.head(2).to_dict('records'))
    
    elif count_home == 1 and count_away == 3:
        dominant_risks = top_4_iniziale[top_4_iniziale['Squadra'] == away_team]
        minor_risks = top_4_iniziale[top_4_iniziale['Squadra'] == home_team]
        
        if len(dominant_risks) >= 3 and len(minor_risks) >= 1:
            risk_dominant_3rd = dominant_risks.iloc[2][risk_col]
            risk_minor_2nd = home_risks.iloc[1][risk_col] if len(home_risks) > 1 else 0
            
            if risk_dominant_3rd > (risk_minor_2nd + RISK_DIFFERENCE_THRESHOLD):
                top_4_ottimizzato = top_4_iniziale.to_dict('records')
            else:
                top_4_ottimizzato = []
                top_4_ottimizzato.extend(home_risks.head(2).to_dict('records'))
                top_4_ottimizzato.extend(away_risks.head(2).to_dict('records'))

    elif count_home == 2 and count_away == 2:
        top_4_ottimizzato = top_4_iniziale.to_dict('records')
    
    else:
        top_4_ottimizzato = []
        top_4_ottimizzato.extend(home_risks.head(min(2, len(home_risks))).to_dict('records'))
        top_4_ottimizzato.extend(away_risks.head(min(2, len(away_risks))).to_dict('records'))
        top_4_ottimizzato = sorted(top_4_ottimizzato, key=lambda x: x[risk_col], reverse=True)[:4]

    final_df = pd.DataFrame(top_4_ottimizzato).sort_values(risk_col, ascending=False)
    
    # FIX: Assicura che 'Ruolo' esista
    if 'Ruolo' not in final_df.columns:
        final_df['Ruolo'] = 'N/A'
    
    return final_df[['Player', 'Squadra', risk_col, 'Ruolo']].to_dict('records')

def get_risk_color(risk_score):
    """Restituisce colore basato sul rischio."""
    if risk_score >= 0.70:
        return "#f44336"
    elif risk_score >= 0.55:
        return "#ff9800"
    elif risk_score >= 0.40:
        return "#ffc107"
    else:
        return "#4caf50"

# =========================================================================
# DISPLAY TOP 4 DINAMICO
# =========================================================================
def display_dynamic_top_4():
    """Visualizza TOP 4 con pulsante Escludi."""
    if 'result' not in st.session_state or st.session_state['result'] is None:
        return

    result = st.session_state.get('recalculated_result', st.session_state['result'])
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
    
    # FIX: Determina quale campo di rischio usare
    risk_field = 'Rischio_Finale' if 'Rischio_Finale' in current_top_4[0] else 'Rischio'
    
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
                key_hash = hashlib.md5(f"{player_name}_{i}".encode()).hexdigest()[:8]
                button_key = f"exclude_btn_{key_hash}"
                
                if st.button(f"❌ Escludi", key=button_key, use_container_width=False):
                    st.session_state['scrolled_exclusions'].append(player_name)
                    
                    excluded_players = st.session_state['scrolled_exclusions']
                    new_top_predictions_df = all_predictions_df[
                        ~all_predictions_df['Player'].isin(excluded_players)
                    ].copy()
                    
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
def display_match_analysis(result):
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
    
    with col4:
        confidence_color = "🟢" if match_info['algorithm_confidence'] == 'High' else "🟡"
        st.markdown(f"""
        <div class='metric-box'>
            <h4>📊 Confidenza Algoritmo</h4>
            <h3>{confidence_color} {match_info['algorithm_confidence']}</h3>
        </div>
        """, unsafe_allow_html=True)

def display_referee_analysis(referee_profile):
    """Visualizza analisi arbitro."""
    severity_emoji = {
        'strict': '🚫',
        'medium': '⚖️',
        'permissive': '✅'
    }
    
    emoji = severity_emoji.get(referee_profile['severity_level'], '⚖️')
    
    st.markdown(f"""
    <div class='referee-info'>
        <h4>{emoji} Profilo Arbitro: {referee_profile['name']}</h4>
        <p><strong>Cartellini Gialli per partita:</strong> {referee_profile['cards_per_game']:.1f}</p>
        <p><strong>Fattore severità:</strong> {referee_profile['strictness_factor']:.2f}</p>
        <p><strong>Classificazione:</strong> {referee_profile['severity_level'].title()}</p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================================
# INTERFACCIA PRINCIPALE
# =========================================================================
def main_prediction_interface(df_players, df_referees):
    """Interfaccia principale."""
    
    st.markdown("## 🚀 Sistema Avanzato Predizione Cartellini")
    
    all_referees = sorted(df_referees['Nome'].unique())
    all_teams = sorted(df_players['Squadra'].unique())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_team = st.selectbox("🏠 Squadra Casa", ['Seleziona...'] + all_teams, key='home')
    
    with col2:
        away_team = st.selectbox("✈️ Squadra Trasferta", ['Seleziona...'] + all_teams, key='away')
    
    with col3:
        referee = st.selectbox("⚖️ Arbitro", ['Seleziona...'] + all_referees, key='ref')
    
    if home_team == away_team and home_team != 'Seleziona...':
        st.error("⚠️ Le squadre devono essere diverse!")
        return
    
    if (home_team != 'Seleziona...' and away_team != 'Seleziona...' and referee != 'Seleziona...'):
        
        initial_home_df = df_players[df_players['Squadra'] == home_team].copy()
        initial_away_df = df_players[df_players['Squadra'] == away_team].copy()
        ref_df = df_referees[df_referees['Nome'] == referee].copy()
        
        if initial_home_df.empty or initial_away_df.empty:
            st.error("❌ Dati insufficienti per le squadre selezionate.")
            return
        
        # FASE 1: Verifica titolarità
        high_risk_victims = identify_high_risk_victims(initial_home_df, initial_away_df)
        
        excluded_pre = []
        if high_risk_victims:
            excluded_pre = display_starter_verification(high_risk_victims)
        else:
            st.info("ℹ️ Nessun giocatore ad alto rischio rilevato.")
        
        high_risk_victims_filtered = [
            victim['Player'] for victim in high_risk_victims 
            if victim['Player'] not in excluded_pre
        ]
        
        home_df_filtered = initial_home_df[~initial_home_df['Player'].isin(excluded_pre)]
        away_df_filtered = initial_away_df[~initial_away_df['Player'].isin(excluded_pre)]
        
        if excluded_pre:
            st.info(f"🔧 {len(excluded_pre)} giocatori esclusi (FASE 1).")
        
        if st.button("🎯 Elabora Pronostico", type="primary", use_container_width=True):
            with st.spinner("🔄 Analisi in corso..."):
                try:
                    model = SuperAdvancedCardPredictionModel()
                    result = model.predict_match_cards(
                        home_df_filtered, 
                        away_df_filtered, 
                        ref_df, 
                        high_risk_victims_filtered
                    )
                    
                    st.success("✅ Analisi completata!")
                    
                    st.session_state['result'] = result
                    st.session_state['home_team'] = home_team
                    st.session_state['away_team'] = away_team
                    st.session_state['referee'] = referee
                    st.session_state['recalculated_result'] = result
                    st.session_state['scrolled_top_4'] = None
                    st.session_state['scrolled_exclusions'] = []
                    
                except Exception as e:
                    st.error(f"❌ Errore: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
        
        if st.session_state['result'] is not None:
            display_dynamic_top_4()
            
            st.markdown("---")
            
            result = st.session_state.get('recalculated_result', st.session_state['result'])
            
            display_match_analysis(result)
            st.markdown("---")
            display_referee_analysis(result['referee_profile'])
            st.markdown("---")
            
            csv_data = result['all_predictions'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Scarica Predizioni Complete (CSV)",
                data=csv_data,
                file_name=f"predizioni_{home_team}_vs_{away_team}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    else:
        st.info("👆 Seleziona squadre casa, trasferta e arbitro per iniziare.")

# =========================================================================
# MAIN
# =========================================================================
def main():
    init_session_state()
    
    st.markdown("""
    <div class='main-header'>
        <h1>⚽ Il Mostro 5.0 - Sistema Predizione Cartellini</h1>
        <p>Algoritmo ottimizzato per predire i 4 giocatori più probabili da ammonire</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not MODEL_LOADED:
        return
    
    with st.spinner("📊 Caricamento dati..."):
        data = load_excel_data()
    
    if data is None:
        st.error("❌ Impossibile caricare i dati. Verifica che il file 'Il Mostro 5.0.xlsx' sia presente.")
        return
    
    df_players = data['players']
    df_referees = data['referees']
    
    if st.session_state['full_df_players'] is None:
        st.session_state['full_df_players'] = df_players
    
    total_players_after_filter = len(df_players)
    st.sidebar.success(f"✅ Dati caricati: {total_players_after_filter} giocatori (≥5 part.), {len(df_referees)} arbitri")
    
    with st.sidebar.expander("📋 Lista Arbitri Caricati"):
        for idx, ref in df_referees.iterrows():
            st.write(f"• {ref['Nome']} - {ref['Gialli a partita']:.2f} gialli/partita")
    
    main_prediction_interface(df_players, df_referees)

if __name__ == '__main__':
    main(), unsafe_allow_html=True)
    
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
        """