import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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
# STILE PERSONALIZZATO
# =========================================================================
st.markdown("""
<style>
    /* Intestazione principale - Tonalità Giallo/Arancio (Cartellino) */
    .main-header {
        background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: #333; /* Testo scuro sul giallo */
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1, .main-header p {
        color: #333;
    }

    /* Stile per il TOP 4 Dinamico */
    .dynamic-top-prediction {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    /* Sezione Dati Partita - Tonalità Blu (Informazione) */
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
    /* Arbitro - Tonalità Viola */
    .referee-info {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Contenitore del Giocatore nel TOP 4 (Card Nette e Semplici) */
    .player-card {
        border-left: 6px solid; /* Colore dinamico per il rischio */
        padding: 15px 15px;
        margin-bottom: 15px;
        border-radius: 10px;
        background-color: #f7f9fc; /* Sfondo leggermente grigio */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Ombreggiatura netta */
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: background-color 0.2s;
    }
    .player-card:hover {
        background-color: #f0f3f7; /* Leggero hover */
    }
    .player-details h4 {
        margin-bottom: 3px !important;
        font-size: 1.1em;
        font-weight: 600; /* Manteniamo grassetto tramite CSS */
    }
    .player-details p {
        font-size: 0.9em;
        color: #777;
    }
    /* Stilizzazione del numero di Rank */
    .player-details .rank-number {
        font-size: 1.2em; 
        font-weight: 900; 
        margin-right: 5px;
    }
    
    /* Stile per il pulsante 'Escludi' nella card */
    .stButton>button {
        background-color: #f44336; /* Rosso vivo per Escludi */
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #d32f2f;
    }
    
</style>
""", unsafe_allow_html=True)

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
    
    # Carica arbitri
    default_referee_data = {
        'Nome': ['Doveri', 'Orsato', 'Mariani', 'Pairetto', 'Massa', 'Guida'],
        'Gialli a partita': [4.2, 3.8, 5.1, 4.5, 3.2, 4.8]
    }

    df_referees = None
    try:
        if REFEREE_SHEET_NAME in available_sheets:
            df_referees = pd.read_excel(xls, REFEREE_SHEET_NAME)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Errore caricamento arbitri: {e}")

    if df_referees is None or df_referees.empty:
        df_referees = pd.DataFrame(default_referee_data)
    else:
        # Normalizza colonne
        if 'Nome' not in df_referees.columns:
            name_cols = [col for col in df_referees.columns if 'nome' in col.lower() or 'arbitro' in col.lower()]
            if name_cols:
                df_referees.rename(columns={name_cols[0]: 'Nome'}, inplace=True)
        
        if 'Gialli a partita' not in df_referees.columns:
            card_cols = [col for col in df_referees.columns if 'gialli' in col.lower() and 'partita' in col.lower()]
            if card_cols:
                df_referees.rename(columns={card_cols[0]: 'Gialli a partita'}, inplace=True)
            else:
                df_referees = pd.DataFrame(default_referee_data)

    data['referees'] = df_referees
    data = preprocess_data(data)
    
    return data

def preprocess_data(data):
    """Preprocessa i dati."""
    df_players = data['players']
    
    # Normalizza colonne numeriche
    numeric_columns = [
        'Media Falli Subiti 90s Totale', 'Media Falli Fatti 90s Totale',
        'Cartellini Gialli Totali', 'Media Falli per Cartellino Totale',
        'Media 90s per Cartellino Totale', 'Ritardo Cartellino (Minuti)',
        'Minuti Giocati Totali', '90s Giocati Totali'
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
    
    df_players['Posizione_Primaria'] = df_players['Posizione_Primaria'].astype(str).str.strip().str.upper()
    
    # Heatmap
    if 'Heatmap' not in df_players.columns:
        df_players['Heatmap'] = 'Central activity'
    df_players['Heatmap'] = df_players['Heatmap'].astype(str)
    
    data['players'] = df_players
    return data

# =========================================================================
# FUNZIONI GESTIONE TITOLARI (FASE 1)
# =========================================================================
def get_fouls_suffered_metric(df):
    """Estrae la metrica falli subiti."""
    df = df.copy()
    has_total = 'Media Falli Subiti 90s Totale' in df.columns
    has_seasonal = 'Media Falli Subiti 90s Stagionale' in df.columns
    
    if has_total:
        df['Falli_Subiti_Used'] = pd.to_numeric(df['Media Falli Subiti 90s Totale'], errors='coerce').fillna(0)
        if has_seasonal:
            mask_zero = df['Falli_Subiti_Used'] == 0
            seasonal_values = pd.to_numeric(df['Media Falli Subiti 90s Stagionale'], errors='coerce').fillna(0)
            df.loc[mask_zero, 'Falli_Subiti_Used'] = seasonal_values[mask_zero]
            df['Falli_Subiti_Source'] = df.apply(
                lambda row: 'Stagionale' if row['Falli_Subiti_Used'] > 0 and (pd.isna(row.get('Media Falli Subiti 90s Totale')) or row.get('Media Falli Subiti 90s Totale', 0) == 0) else 'Totale',
                axis=1
            )
        else:
            df['Falli_Subiti_Source'] = 'Totale'
    elif has_seasonal:
        df['Falli_Subiti_Used'] = pd.to_numeric(df['Media Falli Subiti 90s Stagionale'], errors='coerce').fillna(0)
        df['Falli_Subiti_Source'] = 'Stagionale'
    else:
        df['Falli_Subiti_Used'] = 0
        df['Falli_Subiti_Source'] = 'N/A'
    return df

def identify_high_risk_victims(home_df, away_df):
    """Identifica SOLO giocatori che SUBISCONO molti falli."""
    high_risk_victims = []
    for df, team_type in [(home_df, 'Casa'), (away_df, 'Trasferta')]:
        df = get_fouls_suffered_metric(df)
        df_valid = df[df['Falli_Subiti_Used'] > 0].copy()
        if df_valid.empty:
            continue
        threshold_suffered = df_valid['Falli_Subiti_Used'].quantile(0.70)
        victims = df_valid[df_valid['Falli_Subiti_Used'] >= threshold_suffered].copy()
        victims = victims.sort_values('Falli_Subiti_Used', ascending=False)
        for _, player in victims.iterrows():
            high_risk_victims.append({
                'Player': player['Player'],
                'Squadra': player['Squadra'],
                'Team_Type': team_type,
                'Falli_Subiti_90': player['Falli_Subiti_Used'],
                'Falli_Source': player['Falli_Subiti_Source'],
                'Posizione': player.get('Posizione_Primaria', 'N/A'),
                'Ruolo': player.get('Ruolo', 'N/A')
            })
    return high_risk_victims

def display_starter_verification(high_risk_victims):
    """Mostra interfaccia verifica titolarità per FASE 1."""
    st.markdown("---")
    st.markdown("### 🔍 FASE 1: Verifica Titolarità Giocatori che Subiscono Molti Falli")
    st.markdown("""
    <div class='verification-box'>
        <h4>⚠️ Giocatori ad Alto Rischio di Subire Falli (Duelli Critici)</h4>
        <p><strong>👉 Seleziona i giocatori NON TITOLARI</strong> per escluderli dall'analisi iniziale.</p>
    </div>
    """, unsafe_allow_html=True)
    
    excluded = []
    if not high_risk_victims:
        st.info("✅ Nessun giocatore ad alto rischio identificato.")
        return excluded
    
    home_victims = [p for p in high_risk_victims if p['Team_Type'] == 'Casa']
    away_victims = [p for p in high_risk_victims if p['Team_Type'] == 'Trasferta']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if home_victims:
            st.markdown(f"#### 🏠 {home_victims[0]['Squadra']}")
            for player in home_victims:
                risk_emoji = "🔴" if player['Falli_Subiti_90'] > 2.5 else "🟡"
                is_excluded = st.checkbox(
                    f"{risk_emoji} {player['Player']} ({player['Ruolo']}) - "
                    f"{player['Falli_Subiti_90']:.1f} FS/90",
                    key=f"pre_home_{player['Player']}"
                )
                if is_excluded:
                    excluded.append(player['Player'])
    
    with col2:
        if away_victims:
            st.markdown(f"#### ✈️ {away_victims[0]['Squadra']}")
            for player in away_victims:
                risk_emoji = "🔴" if player['Falli_Subiti_90'] > 2.5 else "🟡"
                is_excluded = st.checkbox(
                    f"{risk_emoji} {player['Player']} ({player['Ruolo']}) - "
                    f"{player['Falli_Subiti_90']:.1f} FS/90",
                    key=f"pre_away_{player['Player']}"
                )
                if is_excluded:
                    excluded.append(player['Player'])
    
    st.markdown("---")
    
    if excluded:
        st.warning(f"⚠️ **{len(excluded)} giocatori NON TITOLARI esclusi:** {', '.join(excluded)}")
        st.session_state['excluded_pre'] = excluded
    else:
        st.success("✅ Nessuna esclusione (FASE 1).")
        st.session_state['excluded_pre'] = []
    
    return excluded
# =========================================================================
# VISUALIZZAZIONI DINAMICHE
# =========================================================================

def apply_balancing_logic(predictions_df, home_team_name, away_team_name):
    """
    Applica la logica di distribuzione ottimizzata al DataFrame filtrato.
    """
    home_risks = predictions_df[predictions_df['Squadra'] == home_team_name]
    away_risks = predictions_df[predictions_df['Squadra'] == away_team_name]
    
    top_4_ottimizzato = []
    
    top_4_iniziale = predictions_df.head(4)
    count_home = (top_4_iniziale['Squadra'] == home_team_name).sum()
    count_away = (top_4_iniziale['Squadra'] == away_team_name).sum()

    RISK_DIFFERENCE_THRESHOLD = 0.40 

    # La logica ottimizzata (2-2 preferito, 3-1 accettato solo se rischio netto) rimane intatta
    if (count_home == 4 or count_away == 4) and len(home_risks) >= 2 and len(away_risks) >= 2:
        # Se troppo sbilanciato (4-0/0-4) -> FORZA 2-2
        top_4_ottimizzato.extend(home_risks.head(2).to_dict('records'))
        top_4_ottimizzato.extend(away_risks.head(2).to_dict('records'))
    
    elif (count_home == 3 and count_away == 1) or (count_home == 1 and count_away == 3):
        
        dominant_risks = home_risks if count_home == 3 else away_risks
        minor_risks = away_risks if count_home == 3 else home_risks
        
        if len(minor_risks) < 2:
            top_4_ottimizzato = top_4_iniziale.to_dict('records')
        else:
            risk_dominant_3rd = dominant_risks.iloc[2]['Rischio_Finale']
            risk_minor_2nd = minor_risks.iloc[1]['Rischio_Finale']
            
            if risk_dominant_3rd > (risk_minor_2nd + RISK_DIFFERENCE_THRESHOLD):
                 # Accetta 3-1 se la differenza è netta
                 top_4_ottimizzato = top_4_iniziale.to_dict('records')
            else:
                 # Se la differenza non è netta, forza il 2-2
                 top_4_ottimizzato.extend(home_risks.head(2).to_dict('records'))
                 top_4_ottimizzato.extend(away_risks.head(2).to_dict('records'))

    elif count_home == 2 and count_away == 2:
        # Mantiene il 2-2 se è già presente
        top_4_ottimizzato = top_4_iniziale.to_dict('records')
    
    else:
        # Ogni altro caso -> FORZA 2-2 (se possibile)
        top_4_ottimizzato.extend(home_risks.head(min(2, len(home_risks))).to_dict('records'))
        top_4_ottimizzato.extend(away_risks.head(min(2, len(away_risks))).to_dict('records'))
        top_4_ottimizzato = sorted(top_4_ottimizzato, key=lambda x: x['Rischio_Finale'], reverse=True)[:4]

    # Riordina il TOP 4 ottimizzato in base al Rischio_Finale
    final_df = pd.DataFrame(top_4_ottimizzato).sort_values(
        'Rischio_Finale', ascending=False
    )
    
    return final_df[['Player', 'Squadra', 'Rischio_Finale', 'Quota_Stimata', 'Zona_Campo', 'Ruolo']].to_dict('records')


def get_risk_color(risk_score):
    """Restituisce un colore basato sul rischio finale."""
    if risk_score >= 0.70:
        return "#f44336"  # Rosso
    elif risk_score >= 0.55:
        return "#ff9800"  # Arancio scuro
    elif risk_score >= 0.40:
        return "#ffc107"  # Giallo
    else:
        return "#4caf50"  # Verde (rischio basso)


def display_dynamic_top_4():
    """
    Visualizza il TOP 4 con la nuova grafica pulita e il pulsante Escludi integrato.
    """
    
    if 'result' not in st.session_state or st.session_state['result'] is None:
        return 

    result = st.session_state.get('recalculated_result', st.session_state['result'])
    all_predictions_df = result['all_predictions']
    
    # Decide quale TOP 4 mostrare: scorrimento > ricalcolo > originale
    if 'scrolled_top_4' in st.session_state and st.session_state['scrolled_top_4'] is not None:
        current_top_4 = st.session_state['scrolled_top_4']
    else:
        current_top_4 = result['top_4_predictions']
        
    st.markdown("## 🎯 TOP 4 PRONOSTICO CARTELLINI")
    st.markdown("Clicca sul pulsante '❌ Escludi' all'interno di ogni card per rimuovere un giocatore non titolare e far scorrere la graduatoria in modo ottimizzato.")


    if 'scrolled_exclusions' not in st.session_state:
        st.session_state['scrolled_exclusions'] = []
    
    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    
    # Contenitore principale per le card dei giocatori
    cols = st.columns(2)
    
    # Visualizza i giocatori attuali del TOP 4
    for i, prediction in enumerate(current_top_4, 1):
        player_name = prediction.get('Player', 'Sconosciuto')
        squadra = prediction.get('Squadra', 'N/A')
        ruolo = prediction.get('Ruolo', 'N/A')
        rischio = prediction.get('Rischio_Finale', 0.0)
        
        card_color = get_risk_color(rischio)
        
        # Sceglie la colonna in base all'indice
        with cols[(i-1) % 2]:
            
            # Inizio Card HTML (Rimosso ogni riferimento al grassetto inutile e bilanciamento)
            st.markdown(f"""
            <div class='player-card' style='border-left-color: {card_color};'>
                <div class='player-details'>
                    <div>
                        <h4 style='color: #2c3e50; margin-bottom: 3px !important;'>
                            <span class='rank-number' style='color: {card_color};'>#{i}</span>
                            {player_name} </h4>
                        <p style='margin: 0;'>
                            {squadra} • Ruolo: {ruolo} 
                        </p>
                    </div>
                    <div style='text-align: right;'>
            """, unsafe_allow_html=True)

            # Pulsante Escludi (dentro il blocco markdown per essere nella card)
            if player_name not in st.session_state['scrolled_exclusions']:
                # Usa una chiave unica e breve per Streamlit
                if st.button(f"❌ Escludi", key=f"exclude_btn_{player_name}", use_container_width=False):
                    
                    newly_excluded = player_name
                    st.session_state['scrolled_exclusions'].append(newly_excluded)
                    
                    excluded_players = st.session_state['scrolled_exclusions']
                    
                    # DataFrame filtrato
                    new_top_predictions_df = all_predictions_df[
                        ~all_predictions_df['Player'].isin(excluded_players)
                    ].copy() 
                    
                    # APPLICA LA LOGICA DI DISTRIBUZIONE OTTIMIZZATA
                    new_top_4 = apply_balancing_logic(
                        new_top_predictions_df, 
                        st.session_state['home_team'], 
                        st.session_state['away_team']
                    )

                    st.session_state['scrolled_top_4'] = new_top_4
                    st.rerun() # Forza il ricaricamento
            else:
                 st.markdown("<span style='color: #f44336; font-weight: bold; font-size: 0.9em;'>ESCLUSO</span>", unsafe_allow_html=True)


            # Fine Card HTML
            st.markdown("</div></div>", unsafe_allow_html=True)


    # Messaggio di stato
    if st.session_state['scrolled_exclusions']:
        # Rimosso il riferimento esplicito a "logica bilanciata"
        st.warning(f"⚠️ **ATTENZIONE:** TOP 4 modificato per scorrimento (distribuzione ottimizzata applicata). Esclusi (post-analisi): {', '.join(st.session_state['scrolled_exclusions'])}")
        if st.button("↩️ Ripristina TOP 4 Originale", key='reset_scrolling', type='primary'):
            if 'scrolled_top_4' in st.session_state:
                del st.session_state['scrolled_top_4']
            if 'scrolled_exclusions' in st.session_state:
                del st.session_state['scrolled_exclusions']
            st.rerun()
    st.markdown("---") # Separatore dopo il TOP 4

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
    
    # Selezione squadre e arbitro
    all_teams = sorted(df_players['Squadra'].unique())
    all_referees = sorted(df_referees['Nome'].unique())
    
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
        
        # Prepara DataFrame della partita
        initial_home_df = df_players[df_players['Squadra'] == home_team].copy()
        initial_away_df = df_players[df_players['Squadra'] == away_team].copy()
        ref_df = df_referees[df_referees['Nome'] == referee].copy()
        
        if initial_home_df.empty or initial_away_df.empty:
            st.error("❌ Dati insufficienti per le squadre selezionate.")
            return
        
        # === FASE 1: VERIFICA TITOLARITÀ PRE-ELABORAZIONE ===
        high_risk_victims = identify_high_risk_victims(initial_home_df, initial_away_df)
        
        excluded_pre = []
        if high_risk_victims:
            excluded_pre = display_starter_verification(high_risk_victims)
        else:
            st.info("ℹ️ Nessun giocatore ad alto rischio (falli subiti) rilevato. Procedi con l'elaborazione.")
        
        # Applica esclusioni PRE ai DataFrame che saranno passati al modello
        home_df_filtered = initial_home_df[~initial_home_df['Player'].isin(excluded_pre)]
        away_df_filtered = initial_away_df[~initial_away_df['Player'].isin(excluded_pre)]
        
        if excluded_pre:
            st.info(f"🔧 **Analisi configurata:** {len(excluded_pre)} giocatori esclusi come non titolari (FASE 1).")
        
        # Pulsante elaborazione
        if st.button("🎯 Elabora Pronostico", type="primary", use_container_width=True):
            with st.spinner("🔄 Analisi in corso... Elaborazione algoritmo avanzato..."):
                try:
                    model = SuperAdvancedCardPredictionModel()
                    result = model.predict_match_cards(home_df_filtered, away_df_filtered, ref_df)
                    
                    st.success("✅ Analisi completata! Usa i pulsanti '❌ Escludi' per lo scorrimento del TOP 4.")
                    
                    st.session_state['result'] = result
                    st.session_state['home_team'] = home_team
                    st.session_state['away_team'] = away_team
                    st.session_state['referee'] = referee
                    
                    # Resetta ricalcolo e scorrimento al risultato iniziale
                    st.session_state['recalculated_result'] = result
                    if 'scrolled_top_4' in st.session_state:
                         del st.session_state['scrolled_top_4']
                    if 'scrolled_exclusions' in st.session_state:
                         del st.session_state['scrolled_exclusions']
                    
                except Exception as e:
                    st.error(f"❌ Errore: {str(e)}")
                    return
        
        # Mostra risultati se disponibili
        if st.session_state['result'] is not None:
            
            # TOP 4 VISUALIZZAZIONE DINAMICA (Semplificata)
            display_dynamic_top_4()
            
            st.markdown("---")
            
            # Il resto delle analisi usa l'ultimo risultato completo (originale o ricalcolato)
            result = st.session_state.get('recalculated_result', st.session_state['result'])
            
            # Analisi partita
            display_match_analysis(result)
            
            st.markdown("---")
            
            # Arbitro
            display_referee_analysis(result['referee_profile'])
            
            st.markdown("---")
            
            # Download (sempre dalla graduatoria completa originale/ricalcolata)
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
    
    # === INIZIALIZZAZIONE SESSION STATE ROBUSTA ===
    if 'full_df_players' not in st.session_state:
        st.session_state['full_df_players'] = None
    if 'excluded_pre' not in st.session_state:
         st.session_state['excluded_pre'] = []
    if 'result' not in st.session_state:
         st.session_state['result'] = None
    if 'recalculated_result' not in st.session_state:
         st.session_state['recalculated_result'] = None
    if 'scrolled_top_4' not in st.session_state:
         st.session_state['scrolled_top_4'] = None
    if 'scrolled_exclusions' not in st.session_state:
         st.session_state['scrolled_exclusions'] = []
    if 'home_team' not in st.session_state:
         st.session_state['home_team'] = None
    if 'away_team' not in st.session_state:
         st.session_state['away_team'] = None
    # ===============================================

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

    # === SALVATAGGIO IN SESSION STATE SOLO DOPO IL CARICAMENTO ===
    if st.session_state['full_df_players'] is None:
        st.session_state['full_df_players'] = df_players
    # =========================================================================
    
    st.sidebar.success(f"✅ Dati caricati: {len(df_players)} giocatori, {len(df_referees)} arbitri")
    
    main_prediction_interface(df_players, df_referees)

if __name__ == '__main__':
    main()