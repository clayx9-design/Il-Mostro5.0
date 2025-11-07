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
    
    # =========================================================================
    # CARICAMENTO MIGLIORATO ARBITRI
    # =========================================================================
    default_referee_data = {
        'Nome': ['Doveri', 'Orsato', 'Mariani', 'Pairetto', 'Massa', 'Guida'],
        'Gialli a partita': [4.2, 3.8, 5.1, 4.5, 3.2, 4.8]
    }
    
    SERIE_A_AVG_CARDS = 4.2  # Media Serie A come fallback

    df_referees = None
    try:
        if REFEREE_SHEET_NAME in available_sheets:
            # Leggi il foglio arbitri
            df_referees = pd.read_excel(xls, REFEREE_SHEET_NAME)
            
            # Debug: mostra le colonne disponibili
            st.sidebar.info(f"📋 Colonne trovate nel foglio Arbitri: {list(df_referees.columns)}")
            
            # Identifica automaticamente la colonna del nome
            nome_col = None
            for col in df_referees.columns:
                col_lower = str(col).lower().strip()
                if any(keyword in col_lower for keyword in ['nome', 'arbitro', 'name', 'referee']):
                    nome_col = col
                    break
            
            # Identifica automaticamente la colonna dei gialli
            gialli_col = None
            for col in df_referees.columns:
                col_lower = str(col).lower().strip()
                if 'giall' in col_lower and 'partita' in col_lower:
                    gialli_col = col
                    break
                elif 'card' in col_lower or 'yellow' in col_lower:
                    gialli_col = col
                    break
            
            # Se non trova le colonne, prova con la prima e seconda colonna
            if nome_col is None and len(df_referees.columns) > 0:
                nome_col = df_referees.columns[0]
                st.sidebar.warning(f"⚠️ Colonna nome non trovata, uso: {nome_col}")
            
            if gialli_col is None and len(df_referees.columns) > 1:
                gialli_col = df_referees.columns[1]
                st.sidebar.warning(f"⚠️ Colonna gialli non trovata, uso: {gialli_col}")
            
            # Rinomina le colonne
            if nome_col and gialli_col:
                df_referees = df_referees[[nome_col, gialli_col]].copy()
                df_referees.columns = ['Nome', 'Gialli a partita']
                
                # Pulizia dati
                # 1. Converti la colonna Nome in stringa e rimuovi spazi
                df_referees['Nome'] = df_referees['Nome'].astype(str).str.strip()
                
                # 2. Rimuovi righe con nome vuoto, NaN o 'nan'
                df_referees = df_referees[
                    (df_referees['Nome'].notna()) & 
                    (df_referees['Nome'] != '') & 
                    (df_referees['Nome'] != 'nan') &
                    (df_referees['Nome'].str.lower() != 'unnamed')
                ]
                
                # 3. Converti la colonna Gialli in numerico
                df_referees['Gialli a partita'] = pd.to_numeric(
                    df_referees['Gialli a partita'], 
                    errors='coerce'
                )
                
                # 4. Sostituisci valori NaN con la media Serie A
                df_referees['Gialli a partita'].fillna(SERIE_A_AVG_CARDS, inplace=True)
                
                # 5. Rimuovi duplicati
                df_referees.drop_duplicates(subset=['Nome'], keep='first', inplace=True)
                
                # 6. Reset dell'indice
                df_referees.reset_index(drop=True, inplace=True)
                
                st.sidebar.success(f"✅ Caricati {len(df_referees)} arbitri dal foglio Excel")
            else:
                st.sidebar.error("❌ Impossibile identificare le colonne Nome e Gialli")
                df_referees = pd.DataFrame(default_referee_data)
                
    except Exception as e:
        st.sidebar.error(f"⚠️ Errore caricamento arbitri: {e}")
        df_referees = None

    # Se il caricamento è fallito, usa i dati di default
    if df_referees is None or df_referees.empty:
        st.sidebar.warning("⚠️ Uso dati arbitri di default")
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
    
    # Assicuriamo che la colonna Ruolo esista per la FASE 1
    if 'Ruolo' not in df_players.columns:
        try:
            from optimized_prediction_model import get_player_role as get_role
            df_players['Ruolo'] = df_players['Posizione_Primaria'].apply(get_role)
        except ImportError:
            df_players['Ruolo'] = 'MF'
            
    df_players['Posizione_Primaria'] = df_players['Posizione_Primaria'].astype(str).str.strip().str.upper()
    
    # Heatmap
    if 'Heatmap' not in df_players.columns:
        df_players['Heatmap'] = 'Central activity'
    df_players['Heatmap'] = df_players['Heatmap'].astype(str)
    
    data['players'] = df_players
    return data

# =========================================================================
# FUNZIONI GESTIONE TITOLARI (FASE 1) - LOGICA AGGIORNATA
# =========================================================================
def get_fouls_suffered_metric(df):
    """
    Estrae le metriche dei falli subiti (Totale e Stagionale) 
    e calcola la metrica 'Falli_Subiti_Used' (Totale > Stagionale).
    """
    df = df.copy()
    
    # Assicura le colonne per il calcolo
    total_col = 'Media Falli Subiti 90s Totale'
    seasonal_col = 'Media Falli Subiti 90s Stagionale'
    
    has_total = total_col in df.columns
    has_seasonal = seasonal_col in df.columns
    
    # Inizializza le colonne per il calcolo della differenza
    df['Falli_Subiti_Totale'] = df.get(total_col, 0.0)
    df['Falli_Subiti_Stagionale'] = df.get(seasonal_col, 0.0)
    df['90s Giocati Totali'] = df.get('90s Giocati Totali', 0.0) 
    
    # Logica di base per 'Falli_Subiti_Used' (priorità a Totale)
    if has_total:
        df['Falli_Subiti_Used'] = df['Falli_Subiti_Totale']
        if has_seasonal:
            # Se la media totale è zero o nulla, usa la stagionale
            mask_zero = (df['Falli_Subiti_Used'] == 0) | df['Falli_Subiti_Used'].isna()
            df.loc[mask_zero, 'Falli_Subiti_Used'] = df.loc[mask_zero, 'Falli_Subiti_Stagionale']
            df['Falli_Subiti_Source'] = df.apply(
                lambda row: 'Stagionale' if row['Falli_Subiti_Used'] > 0 and mask_zero[row.name] else 'Totale',
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
    """
    Identifica i giocatori che SUBISCONO molti falli, includendo gli attaccanti chiave.
    VERSIONE 5.1: Soglie più strict per ridurre il numero (focus su alto rischio reale).
    """
    all_victims = []
    
    for df, team_type in [(home_df, 'Casa'), (away_df, 'Trasferta')]:
        df = get_fouls_suffered_metric(df)
        
        # Filtra i giocatori con dati giocati significativi (≥1 90s)
        df_valid = df[(df['Falli_Subiti_Used'] > 0) & (df['90s Giocati Totali'] >= 1)].copy()
        
        if df_valid.empty:
            continue
            
        # Calcola lo Spread Stagionale (la differenza diretta)
        df_valid['Stagional_Spread'] = np.where(
            (df_valid['Falli_Subiti_Stagionale'] > 0) & (df_valid['Falli_Subiti_Totale'] > 0),
            (df_valid['Falli_Subiti_Stagionale'] - df_valid['Falli_Subiti_Totale']),
            0
        )
        
        # SOGLIE AGGIORNATE (più strict)
        SPREAD_THRESHOLD_HIGH = 0.5  # Invariato (incremento stagionale forte)
        MIN_FOULS_STANDARD = 2.0     # Aumentato da 1.5: Solo alto rischio
        MIN_90S_ACTIVE = 3.0         # Aumentato da 2.0: Più esperienza
        MIN_90S_TOP_PLAYER = 5.0     # Invariato
        MIN_SEASONAL_FOULS = 2.0     # Aumentato da 1.5
        
        # 1. Rilevazione Estrema (Forte incremento stagionale)
        victims_forced_seasonal = df_valid[
            (df_valid['Stagional_Spread'] >= SPREAD_THRESHOLD_HIGH) &
            (df_valid['Falli_Subiti_Stagionale'] >= MIN_SEASONAL_FOULS) &
            (df_valid['90s Giocati Totali'] >= MIN_90S_ACTIVE)
        ].copy()
        
        # 2. Rilevazione Standard (Media Falli Subiti Alta, >=2.0 + ≥2 90s per qualità)
        victims_standard = df_valid[
            (df_valid['Falli_Subiti_Used'] >= MIN_FOULS_STANDARD) &
            (df_valid['90s Giocati Totali'] >= 2.0)  # NOVITÀ: Filtro extra per standard
        ].copy()
        
        # 3. Rilevazione Top Player (Attaccanti chiave con alto volume di gioco)
        victims_top_player = df_valid[
            (df_valid['Ruolo'] == 'ATT') &
            (df_valid['90s Giocati Totali'] >= MIN_90S_TOP_PLAYER) &
            (df_valid['Falli_Subiti_Used'] >= 1.5)  # NOVITÀ: Aggiunto soglia minima per ATT
        ].copy()

        # Combina le tre liste e rimuovi duplicati
        all_victims_df = pd.concat([victims_standard, victims_forced_seasonal, victims_top_player]).drop_duplicates(subset=['Player'])

        # Processa i risultati combinati
        for _, player in all_victims_df.iterrows():
            all_victims.append({
                'Player': player['Player'],
                'Squadra': player['Squadra'],
                'Falli_Subiti_Used': player['Falli_Subiti_Used'],
                '90s Giocati Totali': player['90s Giocati Totali'],
                'Categoria': 'Standard' if player['Falli_Subiti_Used'] >= MIN_FOULS_STANDARD else 'Top Player' if player['Ruolo'] == 'ATT' else 'Forced Seasonal'
            })
    
    return all_victims

def display_starter_verification(high_risk_victims):
    """Visualizza la verifica titolarità per vittime ad alto rischio."""
    excluded = []
    st.markdown("### 🔍 FASE 1: Verifica Titolarità - Vittime ad Alto Rischio (Falli Subiti)")
    st.info(f"Identificate {len(high_risk_victims)} potenziali 'magneti per falli'. Escludi se non titolari per raffinare i duelli.")
    
    for victim in high_risk_victims:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{victim['Player']}** ({victim['Squadra']}) - {victim['Falli_Subiti_Used']:.2f} falli subiti/90s | {victim['Categoria']}")
        with col2:
            if st.button(f"❌ Escludi {victim['Player']}", key=f"exclude_{victim['Player']}"):
                excluded.append(victim['Player'])
    
    if excluded:
        st.warning(f"🔧 Esclusi: {', '.join(excluded)}")
    return excluded

# =========================================================================
# FUNZIONI LOGICA BILANCIAMENTO TOP 4
# =========================================================================
def apply_balancing_logic(all_predictions_df, home_team, away_team):
    """
    Applica logica di bilanciamento ottimizzata per il TOP 4 dopo esclusioni.
    """
    RISK_DIFFERENCE_THRESHOLD = 0.2
    
    home_risks = all_predictions_df[all_predictions_df['Squadra'] == home_team].sort_values('Rischio_Finale', ascending=False)
    away_risks = all_predictions_df[all_predictions_df['Squadra'] == away_team].sort_values('Rischio_Finale', ascending=False)
    
    count_home = len(home_risks)
    count_away = len(away_risks)
    
    top_4_ottimizzato = []
    top_4_iniziale = pd.concat([home_risks, away_risks]).head(4)
    
    if count_home >= 4 or count_away >= 4:
        # Se una squadra ha 4+, usa i primi 4 di quella e 0 dell'altra
        if count_home >= 4:
            top_4_ottimizzato = home_risks.head(4).to_dict('records')
        else:
            top_4_ottimizzato = away_risks.head(4).to_dict('records')
    
    elif (count_home == 3 and count_away >= 1) or (count_away == 3 and count_home >= 1):
        # 3-1: Verifica se la differenza giustifica lo sbilanciamento
        top_4_ottimizzato = top_4_iniziale.to_dict('records')
        
        # Controlla se forzare 2-2
        if len(away_risks) >= 2 and len(home_risks) >= 2:
            risk_dominant_3rd = home_risks.iloc[2]['Rischio_Finale'] if count_home == 3 else away_risks.iloc[2]['Rischio_Finale']
            risk_minor_2nd = away_risks.iloc[1]['Rischio_Finale'] if count_home == 3 else home_risks.iloc[1]['Rischio_Finale']
            
            if risk_dominant_3rd < (risk_minor_2nd + RISK_DIFFERENCE_THRESHOLD):
                top_4_ottimizzato = []
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
                 top_4_ottimizzato = []
                 top_4_ottimizzato.extend(home_risks.head(2).to_dict('records'))
                 top_4_ottimizzato.extend(away_risks.head(2).to_dict('records'))

    elif count_home == 2 and count_away == 2:
        # Mantiene il 2-2 se è già presente
        top_4_ottimizzato = top_4_iniziale.to_dict('records')
    
    else:
        # Ogni altro caso -> FORZA 2-2 (se possibile)
        top_4_ottimizzato = []
        top_4_ottimizzato.extend(home_risks.head(min(2, len(home_risks))).to_dict('records'))
        top_4_ottimizzato.extend(away_risks.head(min(2, len(away_risks))).to_dict('records'))
        top_4_ottimizzato = sorted(top_4_ottimizzato, key=lambda x: x['Rischio_Finale'], reverse=True)[:4]

    # Riordina il TOP 4 ottimizzato in base al Rischio_Finale
    final_df = pd.DataFrame(top_4_ottimizzato).sort_values(
        'Rischio_Finale', ascending=False
    )
    
    return final_df[['Player', 'Squadra', 'Rischio_Finale', 'Ruolo']].to_dict('records')  # Aggiustato: rimuovi colonne non esistenti


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
        rischio = prediction.get('Rischio', 0.0)  # Nota: nel modello è 'Rischio', non 'Rischio_Finale'
        
        card_color = get_risk_color(rischio)
        
        # Sceglie la colonna in base all'indice
        with cols[(i-1) % 2]:
            
            # Inizio Card HTML
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
                if st.button(f"❌ Escludi", key=f"exclude_btn_{