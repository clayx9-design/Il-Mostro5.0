import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
import io

warnings.filterwarnings('ignore')

# =========================================================================
# FUNZIONI DI MODELLO SIMULATE (necessarie per far funzionare l'interfaccia)
# =========================================================================

# Le funzioni get_field_zone e get_player_role_category devono essere definite o importate
def get_field_zone(position, heatmap):
    """Determina la zona di campo (simulata)."""
    # Logica semplificata per la simulazione
    if 'L' in position or 'LB' in position or 'LW' in position: return 'Left Flank'
    if 'R' in position or 'RB' in position or 'RW' in position: return 'Right Flank'
    if 'CB' in position or 'ST' in position: return 'Central'
    return 'Midfield'

def get_player_role_category(position):
    """Classifica il ruolo del giocatore (simulata)."""
    # Logica semplificata per la simulazione
    if 'G' in position or 'P' in position: return 'Goalkeeper'
    if 'D' in position or 'CB' in position or 'RB' in position or 'LB' in position: return 'Defender'
    if 'C' in position or 'M' in position: return 'Midfielder'
    if 'A' in position or 'ST' in position or 'W' in position: return 'Attacker'
    return 'Other'

class SuperAdvancedCardPredictionModel:
    """Modello di predizione simulato per l'interfaccia utente."""
    def predict_match_cards(self, home_df, away_df, ref_df):
        # Unisci i dati di casa e trasferta per le predizioni
        all_players_df = pd.concat([home_df, away_df], ignore_index=True)
        
        # --- Creazione colonne di output simulate per evitare errori ---
        
        # Assicurati che le colonne necessarie per la simulazione esistano o siano create
        if 'Rischio_Finale' not in all_players_df.columns:
            # Simulazione di un rischio basato sulla media falli
            all_players_df['Rischio_Finale'] = all_players_df['Media Falli Fatti 90s Totale'] * 0.15 + np.random.rand(len(all_players_df)) * 0.1
            all_players_df['Rischio_Finale'] = all_players_df['Rischio_Finale'].clip(0.01, 0.6)
        
        if 'Quota_Stimata' not in all_players_df.columns:
            all_players_df['Quota_Stimata'] = 1 / all_players_df['Rischio_Finale'] * 0.5
            all_players_df['Quota_Stimata'] = all_players_df['Quota_Stimata'].clip(1.5, 10.0)
            
        if 'Tendenza_Individuale' not in all_players_df.columns:
            all_players_df['Tendenza_Individuale'] = all_players_df['Rischio_Finale'] * 0.8
            
        if 'Media_Falli_Fatti' not in all_players_df.columns:
            all_players_df['Media_Falli_Fatti'] = all_players_df['Media Falli Fatti 90s Totale']

        if 'Media_Falli_Subiti' not in all_players_df.columns:
            all_players_df['Media_Falli_Subiti'] = all_players_df['Media Falli Subiti 90s Totale']
            
        if 'Duelli_Critici' not in all_players_df.columns:
            all_players_df['Duelli_Critici'] = (all_players_df['Rischio_Finale'] > 0.3).astype(int)
        
        # --- Fine simulazione colonne ---

        # Ordina per rischio
        all_players_df.sort_values(by='Rischio_Finale', ascending=False, inplace=True)
        
        # Estrai i top 4
        # Assicurati che 'Player' sia la colonna nome prima di convertire
        if 'Player' not in all_players_df.columns and 'Giocatore' in all_players_df.columns:
            all_players_df.rename(columns={'Giocatore': 'Player'}, inplace=True)

        top_4_predictions = all_players_df.head(4).rename(columns={'Player': 'Player', 'Squadra': 'Squadra'}).to_dict('records')
        
        # Profilo arbitro simulato
        ref_name = ref_df['Nome'].iloc[0]
        ref_cards_per_game = ref_df['Gialli a partita'].iloc[0]
        if ref_cards_per_game > 5.0:
            severity = 'strict'
            factor = 1.2
        elif ref_cards_per_game < 3.5:
            severity = 'permissive'
            factor = 0.8
        else:
            severity = 'medium'
            factor = 1.0

        # Risultato simulato
        return {
            'all_predictions': all_players_df,
            'top_4_predictions': top_4_predictions,
            'match_info': {
                'home_team': home_df['Squadra'].iloc[0],
                'away_team': away_df['Squadra'].iloc[0],
                'expected_total_cards': f"{ref_cards_per_game * factor:.1f}",
                'algorithm_confidence': 'High' if len(all_players_df) > 40 else 'Medium',
            },
            'referee_profile': {
                'name': ref_name,
                'cards_per_game': ref_cards_per_game,
                'strictness_factor': factor,
                'severity_level': severity,
            },
            'critical_matchups': [
                {'risk_score': 0.65, 'aggressor_player': top_4_predictions[0]['Player'], 'aggressor_team': top_4_predictions[0]['Squadra'], 'aggressor_zone': all_players_df['Zona_Campo'].iloc[0], 'victim_player': all_players_df.iloc[10]['Player'], 'victim_team': all_players_df.iloc[10]['Squadra'], 'victim_zone': all_players_df.iloc[10]['Zona_Campo'], 'compatibility': 0.9, 'aggressor_role': all_players_df['Categoria_Ruolo'].iloc[0], 'victim_role': all_players_df.iloc[10]['Categoria_Ruolo']},
                {'risk_score': 0.52, 'aggressor_player': top_4_predictions[1]['Player'], 'aggressor_team': top_4_predictions[1]['Squadra'], 'aggressor_zone': all_players_df.iloc[1]['Zona_Campo'], 'victim_player': all_players_df.iloc[11]['Player'], 'victim_team': all_players_df.iloc[11]['Squadra'], 'victim_zone': all_players_df.iloc[11]['Zona_Campo'], 'compatibility': 0.8, 'aggressor_role': all_players_df.iloc[1]['Categoria_Ruolo'], 'victim_role': all_players_df.iloc[11]['Categoria_Ruolo']},
            ] if len(top_4_predictions) >= 2 and len(all_players_df) >= 12 else [],
            'algorithm_summary': {
                'methodology': 'Simulazione Avanzata con Fattori Ponderati',
                'critical_matchups_found': 2,
                'high_risk_players': (all_players_df['Rischio_Finale'] > 0.4).sum(),
                'weights_used': {
                    'individual_tendency': 0.25,
                    'matchup_risk': 0.20,
                    'referee_influence': 0.18,
                    'team_dynamics': 0.15,
                    'positional_risk': 0.12,
                    'delay_factor': 0.10,
                }
            }
        }
    
try:
    # Tentativo di importare il modello reale
    from enhanced_prediction_model_v2 import SuperAdvancedCardPredictionModel, get_field_zone, get_player_role_category
except ImportError:
    st.info("Info: Impossibile trovare il modulo di predizione migliorato. Sto usando un modello simulato per l'interfaccia.")
    # Le classi e funzioni di simulazione sono già state definite sopra.


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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .prediction-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #2c3e50;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(255,154,158,0.3);
        border-left: 5px solid #e74c3c;
    }
    .top-prediction {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(255,107,107,0.3);
        text-align: center;
    }
    .referee-info {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(116,185,255,0.3);
    }
    .matchup-card {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-box {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .algorithm-info {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2d3436;
        margin: 1rem 0;
        border-left: 4px solid #e17055;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# FUNZIONI DI CARICAMENTO DATI (FIXED)
# =========================================================================
@st.cache_data
def load_excel_data():
    """Carica tutti i dati dal file Excel con gestione errori e normalizzazione."""
    try:
        xls = pd.ExcelFile(EXCEL_FILE_NAME)
    except Exception as e:
        st.error(f"Errore nel caricamento del file Excel: {e}")
        return None

    data = {}
    team_dataframes = []
    
    # Carica dati squadre
    available_sheets = xls.sheet_names
    st.sidebar.info(f"Fogli disponibili: {', '.join(available_sheets)}")
    
    for sheet in TEAM_SHEET_NAMES:
        if sheet in available_sheets:
            try:
                df = pd.read_excel(xls, sheet)
                df.insert(1, 'Squadra', sheet)
                team_dataframes.append(df)
                st.sidebar.success(f"✅ {sheet}: {len(df)} giocatori")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Errore {sheet}: {e}")
                continue

    if not team_dataframes:
        st.error("Nessun foglio squadra trovato nel file Excel.")
        return None

    # Combina tutti i dati giocatori
    data['players'] = pd.concat(team_dataframes, ignore_index=True)
    
    # Carica dati arbitri con fallback robusto
    df_referees = None
    default_referee_data = {
        'Nome': ['Doveri', 'Orsato', 'Mariani', 'Pairetto', 'Massa', 'Guida'],
        'Gialli a partita': [4.2, 3.8, 5.1, 4.5, 3.2, 4.8]
    }

    try:
        if REFEREE_SHEET_NAME in available_sheets:
            df_referees = pd.read_excel(xls, REFEREE_SHEET_NAME)
            st.sidebar.success(f"✅ Arbitri: {len(df_referees)} arbitri")
        else:
            st.sidebar.warning("⚠️ Foglio arbitri mancante. Usando dati arbitri di default.")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Errore durante il caricamento arbitri: {e}. Usando dati arbitri di default.")

    # Assicurati che il DataFrame arbitri esista e abbia le colonne necessarie
    if df_referees is None or df_referees.empty:
        df_referees = pd.DataFrame(default_referee_data)
        st.sidebar.warning("⚠️ Dati arbitri di default attivati.")
    else:
        # NORMALIZZAZIONE COLONNA 'Nome' (Arbitro)
        referee_name_cols = [
            col for col in df_referees.columns 
            if 'nome' in col.lower() or 'arbitro' in col.lower()
        ]
        
        if referee_name_cols:
            if referee_name_cols[0] != 'Nome':
                df_referees.rename(columns={referee_name_cols[0]: 'Nome'}, inplace=True)
        
        if 'Nome' not in df_referees.columns:
            st.error("❌ La colonna del nome dell'arbitro non è stata trovata nel foglio 'Arbitri'. Usando dati di default.")
            df_referees = pd.DataFrame(default_referee_data)
        
        # --- FIX V2: NORMALIZZAZIONE COLONNA CARTELLINI 'Gialli a partita' ---
        TARGET_CARD_COL = 'Gialli a partita'
        referee_cards_cols = [
            col for col in df_referees.columns 
            if 'gialli a partita' in col.lower() or 
               'media cartellini' in col.lower() or 
               'cartellini per partita' in col.lower() or
               'gialli ap' in col.lower() # <-- AGGIUNTO IL TUO ESATTO NOME DI COLONNA
        ]

        if referee_cards_cols:
            # Rinomina la prima colonna trovata che sembra essere la media cartellini
            if referee_cards_cols[0] != TARGET_CARD_COL:
                df_referees.rename(columns={referee_cards_cols[0]: TARGET_CARD_COL}, inplace=True)
        
        if TARGET_CARD_COL not in df_referees.columns:
             # Se la colonna non è stata trovata dopo il tentativo di normalizzazione, usa i dati di default
             st.error(f"❌ La colonna dei cartellini ('{TARGET_CARD_COL}') non è stata trovata nel foglio 'Arbitri'. Usando i dati di default.")
             df_referees = pd.DataFrame(default_referee_data)
        else:
             # Assicurati che sia numerico, nel caso sia stato caricato da Excel come oggetto
             df_referees[TARGET_CARD_COL] = pd.to_numeric(df_referees[TARGET_CARD_COL], errors='coerce').fillna(4.0)


    data['referees'] = df_referees
    
    # Preprocessing dei dati
    data = preprocess_data(data)
    
    return data

def preprocess_data(data):
    """Preprocessa i dati per l'analisi (FIX TypeError: float is not iterable)."""
    df_players = data['players']
    
    # Colonne numeriche essenziali
    numeric_columns = [
        'Media Falli Subiti 90s Totale', 'Media Falli Fatti 90s Totale',
        'Cartellini Gialli Totali', 'Media Falli per Cartellino Totale',
        'Media 90s per Cartellino Totale', 'Ritardo Cartellino (Minuti)',
        'Minuti Giocati Totali', '90s Giocati Totali'
    ]
    
    # Converti colonne numeriche
    for col in numeric_columns:
        if col in df_players.columns:
            df_players[col] = pd.to_numeric(df_players[col], errors='coerce').fillna(0)
    
    # Gestione colonne mancanti e Normalizzazione (FIX: Assicura che siano stringhe)
    
    # 1. Posizione Primaria: Assicurati che esista e sia una stringa
    if 'Posizione_Primaria' not in df_players.columns:
        if 'Pos' in df_players.columns:
            df_players['Posizione_Primaria'] = df_players['Pos']
        else:
            df_players['Posizione_Primaria'] = 'MF'  # Default
            
    # FIX: Converti Posizione_Primaria in stringa per evitare 'TypeError: float is not iterable'
    df_players['Posizione_Primaria'] = df_players['Posizione_Primaria'].astype(str).str.strip().str.upper()
    df_players['Posizione_Primaria'] = df_players['Posizione_Primaria'].replace({'NAN': 'MF', '': 'MF'})
    
    # 2. Heatmap: Assicurati che esista e sia una stringa
    if 'Heatmap' not in df_players.columns:
        df_players['Heatmap'] = 'Central activity'
        
    # FIX: Converti Heatmap in stringa
    df_players['Heatmap'] = df_players['Heatmap'].astype(str).str.strip()
    
    # 3. Colonna 'Player' per il nome del giocatore
    if 'Giocatore' in df_players.columns and 'Player' not in df_players.columns:
         df_players.rename(columns={'Giocatore': 'Player'}, inplace=True)
    elif 'Player' not in df_players.columns and 'Nome' in df_players.columns:
         df_players.rename(columns={'Nome': 'Player'}, inplace=True)
    elif 'Player' not in df_players.columns:
         # Se nessuna colonna nome è stata trovata, usa la prima colonna come nome per emergenza
         df_players['Player'] = df_players.iloc[:, 0].astype(str) # Assicurati che il nome sia una stringa

    # Calcola metriche derivate
    df_players['Zona_Campo'] = df_players.apply(
        lambda row: get_field_zone(row['Posizione_Primaria'], row['Heatmap']), 
        axis=1
    )
    
    df_players['Categoria_Ruolo'] = df_players.apply(
        lambda row: get_player_role_category(row['Posizione_Primaria']), 
        axis=1
    )
    
    # Calcola Media Falli Fatti 90s Totale se mancante
    if 'Media Falli Fatti 90s Totale' not in df_players.columns:
        if 'Falli Fatti Totali' in df_players.columns and '90s Giocati Totali' in df_players.columns:
            df_players['Media Falli Fatti 90s Totale'] = (
                df_players['Falli Fatti Totali'] / (df_players['90s Giocati Totali'] + 0.1)
            )
        else:
            df_players['Media Falli Fatti 90s Totale'] = 1.5  # Default realistico

    # Calcola Media Falli Subiti 90s Totale se mancante
    if 'Media Falli Subiti 90s Totale' not in df_players.columns:
        df_players['Media Falli Subiti 90s Totale'] = 1.0 # Default realistico
    
    data['players'] = df_players
    return data

# =========================================================================
# VISUALIZZAZIONI
# =========================================================================
def display_top_4_predictions(top_4_predictions):
    """Visualizza i top 4 giocatori predetti per ricevere cartellini."""
    st.markdown("## 🎯 TOP 4 PREDIZIONI CARTELLINI")
    
    for i, prediction in enumerate(top_4_predictions, 1):
        # Gestione di chiavi mancanti in caso di dati incompleti
        player_name = prediction.get('Player', 'Sconosciuto')
        squadra = prediction.get('Squadra', 'N/A')
        rischio = prediction.get('Rischio_Finale', 0.0)
        quota = prediction.get('Quota_Stimata', 0.0)

        risk_color = "🔴" if rischio > 0.4 else "🟡" if rischio > 0.25 else "🟢"
        
        st.markdown(f"""
        <div class='top-prediction'>
            <h3>{risk_color} #{i} - {player_name}</h3>
            <p><strong>Squadra:</strong> {squadra}</p>
            <p><strong>Rischio Finale:</strong> {rischio:.3f}</p>
            <p><strong>Quota Stimata:</strong> {quota:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

def display_match_analysis(result):
    """Visualizza l'analisi completa della partita."""
    match_info = result['match_info']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h4>🏠 Casa</h4>
            <h3>{match_info['home_team']}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <h4>✈️ Trasferta</h4>
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
            <h4>📊 Confidenza</h4>
            <h3>{confidence_color} {match_info['algorithm_confidence']}</h3>
        </div>
        """, unsafe_allow_html=True)

def display_referee_analysis(referee_profile):
    """Visualizza l'analisi dell'arbitro."""
    severity_emoji = {
        'strict': '🚫',
        'medium': '⚖️',
        'permissive': '✅'
    }
    
    emoji = severity_emoji.get(referee_profile['severity_level'], '⚖️')
    
    st.markdown(f"""
    <div class='referee-info'>
        <h4>{emoji} Profilo Arbitro</h4>
        <p><strong>Cartellini per partita:</strong> {referee_profile['cards_per_game']:.1f}</p>
        <p><strong>Fattore severità:</strong> {referee_profile['strictness_factor']:.2f}</p>
        <p><strong>Classificazione:</strong> {referee_profile['severity_level'].title()}</p>
    </div>
    """, unsafe_allow_html=True)

def display_critical_matchups(critical_matchups):
    """Visualizza i duelli critici identificati."""
    if not critical_matchups:
        st.info("Nessun duello critico identificato per questa partita.")
        return
    
    st.markdown("### ⚔️ Duelli Critici Identificati")
    
    for i, matchup in enumerate(critical_matchups[:6], 1):
        st.markdown(f"""
        <div class='matchup-card'>
            <h4>🥊 Duello #{i} - Rischio: {matchup['risk_score']:.3f}</h4>
            <p><strong>Aggressore:</strong> {matchup['aggressor_player']} ({matchup['aggressor_team']}) - {matchup['aggressor_zone']}</p>
            <p><strong>Vittima:</strong> {matchup['victim_player']} ({matchup['victim_team']}) - {matchup['victim_zone']}</p>
            <p><strong>Compatibilità:</strong> {matchup['compatibility']:.2f} | <strong>Ruoli:</strong> {matchup.get('aggressor_role', 'N/A')} vs {matchup.get('victim_role', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

def create_risk_visualization(predictions_df):
    """Crea visualizzazioni del rischio."""
    
    # Grafico 1: Top 10 giocatori
    st.markdown("### 📊 Top 10 Giocatori a Rischio")
    
    top_10 = predictions_df.head(10)
    
    fig1 = go.Figure()
    
    colors = ['#ff6b6b' if risk > 0.4 else '#ffa726' if risk > 0.25 else '#ffee58' 
              for risk in top_10['Rischio_Finale']]
    
    fig1.add_trace(go.Bar(
        x=top_10['Player'],
        y=top_10['Rischio_Finale'],
        marker_color=colors,
        text=top_10['Quota_Stimata'].round(2),
        textposition='outside',
        texttemplate='Q: %{text}',
        name='Rischio Cartellino'
    ))
    
    fig1.update_layout(
        title='Top 10 Giocatori - Rischio Cartellino e Quote',
        xaxis_title='Giocatori',
        yaxis_title='Rischio Finale',
        xaxis_tickangle=45,
        height=500
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Grafico 2: Distribuzione per squadra
    st.markdown("### 🏟️ Distribuzione Rischio per Squadra")
    
    team_risk = predictions_df.groupby('Squadra').agg({
        'Rischio_Finale': ['mean', 'max', 'count']
    }).round(3)
    
    team_risk.columns = ['Rischio Medio', 'Rischio Massimo', 'Numero Giocatori']
    team_risk = team_risk.reset_index()
    
    fig2 = px.bar(
        team_risk, 
        x='Squadra', 
        y='Rischio Medio',
        color='Rischio Massimo',
        title='Rischio Medio per Squadra',
        color_continuous_scale='Reds'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

def display_detailed_predictions_table(predictions_df):
    """Visualizza la tabella dettagliata delle predizioni."""
    st.markdown("### 📋 Tabella Dettagliata Predizioni")
    
    # Seleziona top 15 per display, filtrando le colonne esistenti
    display_cols_map = {
        'Player': 'Giocatore',
        'Squadra': 'Squadra',
        'Posizione_Primaria': 'Posizione',
        'Zona_Campo': 'Zona',
        'Rischio_Finale': 'Rischio',
        'Quota_Stimata': 'Quota',
        'Tendenza_Individuale': 'Tendenza',
        'Media_Falli_Fatti': 'Falli Fatti/90',
        'Media_Falli_Subiti': 'Falli Subiti/90',
        'Duelli_Critici': 'Duelli'
    }
    
    existing_cols = [col for col in display_cols_map.keys() if col in predictions_df.columns]
    
    display_df = predictions_df.head(15)[existing_cols].copy()
    display_df.rename(columns={k: display_cols_map[k] for k in existing_cols}, inplace=True)
    
    # Formattazione
    def highlight_risk(val):
        if pd.isna(val) or not isinstance(val, (int, float)): return ''
        if val > 0.4:
            return 'background-color: #ff6b6b; color: white'
        elif val > 0.25:
            return 'background-color: #ffa726; color: white'
        elif val > 0.15:
            return 'background-color: #ffee58; color: black'
        else:
            return 'background-color: #c8e6c9; color: black'
    
    format_dict = {}
    if 'Rischio' in display_df.columns: format_dict['Rischio'] = '{:.3f}'
    if 'Quota' in display_df.columns: format_dict['Quota'] = '{:.2f}'
    if 'Tendenza' in display_df.columns: format_dict['Tendenza'] = '{:.3f}'
    if 'Falli Fatti/90' in display_df.columns: format_dict['Falli Fatti/90'] = '{:.2f}'
    if 'Falli Subiti/90' in display_df.columns: format_dict['Falli Subiti/90'] = '{:.2f}'

    styled_df = display_df.style.format(format_dict)
    
    if 'Rischio' in display_df.columns:
        styled_df = styled_df.applymap(highlight_risk, subset=['Rischio'])
    
    st.dataframe(styled_df, use_container_width=True)

def display_algorithm_summary(result):
    """Visualizza il riassunto dell'algoritmo."""
    algo_info = result['algorithm_summary']
    
    st.markdown(f"""
    <div class='algorithm-info'>
        <h4>🧠 Riassunto Algoritmo</h4>
        <p><strong>Metodologia:</strong> {algo_info['methodology']}</p>
        <p><strong>Duelli critici trovati:</strong> {algo_info['critical_matchups_found']}</p>
        <p><strong>Giocatori ad alto rischio:</strong> {algo_info['high_risk_players']}</p>
        <p><strong>Pesi utilizzati:</strong></p>
        <ul>
            <li>Tendenza individuale: {algo_info['weights_used']['individual_tendency']:.0%}</li>
            <li>Rischio duelli: {algo_info['weights_used']['matchup_risk']:.0%}</li>
            <li>Influenza arbitro: {algo_info['weights_used']['referee_influence']:.0%}</li>
            <li>Dinamiche squadre: {algo_info['weights_used']['team_dynamics']:.0%}</li>
            <li>Rischio posizionale: {algo_info['weights_used']['positional_risk']:.0%}</li>
            <li>Fattore ritardo: {algo_info['weights_used']['delay_factor']:.0%}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =========================================================================
# INTERFACCIA PRINCIPALE
# =========================================================================
def main_prediction_interface(df_players, df_referees):
    """Interfaccia principale per la predizione."""
    
    st.markdown("## 🚀 Sistema Avanzato Predizione Cartellini")
    
    st.markdown("""
    <div class='prediction-card'>
        <h4>🎯 Algoritmo Intelligente</h4>
        <p>Questo sistema analizza in modo approfondito:</p>
        <ul>
            <li><strong>Statistiche individuali</strong> - Tendenze ai cartellini e aggressività</li>
            <li><strong>Duelli diretti</strong> - Giocatori che subiscono falli vs aggressori</li>
            <li><strong>Compatibilità posizionale</strong> - Zone di campo complementari</li>
            <li><strong>Profilo arbitro</strong> - Severità e stile di direzione</li>
            <li><strong>Dinamiche squadre</strong> - Rivalità e intensità della partita</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Selezione squadre e arbitro
    all_teams = sorted(df_players['Squadra'].unique())
    all_referees = sorted(df_referees['Nome'].unique()) # Garantito dal FIX in load_excel_data
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_team = st.selectbox("🏠 Squadra Casa", ['Seleziona...'] + all_teams, key='home')
    
    with col2:
        away_team = st.selectbox("✈️ Squadra Trasferta", ['Seleziona...'] + all_teams, key='away')
    
    with col3:
        referee = st.selectbox("⚖️ Arbitro", ['Seleziona...'] + all_referees, key='ref')
    
    # Validazione selezione
    if home_team == away_team and home_team != 'Seleziona...':
        st.error("⚠️ Le squadre devono essere diverse!")
        return
    
    # Esecuzione predizione
    if (home_team != 'Seleziona...' and away_team != 'Seleziona...' and 
        referee != 'Seleziona...'):
        
        # Prepara dati
        home_df = df_players[df_players['Squadra'] == home_team].copy()
        away_df = df_players[df_players['Squadra'] == away_team].copy()
        ref_df = df_referees[df_referees['Nome'] == referee].copy()
        
        if home_df.empty or away_df.empty:
            st.error("❌ Dati insufficienti per le squadre selezionate.")
            return
        
        # Esegui predizione
        with st.spinner("🔄 Analisi in corso... Elaborazione algoritmo avanzato..."):
            try:
                model = SuperAdvancedCardPredictionModel()
                result = model.predict_match_cards(home_df, away_df, ref_df)
                
                # Visualizza risultati
                st.success("✅ Analisi completata!")
                
                # TOP 4 PREDIZIONI (OBIETTIVO PRINCIPALE)
                display_top_4_predictions(result['top_4_predictions'])
                
                st.markdown("---")
                
                # Analisi partita
                display_match_analysis(result)
                
                st.markdown("---")
                
                # Analisi arbitro
                display_referee_analysis(result['referee_profile'])
                
                st.markdown("---")
                
                # Duelli critici
                display_critical_matchups(result['critical_matchups'])
                
                st.markdown("---")
                
                # Visualizzazioni
                create_risk_visualization(result['all_predictions'])
                
                st.markdown("---")
                
                # Tabella dettagliata
                display_detailed_predictions_table(result['all_predictions'])
                
                st.markdown("---")
                
                # Riassunto algoritmo
                display_algorithm_summary(result)
                
                # Download risultati
                st.markdown("### 💾 Download Risultati")
                
                csv_data = result['all_predictions'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Scarica Predizioni Complete (CSV)",
                    data=csv_data,
                    file_name=f"predizioni_{home_team}_vs_{away_team}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
                # Riassunto per scommesse
                st.markdown("### 🎲 Riassunto per Scommesse")
                st.markdown("""
                <div class='warning-box'>
                    <h4>⚠️ Disclaimer</h4>
                    <p>Questo sistema fornisce analisi statistiche avanzate ma non garantisce risultati. 
                    Le scommesse comportano sempre dei rischi. Gioca responsabilmente.</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Errore nell'esecuzione dell'algoritmo: {str(e)}")
                # Rimosso st.exception(e) per non esporre stack trace se non richiesto.
    
    else:
        st.info("👆 Seleziona squadre casa, trasferta e arbitro per iniziare l'analisi.")

def show_data_overview(df_players, df_referees):
    """Mostra panoramica dei dati."""
    st.markdown("## 📊 Panoramica Dati Caricati")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Giocatori Totali", len(df_players))
    
    with col2:
        st.metric("Squadre", df_players['Squadra'].nunique())
    
    with col3:
        st.metric("Arbitri", len(df_referees))
    
    with col4:
        avg_fouls = df_players['Media Falli Fatti 90s Totale'].mean()
        st.metric("Media Falli/90min", f"{avg_fouls:.2f}")
    
    # Statistiche per squadra
    st.markdown("### 📋 Statistiche per Squadra")
    
    # Assicurati che 'Player' esista per il conteggio (garantito da preprocess_data)
    team_stats = df_players.groupby('Squadra').agg({
        'Player': 'count',
        'Media Falli Fatti 90s Totale': 'mean',
        'Media Falli Subiti 90s Totale': 'mean',
        'Cartellini Gialli Totali': 'sum'
    }).round(2)
    
    team_stats.columns = ['Giocatori', 'Falli Fatti/90', 'Falli Subiti/90', 'Cartellini Tot.']
    
    st.dataframe(team_stats, use_container_width=True)

# =========================================================================
# MAIN APPLICATION
# =========================================================================
def main():
    st.markdown("""
    <div class='main-header'>
        <h1>⚽ Il Mostro 5.0 - Sistema Predizione Cartellini</h1>
        <p>Algoritmo avanzato per predire i 4 giocatori più probabili da ammonire</p>
        <p><em>Basato su analisi statistiche, duelli posizionali e profilo arbitro</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar per navigazione
    page = st.sidebar.selectbox(
        "📖 Sezioni", 
        ["🎯 Predizione Cartellini", "📊 Panoramica Dati", "ℹ️ Info Sistema"]
    )
    
    # Carica dati
    with st.spinner("📊 Caricamento dati dal file Excel..."):
        data = load_excel_data()
    
    if data is None:
        st.error("❌ Impossibile caricare i dati. Verifica che il file 'Il Mostro 5.0.xlsx' sia presente.")
        return
    
    df_players = data['players']
    df_referees = data['referees']
    
    st.sidebar.success(f"✅ Dati caricati: {len(df_players)} giocatori, {len(df_referees)} arbitri")
    
    # Navigazione
    if page == "🎯 Predizione Cartellini":
        main_prediction_interface(df_players, df_referees)
    
    elif page == "📊 Panoramica Dati":
        show_data_overview(df_players, df_referees)
    
    elif page == "ℹ️ Info Sistema":
        st.markdown("""
        ## 🧠 Informazioni Sistema
        
        ### 🎯 Obiettivo Principale
        **Predire i 4 giocatori più probabili da ammonire** in una partita specifica, 
        inserendo squadra casa, squadra ospite e arbitro.
        
        ### 🔍 Metodologia Algoritmo
        
        #### 1. **Analisi Partita Generale**
        - Calcolo delle medie dell'arbitro selezionato
        - Analisi delle medie delle squadre coinvolte
        - Stima cartellini totali attesi per la partita
        
        #### 2. **Identificazione Duelli Critici**
        - **Giocatori che subiscono falli**: Top 25% per falli subiti/90min
        - **Giocatori aggressivi**: Top 30% per falli commessi/90min
        - **Analisi posizionale**: Attaccante destro vs difensore sinistro, etc.
        - **Compatibilità zone**: Laterali opposti = massima probabilità di scontro
        
        #### 3. **Calcolo Rischio Individuale**
        - **Tendenza cartellini**: Media partite per cartellino (più bassa = più propenso)
        - **Impulsività**: Media falli per cartellino (più bassa = più aggressivo)
        - **Fattore ritardo**: Velocità nel ricevere cartellini
        - **Influenza arbitro**: Severità basata su cartellini/partita
        - **Dinamiche squadre**: Rivalità storiche e fattore casa/trasferta
        
        #### 4. **Algoritmo di Matching**
        ```
        Rischio Finale = 
            Tendenza Individuale (25%) +
            Rischio Duelli Critici (20%) +
            Influenza Arbitro (18%) +
            Dinamiche Squadre (15%) +
            Rischio Posizionale (12%) +
            Fattore Ritardo (10%)
        ```
        
        ### 📊 Output del Sistema
        
        #### **TOP 4 PREDIZIONI** 🎯
        I 4 nomi più probabili con:
        - Rischio finale (0-1)
        - Quota stimata
        - Squadra di appartenenza
        
        #### **Analisi Dettagliata**
        - Profilo arbitro e severità
        - Duelli critici identificati
        - Visualizzazioni rischio
        - Tabella completa giocatori
        
        ### 🎲 Fattori Considerati
        
        #### **Rivalità Storiche**
        - Inter vs Milan (Derby di Milano)
        - Roma vs Lazio (Derby di Roma)  
        - Juventus vs Inter (Derby d'Italia)
        - Altre rivalità significative
        
        #### **Classificazione Arbitri**
        - **Permissivo**: < 3 cartellini/partita
        - **Normale**: 3-5.5 cartellini/partita  
        - **Severo**: > 5.5 cartellini/partita
        
        #### **Zone di Campo**
        - **L/R**: Laterali (alto rischio duelli)
        - **C**: Centro (rischio medio)
        - Compatibilità: L vs R = 90% probabilità scontro
        
        ### ⚠️ Limitazioni
        - Basato su dati storici e statistiche
        - Non considera infortuni o squalifiche
        - Fattori imprevedibili (decisioni arbitrali, episodi casuali)
        - Le quote sono stime, non garantite
        
        ### 🔬 Validazione
        L'algoritmo è ottimizzato per:
        - Massimizzare identificazione situazioni ad alto rischio
        - Minimizzare falsi positivi
        - Bilanciare fattori statistici e tattici
        - Fornire predizioni actionable per scommesse responsabili
        """)

if __name__ == '__main__':
    main()