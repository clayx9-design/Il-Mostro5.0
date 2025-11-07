"""
MODELLO OTTIMIZZATO PER PREDIZIONE CARTELLINI
==============================================
Sistema semplificato, efficace e realistico per predire i 4 giocatori
più probabili a ricevere cartellino giallo in una partita.

NOVITÀ V5.1:
- Duelli BIDIREZIONALI (home vs away e viceversa).
- Boost duelli basato su FALLI SUBITI della vittima (non più su rischio base della vittima).
- Metodo predict_match_cards per compatibilità con app.py.
- Integrazione high_risk_victims da app.py per prioritarizzare.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# =========================================================================
# COSTANTI E CONFIGURAZIONE
# =========================================================================

# Medie Serie A (stagione tipo)
SERIE_A_AVG_CARDS_PER_GAME = 4.2
SERIE_A_AVG_FOULS_PER_PLAYER = 1.8  # Media Falli fatti/90s
SERIE_A_AVG_FOULS_SUFFIRED = 1.8   # NOVITÀ: Media Falli subiti/90s (per vittime)
SERIE_A_AVG_CARDS_PER_PLAYER_SEASON = 3.5

# Pesi del modello (ottimizzati per accuratezza)
WEIGHTS = {
    'historical_tendency': 0.35,      # Storico cartellini del giocatore
    'foul_rate': 0.25,                # Frequenza Falli fatti/90s
    'referee_impact': 0.15,           # Impatto dell'arbitro
    'role_risk': 0.15,                # Rischio intrinseco per ruolo
    'aggression_score': 0.10          # Score aggressività della squadra
}

# Pesi aggiuntivi per duelli e ruolo
POSITION_WEIGHTS = {
    'DF': 1.0, # Rischio base Difensore più alto
    'MF': 0.8,
    'FW': 0.6
}

# Mappatura Zone (per duelli)
ZONE_MAP = {
    'Low activity in defensive third': 'Def_Central',
    'High activity in defensive third, focused on tackles and aerial duels': 'Def_Central',
    'High activity on left flank': 'Wide_Def',
    'High activity on right flank': 'Wide_Def',
    'High activity on left flank in midfield': 'Wide_Mid',
    'High activity on right flank in midfield': 'Wide_Mid',
    'High activity in central midfield': 'Central_Mid',
    'High activity in central areas, focused on passing and transitions': 'Central_Mid',
    'High activity in central attacking third': 'Att_Central',
    'High activity on left flank in attacking third': 'Wide_Att',
    'High activity on right flank in attacking third': 'Wide_Att',
    'High activity in penalty area': 'Att_Central'
}

# Duelli Critici (Aggressore vs Vittima) - Fattore di Rischio Moltiplicativo
ZONE_MATCHUPS = {
    ('Def_Central', 'Att_Central'): 1.3, # DC vs ST
    ('Wide_Def', 'Wide_Att'): 1.5,       # Terzino vs Ala (Duello più caldo)
    ('Central_Mid', 'Central_Mid'): 1.1, # CC vs CC
    ('Wide_Mid', 'Wide_Mid'): 1.2        # Laterale CC vs Laterale CC
}

# Soglia per forzare il bilanciamento 2-2
RISK_DIFFERENCE_THRESHOLD = 0.2

# =========================================================================
# FUNZIONI DI SUPPORTO PER POSIZIONE E ZONA
# =========================================================================

def get_player_role(position: str) -> str:
    """Restituisce la categoria di ruolo basata sulla posizione primaria."""
    # Estrae la prima posizione e la normalizza
    pos = position.split(',')[0].strip().upper()
    if pos in ['CB', 'FB', 'WB', 'DF']: return 'DF'
    if pos in ['DM', 'CM', 'AM', 'MF']: return 'MF'
    if pos in ['LW', 'RW', 'ST', 'FW']: return 'FW'
    return 'MF' # Default

def get_player_zone(position: str, heatmap: str) -> str:
    """Restituisce la zona di attività basata sulla heatmap."""
    for key, zone in ZONE_MAP.items():
        if key in heatmap:
            return zone
    
    # Fallback basato sulla posizione
    role = get_player_role(position)
    if role == 'DF': return 'Def_Central'
    if role == 'MF': return 'Central_Mid'
    if role == 'FW': return 'Att_Central'
    return 'Central_Mid' # Default

# Wrapper per compatibilità con app.py
def get_field_zone(position, heatmap):
    return get_player_zone(position, heatmap)

def get_player_role_category(position):
    return get_player_role(position)

# NOVITÀ: Funzione per calcolare Falli_Subiti_Used (compatibile con app.py)
def calculate_fouls_suffered_used(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola 'Falli_Subiti_Used' se non presente (logica da app.py)."""
    df = df.copy()
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
        df['Falli_Subiti_Source'] = 'Totale'  # Semplificato
    elif has_seasonal:
        df['Falli_Subiti_Used'] = df['Falli_Subiti_Stagionale']
        df['Falli_Subiti_Source'] = 'Stagionale'
    else:
        df['Falli_Subiti_Used'] = 0
        df['Falli_Subiti_Source'] = 'N/A'
    
    df['Falli_Subiti_Used'] = df['Falli_Subiti_Used'].fillna(0)
    return df

# =========================================================================
# MODELLO AVANZATO DI PREDIZIONE
# =========================================================================

class SuperAdvancedCardPredictionModel:
    """
    Modello Ottimizzato 5.1 per la predizione dei cartellini.
    Integra storico, rate di falli, impatto arbitro e duelli dinamici CON FALLI SUBITI.
    """
    def __init__(self):
        pass  # Inizializzazione vuota per compatibilità con app.py

    def _get_player_risk(self, df_player: pd.Series, referee_avg_yellow: float, team_aggression_score: float) -> float:
        """Calcola il rischio base per un singolo giocatore (invariato, basato su falli FATTI)."""
        
        # 1. Tendenza Storica (35%)
        if df_player['Media 90s per Cartellino Totale'] == 0:
            historical_risk = 0.5
        else:
            frequency_ratio = (SERIE_A_AVG_CARDS_PER_PLAYER_SEASON / df_player['90s Giocati Totali']) / (1 / df_player['Media 90s per Cartellino Totale'])
            historical_risk = min(frequency_ratio, 1.0)
        
        # 2. Foul Rate (25%) - Falli FATTI
        if df_player['90s Giocati Totali'] > 5:
            foul_rate_90s = df_player.get('Falli Fatti Totali', 0) / df_player['90s Giocati Totali']
            foul_risk = min(foul_rate_90s / SERIE_A_AVG_FOULS_PER_PLAYER, 1.0)
        else:
            foul_risk = 0.5

        # 3. Impatto Arbitro (15%)
        referee_factor = referee_avg_yellow / SERIE_A_AVG_CARDS_PER_GAME
        referee_risk = 0.5 * referee_factor

        # 4. Rischio di Ruolo (15%)
        role = get_player_role(df_player['Posizione_Primaria'])
        role_risk = POSITION_WEIGHTS.get(role, 0.7) * 0.5

        # 5. Aggressività di Squadra (10%)
        team_risk = team_aggression_score * 0.1

        # Calcolo del rischio base
        base_risk = (
            historical_risk * WEIGHTS['historical_tendency'] +
            foul_risk * WEIGHTS['foul_rate'] +
            referee_risk * WEIGHTS['referee_impact'] +
            role_risk * WEIGHTS['role_risk'] +
            team_risk * WEIGHTS['aggression_score']
        )

        return base_risk

    def _calculate_critical_matchups(self, aggressor_df: pd.DataFrame, victim_df: pd.DataFrame, high_risk_victims: List[str] = None) -> List[Dict]:
        """Identifica e valuta i duelli critici (aggressori vs vittime). NOVITÀ: Usa falli subiti della vittima."""
        
        critical_matchups = []
        
        # Assicura Falli_Subiti_Used
        victim_df = calculate_fouls_suffered_used(victim_df)
        
        # Prioritarizza vittime ad alto rischio se fornite
        victim_priority_bonus = 1.2 if high_risk_victims else 1.0
        
        for a_row in aggressor_df.itertuples():
            a_zone = get_player_zone(a_row.Posizione_Primaria, getattr(a_row, 'Heatmap', ''))
            a_role = get_player_role(a_row.Posizione_Primaria)

            for v_row in victim_df.itertuples():
                v_zone = get_player_zone(v_row.Posizione_Primaria, getattr(v_row, 'Heatmap', ''))
                v_role = get_player_role(v_row.Posizione_Primaria)
                
                # Cerca incroci critici
                if (a_zone, v_zone) in ZONE_MATCHUPS:
                    matchup_risk = ZONE_MATCHUPS.get((a_zone, v_zone))
                    
                    # NOVITÀ: Victim risk basato su FALLI SUBITI (non rischio base)
                    victim_fouls_rate = v_row.Falli_Subiti_Used  # O fallback a Media Falli Subiti 90s Totale
                    victim_risk = min(victim_fouls_rate / SERIE_A_AVG_FOULS_SUFFIRED, 2.0)  # Normalizza, cap 2x
                    
                    # Bonus se vittima è high_risk
                    if high_risk_victims and v_row.Player in high_risk_victims:
                        victim_risk *= victim_priority_bonus
                    
                    # Rischio Duello: Moltiplica per victim_risk
                    final_risk_factor = matchup_risk * (1 + victim_risk * 0.3)  # Fattore 0.3 per boost moderato

                    critical_matchups.append({
                        'aggressor_player': a_row.Player,  # Assumo 'Player' come nome (da app.py)
                        'aggressor_team': a_row.Squadra,
                        'aggressor_zone': a_zone,
                        'aggressor_role': a_role,
                        'victim_player': v_row.Player,
                        'victim_team': v_row.Squadra,
                        'victim_zone': v_zone,
                        'victim_role': v_role,
                        'zone_compatibility': matchup_risk,
                        'victim_fouls_rate': victim_fouls_rate,
                        'risk_factor': final_risk_factor
                    })

        return critical_matchups

    def predict_match_cards(self, home_df: pd.DataFrame, away_df: pd.DataFrame, ref_df: pd.DataFrame, high_risk_victims: List[str] = None) -> Dict:
        """
        Predizione completa per match cards (compatibile con app.py).
        Riceve df filtrati direttamente.
        """
        if high_risk_victims is None:
            high_risk_victims = []
        
        # Arbitro
        referee_avg_yellow = ref_df['Gialli a partita'].iloc[0] if not ref_df.empty else SERIE_A_AVG_CARDS_PER_GAME

        # Score Aggressività di Squadra (basato su falli FATTI)
        home_aggression = home_df['Falli Fatti Totali'].sum() / home_df['90s Giocati Totali'].sum() if home_df['90s Giocati Totali'].sum() > 0 else 0
        away_aggression = away_df['Falli Fatti Totali'].sum() / away_df['90s Giocati Totali'].sum() if away_df['90s Giocati Totali'].sum() > 0 else 0
        
        home_aggression_score = min(home_aggression / SERIE_A_AVG_FOULS_PER_PLAYER, 1.5)
        away_aggression_score = min(away_aggression / SERIE_A_AVG_FOULS_PER_PLAYER, 1.5)
        
        # Calcolo Rischio Base
        home_df['Rischio_Base'] = home_df.apply(
            lambda row: self._get_player_risk(row, referee_avg_yellow, home_aggression_score), axis=1
        )
        away_df['Rischio_Base'] = away_df.apply(
            lambda row: self._get_player_risk(row, referee_avg_yellow, away_aggression_score), axis=1
        )
        
        # NOVITÀ: Duelli BIDIREZIONALI
        critical_matchups_home = self._calculate_critical_matchups(home_df, away_df, high_risk_victims)
        critical_matchups_away = self._calculate_critical_matchups(away_df, home_df, high_risk_victims)
        critical_matchups = critical_matchups_home + critical_matchups_away
        
        # Inizializza rischio finale
        home_df['Rischio_Finale'] = home_df['Rischio_Base']
        away_df['Rischio_Finale'] = away_df['Rischio_Base']
        
        # Applica boost (per aggressori)
        for m in critical_matchups:
            player = m['aggressor_player']
            risk_boost_factor = m['risk_factor']
            
            # Applica al df corretto
            if player in home_df['Player'].values:
                current_risk = home_df.loc[home_df['Player'] == player, 'Rischio_Base'].iloc[0]
                home_df.loc[home_df['Player'] == player, 'Rischio_Finale'] = current_risk * risk_boost_factor
            elif player in away_df['Player'].values:
                current_risk = away_df.loc[away_df['Player'] == player, 'Rischio_Base'].iloc[0]
                away_df.loc[away_df['Player'] == player, 'Rischio_Finale'] = current_risk * risk_boost_factor

        # Bilanciamento e TOP 4 (invariato)
        all_players = pd.concat([home_df, away_df]).sort_values(by='Rischio_Finale', ascending=False)
        
        top_4 = all_players.head(4).copy()
        home_team = home_df['Squadra'].iloc[0] if not home_df.empty else ''
        away_team = away_df['Squadra'].iloc[0] if not away_df.empty else ''
        
        home_count = len(top_4[top_4['Squadra'] == home_team])
        away_count = len(top_4[top_4['Squadra'] == away_team])

        if home_count == 4 or away_count == 4 or home_count == 3 or away_count == 3:
            home_top_risk_sum = home_df.head(4)['Rischio_Finale'].sum()
            away_top_risk_sum = away_df.head(4)['Rischio_Finale'].sum()
            risk_diff_norm = abs(home_top_risk_sum - away_top_risk_sum) / max(home_top_risk_sum, away_top_risk_sum)
            
            if risk_diff_norm < RISK_DIFFERENCE_THRESHOLD:
                top_home_2 = home_df.sort_values(by='Rischio_Finale', ascending=False).head(2).copy()
                top_away_2 = away_df.sort_values(by='Rischio_Finale', ascending=False).head(2).copy()
                if not top_home_2.empty and not top_away_2.empty:
                    top_4 = pd.concat([top_home_2, top_away_2]).sort_values(by='Rischio_Finale', ascending=False)
                else:
                    top_4 = all_players.head(4)
            else:
                top_4 = all_players.head(4)

        # Preparazione Output
        predictions = top_4[['Player', 'Squadra', 'Posizione_Primaria', 'Rischio_Finale']].copy()  # Usa 'Player' da app.py
        predictions.rename(columns={'Posizione_Primaria': 'Posizione', 'Rischio_Finale': 'Rischio'}, inplace=True)
        predictions['Rischio'] = predictions['Rischio'].round(3)

        return {
            'all_predictions': all_players,  # Per app.py (tutto il df)
            'top_4_predictions': predictions.to_dict('records'),
            'critical_matchups': [  # Semplificato
                {
                    'aggressor_player': m['aggressor_player'],
                    'aggressor_team': m['aggressor_team'],
                    'victim_player': m['victim_player'],
                    'victim_team': m['victim_team'],
                    'risk_factor': m['risk_factor'],
                    'victim_fouls_rate': m['victim_fouls_rate']
                }
                for m in critical_matchups
            ],
            'high_risk_victims_used': high_risk_victims,  # NOVITÀ: Per debug
            'algorithm_summary': {
                'methodology': 'Modello Ottimizzato 5.1 - Duelli Bidirezionali con Falli Subiti',
                'critical_matchups_found': len(critical_matchups),
                'high_risk_victims_integrated': len([v for v in high_risk_victims if any(m['victim_player'] == v for m in critical_matchups)]),
                'weights_used': WEIGHTS,
                'aggression_diff_factor': abs(home_aggression_score - away_aggression_score),
                'dynamic_risk_threshold_applied': RISK_DIFFERENCE_THRESHOLD
            },
            'match_info': {  # Per app.py
                'home_team': home_team,
                'away_team': away_team,
                'expected_total_cards': referee_avg_yellow,
                'algorithm_confidence': 'High' if len(critical_matchups) > 5 else 'Medium'
            },
            'referee_profile': {  # Per app.py
                'name': ref_df['Nome'].iloc[0] if not ref_df.empty else 'Default',
                'cards_per_game': referee_avg_yellow,
                'strictness_factor': referee_avg_yellow / SERIE_A_AVG_CARDS_PER_GAME,
                'severity_level': 'strict' if referee_avg_yellow > 4.5 else 'medium' if referee_avg_yellow > 3.5 else 'permissive'
            }
        }