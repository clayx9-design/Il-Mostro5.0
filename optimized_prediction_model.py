import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, List, Tuple

warnings.filterwarnings('ignore')

# =========================================================================
# COSTANTI E FUNZIONI AUSILIARIE
# =========================================================================

# Ponderazioni utilizzate nel calcolo del rischio AGGIORNATE
WEIGHTS = {
    'Falli_Fatti': 0.30,           # Aggressività del giocatore
    'Falli_per_Cartellino': 0.20,  # Efficienza nel prendere cartellini
    '90s_per_Cartellino': 0.15,    # Frequenza cartellini
    'Falli_Subiti': 0.20,          # Quanto viene targetizzato dagli avversari
    'Matchup_Risk': 0.10,          # Rischio da accoppiamento tattico
    'Ruolo': 0.05                   # Peso ridotto del ruolo
}

# Soglie per categorizzazione
THRESHOLDS = {
    'high_fouls_made': 2.5,        # Falli fatti per 90' per essere "aggressivo"
    'high_fouls_suffered': 2.0,    # Falli subiti per 90' per essere "vittima"
    'min_90s_played': 5,           # Minimo partite giocate
    'card_efficiency': 3.5,        # Falli per cartellino (meno = più pericoloso)
    'frequent_cards': 5.0          # 90' per cartellino (meno = più pericoloso)
}

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizza i dati prima del calcolo."""
    df = df.copy()
    
    # Assicura che le colonne numeriche siano float
    numeric_cols = [
        'Media Falli Fatti 90s Totale', 'Media Falli Subiti 90s Totale',
        'Media Falli per Cartellino Totale', 'Media 90s per Cartellino Totale',
        'Cartellini Gialli Totali', '90s Giocati Totali'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Gestione colonna Player se non presente (compatibilità)
    if 'Player' not in df.columns:
        df['Player'] = df.get('Nome Giocatore', df.get('Nome', '')).astype(str)
    
    return df

def get_player_role(pos: str) -> str:
    """Mappa la posizione al ruolo principale."""
    pos = str(pos).upper().strip()
    if 'D' in pos or 'DF' in pos:
        return 'DIF'
    if 'A' in pos or 'FW' in pos or 'ST' in pos:
        return 'ATT'
    if 'M' in pos or 'MF' in pos:
        return 'CEN'
    return 'CEN'

def get_field_zone(heatmap: str) -> str:
    """Estrae la zona del campo dalla heatmap."""
    heatmap = str(heatmap).lower()
    if 'attack' in heatmap or 'forward' in heatmap:
        return 'attack'
    elif 'defense' in heatmap or 'back' in heatmap:
        return 'defense'
    else:
        return 'midfield'

def get_player_role_category(role: str) -> str:
    """Categoria di ruolo per UI."""
    role_map = {
        'DIF': 'Difensore',
        'CEN': 'Centrocampista',
        'ATT': 'Attaccante'
    }
    return role_map.get(role, role)

# =========================================================================
# CLASSE MODELLO AVANZATO
# =========================================================================

class OptimizedCardPredictionModel:
    """
    Modello avanzato che analizza:
    1. Aggressività individuale (falli fatti)
    2. Esposizione al rischio (falli subiti) 
    3. Efficienza nel ricevere cartellini
    4. Matchup tattici (aggressori vs vittime)
    """
    
    def __init__(self):
        self.weights = WEIGHTS
        self.thresholds = THRESHOLDS

    def identify_aggressive_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifica giocatori con alto tasso di falli fatti."""
        df['Is_Aggressive'] = (
            df.get('Media Falli Fatti 90s Totale', 0) >= self.thresholds['high_fouls_made']
        )
        return df

    def identify_victim_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifica giocatori che subiscono molti falli."""
        df['Is_Victim'] = (
            df.get('Media Falli Subiti 90s Totale', 0) >= self.thresholds['high_fouls_suffered']
        )
        return df

    def calculate_matchup_risk(
        self, 
        home_df: pd.DataFrame, 
        away_df: pd.DataFrame,
        high_risk_victims: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calcola il rischio derivante dagli accoppiamenti tattici.
        
        Logica:
        - Difensori di casa vs Attaccanti trasferta (e viceversa)
        - Centrocampisti centrali contro centrocampisti avversari
        - Bonus se un aggressore marca una "vittima" nota
        """
        high_risk_victims = high_risk_victims or []
        
        # Identifica ruoli e zone
        home_df['Zone'] = home_df['Heatmap'].apply(get_field_zone)
        away_df['Zone'] = away_df['Heatmap'].apply(get_field_zone)
        
        # Aggiungi bonus matchup
        home_df['Matchup_Bonus'] = 0.0
        away_df['Matchup_Bonus'] = 0.0
        
        # CASA: Difensori contro attaccanti trasferta che sono vittime
        home_defenders = home_df[home_df['Ruolo'] == 'DIF']
        away_attackers_victims = away_df[
            (away_df['Ruolo'] == 'ATT') & 
            (away_df['Player'].isin(high_risk_victims))
        ]
        
        if len(home_defenders) > 0 and len(away_attackers_victims) > 0:
            # I difensori casa aggressivi che marcano attaccanti-vittime hanno bonus
            home_df.loc[
                (home_df['Ruolo'] == 'DIF') & 
                (home_df['Is_Aggressive'] == True),
                'Matchup_Bonus'
            ] = 0.15
        
        # TRASFERTA: Difensori contro attaccanti casa che sono vittime
        away_defenders = away_df[away_df['Ruolo'] == 'DIF']
        home_attackers_victims = home_df[
            (home_df['Ruolo'] == 'ATT') & 
            (home_df['Player'].isin(high_risk_victims))
        ]
        
        if len(away_defenders) > 0 and len(home_attackers_victims) > 0:
            away_df.loc[
                (away_df['Ruolo'] == 'DIF') & 
                (away_df['Is_Aggressive'] == True),
                'Matchup_Bonus'
            ] = 0.15
        
        # CENTROCAMPO: Centrocampisti aggressivi contro zone centrali avversarie
        home_central_aggressive = home_df[
            (home_df['Ruolo'] == 'CEN') & 
            (home_df['Is_Aggressive'] == True) &
            (home_df['Zone'] == 'midfield')
        ]
        away_central_victims = away_df[
            (away_df['Zone'] == 'midfield') &
            (away_df['Is_Victim'] == True)
        ]
        
        if len(home_central_aggressive) > 0 and len(away_central_victims) > 0:
            home_df.loc[home_central_aggressive.index, 'Matchup_Bonus'] += 0.10
        
        away_central_aggressive = away_df[
            (away_df['Ruolo'] == 'CEN') & 
            (away_df['Is_Aggressive'] == True) &
            (away_df['Zone'] == 'midfield')
        ]
        home_central_victims = home_df[
            (home_df['Zone'] == 'midfield') &
            (home_df['Is_Victim'] == True)
        ]
        
        if len(away_central_aggressive) > 0 and len(home_central_victims) > 0:
            away_df.loc[away_central_aggressive.index, 'Matchup_Bonus'] += 0.10
        
        return home_df, away_df

    def calculate_risk_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola i fattori di rischio individuali con la nuova logica.
        
        Componenti del rischio:
        1. Rischio_Falli: Normalizzato da Media Falli Fatti
        2. Rischio_Efficacia: Inverso di Falli per Cartellino (meno falli = più pericoloso)
        3. Rischio_Frequenza: Inverso di 90s per Cartellino (meno tempo = più frequente)
        4. Rischio_Vittima: Falli subiti (chi subisce falli è più a rischio ammonizione per reazione)
        5. Matchup_Bonus: Da accoppiamenti tattici
        """
        
        # 1. Rischio da Falli Fatti (normalizzato 0-1)
        max_fouls = df['Media Falli Fatti 90s Totale'].max()
        df['Rischio_Falli'] = df['Media Falli Fatti 90s Totale'] / max_fouls if max_fouls > 0 else 0
        
        # 2. Rischio da Efficacia Cartellini (inverso: meno falli per cartellino = più pericoloso)
        fouls_per_card = df['Media Falli per Cartellino Totale'].replace(0, 999)
        df['Rischio_Efficacia'] = fouls_per_card.apply(
            lambda x: min(1.0, self.thresholds['card_efficiency'] / x)
        )
        
        # 3. Rischio da Frequenza Cartellini (inverso: meno 90' tra cartellini = più frequente)
        nineties_per_card = df['Media 90s per Cartellino Totale'].replace(0, 999)
        df['Rischio_Frequenza'] = nineties_per_card.apply(
            lambda x: min(1.0, self.thresholds['frequent_cards'] / x)
        )
        
        # 4. Rischio da Falli Subiti (normalizzato)
        max_suffered = df['Media Falli Subiti 90s Totale'].max()
        df['Rischio_Vittima'] = df['Media Falli Subiti 90s Totale'] / max_suffered if max_suffered > 0 else 0
        
        # 5. Bonus Ruolo (ridotto rispetto a prima)
        role_bonus = {
            'DIF': 0.10,  # Difensori: rischio moderato
            'CEN': 0.15,  # Centrocampisti: rischio più alto (zona calda)
            'ATT': 0.05   # Attaccanti: rischio basso (subiscono più che fare falli)
        }
        df['Rischio_Ruolo'] = df['Ruolo'].map(role_bonus).fillna(0.10)
        
        # CALCOLO RISCHIO FINALE PONDERATO
        df['Rischio'] = (
            df['Rischio_Falli'] * self.weights['Falli_Fatti'] +
            df['Rischio_Efficacia'] * self.weights['Falli_per_Cartellino'] +
            df['Rischio_Frequenza'] * self.weights['90s_per_Cartellino'] +
            df['Rischio_Vittima'] * self.weights['Falli_Subiti'] +
            df.get('Matchup_Bonus', 0) * self.weights['Matchup_Risk'] +
            df['Rischio_Ruolo'] * self.weights['Ruolo']
        )
        
        # Normalizzazione finale (0-1)
        if df['Rischio'].max() > 0:
            df['Rischio'] = df['Rischio'] / df['Rischio'].max()
        else:
            df['Rischio'] = 0
        
        # Rinomina per compatibilità con UI
        df['Rischio_Finale'] = df['Rischio']
        
        return df

    def predict_match_cards(
        self,
        home_df: pd.DataFrame,
        away_df: pd.DataFrame,
        referee_df: pd.DataFrame,
        high_risk_victims: List[str] = None
    ) -> Dict[str, Any]:
        """
        Esegue la predizione completa per una partita con la nuova logica.
        
        Args:
            home_df: DataFrame giocatori casa
            away_df: DataFrame giocatori trasferta
            referee_df: DataFrame arbitro
            high_risk_victims: Lista nomi giocatori vittime dalla Fase 1
        """
        
        # 1. Normalizza e filtra i dati
        home_df = normalize_data(home_df)
        away_df = normalize_data(away_df)
        
        initial_home = len(home_df)
        initial_away = len(away_df)
        
        # Filtro minimo partite
        home_df = home_df[home_df.get('90s Giocati Totali', 0) >= self.thresholds['min_90s_played']].copy()
        away_df = away_df[away_df.get('90s Giocati Totali', 0) >= self.thresholds['min_90s_played']].copy()
        
        excluded_home = initial_home - len(home_df)
        excluded_away = initial_away - len(away_df)
        
        if home_df.empty or away_df.empty:
            return {
                'error': f'Dati insufficienti dopo filtro (Casa: {excluded_home} esclusi, Trasferta: {excluded_away} esclusi)',
                'excluded_count': {'home': excluded_home, 'away': excluded_away}
            }
        
        # Assegna ruoli ai giocatori
        home_df['Ruolo'] = home_df['Posizione_Primaria'].apply(get_player_role)
        away_df['Ruolo'] = away_df['Posizione_Primaria'].apply(get_player_role)
        
        # 2. Identifica categorie di giocatori
        home_df = self.identify_aggressive_players(home_df)
        away_df = self.identify_aggressive_players(away_df)
        home_df = self.identify_victim_players(home_df)
        away_df = self.identify_victim_players(away_df)
        
        # 3. Calcola rischio da matchup tattici
        home_df, away_df = self.calculate_matchup_risk(home_df, away_df, high_risk_victims or [])
        
        # 4. Calcola fattori di rischio individuali
        home_df = self.calculate_risk_factors(home_df)
        away_df = self.calculate_risk_factors(away_df)
        
        # 5. Combina tutte le predizioni
        all_predictions_df = pd.concat([home_df, away_df], ignore_index=True)
        all_predictions_df = all_predictions_df.sort_values('Rischio_Finale', ascending=False).reset_index(drop=True)
        
        # 6. Determina profilo arbitro
        referee_name = referee_df['Nome'].iloc[0]
        referee_avg = float(referee_df['Gialli a partita'].iloc[0])
        
        if referee_avg > 4.8:
            referee_severity = 'strict'
        elif referee_avg < 3.8:
            referee_severity = 'permissive'
        else:
            referee_severity = 'medium'
        
        # 7. Calcola cartellini attesi
        avg_risk = all_predictions_df['Rischio_Finale'].mean()
        top_4_avg_risk = all_predictions_df.head(4)['Rischio_Finale'].mean()
        
        # Formula migliorata: considera sia media generale che top 4
        expected_total_cards = round(
            referee_avg * (1 + (avg_risk * 0.3 + top_4_avg_risk * 0.2)), 
            1
        )
        
        # 8. Statistiche aggiuntive
        aggressive_home = home_df[home_df['Is_Aggressive'] == True].shape[0]
        aggressive_away = away_df[away_df['Is_Aggressive'] == True].shape[0]
        victims_home = home_df[home_df['Is_Victim'] == True].shape[0]
        victims_away = away_df[away_df['Is_Victim'] == True].shape[0]
        
        # 9. Genera output
        return {
            'match_info': {
                'home_team': home_df['Squadra'].iloc[0],
                'away_team': away_df['Squadra'].iloc[0],
                'expected_total_cards': f"{expected_total_cards:.1f}",
                'algorithm_confidence': 'High' if top_4_avg_risk > 0.6 else 'Medium',
                'aggressive_players': {
                    'home': aggressive_home,
                    'away': aggressive_away
                },
                'victim_players': {
                    'home': victims_home,
                    'away': victims_away
                }
            },
            'referee_profile': {
                'name': referee_name,
                'Nome': referee_name,
                'cards_per_game': referee_avg,
                'Gialli_a_partita': referee_avg,
                'severity_level': referee_severity,
                'Severity': referee_severity,
                'strictness_factor': referee_avg / 4.2,  # Media Serie A
                'Description': f"Arbitro con media di {referee_avg:.1f} gialli/partita ({referee_severity})"
            },
            'all_predictions': all_predictions_df,
            'top_4_predictions': [],  # Verrà popolato da apply_balancing_logic in app.py
            'algorithm_summary': {
                'methodology': 'Modello Avanzato v2.0 - Matchup Tattici + Falli Subiti',
                'weights_used': self.weights,
                'thresholds_used': self.thresholds,
                'min_games_filter_applied': self.thresholds['min_90s_played'],
                'players_after_filter': {
                    'home': len(home_df), 
                    'away': len(away_df)
                },
                'high_risk_victims_used': len(high_risk_victims) if high_risk_victims else 0
            }
        }


# Alias per compatibilità con app.py
SuperAdvancedCardPredictionModel = OptimizedCardPredictionModel