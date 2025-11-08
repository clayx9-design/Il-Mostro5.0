# optimized_prediction_model.py
import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any

# =========================================================================
# COSTANTI E FUNZIONI AUSILIARIE
# =========================================================================

# Ponderazioni utilizzate nel calcolo del rischio (versione semplificata)
WEIGHTS = {
    'Falli_Fatti': 0.35, 
    'Falli_per_Cartellino': 0.25, 
    '90s_per_Cartellino': 0.20,
    'Ruolo': 0.10,
    'Heatmap': 0.10
}

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Funzione placeholder per la normalizzazione dei dati prima del calcolo."""
    df = df.copy()
    numeric_cols = [
        'Media Falli Fatti 90s Totale', 'Media Falli Subiti 90s Totale',
        'Media Falli per Cartellino Totale', 'Media 90s per Cartellino Totale',
        'Cartellini Gialli Totali', '90s Giocati Totali'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if 'Player' not in df.columns:
        df['Player'] = df.get('Nome Giocatore', df.get('Nome', '')).astype(str)
    if 'Squadra' not in df.columns:
        df['Squadra'] = 'Default Team'
    return df

def get_player_role(pos: str) -> str:
    """Mappa la posizione (Posizione_Primaria) al ruolo principale."""
    pos = str(pos).upper().strip()
    if 'D' in pos or 'DF' in pos: return 'DIF'
    if 'A' in pos or 'FW' in pos or 'ST' in pos: return 'ATT'
    return 'CEN' 

def get_field_zone(heatmap: str) -> str:
    """Funzione placeholder per la zona del campo (usata per il rischio)"""
    heatmap = str(heatmap).lower()
    if 'attack' in heatmap or 'forward' in heatmap:
        return 'attack'
    elif 'defense' in heatmap or 'back' in heatmap:
        return 'defense'
    else:
        return 'midfield'

def get_player_role_category(role: str) -> str:
    """Funzione placeholder per la categoria di ruolo (es. Attaccante, Difensore)."""
    role_map = {
        'DIF': 'Difensore',
        'CEN': 'Centrocampista',
        'ATT': 'Attaccante'
    }
    return role_map.get(role, role)

# =========================================================================
# CLASSE MODELLO
# =========================================================================

class OptimizedCardPredictionModel:
    
    def __init__(self):
        self.weights = WEIGHTS

    def calculate_risk_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcola i fattori di rischio base per i giocatori."""
        df = normalize_data(df)
        df['Rischio_Falli'] = df.get('Media Falli Fatti 90s Totale', 0)
        
        # Calcola l'inverso per Falli per Cartellino e 90s per Cartellino 
        fouls_per_card = df.get('Media Falli per Cartellino Totale', 999).replace(0, 999)
        df['Rischio_Efficacia'] = fouls_per_card.rdiv(1)
        nineties_per_card = df.get('Media 90s per Cartellino Totale', 999).replace(0, 999)
        df['Rischio_Frequenza'] = nineties_per_card.rdiv(1)
        
        # Bonus ruolo
        if 'Posizione_Primaria' in df.columns:
            df['Ruolo'] = df['Posizione_Primaria'].apply(get_player_role)
        else:
            df['Ruolo'] = 'CEN'
        role_bonus = {'DIF': 0.10, 'CEN': 0.15, 'ATT': 0.05}
        df['Rischio_Ruolo'] = df['Ruolo'].map(role_bonus).fillna(0.10)
        
        # Bonus heatmap
        df['Zone'] = df.get('Heatmap', 'midfield').apply(get_field_zone)
        heatmap_bonus = {'attack': 0.05, 'midfield': 0.15, 'defense': 0.10}
        df['Rischio_Heatmap'] = df['Zone'].map(heatmap_bonus).fillna(0.10)
        
        # Combinazione di rischio ponderata
        df['Rischio'] = (
            df['Rischio_Falli'] * self.weights['Falli_Fatti'] +
            df['Rischio_Efficacia'] * self.weights['Falli_per_Cartellino'] * 0.5 + 
            df['Rischio_Frequenza'] * self.weights['90s_per_Cartellino'] * 0.5 +
            df['Rischio_Ruolo'] * self.weights['Ruolo'] +
            df['Rischio_Heatmap'] * self.weights['Heatmap']
        )
        
        # Normalizzazione: porta il rischio massimo a 1.0
        max_risk = df['Rischio'].max()
        if max_risk > 0:
            df['Rischio'] = df['Rischio'] / max_risk
        else:
            df['Rischio'] = 0
            
        df['Rischio_Finale'] = df['Rischio']
        return df


    def predict_match_cards(
        self,
        home_df: pd.DataFrame,
        away_df: pd.DataFrame,
        referee_df: pd.DataFrame
    ) -> Dict:
        """Esegue la predizione completa per una partita."""
        
        # 1. Normalizza e filtra i dati
        home_df = normalize_data(home_df)
        away_df = normalize_data(away_df)
        
        # Filtro >=5 per coerenza
        initial_home = len(home_df)
        initial_away = len(away_df)
        home_df = home_df[home_df.get('90s Giocati Totali', 0) >= 5]
        away_df = away_df[away_df.get('90s Giocati Totali', 0) >= 5]
        
        excluded_home = initial_home - len(home_df)
        excluded_away = initial_away - len(away_df)
        
        if home_df.empty or away_df.empty:
            return {
                'error': 'Dati squadra insufficienti dopo il filtro delle 5 partite minime.',
                'excluded_count': {'home': excluded_home, 'away': excluded_away}
            }
            
        # 2. Calcola i rischi
        home_df = self.calculate_risk_factors(home_df)
        away_df = self.calculate_risk_factors(away_df)
        
        all_predictions_df = pd.concat([home_df, away_df], ignore_index=True)

        # 3. Determina profilo arbitro
        if referee_df.empty:
            referee_name = 'Arbitro Default'
            referee_avg = 4.2
        else:
            referee_name = str(referee_df['Nome'].iloc[0])
            referee_avg = float(referee_df['Gialli a partita'].iloc[0]) if 'Gialli a partita' in referee_df.columns else 4.2
        
        referee_severity = 'medium'
        if referee_avg > 4.8: referee_severity = 'strict'
        elif referee_avg < 3.8: referee_severity = 'permissive'
        
        # 4. Calcola Cartellini Totali Attesi
        avg_risk = all_predictions_df['Rischio'].mean()
        expected_total_cards = round(referee_avg * (1 + avg_risk * 0.5), 1)

        # 5. Genera Output
        return {
            'match_info': {
                'home_team': str(home_df['Squadra'].iloc[0]),
                'away_team': str(away_df['Squadra'].iloc[0]),
                'expected_total_cards': f"{expected_total_cards:.1f}",
                'algorithm_confidence': 'High', 
            },
            'referee_profile': {
                'Nome': referee_name,
                'Gialli_a_partita': referee_avg,
                'Severity': referee_severity,
                'Description': f"Arbitro con media di {referee_avg:.1f} cartellini a partita.",
            },
            'all_predictions': all_predictions_df.sort_values('Rischio', ascending=False).reset_index(drop=True),
            'algorithm_summary': {
                'methodology': 'Modello Ottimizzato - Filtro 5 Partite',
                'weights_used': self.weights,
                'min_games_filter_applied': 5,
                'players_after_filter': {'home': len(home_df), 'away': len(away_df)}
            }
        }