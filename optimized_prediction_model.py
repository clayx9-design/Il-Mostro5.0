import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any

# =========================================================================
# CONSTANTI E FUNZIONI AUSILIARIE (Definite per l'import in app.py)
# =========================================================================

# Assumo che 'WEIGHTS' sia una costante definita qui o importata.
WEIGHTS = {
    'Falli_Fatti': 0.35, 
    'Falli_per_Cartellino': 0.25, 
    '90s_per_Cartellino': 0.20,
    'Ruolo': 0.10,
    'Heatmap': 0.10
}

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Funzione placeholder per la normalizzazione dei dati prima del calcolo."""
    return df.copy()

def get_player_role(pos: str) -> str:
    """Mappa la posizione (Posizione_Primaria) al ruolo principale."""
    pos = str(pos).upper().strip()
    if 'D' in pos: return 'DIF'
    if 'A' in pos: return 'ATT'
    # Copertura per centrocampo, ecc.
    return 'CEN' 

def get_field_zone(heatmap: str) -> str:
    """Funzione placeholder per la zona del campo (usata per il rischio)"""
    return 'Central activity'

def get_player_role_category(role: str) -> str:
    """Funzione placeholder per la categoria di ruolo (es. Attaccante, Difensore)."""
    return role

# =========================================================================
# CLASSE MODELLO
# =========================================================================

class OptimizedCardPredictionModel:
    
    def __init__(self):
        self.weights = WEIGHTS

    def calculate_risk_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcola i fattori di rischio base per i giocatori."""
        df['Rischio_Falli'] = df.get('Media Falli Fatti 90s Totale', 0)
        
        # Calcola l'inverso per Falli per Cartellino e 90s per Cartellino 
        # (più basso è il valore, più alto è il rischio)
        df['Rischio_Efficacia'] = df.get('Media Falli per Cartellino Totale', 999).replace(0, 999).rdiv(1)
        df['Rischio_Frequenza'] = df.get('Media 90s per Cartellino Totale', 999).replace(0, 999).rdiv(1)
        
        # Combinazione di rischio ponderata (semplificata)
        df['Rischio'] = (
            df['Rischio_Falli'] * self.weights['Falli_Fatti'] +
            df['Rischio_Efficacia'] * self.weights['Falli_per_Cartellino'] * 0.5 + 
            df['Rischio_Frequenza'] * self.weights['90s_per_Cartellino'] * 0.5
        )
        
        # Normalizzazione: porta il rischio massimo a 1.0
        if df['Rischio'].max() > 0:
            df['Rischio'] = df['Rischio'] / df['Rischio'].max()
        else:
            df['Rischio'] = 0
            
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
        referee_name = referee_df['Nome'].iloc[0]
        referee_avg = referee_df['Gialli a partita'].iloc[0]
        referee_severity = 'medium'
        if referee_avg > 4.8: referee_severity = 'strict'
        elif referee_avg < 3.8: referee_severity = 'permissive'
        
        # 4. Calcola Cartellini Totali Attesi
        avg_risk = all_predictions_df['Rischio'].mean()
        expected_total_cards = round(referee_avg * (1 + avg_risk * 0.5), 1)

        # 5. Genera Output
        return {
            'match_info': {
                'home_team': home_df['Squadra'].iloc[0],
                'away_team': away_df['Squadra'].iloc[0],
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