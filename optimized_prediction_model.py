import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any

# ... (import e costanti invariati - Aggiungi qui le tue costanti, e.g. WEIGHTS, se mancanti)

# Esempio di funzioni che potrebbero mancare, basate sull'app.py:
def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    # Questa funzione deve essere definita nel tuo modello.
    # Assumo che esista e gestisca la preparazione del dataframe.
    return df

def get_player_role(pos: str) -> str:
    # Questa funzione deve essere definita nel tuo modello.
    pos = str(pos).upper().strip()
    if 'D' in pos: return 'DIF'
    if 'A' in pos: return 'ATT'
    return 'CEN'

def get_field_zone(heatmap: str) -> str:
    # Placeholder
    return 'Central activity'

def get_player_role_category(role: str) -> str:
    # Placeholder
    return role


class OptimizedCardPredictionModel:
    # Il resto dei metodi e dell'init del tuo modello...
    
    # Assumo che 'WEIGHTS' sia una costante definita qui o importata.
    WEIGHTS = {
        'Falli_Fatti': 0.35, 
        'Falli_per_Cartellino': 0.25, 
        '90s_per_Cartellino': 0.20,
        'Ruolo': 0.10,
        'Heatmap': 0.10
    }

    def __init__(self):
        # Inizializzazione del modello, se necessario
        pass
        
    # Esempio di un metodo di supporto che potrebbe mancare:
    def calculate_risk_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcola i fattori di rischio base per i giocatori."""
        df['Rischio_Falli'] = df.get('Media Falli Fatti 90s Totale', 0)
        df['Rischio_Efficacia'] = df.get('Media Falli per Cartellino Totale', 999).replace(0, 999).rdiv(1)
        df['Rischio_Frequenza'] = df.get('Media 90s per Cartellino Totale', 999).replace(0, 999).rdiv(1)
        
        # Combinazione di rischio (semplificata per l'esempio)
        df['Rischio'] = (
            df['Rischio_Falli'] * self.WEIGHTS['Falli_Fatti'] +
            df['Rischio_Efficacia'] * self.WEIGHTS['Falli_per_Cartellino'] +
            df['Rischio_Frequenza'] * self.WEIGHTS['90s_per_Cartellino'] 
            # Aggiungi Ruolo e Heatmap se hai le logiche di ponderazione
        )
        # Normalizzazione: assicurati che sia tra 0 e 1, o usa il tuo metodo di normalizzazione
        df['Rischio'] = df['Rischio'] / df['Rischio'].max() if df['Rischio'].max() > 0 else 0
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
        
        # *** RAFFORZATO: Filtro >=5 con warning se dataset troppo piccolo ***
        initial_home = len(home_df)
        initial_away = len(away_df)
        home_df = home_df[home_df['90s Giocati Totali'] >= 5]
        away_df = away_df[away_df['90s Giocati Totali'] >= 5]
        
        excluded_home = initial_home - len(home_df)
        excluded_away = initial_away - len(away_df)
        if excluded_home > initial_home * 0.5 or excluded_away > initial_away * 0.5:  # >50% esclusi?
            warnings.warn(f"⚠️ Troppi esclusi per 90s <5: Casa {excluded_home}/{initial_home}, Trasferta {excluded_away}/{initial_away}")
        
        if home_df.empty or away_df.empty:
            return {
                'error': 'Dati insufficienti dopo il filtro delle 5 partite minime.',
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
        # Logica semplificata: (media arbitro + media rischi squadre) / 2
        avg_risk = all_predictions_df['Rischio'].mean()
        expected_total_cards = round(referee_avg * (1 + avg_risk * 0.5), 1)

        # 5. Genera Output
        return {
            'match_info': {
                'home_team': home_df['Squadra'].iloc[0],
                'away_team': away_df['Squadra'].iloc[0],
                'expected_total_cards': f"{expected_total_cards:.1f}",
                'algorithm_confidence': 'High', # Placeholder
            },
            'referee_profile': {
                'Nome': referee_name,
                'Gialli_a_partita': referee_avg,
                'Severity': referee_severity,
                'Emoji': referee_severity, # app.py manca il mapping, ma lo includo qui
                'Description': f"Arbitro con media di {referee_avg:.1f} cartellini a partita.",
            },
            'all_predictions': all_predictions_df.sort_values('Rischio', ascending=False).reset_index(drop=True),
            'algorithm_summary': {
                'methodology': 'Modello Ottimizzato - Filtro 5 Partite + Rischio Falli Minimo 1.5',
                'weights_used': self.WEIGHTS,
                'min_games_filter_applied': 5,
                'aggression_factor_min': 1.5,
                'players_after_filter': {'home': len(home_df), 'away': len(away_df)}
            }
        }