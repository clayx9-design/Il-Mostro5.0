# app.py
import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, List, Tuple
from optimized_prediction_model import OptimizedCardPredictionModel  # Importa il modello dal file separato

warnings.filterwarnings('ignore')

# =========================================================================
# COSTANTI E FUNZIONI AUSILIARIE AVANZATE
# =========================================================================

# Ponderazioni utilizzate nel calcolo del rischio AGGIORNATE (per estensione)
ADVANCED_WEIGHTS = {
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

def advanced_normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizza i dati per il modello avanzato."""
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
    
    # Assicura presenza di 'Squadra' per compatibilità
    if 'Squadra' not in df.columns:
        df['Squadra'] = 'Default Team'
    
    return df

# =========================================================================
# ESTENSIONE AVANZATA DEL MODELLO
# =========================================================================

class SuperAdvancedCardPredictionModel(OptimizedCardPredictionModel):
    """
    Estensione avanzata del modello base che include:
    1. Identificazione aggressori e vittime
    2. Calcolo matchup tattici
    3. Integrazione falli subiti
    """
    
    def __init__(self):
        super().__init__()
        self.weights = ADVANCED_WEIGHTS
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
        """
        high_risk_victims = high_risk_victims or []
        
        # Identifica ruoli e zone (gestisci assenza Heatmap)
        if 'Heatmap' in home_df.columns:
            home_df['Zone'] = home_df['Heatmap'].apply(get_field_zone)
        else:
            home_df['Zone'] = 'midfield'
        
        if 'Heatmap' in away_df.columns:
            away_df['Zone'] = away_df['Heatmap'].apply(get_field_zone)
        else:
            away_df['Zone'] = 'midfield'
        
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
            home_mask = (home_df['Ruolo'] == 'DIF') & (home_df['Is_Aggressive'] == True)
            home_df.loc[home_mask, 'Matchup_Bonus'] = 0.15
        
        # TRASFERTA: Difensori contro attaccanti casa che sono vittime
        away_defenders = away_df[away_df['Ruolo'] == 'DIF']
        home_attackers_victims = home_df[
            (home_df['Ruolo'] == 'ATT') & 
            (home_df['Player'].isin(high_risk_victims))
        ]
        
        if len(away_defenders) > 0 and len(home_attackers_victims) > 0:
            away_mask = (away_df['Ruolo'] == 'DIF') & (away_df['Is_Aggressive'] == True)
            away_df.loc[away_mask, 'Matchup_Bonus'] = 0.15
        
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

    def advanced_calculate_risk_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola i fattori di rischio avanzati.
        """
        df = advanced_normalize_data(df)
        
        # 1. Rischio da Falli Fatti (normalizzato 0-1)
        max_fouls = df['Media Falli Fatti 90s Totale'].max()
        df['Rischio_Falli'] = 0.0
        if max_fouls > 0:
            df['Rischio_Falli'] = df['Media Falli Fatti 90s Totale'] / max_fouls
        
        # 2. Rischio da Efficacia Cartellini
        fouls_per_card = df['Media Falli per Cartellino Totale'].replace(0, 999)
        df['Rischio_Efficacia'] = fouls_per_card.apply(
            lambda x: min(1.0, self.thresholds['card_efficiency'] / x)
        )
        
        # 3. Rischio da Frequenza Cartellini
        nineties_per_card = df['Media 90s per Cartellino Totale'].replace(0, 999)
        df['Rischio_Frequenza'] = nineties_per_card.apply(
            lambda x: min(1.0, self.thresholds['frequent_cards'] / x)
        )
        
        # 4. Rischio da Falli Subiti (normalizzato)
        max_suffered = df['Media Falli Subiti 90s Totale'].max()
        df['Rischio_Vittima'] = 0.0
        if max_suffered > 0:
            df['Rischio_Vittima'] = df['Media Falli Subiti 90s Totale'] / max_suffered
        
        # 5. Bonus Ruolo
        role_bonus = {
            'DIF': 0.10, 'CEN': 0.15, 'ATT': 0.05
        }
        df['Rischio_Ruolo'] = df['Ruolo'].map(role_bonus).fillna(0.10)
        
        # Assicura presenza di Matchup_Bonus
        if 'Matchup_Bonus' not in df.columns:
            df['Matchup_Bonus'] = 0.0
        
        # CALCOLO RISCHIO FINALE PONDERATO
        df['Rischio'] = (
            df['Rischio_Falli'] * self.weights['Falli_Fatti'] +
            df['Rischio_Efficacia'] * self.weights['Falli_per_Cartellino'] +
            df['Rischio_Frequenza'] * self.weights['90s_per_Cartellino'] +
            df['Rischio_Vittima'] * self.weights['Falli_Subiti'] +
            df['Matchup_Bonus'] * self.weights['Matchup_Risk'] +
            df['Rischio_Ruolo'] * self.weights['Ruolo']
        )
        
        # Normalizzazione finale (0-1)
        max_risk = df['Rischio'].max()
        if max_risk > 0:
            df['Rischio'] = df['Rischio'] / max_risk
        else:
            df['Rischio'] = 0.0
        
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
        Predizione avanzata con matchup.
        """
        # Usa il metodo base per normalizzazione e filtro
        home_df = advanced_normalize_data(home_df)
        away_df = advanced_normalize_data(away_df)
        
        initial_home = len(home_df)
        initial_away = len(away_df)
        
        home_df = home_df[home_df.get('90s Giocati Totali', 0) >= self.thresholds['min_90s_played']].copy()
        away_df = away_df[away_df.get('90s Giocati Totali', 0) >= self.thresholds['min_90s_played']].copy()
        
        excluded_home = initial_home - len(home_df)
        excluded_away = initial_away - len(away_df)
        
        if home_df.empty or away_df.empty:
            return {
                'error': f'Dati insufficienti dopo filtro (Casa: {excluded_home} esclusi, Trasferta: {excluded_away} esclusi)',
                'excluded_count': {'home': excluded_home, 'away': excluded_away}
            }
        
        # Assegna ruoli
        if 'Posizione_Primaria' in home_df.columns:
            home_df['Ruolo'] = home_df['Posizione_Primaria'].apply(get_player_role)
        else:
            home_df['Ruolo'] = 'CEN'
        
        if 'Posizione_Primaria' in away_df.columns:
            away_df['Ruolo'] = away_df['Posizione_Primaria'].apply(get_player_role)
        else:
            away_df['Ruolo'] = 'CEN'
        
        # Identifica categorie
        home_df = self.identify_aggressive_players(home_df)
        away_df = self.identify_aggressive_players(away_df)
        home_df = self.identify_victim_players(home_df)
        away_df = self.identify_victim_players(away_df)
        
        # Matchup
        home_df, away_df = self.calculate_matchup_risk(home_df, away_df, high_risk_victims or [])
        
        # Calcola rischi avanzati
        home_df = self.advanced_calculate_risk_factors(home_df)
        away_df = self.advanced_calculate_risk_factors(away_df)
        
        # Combina
        all_predictions_df = pd.concat([home_df, away_df], ignore_index=True)
        all_predictions_df = all_predictions_df.sort_values('Rischio_Finale', ascending=False).reset_index(drop=True)
        
        # Profilo arbitro
        if referee_df.empty:
            referee_name = 'Arbitro Default'
            referee_avg = 4.2
        else:
            referee_name = str(referee_df['Nome'].iloc[0])
            referee_avg = float(referee_df['Gialli a partita'].iloc[0]) if 'Gialli a partita' in referee_df.columns else 4.2
        
        referee_severity = 'medium'
        if referee_avg > 4.8: referee_severity = 'strict'
        elif referee_avg < 3.8: referee_severity = 'permissive'
        
        # Cartellini attesi
        avg_risk = all_predictions_df['Rischio_Finale'].mean()
        top_4_avg_risk = all_predictions_df.head(4)['Rischio_Finale'].mean() if len(all_predictions_df) >= 4 else avg_risk
        
        expected_total_cards = round(
            referee_avg * (1 + (avg_risk * 0.3 + top_4_avg_risk * 0.2)), 
            1
        )
        
        # Statistiche
        aggressive_home = home_df[home_df['Is_Aggressive'] == True].shape[0]
        aggressive_away = away_df[away_df['Is_Aggressive'] == True].shape[0]
        victims_home = home_df[home_df['Is_Victim'] == True].shape[0]
        victims_away = away_df[away_df['Is_Victim'] == True].shape[0]
        
        return {
            'match_info': {
                'home_team': str(home_df['Squadra'].iloc[0]),
                'away_team': str(away_df['Squadra'].iloc[0]),
                'expected_total_cards': f"{expected_total_cards:.1f}",
                'algorithm_confidence': 'High' if top_4_avg_risk > 0.6 else 'Medium',
                'aggressive_players': {'home': aggressive_home, 'away': aggressive_away},
                'victim_players': {'home': victims_home, 'away': victims_away}
            },
            'referee_profile': {
                'name': referee_name,
                'Nome': referee_name,
                'cards_per_game': referee_avg,
                'Gialli_a_partita': referee_avg,
                'severity_level': referee_severity,
                'Severity': referee_severity,
                'strictness_factor': referee_avg / 4.2,
                'Description': f"Arbitro con media di {referee_avg:.1f} gialli/partita ({referee_severity})"
            },
            'all_predictions': all_predictions_df,
            'top_4_predictions': [],
            'algorithm_summary': {
                'methodology': 'Modello Avanzato v2.0 - Matchup Tattici + Falli Subiti',
                'weights_used': self.weights,
                'thresholds_used': self.thresholds,
                'min_games_filter_applied': self.thresholds['min_90s_played'],
                'players_after_filter': {'home': len(home_df), 'away': len(away_df)},
                'high_risk_victims_used': len(high_risk_victims) if high_risk_victims else 0
            }
        }

# Alias
SuperAdvancedCardPredictionModel = SuperAdvancedCardPredictionModel

# Main di test
if __name__ == "__main__":
    print("App caricata correttamente.")
    model = SuperAdvancedCardPredictionModel()
    print("Istanza del modello avanzato creata.")
    
    # Test con dati minimali (come prima)
    home_df = pd.DataFrame({
        'Posizione_Primaria': ['DF', 'MF'],
        'Squadra': ['Casa Team', 'Casa Team'],
        '90s Giocati Totali': [6, 7],
        'Media Falli Fatti 90s Totale': [2.0, 3.0],
        'Media Falli Subiti 90s Totale': [1.0, 1.5],
        'Media Falli per Cartellino Totale': [4.0, 2.0],
        'Media 90s per Cartellino Totale': [6.0, 3.0],
        'Heatmap': ['defense', 'midfield'],
        'Nome': ['Giocatore1', 'Giocatore2']
    })
    away_df = pd.DataFrame({
        'Posizione_Primaria': ['ATT', 'CEN'],
        'Squadra': ['Away Team', 'Away Team'],
        '90s Giocati Totali': [5, 8],
        'Media Falli Fatti 90s Totale': [1.5, 2.8],
        'Media Falli Subiti 90s Totale': [2.5, 1.0],
        'Media Falli per Cartellino Totale': [5.0, 3.0],
        'Media 90s per Cartellino Totale': [7.0, 4.0],
        'Heatmap': ['attack', 'midfield'],
        'Nome': ['Giocatore3', 'Giocatore4']
    })
    referee_df = pd.DataFrame({
        'Nome': ['Arbitro Test'],
        'Gialli a partita': [4.0]
    })

    result = model.predict_match_cards(home_df, away_df, referee_df)
    print("Predizione avanzata completata senza errori.")
    print(result['match_info'])