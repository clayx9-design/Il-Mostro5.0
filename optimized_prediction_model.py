"""
MODELLO OTTIMIZZATO PER PREDIZIONE CARTELLINI
==============================================
Sistema semplificato, efficace e realistico per predire i 4 giocatori
più probabili a ricevere cartellino giallo in una partita.

PRINCIPI CHIAVE:
1. Semplicità > Complessità (meno parametri, più efficacia)
2. Dati storici individuali come base solida
3. Duelli realistici (attaccante vs difensore, laterali opposti)
4. Calibrazione su medie Serie A reali
5. Bilanciamento forzato (2-2 predefinito, 3-1 solo se evidente e supportato dall'aggressività di squadra)
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
SERIE_A_AVG_FOULS_PER_PLAYER = 1.8  # Media Falli fatti/90s della Serie A
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


# =========================================================================
# WRAPPER PER COMPATIBILITÀ (come richiesto in app.py)
# =========================================================================

def get_field_zone(position, heatmap):
    """Wrapper per compatibilità con app.py"""
    return get_player_zone(position, heatmap)

def get_player_role_category(position):
    """Wrapper per compatibilità con app.py"""
    return get_player_role(position)


# =========================================================================
# MODELLO AVANZATO DI PREDIZIONE
# =========================================================================

class SuperAdvancedCardPredictionModel:
    """
    Modello Ottimizzato 5.0 per la predizione dei cartellini.
    Integra storico, rate di falli, impatto arbitro e duelli dinamici.
    """
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.df_players = data['players']
        self.df_referees = data['referees']

    def _get_player_risk(self, df_player: pd.Series, referee_avg_yellow: float, team_aggression_score: float) -> float:
        """Calcola il rischio base per un singolo giocatore."""
        
        # 1. Tendenza Storica (35%)
        # Normalizzazione per portare il rischio tra 0 e 1 (o vicino)
        if df_player['Media 90s per Cartellino Totale'] == 0:
            historical_risk = 0.5 # Rischio medio se non ha abbastanza dati
        else:
            # Calcola la frequenza del cartellino vs la media del campionato
            frequency_ratio = (SERIE_A_AVG_CARDS_PER_PLAYER_SEASON / df_player['90s Giocati Totali']) / (1 / df_player['Media 90s per Cartellino Totale'])
            historical_risk = min(frequency_ratio, 1.0) # Cap a 1.0
        
        # 2. Foul Rate (25%)
        if df_player['90s Giocati Totali'] > 5:
            foul_rate_90s = df_player['Falli Fatti Totali'] / df_player['90s Giocati Totali']
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

        # Calcolo del rischio base (somma pesata)
        base_risk = (
            historical_risk * WEIGHTS['historical_tendency'] +
            foul_risk * WEIGHTS['foul_rate'] +
            referee_risk * WEIGHTS['referee_impact'] +
            role_risk * WEIGHTS['role_risk'] +
            team_risk * WEIGHTS['aggression_score']
        )

        return base_risk


    def _calculate_critical_matchups(self, home_players: pd.DataFrame, away_players: pd.DataFrame) -> List[Dict]:
        """Identifica e valuta i duelli critici campo per campo."""
        
        critical_matchups = []
        
        # Duelli Laterali (Wide_Def vs Wide_Att) - I più importanti
        for h_row in home_players.itertuples():
            h_zone = get_player_zone(h_row.Posizione_Primaria, h_row.Heatmap)
            h_role = get_player_role(h_row.Posizione_Primaria)

            # Esegui la ricerca di duelli
            for a_row in away_players.itertuples():
                a_zone = get_player_zone(a_row.Posizione_Primaria, a_row.Heatmap)
                a_role = get_player_role(a_row.Posizione_Primaria)
                
                # Cerca incroci critici (Duello: Aggressore Difensivo vs Vittima Attaccante)
                if (h_zone, a_zone) in ZONE_MATCHUPS:
                    matchup_risk = ZONE_MATCHUPS.get((h_zone, a_zone))
                    
                    # Calcola il fattore di aggressività individuale nel duello
                    h_risk_base = h_row.Rischio_Base
                    a_risk_base = a_row.Rischio_Base
                    
                    # Rischio Duello (moltiplica il rischio base del difensore/centrocampista)
                    # Aggiunge un fattore proporzionale alla differenza di rischio con l'avversario
                    final_risk_factor = matchup_risk * (1 + (a_risk_base - h_risk_base) * 0.5)

                    critical_matchups.append({
                        'aggressor_player': h_row.Nome,
                        'aggressor_team': h_row.Squadra,
                        'aggressor_zone': h_zone,
                        'aggressor_role': h_role,
                        'victim_player': a_row.Nome,
                        'victim_team': a_row.Squadra,
                        'victim_zone': a_zone,
                        'victim_role': a_role,
                        'zone_compatibility': matchup_risk,
                        'risk_factor': final_risk_factor
                    })

        return critical_matchups


    def predict(self, home_team: str, away_team: str, referee: str) -> Dict:
        """
        Esegue la predizione completa e bilancia il risultato.
        """
        
        # --- PREPARAZIONE DATI ---
        
        df_home = self.df_players[self.df_players['Squadra'] == home_team].copy()
        df_away = self.df_players[self.df_players['Squadra'] == away_team].copy()
        
        # Arbitro
        ref_data = self.df_referees[self.df_referees['Nome'] == referee]
        if ref_data.empty:
            referee_avg_yellow = SERIE_A_AVG_CARDS_PER_GAME
        else:
            referee_avg_yellow = ref_data['Gialli a partita'].iloc[0]

        # Score Aggressività di Squadra
        home_aggression = df_home['Falli Fatti Totali'].sum() / df_home['90s Giocati Totali'].sum()
        away_aggression = df_away['Falli Fatti Totali'].sum() / df_away['90s Giocati Totali'].sum()
        
        home_aggression_score = min(home_aggression / SERIE_A_AVG_FOULS_PER_PLAYER, 1.5)
        away_aggression_score = min(away_aggression / SERIE_A_AVG_FOULS_PER_PLAYER, 1.5)
        
        # --- CALCOLO RISCHIO BASE ---

        df_home['Rischio_Base'] = df_home.apply(
            lambda row: self._get_player_risk(row, referee_avg_yellow, home_aggression_score), axis=1
        )
        df_away['Rischio_Base'] = df_away.apply(
            lambda row: self._get_player_risk(row, referee_avg_yellow, away_aggression_score), axis=1
        )
        
        # --- CALCOLO DUELLI CRITICI (APPLICA BOOST) ---

        critical_matchups = self._calculate_critical_matchups(df_home, df_away)
        
        # Inizializza il rischio finale con il rischio base
        df_home['Rischio_Finale'] = df_home['Rischio_Base']
        df_away['Rischio_Finale'] = df_away['Rischio_Base']
        
        # Applica il boost dei duelli (solo sul giocatore aggressore del duello critico)
        for m in critical_matchups:
            player = m['aggressor_player']
            team = m['aggressor_team']
            risk_boost_factor = m['risk_factor'] # Fattore moltiplicativo

            if team == home_team:
                current_risk = df_home.loc[df_home['Nome'] == player, 'Rischio_Base'].iloc[0]
                df_home.loc[df_home['Nome'] == player, 'Rischio_Finale'] = current_risk * risk_boost_factor
            elif team == away_team:
                current_risk = df_away.loc[df_away['Nome'] == player, 'Rischio_Base'].iloc[0]
                df_away.loc[df_away['Nome'] == player, 'Rischio_Finale'] = current_risk * risk_boost_factor

        # --- BILANCIAMENTO E SCELTA FINALE ---

        all_players = pd.concat([df_home, df_away]).sort_values(by='Rischio_Finale', ascending=False)
        
        # Forza la top 4 a mantenere l'equilibrio 2-2 di default
        top_4 = all_players.head(4).copy()
        
        home_count = len(top_4[top_4['Squadra'] == home_team])
        away_count = len(top_4[top_4['Squadra'] == away_team])

        # Se il risultato è sbilanciato (4-0 o 3-1), verifica se c'è un'evidente differenza di rischio
        if home_count == 4 or away_count == 4 or home_count == 3 or away_count == 3:
            
            home_top_risk_sum = df_home.head(4)['Rischio_Finale'].sum()
            away_top_risk_sum = df_away.head(4)['Rischio_Finale'].sum()
            
            risk_diff_norm = abs(home_top_risk_sum - away_top_risk_sum) / max(home_top_risk_sum, away_top_risk_sum)
            
            # Applica il bilanciamento se la differenza di rischio non è estrema
            if risk_diff_norm < RISK_DIFFERENCE_THRESHOLD:
                
                # Forza un 2-2 prendendo i primi 2 di ogni squadra
                top_home_2 = df_home.sort_values(by='Rischio_Finale', ascending=False).head(2).copy()
                top_away_2 = df_away.sort_values(by='Rischio_Finale', ascending=False).head(2).copy()
                
                if not top_home_2.empty and not top_away_2.empty:
                    top_4 = pd.concat([top_home_2, top_away_2]).sort_values(by='Rischio_Finale', ascending=False)
                else:
                    # Fallback
                    top_4 = all_players.head(4)
            else:
                # Se la differenza è ESTREMA, rispetta il 3-1 o 4-0
                top_4 = all_players.head(4)


        # --- PREPARAZIONE OUTPUT ---
        
        predictions = top_4[['Nome', 'Squadra', 'Posizione_Primaria', 'Rischio_Finale']].copy()
        predictions.rename(columns={'Posizione_Primaria': 'Posizione', 'Rischio_Finale': 'Rischio'}, inplace=True)
        
        # Arrotonda il rischio per l'output
        predictions['Rischio'] = predictions['Rischio'].round(3)

        return {
            'predictions_df': predictions,
            'critical_matchups': [
                {
                    'aggressor_player': m['aggressor_player'],
                    'aggressor_team': m['aggressor_team'],
                    'aggressor_zone': m['aggressor_zone'],
                    'aggressor_role': m['aggressor_role'],
                    'victim_player': m['victim_player'],
                    'victim_team': m['victim_team'],
                    'victim_zone': m['victim_zone'],
                    'victim_role': m['victim_role'],
                    'compatibility': m['zone_compatibility']
                }
                for m in critical_matchups
            ],
            'algorithm_summary': {
                'methodology': 'Modello Ottimizzato - Distribuzione Dinamica',
                'critical_matchups_found': len(critical_matchups),
                'high_risk_players': (all_players['Rischio_Finale'] > 0.4).sum(),
                'weights_used': WEIGHTS,
                'aggression_diff_factor': abs(home_aggression_score - away_aggression_score),
                'dynamic_risk_threshold_applied': RISK_DIFFERENCE_THRESHOLD
            }
        }

# =========================================================================
# CLASSE PRINCIPALE ESPORTATA
# =========================================================================
# La classe SuperAdvancedCardPredictionModel è il tuo modello principale.
# Le funzioni get_field_zone e get_player_role_category sono wrapper per app.py.