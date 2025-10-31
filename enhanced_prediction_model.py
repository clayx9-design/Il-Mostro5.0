import pandas as pd
import numpy as np
import re
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')

# =========================================================================
# FUNZIONI DI SUPPORTO MIGLIORATE
# =========================================================================

def normalize_name(name):
    """Normalizza un nome rimuovendo accenti, spazi e caratteri speciali."""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s-]+', '_', name)
    return name

def get_field_zone(position: str, heatmap: str) -> str:
    """
    Determina la zona di campo del giocatore con maggiore precisione.
    Restituisce: 'L' (Sinistra), 'R' (Destra), 'C' (Centro), 'LC' (Sinistra-Centro), 'RC' (Destra-Centro)
    """
    if pd.isna(position):
        position = ''
    if pd.isna(heatmap):
        heatmap = ''
    
    position = position.upper()
    heatmap = heatmap.lower()
    
    # Analisi posizione primaria
    if 'LW' in position or 'LB' in position or 'LWB' in position:
        return 'L'
    elif 'RW' in position or 'RB' in position or 'RWB' in position:
        return 'R'
    elif 'L' in position and 'R' not in position:
        return 'LC'  # Sinistra-Centro
    elif 'R' in position and 'L' not in position:
        return 'RC'  # Destra-Centro
    
    # Analisi heatmap
    if any(term in heatmap for term in ['left', 'sinistra', 'lwb', 'lb']):
        return 'L'
    elif any(term in heatmap for term in ['right', 'destra', 'rwb', 'rb']):
        return 'R'
    elif 'flank' in heatmap:
        if 'left' in heatmap:
            return 'L'
        elif 'right' in heatmap:
            return 'R'
    
    return 'C'  # Centro

def get_player_role_category(position: str) -> str:
    """Categorizza il ruolo del giocatore per l'analisi delle marcature."""
    if pd.isna(position):
        return 'MF'
    
    position = position.upper()
    
    if any(role in position for role in ['CB', 'LB', 'RB', 'LWB', 'RWB', 'DF']):
        return 'DEF'
    elif any(role in position for role in ['DM', 'CM', 'AM', 'MF']):
        return 'MID'
    elif any(role in position for role in ['LW', 'RW', 'FW', 'CF', 'ST']):
        return 'ATT'
    else:
        return 'MID'

# =========================================================================
# CLASSE DI PREDIZIONE AVANZATA MIGLIORATA
# =========================================================================

class SuperAdvancedCardPredictionModel:
    """
    Modello super avanzato per la predizione dei cartellini con:
    - Analisi zone di campo dettagliata
    - Fattori temporali e ritardi
    - Analisi arbitro approfondita
    - Dinamiche squadra vs squadra
    """
    
    def __init__(self):
        # Soglie dinamiche
        self.fouls_suffered_percentile = 0.75
        self.fouls_committed_percentile = 0.70
        self.card_frequency_threshold = 0.3
        
        # Pesi per il calcolo del rischio
        self.weights = {
            'zone_compatibility': 0.25,
            'card_tendency': 0.20,
            'foul_aggressivity': 0.15,
            'delay_factor': 0.15,
            'referee_influence': 0.15,
            'team_dynamics': 0.10
        }
        
        # Matrice di compatibilità zone (probabilità di scontro)
        self.zone_compatibility_matrix = {
            ('L', 'R'): 0.9,   # Sinistra vs Destra (massima compatibilità)
            ('R', 'L'): 0.9,
            ('LC', 'RC'): 0.8,  # Sinistra-Centro vs Destra-Centro
            ('RC', 'LC'): 0.8,
            ('L', 'RC'): 0.7,   # Sinistra vs Destra-Centro
            ('LC', 'R'): 0.7,
            ('R', 'LC'): 0.7,
            ('RC', 'L'): 0.7,
            ('C', 'C'): 0.6,    # Centro vs Centro
            ('L', 'C'): 0.5,    # Laterali vs Centro
            ('R', 'C'): 0.5,
            ('C', 'L'): 0.5,
            ('C', 'R'): 0.5,
            ('LC', 'C'): 0.6,
            ('RC', 'C'): 0.6,
            ('C', 'LC'): 0.6,
            ('C', 'RC'): 0.6,
            ('L', 'L'): 0.1,    # Stesso lato (bassa probabilità)
            ('R', 'R'): 0.1,
            ('LC', 'LC'): 0.2,
            ('RC', 'RC'): 0.2
        }

    def _calculate_card_tendency_score(self, player_data):
        """
        Calcola il punteggio di tendenza al cartellino basato su:
        - Frequenza cartellini (media 90s per cartellino)
        - Impulsività (media falli per cartellino)
        - Ritardo cartellino
        """
        # 1. Frequenza cartellini (più bassa = più tendenza)
        card_frequency = player_data.get('Media 90s per Cartellino Totale', 999)
        if pd.isna(card_frequency) or card_frequency == 0:
            card_frequency = 999
        
        frequency_score = 1 / (card_frequency + 0.1)  # Evita divisione per zero
        
        # 2. Impulsività (meno falli per cartellino = più impulsivo)
        fouls_per_card = player_data.get('Media Falli per Cartellino Totale', 999)
        if pd.isna(fouls_per_card) or fouls_per_card == 0:
            fouls_per_card = 999
            
        impulsivity_score = 1 / (fouls_per_card + 0.1)
        
        # 3. Ritardo cartellino (più basso = cartellini più rapidi)
        card_delay = player_data.get('Ritardo Cartellino (Minuti)', 90)
        if pd.isna(card_delay):
            card_delay = 90
            
        # Normalizza il ritardo (0-180 minuti tipici)
        delay_score = max(0, (180 - card_delay) / 180)
        
        # Combina i punteggi
        tendency_score = (
            frequency_score * 0.4 +
            impulsivity_score * 0.4 +
            delay_score * 0.2
        )
        
        return min(tendency_score, 1.0)

    def _analyze_referee_profile(self, referee_data):
        """Analizza il profilo dell'arbitro per determinare il fattore di influenza."""
        if referee_data.empty:
            return {
                'cards_per_game': 4.0,
                'strictness_factor': 1.0,
                'card_tendency': 'medium'
            }
        
        cards_per_game = referee_data.get('Gialli a partita', [4.0]).iloc[0] if len(referee_data) > 0 else 4.0
        
        # Categorizza la severità dell'arbitro
        if cards_per_game < 3.0:
            strictness = 'permissive'
            factor = 0.7
        elif cards_per_game > 5.0:
            strictness = 'strict'
            factor = 1.4
        else:
            strictness = 'medium'
            factor = 1.0
            
        return {
            'cards_per_game': cards_per_game,
            'strictness_factor': factor,
            'card_tendency': strictness
        }

    def _calculate_team_dynamics_factor(self, home_team, away_team, player_team):
        """
        Calcola il fattore dinamiche squadra basato su:
        - Rivalità storiche
        - Stile di gioco
        - Fattore casa/trasferta
        """
        # Rivalità note (da personalizzare con dati storici)
        rivalries = {
            ('Inter', 'Milan'): 1.3,
            ('Milan', 'Inter'): 1.3,
            ('Juventus', 'Inter'): 1.2,
            ('Inter', 'Juventus'): 1.2,
            ('Roma', 'Lazio'): 1.4,
            ('Lazio', 'Roma'): 1.4,
            ('Napoli', 'Juventus'): 1.2,
            ('Juventus', 'Napoli'): 1.2
        }
        
        # Fattore rivalità
        rivalry_factor = rivalries.get((home_team, away_team), 1.0)
        
        # Fattore casa/trasferta
        home_advantage = 1.1 if player_team == home_team else 0.95
        
        return rivalry_factor * home_advantage

    def _identify_critical_matchups(self, home_data, away_data):
        """
        Identifica i duelli critici basati su zone di campo e tendenze.
        Versione migliorata con analisi più sofisticata.
        """
        critical_matchups = []
        
        # Calcola soglie dinamiche
        all_players = pd.concat([home_data, away_data])
        
        fouls_suffered_threshold = all_players['Media Falli Subiti 90s Totale'].quantile(self.fouls_suffered_percentile)
        fouls_committed_threshold = all_players['Media Falli Fatti 90s Totale'].quantile(self.fouls_committed_percentile)
        
        # Identifica giocatori ad alto rischio di subire falli
        high_fouled_home = home_data[
            home_data['Media Falli Subiti 90s Totale'] >= fouls_suffered_threshold
        ]
        high_fouled_away = away_data[
            away_data['Media Falli Subiti 90s Totale'] >= fouls_suffered_threshold
        ]
        
        # Identifica giocatori che commettono molti falli
        aggressive_home = home_data[
            home_data['Media Falli Fatti 90s Totale'] >= fouls_committed_threshold
        ]
        aggressive_away = away_data[
            away_data['Media Falli Fatti 90s Totale'] >= fouls_committed_threshold
        ]
        
        # Analizza duelli Home vs Away
        for _, fouled_player in high_fouled_home.iterrows():
            fouled_zone = get_field_zone(fouled_player.get('Posizione_Primaria', ''), 
                                       fouled_player.get('Heatmap', ''))
            fouled_role = get_player_role_category(fouled_player.get('Posizione_Primaria', ''))
            
            for _, aggressive_player in aggressive_away.iterrows():
                aggressive_zone = get_field_zone(aggressive_player.get('Posizione_Primaria', ''), 
                                               aggressive_player.get('Heatmap', ''))
                aggressive_role = get_player_role_category(aggressive_player.get('Posizione_Primaria', ''))
                
                # Calcola compatibilità zone
                zone_compatibility = self.zone_compatibility_matrix.get(
                    (fouled_zone, aggressive_zone), 0.1
                )
                
                if zone_compatibility >= 0.5:  # Soglia di compatibilità
                    # Calcola punteggio duello
                    matchup_score = self._calculate_matchup_score(
                        fouled_player, aggressive_player, zone_compatibility
                    )
                    
                    critical_matchups.append({
                        'fouled_player': fouled_player['Player'],
                        'fouled_team': fouled_player['Squadra'],
                        'fouled_zone': fouled_zone,
                        'fouled_role': fouled_role,
                        'aggressive_player': aggressive_player['Player'],
                        'aggressive_team': aggressive_player['Squadra'],
                        'aggressive_zone': aggressive_zone,
                        'aggressive_role': aggressive_role,
                        'zone_compatibility': zone_compatibility,
                        'matchup_score': matchup_score,
                        'fouled_data': fouled_player,
                        'aggressive_data': aggressive_player
                    })
        
        # Analizza duelli Away vs Home
        for _, fouled_player in high_fouled_away.iterrows():
            fouled_zone = get_field_zone(fouled_player.get('Posizione_Primaria', ''), 
                                       fouled_player.get('Heatmap', ''))
            fouled_role = get_player_role_category(fouled_player.get('Posizione_Primaria', ''))
            
            for _, aggressive_player in aggressive_home.iterrows():
                aggressive_zone = get_field_zone(aggressive_player.get('Posizione_Primaria', ''), 
                                               aggressive_player.get('Heatmap', ''))
                aggressive_role = get_player_role_category(aggressive_player.get('Posizione_Primaria', ''))
                
                zone_compatibility = self.zone_compatibility_matrix.get(
                    (fouled_zone, aggressive_zone), 0.1
                )
                
                if zone_compatibility >= 0.5:
                    matchup_score = self._calculate_matchup_score(
                        fouled_player, aggressive_player, zone_compatibility
                    )
                    
                    critical_matchups.append({
                        'fouled_player': fouled_player['Player'],
                        'fouled_team': fouled_player['Squadra'],
                        'fouled_zone': fouled_zone,
                        'fouled_role': fouled_role,
                        'aggressive_player': aggressive_player['Player'],
                        'aggressive_team': aggressive_player['Squadra'],
                        'aggressive_zone': aggressive_zone,
                        'aggressive_role': aggressive_role,
                        'zone_compatibility': zone_compatibility,
                        'matchup_score': matchup_score,
                        'fouled_data': fouled_player,
                        'aggressive_data': aggressive_player
                    })
        
        # Ordina per punteggio duello
        critical_matchups.sort(key=lambda x: x['matchup_score'], reverse=True)
        
        return critical_matchups[:20]  # Top 20 duelli critici

    def _calculate_matchup_score(self, fouled_player, aggressive_player, zone_compatibility):
        """Calcola il punteggio di un duello specifico."""
        
        # Fattori del giocatore che subisce falli
        fouled_tendency = fouled_player.get('Media Falli Subiti 90s Totale', 0)
        
        # Fattori del giocatore aggressivo
        aggressive_tendency = aggressive_player.get('Media Falli Fatti 90s Totale', 0)
        aggressive_card_tendency = self._calculate_card_tendency_score(aggressive_player)
        
        # Calcola punteggio combinato
        matchup_score = (
            fouled_tendency * 0.3 +
            aggressive_tendency * 0.3 +
            zone_compatibility * 0.2 +
            aggressive_card_tendency * 0.2
        )
        
        return matchup_score

    def calculate_enhanced_player_risk(self, player_data, referee_profile, home_team, away_team):
        """
        Calcola il rischio migliorato per un singolo giocatore considerando tutti i fattori.
        """
        player_team = player_data.get('Squadra', '')
        
        # 1. Tendenza ai cartellini
        card_tendency = self._calculate_card_tendency_score(player_data)
        
        # 2. Aggressività (falli commessi)
        fouls_made = player_data.get('Media Falli Fatti 90s Totale', 0)
        foul_aggressivity = min(fouls_made / 3.0, 1.0)  # Normalizza a max 3 falli/90min
        
        # 3. Fattore ritardo cartellino
        card_delay = player_data.get('Ritardo Cartellino (Minuti)', 90)
        if pd.isna(card_delay):
            card_delay = 90
        
        # Se il ritardo è molto basso, indica cartellini rapidi (alto rischio)
        delay_factor = max(0, (180 - card_delay) / 180)
        
        # 4. Influenza arbitro
        referee_factor = referee_profile['strictness_factor']
        
        # 5. Dinamiche squadra
        team_dynamics = self._calculate_team_dynamics_factor(home_team, away_team, player_team)
        
        # 6. Zona di campo (giocatori in zone critiche hanno più rischio)
        zone = get_field_zone(player_data.get('Posizione_Primaria', ''), 
                             player_data.get('Heatmap', ''))
        zone_risk = 0.8 if zone in ['L', 'R'] else 0.6  # Laterali più a rischio
        
        # Calcola rischio finale
        final_risk = (
            card_tendency * self.weights['card_tendency'] +
            foul_aggressivity * self.weights['foul_aggressivity'] +
            delay_factor * self.weights['delay_factor'] +
            (referee_factor - 1.0 + 1.0) * self.weights['referee_influence'] +
            (team_dynamics - 1.0 + 1.0) * self.weights['team_dynamics'] +
            zone_risk * self.weights['zone_compatibility']
        )
        
        return min(final_risk, 1.0)

    def predict_match_cards(self, home_data: pd.DataFrame, away_data: pd.DataFrame, 
                          referee_data: pd.DataFrame) -> Dict:
        """
        Predice i cartellini per una partita con algoritmo super avanzato.
        """
        
        # Analizza profilo arbitro
        referee_profile = self._analyze_referee_profile(referee_data)
        
        # Ottieni nomi squadre
        home_team = home_data['Squadra'].iloc[0] if not home_data.empty else 'Home'
        away_team = away_data['Squadra'].iloc[0] if not away_data.empty else 'Away'
        
        # Identifica duelli critici
        critical_matchups = self._identify_critical_matchups(home_data, away_data)
        
        # Calcola rischio per tutti i giocatori
        all_players = pd.concat([home_data, away_data], ignore_index=True)
        predictions = []
        
        for _, player in all_players.iterrows():
            base_risk = self.calculate_enhanced_player_risk(
                player, referee_profile, home_team, away_team
            )
            
            # Bonus per duelli critici
            critical_bonus = 0
            for matchup in critical_matchups:
                if (player['Player'] == matchup['aggressive_player'] or 
                    player['Player'] == matchup['fouled_player']):
                    critical_bonus = max(critical_bonus, matchup['matchup_score'] * 0.3)
            
            final_risk = min(base_risk + critical_bonus, 1.0)
            
            # Calcola quota stimata
            quota = 100 if final_risk <= 0.01 else min((1 / final_risk) * 8, 100)
            
            predictions.append({
                'Player': player['Player'],
                'Squadra': player['Squadra'],
                'Posizione_Primaria': player.get('Posizione_Primaria', 'N/A'),
                'Zona_Campo': get_field_zone(player.get('Posizione_Primaria', ''), 
                                           player.get('Heatmap', '')),
                'Rischio_Finale': round(final_risk, 4),
                'Quota_Stimata': round(quota, 2),
                'Tendenza_Cartellini': round(self._calculate_card_tendency_score(player), 3),
                'Media_Falli_Fatti': player.get('Media Falli Fatti 90s Totale', 0),
                'Media_Falli_Subiti': player.get('Media Falli Subiti 90s Totale', 0),
                'Ritardo_Cartellino': player.get('Ritardo Cartellino (Minuti)', 0),
                'Duelli_Critici': len([m for m in critical_matchups 
                                     if player['Player'] in [m['aggressive_player'], m['fouled_player']]])
            })
        
        # Ordina per rischio
        predictions_df = pd.DataFrame(predictions).sort_values('Rischio_Finale', ascending=False)
        
        # Stima cartellini totali partita
        total_expected_cards = (
            predictions_df['Rischio_Finale'].sum() * referee_profile['strictness_factor']
        )
        
        result = {
            'predictions': predictions_df,
            'critical_matchups': critical_matchups,
            'referee_profile': referee_profile,
            'match_info': {
                'home_team': home_team,
                'away_team': away_team,
                'expected_total_cards': round(total_expected_cards, 1),
                'top_risk_players': predictions_df.head(5)['Player'].tolist()
            },
            'algorithm_info': {
                'critical_matchups_found': len(critical_matchups),
                'high_risk_players': len(predictions_df[predictions_df['Rischio_Finale'] > 0.3]),
                'weights_used': self.weights
            }
        }
        
        return result

# Classe di compatibilità
class EnhancedCardPredictionModel(SuperAdvancedCardPredictionModel):
    """Classe di compatibilità che eredita dal modello super avanzato."""
    
    def calculate_match_risk(self, home_data, away_data, referee_data):
        """Wrapper per mantenere compatibilità con l'interfaccia esistente."""
        result = self.predict_match_cards(home_data, away_data, referee_data)
        return result['predictions']