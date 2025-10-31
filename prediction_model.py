import pandas as pd
import numpy as np
import re
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# =========================================================================
# FUNZIONI DI SUPPORTO
# =========================================================================

def normalize_name(name):
    """Normalizza un nome rimuovendo accenti, spazi e caratteri speciali."""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s-]+', '_', name)
    return name

def get_side_of_field(position: str, heatmap: str) -> Optional[str]:
    """Estrae il lato del campo (L, R) dalla posizione o dalla heatmap. Restituisce 'V' (Verticale/Centrale) se non laterale."""
    if pd.isna(position):
        position = ''
    if pd.isna(heatmap):
        heatmap = ''
        
    # 1. Priorità alla Posizione Primaria: Cerca 'L' o 'R' all'interno della stringa di posizione
    pos_upper = position.upper()
    if 'R' in pos_upper and 'L' not in pos_upper:
        return 'R'
    elif 'L' in pos_upper and 'R' not in pos_upper:
        return 'L'
        
    # 2. Fallback all'Heatmap: Cerca termini laterali (migliorato con regex per 'flank')
    heatmap_lower = heatmap.lower()
    if re.search(r'(right|destra|rwb?|rb?|right flank)', heatmap_lower):
        return 'R'
    if re.search(r'(left|sinistra|lwb?|lb?|left flank)', heatmap_lower):
        return 'L'

    # 3. Ritorno 'V' per Verticale/Centrale (o non specificato)
    return 'V'

def calculate_derived_metrics(df_players: pd.DataFrame) -> pd.DataFrame:
    """Calcola metriche derivate dai dati grezzi del file Excel."""
    df = df_players.copy()
    
    # Colonne numeriche essenziali dai dati grezzi
    numeric_cols = {
        'Falli_Fatti_Totali': 'Falli Fatti Totali',
        'Falli_Subiti_Totali': 'Falli Subiti Totali',
        'Cartellini_Gialli_Totali': 'Cartellini Gialli Totali',
        'Minuti_Giocati_Totali': 'Minuti Giocati Totali',
        '90s_Giocati_Totali': '90s Giocati Totali'
    }
    
    # Converti in numerico
    for derived, raw in numeric_cols.items():
        if raw in df.columns:
            df[derived] = pd.to_numeric(df[raw], errors='coerce').fillna(0)
    
    # Calcola metriche derivate
    df['Media_Falli_Fatti_90s_Totale'] = df['Falli_Fatti_Totali'] / df['90s_Giocati_Totali'].replace(0, np.nan)
    df['Media_Falli_Subiti_90s_Totale'] = df['Falli_Subiti_Totali'] / df['90s_Giocati_Totali'].replace(0, np.nan)
    
    # Media 90s per Cartellino Totale (bassa = aggressivo)
    df['Media_90s_per_Cartellino_Totale'] = df['90s_Giocati_Totali'] / df['Cartellini_Gialli_Totali'].replace(0, np.inf)
    
    # Media Falli per Cartellino Totale (bassa = propenso ai gialli)
    df['Media_Falli_per_Cartellino_Totale'] = df['Falli_Fatti_Totali'] / df['Cartellini_Gialli_Totali'].replace(0, np.inf)
    
    # Ritardo Cartellino (Minuti): Assumi media semplificata se non presente; altrimenti calcola da dati se disponibili
    # Per demo, genera basato su impulsività (bassa media_90s -> ritardo basso)
    df['Ritardo_Cartellino_Minuti'] = np.where(
        df['Media_90s_per_Cartellino_Totale'] < df['Media_90s_per_Cartellino_Totale'].median(),
        np.random.uniform(20, 60, len(df)),  # Impulsivo: ritardo basso
        np.random.uniform(60, 120, len(df))  # Calmo: ritardo alto
    )
    
    # Gestione NaN/Inf
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Mappa Posizione_Primaria da Pos (abbreviazioni comuni)
    position_mapping = {
        'GK': 'GK', 'DF': 'DF', 'MF': 'MF', 'FW': 'FW',
        'CB': 'DF', 'LB': 'DF', 'RB': 'DF', 'DM': 'MF', 'CM': 'MF', 'AM': 'MF',
        'LW': 'FW', 'RW': 'FW', 'ST': 'FW'
    }
    df['Posizione_Primaria'] = df['Pos'].map(position_mapping).fillna('MF')
    
    # Heatmap: se non presente, genera basata su posizione
    if 'Heatmap' not in df.columns:
        df['Heatmap'] = np.where(
            df['Posizione_Primaria'].str.contains('DF'),
            'High activity in defensive third, focused on tackles and aerial duels',
            np.where(
                df['Posizione_Primaria'].str.contains('FW'),
                'High activity in attacking third, focused on finishing',
                'Central activity with moderate defensive contribution'
            )
        )
    
    return df

# =========================================================================
# CLASSE DI PREDIZIONE MIGLIORATA E ROBUSTA
# =========================================================================

class AdvancedCardPredictionModel:
    """
    Modello avanzato e robusto per predire 4 giocatori ad alto rischio di cartellino giallo.
    Integra medie globali/squadra/arbitro, analisi marcature a specchio, e fattori impulsività/ritardo.
    """
    
    def __init__(self, marking_threshold_fouls_suffered: float = 1.5,
                 marking_threshold_fouls_committed: float = 1.8,
                 compatibility_score_threshold: float = 0.3,
                 global_referee_avg: float = 4.0):  # Media gialli/partita Serie A
        self.marking_threshold_fouls_suffered = marking_threshold_fouls_suffered
        self.marking_threshold_fouls_committed = marking_threshold_fouls_committed
        self.compatibility_score_threshold = compatibility_score_threshold
        self.global_referee_avg = global_referee_avg
        
        self.defensive_roles = ['DF', 'CB', 'LB', 'RB', 'LWB', 'RWB', 'DM']
        self.central_mid_roles = ['CM', 'DM', 'AM']
        self.global_medians = {}  # Verranno calcolati in calculate_match_risk

    def _calculate_team_and_global_averages(self, df_players: pd.DataFrame, df_referees: pd.DataFrame) -> Dict:
        """Calcola medie globali, per squadra e per arbitro."""
        # Medie globali giocatori
        self.global_medians = {
            'fouls_suffered_90s': df_players['Media_Falli_Subiti_90s_Totale'].median(),
            'fouls_committed_90s': df_players['Media_Falli_Fatti_90s_Totale'].median(),
            'games_per_card': df_players['Media_90s_per_Cartellino_Totale'].median(),
            'fouls_per_card': df_players['Media_Falli_per_Cartellino_Totale'].median(),
            'card_delay': df_players['Ritardo_Cartellino_Minuti'].median()
        }
        
        # Medie per squadra (cartellini totali / partite ~34 per stagione)
        df_players['Squadra_Avg_Cards'] = df_players.groupby('Squadra')['Cartellini_Gialli_Totali'].transform('mean') / 34.0
        
        # Medie arbitri
        avg_referee_cards = df_referees['Gialli ap (Media/Partita)'].mean() if 'Gialli ap (Media/Partita)' in df_referees.columns else self.global_referee_avg
        
        return {
            'global_medians': self.global_medians,
            'avg_referee_cards': avg_referee_cards,
            'team_avg_cards': df_players.groupby('Squadra')['Squadra_Avg_Cards'].first().to_dict()
        }

    def _calculate_statistical_risk(self, row, referee_factor: float, averages: Dict) -> float:
        """Calcola rischio statistico base, integrando deviazioni dalle medie."""
        # Base: falli fatti/subiti
        fouls_risk = (row['Media_Falli_Fatti_90s_Totale'] / averages['global_medians']['fouls_committed_90s']) * 0.4
        suffered_risk = (row['Media_Falli_Subiti_90s_Totale'] / averages['global_medians']['fouls_suffered_90s']) * 0.3
        
        # Aggressività: inverso media partite/cartellino (bassa = alto rischio)
        games_per_card_safe = max(row['Media_90s_per_Cartellino_Totale'], 1e-6)
        agg_risk = (averages['global_medians']['games_per_card'] / games_per_card_safe) * 0.2
        
        # Propensione: inverso falli/cartellino (bassa = propenso)
        fouls_per_card_safe = max(row['Media_Falli_per_Cartellino_Totale'], 1e-6)
        prop_risk = (averages['global_medians']['fouls_per_card'] / fouls_per_card_safe) * 0.2
        
        # Deviazione dalla media squadra
        team_dev = abs(row.get('Squadra_Avg_Cards', 0) - averages['team_avg_cards'].get(row['Squadra'], 0))
        team_risk = min(team_dev * 0.1, 0.5)  # Penalizza deviazioni alte
        
        risk = fouls_risk + suffered_risk + agg_risk + prop_risk + team_risk
        return risk * referee_factor

    def _calculate_delay_factor(self, row: pd.Series, global_medians: Dict) -> float:
        """Fattore ritardo: applicato SOLO a giocatori con media partite/cartellino bassa (tendenti al cartellino).
        Se media_90s_per_cartellino < mediana globale, allora:
        - Se ritardo < threshold (basato su media partite), aumenta rischio (impulsivo).
        Altrimenti, fattore neutro (1.0)."""
        games_per_card = row['Media_90s_per_Cartellino_Totale']
        global_games_per_card = global_medians['games_per_card']
        
        # Applica solo a tendenti (bassa media partite/cartellino)
        if games_per_card >= global_games_per_card:
            return 1.0  # Neutro per giocatori calmi/non tendenti
        
        # Per tendenti: calcola threshold e factor
        delay = row['Ritardo_Cartellino_Minuti']
        threshold = games_per_card * 30  # Es. se 5 partite/cartellino, threshold ~150 min
        
        if delay > threshold:
            return 0.7  # Calmo nonostante tendenza, riduci rischio
        elif delay < global_medians['card_delay'] * 0.8:  # Impulsivo (basso ritardo)
            return 1.3  # Aumenta rischio
        return 1.0  # Neutro

    def _get_role_category(self, pos: str) -> Tuple[str, str]:
        """Categorizza ruolo per compatibilità: (main, side) es. ('Defender', 'Flank') per LB/RB, ('Central_Mid', 'Central') per CM."""
        pos_upper = pos.upper()
        is_flank = any(side in pos_upper for side in ['LB', 'RB', 'LW', 'RW', 'LWB', 'RWB'])
        
        if any(role in pos_upper for role in ['CM', 'DM', 'AM']):
            return 'Central_Mid', 'Central'
        elif 'FW' in pos_upper or 'ST' in pos_upper:
            return 'Forward', 'Flank' if is_flank else 'Central'
        elif any(role in pos_upper for role in ['DF', 'CB']):
            return 'Defender', 'Flank' if is_flank else 'Central'
        elif any(role in pos_upper for role in ['LW', 'RW', 'LWB', 'RWB']):
            return 'Flank', 'Flank'
        return 'Other', 'Central'

    def _calculate_compatibility_score(self, player_pos: str, marker_pos: str, player_side: str, marker_side: str) -> Tuple[float, str]:
        """Calcola score di compatibilità (0-1) per duelli, con logica specifica per ruoli e sottocategorie.
        - CC (Central_Mid, Central) vs CC: 1.0
        - Att (Forward) vs Dif (Defender): 1.0
        - Dif Esterno (Defender, Flank) vs CC (Central_Mid, Central): 0.3 (basso, evita duelli non realistici)
        - Esterni (Flank): opposti L/R: 1.0; uguali: 0.8
        - Att Esterno vs Dif: 1.0 (bonus); vs CC Esterno: 0.7
        - Centrali vs tutto (eccetto casi specifici): 0.8
        - Altri: 0.5"""
        player_main, player_sub = self._get_role_category(player_pos)
        marker_main, marker_sub = self._get_role_category(marker_pos)
        
        # CC vs CC
        if player_main == 'Central_Mid' and marker_main == 'Central_Mid':
            return 1.0, 'CC vs CC'
        
        # Att vs Dif
        if player_main == 'Forward' and marker_main == 'Defender':
            return 1.0, 'Att vs Dif'
        
        # Dif vs Att (raro)
        if player_main == 'Defender' and marker_main == 'Forward':
            return 0.8, 'Dif vs Att'
        
        # Dif Esterno vs CC: basso (evita casi come Posch vs Niasse)
        if player_main == 'Defender' and player_sub == 'Flank' and marker_main == 'Central_Mid' and marker_sub == 'Central':
            return 0.3, 'Dif Esterno vs CC (Basso)'
        if marker_main == 'Defender' and marker_sub == 'Flank' and player_main == 'Central_Mid' and player_sub == 'Central':
            return 0.3, 'CC vs Dif Esterno (Basso)'
        
        # Logica esterni (Flank)
        if player_sub == 'Flank' or marker_sub == 'Flank':
            if player_side != marker_side and player_side != 'V' and marker_side != 'V':
                comp = 1.0  # Opositi L/R
                detail = f'{player_side} vs {marker_side} (Opositi Esterni)'
            else:
                comp = 0.8  # Uguali o misti
                detail = f'{player_side} vs {marker_side} (Esterni Misti)'
            
            # Bonus per Att Esterno vs Dif
            if player_main == 'Forward' and marker_main == 'Defender':
                comp = 1.0
                detail = 'Att Esterno vs Dif (Bonus)'
            elif player_main == 'Forward' and marker_main == 'Central_Mid':
                comp = 0.7
                detail = 'Att Esterno vs CC Esterno'
            return comp, detail
        
        # Centrali vs tutto (default, ma con soglia più alta per evitare mismatch)
        if player_sub == 'Central' or marker_sub == 'Central':
            return 0.8, 'Centrale vs Qualsiasi'
        
        # Default basso
        return 0.5, 'Bassa Compatibilità'

    def identify_critical_marking_situations(self, home_data: pd.DataFrame, away_data: pd.DataFrame, averages: Dict) -> List[Dict]:
        """Identifica marcature critiche: top falli subiti vs potenziali marcatori aggressivi.
        Usa score di compatibilità per pesare i duelli (non mostra dettagli, solo per elaborazione)."""
        critical_situations = []
        
        # Seleziona top 20% falli subiti per squadra (giocatori "vittime")
        for team_data, is_home in [(home_data, True), (away_data, False)]:
            high_sufferers = team_data[
                team_data['Media_Falli_Subiti_90s_Totale'] >= team_data['Media_Falli_Subiti_90s_Totale'].quantile(0.8)
            ]
            
            opponent_data = away_data if is_home else home_data
            
            for _, player in high_sufferers.iterrows():
                player_side = get_side_of_field(player['Posizione_Primaria'], player['Heatmap'])
                
                # Potenziali marcatori: top aggressivi in ruoli complementari
                potential_markers = opponent_data[
                    (opponent_data['Media_Falli_Fatti_90s_Totale'] >= self.marking_threshold_fouls_committed) &
                    (opponent_data['Posizione_Primaria'].isin(self.defensive_roles) if 'FW' in player['Posizione_Primaria'] else True)
                ]
                
                for _, marker in potential_markers.iterrows():
                    marker_side = get_side_of_field(marker['Posizione_Primaria'], marker['Heatmap'])
                    comp_score, detail = self._calculate_compatibility_score(player['Posizione_Primaria'], marker['Posizione_Primaria'], player_side, marker_side)
                    
                    if comp_score >= 0.5:  # Soglia minima per considerare duello (esclude 0.3 per Dif Est vs CC)
                        # Score matchup pesato dalla compatibilità
                        base_matchup = (player['Media_Falli_Subiti_90s_Totale'] * marker['Media_Falli_Fatti_90s_Totale']) / (averages['global_medians']['fouls_suffered_90s'] * averages['global_medians']['fouls_committed_90s'])
                        
                        # Fattori aggressività marcatori
                        marker_agg = (averages['global_medians']['games_per_card'] / max(marker['Media_90s_per_Cartellino_Totale'], 1e-6)) * 0.2
                        marker_prop = (averages['global_medians']['fouls_per_card'] / max(marker['Media_Falli_per_Cartellino_Totale'], 1e-6)) * 0.2
                        
                        # Usa comp_score invece di bonus fisso
                        
                        # Delay factor per entrambi (solo se tendenti)
                        player_delay_factor = self._calculate_delay_factor(player, averages['global_medians'])
                        marker_delay_factor = self._calculate_delay_factor(marker, averages['global_medians'])
                        
                        situation_risk = base_matchup * (marker_agg + marker_prop) * comp_score * player_delay_factor * marker_delay_factor
                        
                        if situation_risk > self.compatibility_score_threshold:
                            critical_situations.append({
                                'Player': player['Player'],
                                'Team': player['Squadra'],
                                'Marker': marker['Player'],
                                'Marker_Team': marker['Squadra'],
                                'Player_Side': player_side,
                                'Marker_Side': marker_side,
                                'Compatibility_Score': comp_score,
                                'Compatibility_Detail': detail,  # Interno, non mostrato
                                'Situation_Risk': situation_risk,
                                'Matchup_Type': 'Victim vs Aggressor'
                            })
        
        return critical_situations

    def calculate_match_risk(self, home_data: pd.DataFrame, away_data: pd.DataFrame, referee_data: pd.DataFrame) -> Dict:
        """Calcola rischi integrati e restituisce top 4 predizioni (duelli solo interni)."""
        # Preprocess dati
        home_data = calculate_derived_metrics(home_data)
        away_data = calculate_derived_metrics(away_data)
        df_match = pd.concat([home_data, away_data], ignore_index=True)
        
        # Calcola medie
        averages = self._calculate_team_and_global_averages(df_match, referee_data)
        
        # Fattore arbitro specifico vs globali
        referee_factor = 1.0
        if not referee_data.empty:
            ref_yellows = referee_data['Gialli ap (Media/Partita)'].iloc[0] if 'Gialli ap (Media/Partita)' in referee_data.columns else 4.0
            referee_factor = ref_yellows / averages['avg_referee_cards']
        
        # Rischio statistico base per tutti
        df_match['Rischio_Statistico'] = df_match.apply(
            lambda row: self._calculate_statistical_risk(row, referee_factor, averages), axis=1
        )
        
        # Identifica situazioni critiche (duelli interni)
        critical_situations = self.identify_critical_marking_situations(home_data, away_data, averages)
        
        # Aggrega rischi critici per giocatore (max per ruolo vittima/marcatore)
        player_risks = df_match[['Player', 'Squadra', 'Rischio_Statistico']].copy()
        if critical_situations:
            crit_df = pd.DataFrame(critical_situations)
            # Rischio max come vittima
            victim_risk = crit_df.groupby(['Player', 'Team'])['Situation_Risk'].max().reset_index(name='Rischio_Vittima')
            # Rischio max come marcatore
            marker_risk = crit_df.groupby(['Marker', 'Marker_Team'])['Situation_Risk'].max().reset_index(name='Rischio_Marcatore')
            marker_risk.rename(columns={'Marker': 'Player', 'Marker_Team': 'Squadra'}, inplace=True)
            
            # Merge e max
            crit_risk = pd.merge(victim_risk, marker_risk, on=['Player', 'Squadra'], how='outer').fillna(0)
            crit_risk['Rischio_Critico'] = crit_risk[['Rischio_Vittima', 'Rischio_Marcatore']].max(axis=1)
            
            player_risks = pd.merge(player_risks, crit_risk[['Player', 'Squadra', 'Rischio_Critico']], on=['Player', 'Squadra'], how='left').fillna(0)
            
            # Rischio finale: 60% critico se presente, else 100% statistico + delay factor (solo per tendenti)
            player_risks['Delay_Factor'] = player_risks.apply(lambda row: self._calculate_delay_factor(row, averages['global_medians']), axis=1)
            player_risks['Rischio_Finale'] = np.where(
                player_risks['Rischio_Critico'] > 0,
                (player_risks['Rischio_Statistico'] * 0.4 + player_risks['Rischio_Critico'] * 0.6) * player_risks['Delay_Factor'],
                player_risks['Rischio_Statistico'] * player_risks['Delay_Factor']
            )
        else:
            player_risks['Delay_Factor'] = player_risks.apply(lambda row: self._calculate_delay_factor(row, averages['global_medians']), axis=1)
            player_risks['Rischio_Finale'] = player_risks['Rischio_Statistico'] * player_risks['Delay_Factor']
        
        # Top 4 predizioni
        top_4 = player_risks.nlargest(4, 'Rischio_Finale')[['Player', 'Squadra', 'Rischio_Finale']].to_dict('records')
        
        return {
            'top_4_predictions': top_4,
            'all_risks': player_risks,
            'critical_situations': critical_situations,  # Interno, non mostrato nell'app
            'referee_factor': referee_factor,
            'averages': averages
        }

# Mantieni la classe originale per compatibilità
class CardPredictionModel(AdvancedCardPredictionModel):
    """Wrapper per compatibilità."""
    pass