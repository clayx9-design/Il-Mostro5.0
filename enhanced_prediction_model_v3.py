import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# =========================================================================
# FUNZIONI DI SUPPORTO AVANZATE MIGLIORATE
# =========================================================================

def get_field_zone_advanced(position: str, heatmap: str) -> str:
    """
    Determina la zona di campo con logica avanzata migliorata.
    Classificazione prioritaria per fascia: Left Flank, Right Flank, Central Attacking, Central Defensive.
    """
    position = str(position).strip().upper() if pd.notna(position) else ''
    heatmap = str(heatmap).strip().lower() if pd.notna(heatmap) else ''
    
    # Analisi Heatmap avanzata: Cerca pattern specifici per zone
    if any(term in heatmap for term in ['left', 'sinistra', 'lateral left', 'left flank', 'left wing']):
        return 'Left Flank'
    elif any(term in heatmap for term in ['right', 'destra', 'lateral right', 'right flank', 'right wing']):
        return 'Right Flank'
    elif any(term in heatmap for term in ['central attacking', 'attacking third', 'final third', 'box']):
        return 'Central Attacking'
    elif any(term in heatmap for term in ['central defensive', 'defensive third', 'own half', 'tackles defensive']):
        return 'Central Defensive'
    elif 'central midfield' in heatmap or 'half spaces' in heatmap:
        return 'Central Midfield'
    
    # Fallback a Posizione primaria con priorità laterali e centrali
    if any(x in position for x in ['LB', 'LWB', 'LM', 'LW', 'LDM']): return 'Left Flank'
    if any(x in position for x in ['RB', 'RWB', 'RM', 'RW', 'RDM']): return 'Right Flank'
    if any(x in position for x in ['CB', 'CDM', 'DC']): return 'Central Defensive'
    if any(x in position for x in ['ST', 'CF', 'CAM', 'AM']): return 'Central Attacking'
    if any(x in position for x in ['CM', 'M']): return 'Central Midfield'
    
    return 'Central Midfield'

def get_player_role_category(position: str) -> str:
    """Classifica il ruolo per l'analisi dei duelli."""
    position = str(position).strip().upper() if pd.notna(position) else ''
    
    if any(role in position for role in ['GK', 'P']): return 'Goalkeeper'
    if any(role in position for role in ['CB', 'LB', 'RB', 'LWB', 'RWB', 'DF', 'D']): return 'Defender'
    if any(role in position for role in ['ST', 'CF', 'LW', 'RW', 'FW', 'A']): return 'Attacker'
    if any(role in position for role in ['CM', 'DM', 'AM', 'MF', 'M']): return 'Midfielder'
    
    return 'Midfielder'

def get_player_aggression_profile(row: pd.Series) -> Dict[str, float]:
    """Calcola il profilo di aggressività del giocatore (omesso per brevità, resta invariato)."""
    fouls_made = row.get('Media Falli Fatti 90s Totale', 0)
    fouls_suffered = row.get('Media Falli Subiti 90s Totale', 0)
    cards_total = row.get('Cartellini Gialli Totali', 0)
    minutes_played = row.get('90s Giocati Totali', 1)
    fouls_per_card = row.get('Media Falli per Cartellino Totale', 10)
    delay_factor = row.get('Ritardo Cartellino (Minuti)', 45)
    
    offensive_aggression = min(fouls_made / 3.0, 1.0)
    vulnerability = min(fouls_suffered / 2.5, 1.0)
    impulsivity = max(0, 1 - (fouls_per_card / 8.0))
    card_frequency = min((cards_total / minutes_played) * 10, 1.0) if minutes_played > 0 else 0
    delay_risk = max(0, 1 - (delay_factor / 60.0))
    
    overall_risk = (
        offensive_aggression * 0.30 +
        vulnerability * 0.20 +
        impulsivity * 0.25 +
        card_frequency * 0.15 +
        delay_risk * 0.10
    )
    
    return {
        'offensive_aggression': offensive_aggression,
        'vulnerability': vulnerability,
        'impulsivity': impulsivity,
        'card_frequency': card_frequency,
        'delay_risk': delay_risk,
        'overall_risk': overall_risk
    }

def calculate_positional_matchup_probability(zone1: str, zone2: str, role1: str, role2: str) -> float:
    """
    Migliora la probabilità di scontro con enfasi sulla logica di specchiatura L vs R.
    zone1 è la zona del giocatore 1 (es. Aggressore), zone2 è la zona del giocatore 2 (es. Vittima).
    """
    # 1. Logica di Specchiatura (Attaccante Dx vs Difensore Sx)
    is_mirrored_flank = (
        (zone1 == 'Left Flank' and zone2 == 'Right Flank') or
        (zone1 == 'Right Flank' and zone2 == 'Left Flank')
    )
    
    # 2. Logica di Ruolo Complementare
    is_attacker_vs_defender_mid = (
        (role1 == 'Attacker' and role2 in ['Defender', 'Midfielder']) or
        (role2 == 'Attacker' and role1 in ['Defender', 'Midfielder'])
    )

    if is_mirrored_flank and is_attacker_vs_defender_mid:
        # Priorità MASSIMA per duelli Attacker/Flank vs Defender/Flank opposto
        return 0.98  # Probabilità quasi certa di scontro
        
    # 3. Logica Basata sulla Zona (invariata/secondaria)
    zone_compatibility = {
        ('Central Attacking', 'Central Defensive'): 0.92,
        ('Left Flank', 'Central Defensive'): 0.85,
        ('Right Flank', 'Central Defensive'): 0.85,
        ('Central Midfield', 'Central Midfield'): 0.75,
        ('Left Flank', 'Right Flank'): 0.60, # Abbassata, la specchiatura è ora gestita dal 0.98
        ('Right Flank', 'Left Flank'): 0.60,
    }
    
    base_prob = zone_compatibility.get((zone1, zone2), zone_compatibility.get((zone2, zone1), 0.40))

    # 4. Bonus Ruolo (se non si rientra nello specchiato)
    if is_attacker_vs_defender_mid:
        return min(base_prob + 0.15, 0.90) # Bonus significativo per ruoli complementari
        
    return base_prob

def find_critical_matchups_advanced(df_home: pd.DataFrame, df_away: pd.DataFrame) -> List[Dict]:
    """
    Identifica duelli critici con logica ultra-precisa:
    - Massima priorità: Attaccante Vittima (alto FS) vs Difensore Aggressore (alto FF) di fascia opposta.
    - Calcola rischio duello basato su: Aggressività x Vulnerabilità x Probabilità di Scontro.
    """
    matchups = []
    
    # Esegui calcoli del profilo di aggressività se mancanti
    for df in [df_home, df_away]:
        if 'offensive_aggression' not in df.columns:
            df['offensive_aggression'] = df.apply(lambda row: get_player_aggression_profile(row)['offensive_aggression'], axis=1)
        if 'vulnerability' not in df.columns:
            df['vulnerability'] = df.apply(lambda row: get_player_aggression_profile(row)['vulnerability'], axis=1)
        # Rischio individuale è essenziale per il filtro
        if 'Tendenza_Individuale' not in df.columns:
            df['Tendenza_Individuale'] = df.apply(lambda row: get_player_aggression_profile(row)['overall_risk'], axis=1)

    # Identifica TOP candidati per FALLI SUBITI (Vittime) e FALLI FATTI (Aggressori)
    # Filtra i TOP 10 (per Falli Subiti) come Vittime e TOP 10 (per Falli Fatti) come Aggressori
    
    victims_home = df_home.nlargest(10, 'Media Falli Subiti 90s Totale')
    victims_away = df_away.nlargest(10, 'Media Falli Subiti 90s Totale')
    
    aggressors_home = df_home.nlargest(10, 'Media Falli Fatti 90s Totale')
    aggressors_away = df_away.nlargest(10, 'Media Falli Fatti 90s Totale')

    # Duelli: Aggressore Away vs Vittima Home & Aggressore Home vs Vittima Away
    
    for agg_df, vic_df, agg_team_name, vic_team_name in [
        (aggressors_away, victims_home, 'away', 'home'),
        (aggressors_home, victims_away, 'home', 'away')
    ]:
        for _, aggressor in agg_df.iterrows():
            agg_zone = aggressor['Zona_Campo']
            agg_role = aggressor['Categoria_Ruolo']
            
            # Filtro Aggressore: Deve essere Difensore o Centrocampista
            if agg_role not in ['Defender', 'Midfielder']:
                continue
            
            for _, victim in vic_df.iterrows():
                vic_zone = victim['Zona_Campo']
                vic_role = victim['Categoria_Ruolo']
                
                # Filtro Vittima: Deve essere Attaccante o Centrocampista
                if vic_role not in ['Attacker', 'Midfielder']:
                    continue

                matchup_prob = calculate_positional_matchup_probability(
                    agg_zone, vic_zone, agg_role, vic_role
                )
                
                if matchup_prob < 0.65: # Soglia alta per duelli ultra-critici
                    continue
                
                # Calcolo del Rischio Duello (pesato):
                # Rischio Duello = Falli Fatti Aggressore * Falli Subiti Vittima * Probabilità Matchup * Rischio Individuale Aggressore
                
                duel_risk_score = (
                    aggressor['offensive_aggression'] * victim['vulnerability'] * matchup_prob *
                    aggressor['Tendenza_Individuale'] * 0.5 # Pondera il rischio individuale
                )
                
                # Bonus per specchiatura laterale (Left vs Right)
                is_mirrored = (
                    ('Left Flank' in agg_zone and 'Right Flank' in vic_zone) or
                    ('Right Flank' in agg_zone and 'Left Flank' in vic_zone)
                )
                if is_mirrored:
                    duel_risk_score *= 1.25 # Bonus extra per la specchiatura richiesta
                
                if duel_risk_score > 0.20: # Soglia minima per considerare CRITICO
                    matchups.append({
                        'aggressor_player': aggressor['Player'],
                        'aggressor_team': aggressor['Squadra'],
                        'aggressor_zone': agg_zone,
                        'aggressor_role': agg_role,
                        'victim_player': victim['Player'],
                        'victim_team': victim['Squadra'],
                        'victim_zone': vic_zone,
                        'victim_role': vic_role,
                        'matchup_probability': matchup_prob,
                        'duel_risk': round(duel_risk_score, 3),
                        'mirrored': 'Yes' if is_mirrored else 'No'
                    })
    
    # Ordina per rischio e limita a top 10
    matchups.sort(key=lambda x: x['duel_risk'], reverse=True)
    return matchups[:10]

# =========================================================================
# CLASSE MODELLO ULTRA-AVANZATO MIGLIORATO (omessa per brevità, solo riferimenti ai metodi)
# =========================================================================

class SuperAdvancedCardPredictionModelV3:
    """Modello ultra-avanzato migliorato per predizione cartellini."""
    
    def __init__(self):
        self.weights = {
            'individual_tendency': 0.30,
            'matchup_risk': 0.25,      # Aumentato per enfasi duelli
            'referee_influence': 0.15,
            'positional_risk': 0.15,
            'aggression_profile': 0.10,
            'delay_factor': 0.05
        }
        self.SERIE_A_AVG_CARDS = 4.2
        self.RISK_BASELINE = 0.25

    def calculate_enhanced_individual_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcola rischio individuale con profili aggressività."""
        df = df.copy()
        df['aggression_profile'] = df.apply(get_player_aggression_profile, axis=1)
        profile_cols = ['offensive_aggression', 'vulnerability', 'impulsivity', 'card_frequency', 'delay_risk', 'overall_risk']
        for col in profile_cols:
            df[col] = df['aggression_profile'].apply(lambda x: x.get(col, 0))
        df['Tendenza_Individuale'] = df['overall_risk']
        df['Categoria_Ruolo'] = df['Posizione_Primaria'].apply(get_player_role_category)
        df.drop('aggression_profile', axis=1, inplace=True)
        return df

    def calculate_positional_risk_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rischio posizionale basato su zone."""
        df = df.copy()
        zone_risks = {
            'Left Flank': 0.85, 'Right Flank': 0.85,
            'Central Attacking': 0.80, 'Central Defensive': 0.75,
            'Central Midfield': 0.65
        }
        df['Rischio_Posizionale'] = df['Zona_Campo'].map(zone_risks).fillna(0.50)
        return df

    def calculate_matchup_risk_advanced(self, df_home: pd.DataFrame, df_away: pd.DataFrame) -> pd.DataFrame:
        """Incorpora rischio duelli con specchiature."""
        all_players = pd.concat([df_home, df_away], ignore_index=True)
        critical_matchups = find_critical_matchups_advanced(df_home, df_away) # Usa la nuova logica
        
        player_duel_risks = {}
        for matchup in critical_matchups:
            agg_risk = matchup['duel_risk']
            player_duel_risks.setdefault(matchup['aggressor_player'], 0)
            player_duel_risks[matchup['aggressor_player']] = max(player_duel_risks[matchup['aggressor_player']], agg_risk)
            player_duel_risks.setdefault(matchup['victim_player'], 0)
            player_duel_risks[matchup['victim_player']] = max(player_duel_risks[matchup['victim_player']], agg_risk * 0.8)
        
        all_players['Rischio_Duello'] = all_players['Player'].map(player_duel_risks).fillna(0)
        return all_players

    def calculate_referee_influence_advanced(self, df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
        """Influenza arbitro (invariata)."""
        if ref_df.empty: return df
        ref_cards = ref_df['Gialli a partita'].iloc[0]
        severity_factor = ref_cards / self.SERIE_A_AVG_CARDS
        df['Fattore_Arbitro'] = severity_factor
        return df

    def calculate_final_risk_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rischio finale con pesi aggiornati (invariata)."""
        df = df.copy()
        ref_factor = df['Fattore_Arbitro'].iloc[0] if 'Fattore_Arbitro' in df.columns else 1.0
        delay_factor = df['delay_risk']
        
        base_risk = (
            df['Tendenza_Individuale'] * self.weights['individual_tendency'] +
            df['Rischio_Duello'] * self.weights['matchup_risk'] +
            df['Rischio_Posizionale'] * self.weights['positional_risk'] +
            df['offensive_aggression'] * self.weights['aggression_profile']
        )
        
        df['Rischio_Finale'] = (
            base_risk * ref_factor * (1 + delay_factor * self.weights['delay_factor'])
        )
        
        if df['Rischio_Finale'].max() > 0:
            df['Rischio_Finale'] = (df['Rischio_Finale'] / df['Rischio_Finale'].max()).clip(0.01, 0.95)
        
        df['Quota_Stimata'] = (1 / df['Rischio_Finale']) * 0.6
        df['Quota_Stimata'] = df['Quota_Stimata'].clip(1.2, 15.0)
        
        return df

    def predict_match_cards(self, home_df: pd.DataFrame, away_df: pd.DataFrame, ref_df: pd.DataFrame) -> Dict:
        """Pipeline completa."""
        df_home = self.calculate_enhanced_individual_risk(home_df.copy())
        df_away = self.calculate_enhanced_individual_risk(away_df.copy())
        
        for df in [df_home, df_away]:
            df['Zona_Campo'] = df.apply(
                lambda row: get_field_zone_advanced(row.get('Posizione_Primaria', ''), row.get('Heatmap', '')), 
                axis=1
            )
        
        df_home = self.calculate_positional_risk_advanced(df_home)
        df_away = self.calculate_positional_risk_advanced(df_away)
        
        # Duelli avanzati
        all_players_df = self.calculate_matchup_risk_advanced(df_home.copy(), df_away.copy()) # Uso copie per evitare side effects
        
        all_players_df = self.calculate_referee_influence_advanced(all_players_df, ref_df)
        all_players_df = self.calculate_final_risk_advanced(all_players_df)
        
        # Ricalcola i duelli con le metriche finali (solo per l'output JSON)
        critical_matchups = find_critical_matchups_advanced(df_home, df_away)
        
        all_players_df.sort_values('Rischio_Finale', ascending=False, inplace=True)
        
        ref_cards = ref_df['Gialli a partita'].iloc[0] if not ref_df.empty else 4.2
        team_avg_risk = all_players_df['Rischio_Finale'].mean()
        expected_cards = min(ref_cards * (1 + team_avg_risk * 2), 8.0)
        
        severity = 'strict' if ref_cards > 5.5 else 'permissive' if ref_cards < 3.5 else 'medium'
        factor = 1.3 if severity == 'strict' else 0.8 if severity == 'permissive' else 1.0
        
        return {
            'all_predictions': all_players_df,
            'top_4_predictions': all_players_df.head(4).to_dict('records'),
            'match_info': {
                'home_team': home_df['Squadra'].iloc[0],
                'away_team': away_df['Squadra'].iloc[0],
                'expected_total_cards': f"{expected_cards:.1f}",
                'algorithm_confidence': 'Ultra High' if len(critical_matchups) >= 3 else 'High',
            },
            'referee_profile': {
                'name': ref_df['Nome'].iloc[0] if not ref_df.empty else 'N/A',
                'cards_per_game': ref_cards,
                'strictness_factor': factor,
                'severity_level': severity,
            },
            'critical_matchups': critical_matchups,
            'algorithm_summary': {
                'methodology': 'Ultra-Avanzato con Priorità Duelli Specchiati L vs R (Attacker/Victim vs Defender/Aggressor)',
                'critical_matchups_found': len(critical_matchups),
                'high_risk_players': (all_players_df['Rischio_Finale'] > 0.5).sum(),
                'weights_used': self.weights,
                'improvements': [
                    'Massima Priorità Duello Specchiato (L vs R) Attacker vs Defender/Midfielder',
                    'Filtri Ruolo/Statistiche più Severi per Matchup Critici',
                    'Bonus per Ruoli Complementari e Fascia Opposta'
                ]
            }
        }

# Compatibilità
def get_field_zone(position, heatmap):
    return get_field_zone_advanced(position, heatmap)

SuperAdvancedCardPredictionModel = SuperAdvancedCardPredictionModelV3