import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# =========================================================================
# FUNZIONI DI SUPPORTO
# =========================================================================

def get_field_zone(position, heatmap):
    """
    Determina la zona di campo (Flank, Central, Midfield).
    Migliorata classificazione per evitare 'Midfield' generico per i difensori.
    """
    position = str(position).strip().upper()
    heatmap = str(heatmap).strip().upper()
    
    # Priorità: Laterali (Flank)
    if 'L' in position or 'LB' in position or 'LW' in position or 'LM' in position: return 'Left Flank'
    if 'R' in position or 'RB' in position or 'RW' in position or 'RM' in position: return 'Right Flank'
    
    # Centrali (Difensori Centrali, Attaccanti Centrali, Mediani Difensivi)
    if 'CB' in position or 'ST' in position or 'DM' in position or 'CDM' in position: return 'Central'
    
    return 'Midfield'

def get_player_role_category(position):
    """Classifica il ruolo del giocatore in categorie ampie."""
    position = str(position).strip().upper()
    if 'G' in position or 'P' in position: return 'Goalkeeper'
    if 'D' in position or 'CB' in position or 'RB' in position or 'LB' in position: return 'Defender'
    if 'C' in position or 'M' in position: return 'Midfielder'
    if 'A' in position or 'ST' in position or 'W' in position: return 'Attacker'
    return 'Other'

def find_critical_matchups(df_home, df_away):
    """
    Identifica i duelli critici basati su logica posizionale a specchio,
    escludendo duelli non produttivi (Defender vs Defender, Midfielder vs Defender generici).
    """
    matchups = []
    
    # Aggressori (High Falli Fatti + Alto Rischio)
    aggressors_home = df_home[df_home['Rischio_Finale'] > 0.15].sort_values(
        by=['Rischio_Finale', 'Media Falli Fatti 90s Totale'], ascending=False
    )
    aggressors_away = df_away[df_away['Rischio_Finale'] > 0.15].sort_values(
        by=['Rischio_Finale', 'Media Falli Fatti 90s Totale'], ascending=False
    )
    
    # Vittime (High Falli Subiti)
    victims_home = df_home[df_home['Media Falli Subiti 90s Totale'] > df_home['Media Falli Subiti 90s Totale'].mean() * 0.8].sort_values(
        by='Media Falli Subiti 90s Totale', ascending=False
    )
    victims_away = df_away[df_away['Media Falli Subiti 90s Totale'] > df_away['Media Falli Subiti 90s Totale'].mean() * 0.8].sort_values(
        by='Media Falli Subiti 90s Totale', ascending=False
    )

    # Definizione degli abbinamenti critici e compatibilità di base
    MATCHING_MAP = {
        # Laterali opposti: Altissima frizione
        ('Left Flank', 'home'): ('Right Flank', 'away', 0.95),  
        ('Right Flank', 'home'): ('Left Flank', 'away', 0.95), 
        ('Left Flank', 'away'): ('Right Flank', 'home', 0.95),  
        ('Right Flank', 'away'): ('Left Flank', 'home', 0.95),  

        # Centrali: Attaccante vs Difensore/Mediano
        ('Central', 'home'): ('Central', 'away', 0.85),
        ('Central', 'away'): ('Central', 'home', 0.85),
        
        # Centrocampo vs Centrocampo
        ('Midfield', 'home'): ('Midfield', 'away', 0.75),
        ('Midfield', 'away'): ('Midfield', 'home', 0.75),
    }

    used_players = set()
    
    for (agg_zone, agg_team), (vic_zone, vic_team, base_compat) in MATCHING_MAP.items():
        
        agg_df = aggressors_home if agg_team == 'home' else aggressors_away
        vic_df = victims_away if vic_team == 'away' else victims_home
        
        agg_candidates = agg_df[agg_df['Zona_Campo'] == agg_zone].head(3)
        vic_candidates = vic_df[vic_df['Zona_Campo'] == vic_zone].head(3)
        
        if not agg_candidates.empty and not vic_candidates.empty:
            
            for agg_idx in agg_candidates.index:
                agg = agg_candidates.loc[agg_idx]
                if agg['Player'] in used_players: continue
                
                for vic_idx in vic_candidates.index:
                    vic = vic_candidates.loc[vic_idx]
                    if vic['Player'] in used_players: continue
                    
                    agg_role = agg['Categoria_Ruolo']
                    vic_role = vic['Categoria_Ruolo']
                    
                    # Logica di ESCLUSIONE/FILTRAGGIO
                    # 1. Escludi Defender vs Defender
                    if agg_role == 'Defender' and vic_role == 'Defender':
                        continue
                        
                    # 2. Escludi duelli Midfielder generici vs Difensore
                    if agg_zone == 'Midfield' and ('Defender' in agg_role or 'Defender' in vic_role):
                        continue
                        
                    # Duello trovato!
                    risk_score = (agg['Rischio_Finale'] + agg['Media Falli Fatti 90s Totale'] * 0.1) * base_compat
                    
                    matchups.append({
                        'risk_score': risk_score,
                        'aggressor_player': agg['Player'],
                        'aggressor_team': agg['Squadra'],
                        'aggressor_zone': agg['Zona_Campo'],
                        'aggressor_role': agg_role,
                        'victim_player': vic['Player'],
                        'victim_team': vic['Squadra'],
                        'victim_zone': vic['Zona_Campo'],
                        'victim_role': vic_role,
                        'compatibility': base_compat
                    })
                    
                    used_players.add(agg['Player'])
                    used_players.add(vic['Player'])
                    break
                if agg['Player'] in used_players: break
    
    matchups.sort(key=lambda x: x['risk_score'], reverse=True)
    return matchups[:6]


# =========================================================================
# MODELLO PRINCIPALE DI PREDIZIONE
# =========================================================================

class SuperAdvancedCardPredictionModel:
    
    def __init__(self):
        self.weights = {
            'individual_tendency': 0.25,
            'matchup_risk': 0.20,
            'referee_influence': 0.18,
            'team_dynamics': 0.15,
            'positional_risk': 0.12,
            'delay_factor': 0.10,
        }
        self.SERIE_A_AVG_CARDS = 4.5
        self.RISK_BASELINE = 0.30

    def calculate_individual_risk(self, df):
        # ... (Logica esistente per il rischio individuale)
        df['Tendenza_Cartellini'] = 1 / (df['Media 90s per Cartellino Totale'].replace(0, np.inf) + 0.01)
        df['Impulsivita'] = 1 / (df['Media Falli per Cartellino Totale'].replace(0, np.inf) + 0.01)
        df['Rischio_Posizionale'] = df['Categoria_Ruolo'].apply(
            lambda x: 0.5 if x in ['Defender', 'Midfielder'] else 0.2
        )
        df['Bilancio_Falli'] = (df['Media Falli Fatti 90s Totale'] - df['Media Falli Subiti 90s Totale']).clip(lower=0)
        df['Tendenza_Norm'] = (df['Tendenza_Cartellini'] / df['Tendenza_Cartellini'].max()).fillna(0)
        df['Impulsivita_Norm'] = (df['Impulsivita'] / df['Impulsivita'].max()).fillna(0)
        df['Bilancio_Norm'] = (df['Bilancio_Falli'] / df['Bilancio_Falli'].max()).fillna(0)
        
        df['Rischio_Individuale'] = (
            df['Tendenza_Norm'] * 0.4 +
            df['Impulsivita_Norm'] * 0.3 +
            df['Bilancio_Norm'] * 0.2 +
            df['Rischio_Posizionale'] * 0.1
        )
        df.rename(columns={'Rischio_Individuale': 'Tendenza_Individuale'}, inplace=True)
        return df

    def calculate_matchup_risk(self, df_players):
        """Calcola il rischio base per i duelli, utilizzato nel calcolo finale."""
        df_players['Rischio_Duello'] = (
            df_players['Tendenza_Individuale'] * 0.6 +
            (df_players['Media Falli Fatti 90s Totale'] / df_players['Media Falli Fatti 90s Totale'].max()).fillna(0) * 0.4
        )
        return df_players
        
    def calculate_final_risk(self, df_players, ref_df):
        """Combina tutti i fattori per il rischio finale."""
        
        ref_cards_per_game = ref_df['Gialli a partita'].iloc[0]
        ref_severity_factor = ref_cards_per_game / self.SERIE_A_AVG_CARDS
        
        df_players['Rischio_Finale'] = (
            df_players['Tendenza_Individuale'] * self.weights['individual_tendency'] +
            df_players['Rischio_Duello'] * self.weights['matchup_risk'] +
            (df_players['Tendenza_Individuale'] * ref_severity_factor) * self.weights['referee_influence'] +
            df_players['Rischio_Posizionale'] * self.weights['positional_risk'] + 
            (1 - (df_players['Ritardo Cartellino (Minuti)'] / df_players['Ritardo Cartellino (Minuti)'].max()).fillna(0.5)) * self.weights['delay_factor']
        )
        
        df_players['Rischio_Finale'] = (df_players['Rischio_Finale'] / df_players['Rischio_Finale'].max()).clip(0.01, 0.6)
        
        df_players['Quota_Stimata'] = (1 / df_players['Rischio_Finale']) * 0.5
        df_players['Quota_Stimata'] = df_players['Quota_Stimata'].clip(1.5, 10.0)
        
        return df_players
    
    def predict_match_cards(self, home_df, away_df, ref_df):
        """Esegue l'intera pipeline di predizione."""
        
        df_home = home_df.copy()
        df_away = away_df.copy()
        all_players_df = pd.concat([df_home, df_away], ignore_index=True)
        
        if 'Giocatore' in all_players_df.columns and 'Player' not in all_players_df.columns:
            all_players_df.rename(columns={'Giocatore': 'Player'}, inplace=True)
            df_home.rename(columns={'Giocatore': 'Player'}, inplace=True)
            df_away.rename(columns={'Giocatore': 'Player'}, inplace=True)

        # Calcolo Rischio
        all_players_df = self.calculate_individual_risk(all_players_df)
        all_players_df = self.calculate_matchup_risk(all_players_df)
        all_players_df = self.calculate_final_risk(all_players_df, ref_df)
        
        home_players = all_players_df[all_players_df['Squadra'] == df_home['Squadra'].iloc[0]].reset_index(drop=True)
        away_players = all_players_df[all_players_df['Squadra'] == df_away['Squadra'].iloc[0]].reset_index(drop=True)
        
        # 3. Identificazione Duelli Critici (Nuova Logica)
        critical_matchups = find_critical_matchups(home_players, away_players)
        
        # 4. Estrazione Risultati Top
        all_players_df.sort_values(by='Rischio_Finale', ascending=False, inplace=True)
        
        # NOTE: Il bilanciamento del pronostico (2-2, 3-1) è stato spostato in app.py
        # per consentire l'esclusione interattiva. Qui si restituisce la graduatoria completa.

        # 5. CALCOLO CARTELLINI ATTESI (Logica normalizzata)
        ref_name = ref_df['Nome'].iloc[0]
        ref_cards_per_game = ref_df['Gialli a partita'].iloc[0]
        ref_normalization_factor = ref_cards_per_game / self.SERIE_A_AVG_CARDS
        team_avg_risk = all_players_df['Rischio_Finale'].mean()
        
        if team_avg_risk > 0:
            team_risk_factor = team_avg_risk / self.RISK_BASELINE
        else:
            team_risk_factor = 1.0
            
        match_influence_factor = (
            (ref_normalization_factor * 0.60) + 
            (team_risk_factor * 0.40)
        )
        expected_cards = self.SERIE_A_AVG_CARDS * match_influence_factor
        expected_cards = min(expected_cards, 7.0)

        # 6. PREPARAZIONE OUTPUT
        if ref_cards_per_game > 5.0:
            severity = 'strict'
            factor = 1.2
        elif ref_cards_per_game < 3.5:
            severity = 'permissive'
            factor = 0.8
        else:
            severity = 'medium'
            factor = 1.0
            

        return {
            'all_predictions': all_players_df,
            # NOTA: top_4_predictions non viene estratto qui per permettere il bilanciamento interattivo in app.py
            'top_4_predictions': all_players_df.head(4).to_dict('records'), 
            'match_info': {
                'home_team': df_home['Squadra'].iloc[0],
                'away_team': df_away['Squadra'].iloc[0],
                'expected_total_cards': f"{expected_cards:.1f}", 
                'algorithm_confidence': 'High' if len(all_players_df) > 40 else 'Medium',
            },
            'referee_profile': {
                'name': ref_name,
                'cards_per_game': ref_cards_per_game,
                'strictness_factor': factor,
                'severity_level': severity,
            },
            'critical_matchups': critical_matchups,
            'algorithm_summary': {
                'methodology': 'Modello Avanzato con Matching Posizionale Realistico',
                'critical_matchups_found': len(critical_matchups),
                'high_risk_players': (all_players_df['Rischio_Finale'] > 0.4).sum(),
                'weights_used': self.weights
            }
        }