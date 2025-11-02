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
    'foul_aggression': 0.25,          # Falli commessi
    'critical_matchup': 0.20,         # Duello critico identificato
    'referee_factor': 0.12,           # Severità arbitro
    'positional_risk': 0.08           # Rischio posizionale
}

# TEAM_WEIGHT_FACTOR rimosso, useremo la logica forzata

# =========================================================================
# FUNZIONI DI SUPPORTO
# =========================================================================

def get_player_zone(position: str, heatmap: str) -> str:
    """
    Determina la zona del giocatore: L (sinistra), R (destra), C (centro).
    Semplificata per essere più robusta.
    """
    if pd.isna(position):
        position = ''
    if pd.isna(heatmap):
        heatmap = ''
    
    pos = str(position).upper()
    heat = str(heatmap).lower()
    
    # Priorità alla posizione
    if any(x in pos for x in ['LW', 'LB', 'LWB', 'LM']):
        return 'L'
    elif any(x in pos for x in ['RW', 'RB', 'RWB', 'RM']):
        return 'R'
    
    # Fallback alla heatmap
    if any(x in heat for x in ['left', 'sinistra']):
        return 'L'
    elif any(x in heat for x in ['right', 'destra']):
        return 'R'
    
    return 'C'


def get_player_role(position: str) -> str:
    """
    Classifica il ruolo: ATT, MID, DEF, GK.
    """
    if pd.isna(position):
        return 'MID'
    
    pos = str(position).upper()
    
    if any(x in pos for x in ['ST', 'CF', 'FW', 'W']):
        return 'ATT'
    elif any(x in pos for x in ['CB', 'LB', 'RB', 'WB', 'DF']):
        return 'DEF'
    elif any(x in pos for x in ['GK', 'P']):
        return 'GK'
    else:
        return 'MID'


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza e pulisce i dati in ingresso.
    Gestisce colonne mancanti e valori anomali.
    """
    df = df.copy()
    
    # Mappa colonne standard
    column_mapping = {
        'Giocatore': 'Player',
        'Nome': 'Player',
        'Pos': 'Posizione_Primaria'
    }
    
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    
    # Assicura colonne essenziali
    essential_cols = {
        'Player': lambda: df.iloc[:, 0].astype(str),
        'Squadra': lambda: 'Unknown',
        'Posizione_Primaria': lambda: 'MF',
        'Heatmap': lambda: 'Central activity',
        'Media Falli Fatti 90s Totale': lambda: 1.5,
        'Media Falli Subiti 90s Totale': lambda: 1.0,
        'Cartellini Gialli Totali': lambda: 2.0,
        'Media 90s per Cartellino Totale': lambda: 10.0,
        'Media Falli per Cartellino Totale': lambda: 5.0,
        'Ritardo Cartellino (Minuti)': lambda: 45.0,
        '90s Giocati Totali': lambda: 20.0,
        'Historical_Risk': lambda: 0.0 # Valore temporaneo/iniziale, verrà ricalcolato
    }
    
    for col, default_func in essential_cols.items():
        if col not in df.columns:
            df[col] = default_func()
        else:
            # Converti a numerico se necessario
            if col not in ['Player', 'Squadra', 'Posizione_Primaria', 'Heatmap']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_func())
    
    # Calcola metriche derivate
    df['Zona_Campo'] = df.apply(
        lambda row: get_player_zone(row['Posizione_Primaria'], row['Heatmap']), 
        axis=1
    )
    df['Ruolo'] = df['Posizione_Primaria'].apply(get_player_role)
    
    return df


# =========================================================================
# CLASSE PRINCIPALE DEL MODELLO
# =========================================================================

class OptimizedCardPredictionModel:
    """
    Modello ottimizzato per predizione cartellini.
    Approccio: semplicità + efficacia + realismo.
    """
    
    def __init__(self):
        self.weights = WEIGHTS
    
    
    def _calculate_historical_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola il rischio basato sullo storico individuale del giocatore.
        Questo è il fattore più importante (35% del peso).
        """
        df = df.copy()
        
        # 1. Frequenza cartellini (più bassa = più a rischio)
        df['Card_Frequency_Score'] = np.where(
            df['Media 90s per Cartellino Totale'] > 0,
            np.clip(5 / df['Media 90s per Cartellino Totale'], 0, 1),
            0
        )
        
        # 2. Impulsività (pochi falli per cartellino = impulsivo)
        df['Impulsivity_Score'] = np.where(
            df['Media Falli per Cartellino Totale'] > 0,
            np.clip(4 / df['Media Falli per Cartellino Totale'], 0, 1),
            0
        )
        
        # 3. Velocità cartellino (ritardo basso = cartellini rapidi)
        df['Speed_Score'] = np.clip(1 - (df['Ritardo Cartellino (Minuti)'] / 90), 0, 1)
        
        # 4. Volume assoluto cartellini (più cartellini = più tendenza)
        valid_cards = df[df['90s Giocati Totali'] > 0]['Cartellini Gialli Totali']
        avg_cards = valid_cards.mean() if not valid_cards.empty else 2.0
        
        df['Volume_Score'] = np.clip(df['Cartellini Gialli Totali'] / (avg_cards * 2), 0, 1)
        
        # Combina i 4 fattori
        df['Historical_Risk'] = (
            df['Card_Frequency_Score'] * 0.35 +
            df['Impulsivity_Score'] * 0.30 +
            df['Speed_Score'] * 0.20 +
            df['Volume_Score'] * 0.15
        )
        
        return df
    
    
    def _calculate_foul_aggression(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola aggressività basata sui falli commessi (25% del peso).
        """
        df = df.copy()
        
        # Media falli normalizzata (> 2.5 falli/90 = molto aggressivo)
        df['Foul_Aggression'] = np.clip(
            df['Media Falli Fatti 90s Totale'] / 2.5,
            0,
            1
        )
        
        return df
    
    
    def _calculate_positional_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rischio posizionale base (8% del peso).
        Difensori e centrocampisti = più a rischio.
        """
        df = df.copy()
        
        risk_map = {
            'DEF': 0.85,  # Difensori: alto rischio
            'MID': 0.70,  # Centrocampisti: rischio medio-alto
            'ATT': 0.50,  # Attaccanti: rischio medio
            'GK': 0.10   # Portieri: basso rischio
        }
        
        df['Positional_Risk'] = df['Ruolo'].map(risk_map).fillna(0.5)
        
        return df
    
    
    def _identify_critical_matchups(
        self, 
        home_df: pd.DataFrame, 
        away_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Identifica i duelli critici TRA SQUADRE AVVERSARIE con logica a specchio.
        """
        matchups = []
        
        # === DUELLI: Attaccanti HOME vs Difensori/Centrocampisti AWAY ===
        victims_home = home_df[
            (home_df['Ruolo'].isin(['ATT', 'MID'])) &
            (home_df['Media Falli Subiti 90s Totale'] >= 1.0)
        ].nlargest(8, 'Media Falli Subiti 90s Totale')
        
        aggressors_away = away_df[
            (away_df['Ruolo'].isin(['DEF', 'MID'])) &
            (away_df['Media Falli Fatti 90s Totale'] >= 1.5)
        ].nlargest(8, 'Media Falli Fatti 90s Totale')
        
        matchups.extend(
            self._match_players(victims_home, aggressors_away)
        )
        
        # === DUELLI: Attaccanti AWAY vs Difensori/Centrocampisti HOME ===
        victims_away = away_df[
            (away_df['Ruolo'].isin(['ATT', 'MID'])) &
            (away_df['Media Falli Subiti 90s Totale'] >= 1.0)
        ].nlargest(8, 'Media Falli Subiti 90s Totale')
        
        aggressors_home = home_df[
            (home_df['Ruolo'].isin(['DEF', 'MID'])) &
            (home_df['Media Falli Fatti 90s Totale'] >= 1.5)
        ].nlargest(8, 'Media Falli Fatti 90s Totale')
        
        matchups.extend(
            self._match_players(victims_away, aggressors_home)
        )
        
        matchups.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return matchups[:10]
    
    
    def _match_players(
        self, 
        victims: pd.DataFrame, 
        aggressors: pd.DataFrame
    ) -> List[Dict]:
        """
        Crea matchup tra vittime e aggressori di SQUADRE DIVERSE.
        Logica a specchio: priorità a laterali opposti (L vs R).
        """
        matchups = []
        used_pairs = set()  # Evita duplicati
        
        for _, victim in victims.iterrows():
            victim_zone = victim['Zona_Campo']
            victim_team = victim['Squadra']
            
            for _, aggressor in aggressors.iterrows():
                aggressor_zone = aggressor['Zona_Campo']
                aggressor_team = aggressor['Squadra']
                
                # VERIFICA CRITICA: Devono essere di squadre DIVERSE
                if victim_team == aggressor_team:
                    continue
                
                # Evita duplicati
                pair_key = tuple(sorted([victim['Player'], aggressor['Player']]))
                if pair_key in used_pairs:
                    continue
                
                # Calcola compatibilità zona (PRIORITÀ MASSIMA a L vs R)
                zone_compat = self._calculate_zone_compatibility(
                    victim_zone,
                    aggressor_zone
                )
                
                # Calcola compatibilità ruolo
                role_compat = self._calculate_role_compatibility(
                    victim['Ruolo'],
                    aggressor['Ruolo']
                )
                
                # Compatibilità totale: zona pesa 70%
                total_compat = (zone_compat * 0.70 + role_compat * 0.30)
                
                # Soglia minima per considerare il duello
                if total_compat < 0.5:
                    continue
                
                # Calcola rischio duello
                risk_score = (
                    victim['Media Falli Subiti 90s Totale'] * 0.35 +
                    aggressor['Media Falli Fatti 90s Totale'] * 0.35 +
                    aggressor['Historical_Risk'] * 0.30 
                ) * total_compat
                
                matchups.append({
                    'victim_player': victim['Player'],
                    'victim_team': victim_team,
                    'victim_zone': victim_zone,
                    'victim_role': victim['Ruolo'],
                    'aggressor_player': aggressor['Player'],
                    'aggressor_team': aggressor_team,
                    'aggressor_zone': aggressor_zone,
                    'aggressor_role': aggressor['Ruolo'],
                    'zone_compatibility': zone_compat,
                    'role_compatibility': role_compat,
                    'risk_score': risk_score
                })
                
                used_pairs.add(pair_key)
        
        return matchups
    
    
    def _calculate_zone_compatibility(self, zone1: str, zone2: str) -> float:
        """Calcola compatibilità tra zone."""
        if (zone1 == 'L' and zone2 == 'R') or (zone1 == 'R' and zone2 == 'L'):
            return 1.0
        if zone1 == 'C' or zone2 == 'C':
            return 0.7
        if zone1 == zone2:
            return 0.3
        return 0.5
    
    
    def _calculate_role_compatibility(self, role1: str, role2: str) -> float:
        """Calcola compatibilità tra ruoli."""
        if (role1 == 'ATT' and role2 == 'DEF') or (role1 == 'DEF' and role2 == 'ATT'):
            return 1.0
        if role1 == 'MID' or role2 == 'MID':
            return 0.8
        if role1 == 'DEF' and role2 == 'DEF':
            return 0.2
        return 0.6
    
    
    def _calculate_referee_factor(self, referee_df: pd.DataFrame) -> float:
        """Calcola il fattore arbitro."""
        if referee_df.empty:
            return 1.0
        ref_cards = referee_df['Gialli a partita'].iloc[0]
        factor = ref_cards / SERIE_A_AVG_CARDS_PER_GAME
        return np.clip(factor, 0.7, 1.4)

    
    def _calculate_team_aggression_factor(self, home_df: pd.DataFrame, away_df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calcola un fattore che indica quale squadra è più aggressiva in base a falli e cartellini.
        """
        # Filtra i titolari (o giocatori con dati validi)
        home_players = home_df[home_df['90s Giocati Totali'] >= 5]
        away_players = away_df[away_df['90s Giocati Totali'] >= 5]
        
        if home_players.empty or away_players.empty:
            return 0, 0

        # Metriche cumulative (su base 90s per normalizzare il volume di gioco)
        home_aggression_score = (
            home_players['Media Falli Fatti 90s Totale'].sum() * 0.5 +
            (home_players['Cartellini Gialli Totali'] / home_players['90s Giocati Totali']).replace([np.inf, -np.inf], 0).sum() * 0.5
        )
        
        away_aggression_score = (
            away_players['Media Falli Fatti 90s Totale'].sum() * 0.5 +
            (away_players['Cartellini Gialli Totali'] / away_players['90s Giocati Totali']).replace([np.inf, -np.inf], 0).sum() * 0.5
        )
        
        total_aggression = home_aggression_score + away_aggression_score
        
        if total_aggression == 0:
            return 0, 0
        
        # Differenza normalizzata, scalata tra -1 (Away domina) e +1 (Home domina)
        diff_factor = (home_aggression_score - away_aggression_score) / total_aggression
        
        return diff_factor, abs(diff_factor)

    
    def predict_match_cards(
        self,
        home_df: pd.DataFrame,
        away_df: pd.DataFrame,
        referee_df: pd.DataFrame
    ) -> Dict:
        """
        Esegue la predizione completa per una partita.
        Applica il bilanciamento forzato 2-2/3-1 in post-processing.
        """
        
        # 1. Normalizza e concatena i dati
        home_df = normalize_data(home_df)
        away_df = normalize_data(away_df)
        all_players = pd.concat([home_df, away_df], ignore_index=True)
        
        # 2. Calcola rischi base
        all_players = self._calculate_historical_risk(all_players)
        all_players = self._calculate_foul_aggression(all_players)
        all_players = self._calculate_positional_risk(all_players)
        
        # 3. Identifica duelli critici
        home_team_name = home_df['Squadra'].iloc[0]
        away_team_name = away_df['Squadra'].iloc[0]
        
        critical_matchups = self._identify_critical_matchups(
            all_players[all_players['Squadra'] == home_team_name], 
            all_players[all_players['Squadra'] == away_team_name]
        )

        # 3.5. Calcola il Fattore di Aggressività di Squadra
        aggression_diff_factor, aggression_abs_diff = self._calculate_team_aggression_factor(home_df, away_df)
        
        # 4. Assegna bonus per duelli critici
        matchup_bonus = {}
        for matchup in critical_matchups:
            aggressor = matchup['aggressor_player']
            victim = matchup['victim_player']
            matchup_bonus[aggressor] = max(matchup_bonus.get(aggressor, 0), matchup['risk_score'] * 0.5)
            matchup_bonus[victim] = max(matchup_bonus.get(victim, 0), matchup['risk_score'] * 0.3)
        
        all_players['Matchup_Bonus'] = all_players['Player'].map(matchup_bonus).fillna(0)
        if all_players['Matchup_Bonus'].max() > 0:
            all_players['Matchup_Bonus'] = (
                all_players['Matchup_Bonus'] / all_players['Matchup_Bonus'].max() * 0.4
            )
        
        # 5. Calcola fattore arbitro
        referee_factor = self._calculate_referee_factor(referee_df)
        
        # 6. CALCOLA RISCHIO FINALE (Rimosso Team_Risk_Factor)
        all_players['Rischio_Finale'] = (
            all_players['Historical_Risk'] * self.weights['historical_tendency'] +
            all_players['Foul_Aggression'] * self.weights['foul_aggression'] +
            all_players['Matchup_Bonus'] * self.weights['critical_matchup'] +
            all_players['Positional_Risk'] * self.weights['positional_risk']
        ) * (referee_factor ** self.weights['referee_factor'])
        
        all_players['Rischio_Finale'] = np.clip(all_players['Rischio_Finale'], 0.01, 0.95)
        
        # 7. Calcola quota stimata
        all_players['Quota_Stimata'] = np.where(
            all_players['Rischio_Finale'] > 0.01,
            np.clip((1 / all_players['Rischio_Finale']) * 0.45, 1.5, 20.0),
            20.0
        )
        
        # 8. Ordina la graduatoria completa (serve per lo scorrimento)
        all_players = all_players.sort_values('Rischio_Finale', ascending=False).reset_index(drop=True)
        
        # =================================================================
        # 9. BILANCIAMENTO FORZATO 2-2 / 3-1 (Post-Processing)
        #    La soglia del 3-1 è ora dinamica in base all'Aggressività di Squadra.
        # =================================================================
        
        
        # Separa i giocatori ordinati per squadra
        home_risks = all_players[all_players['Squadra'] == home_team_name]
        away_risks = all_players[all_players['Squadra'] == away_team_name]
        
        top_4_bilanciato = []
        
        # Caso 1: Verifica TOP 4 iniziale per vedere la distribuzione
        top_4_iniziale = all_players.head(4)
        count_home = (top_4_iniziale['Squadra'] == home_team_name).sum()
        count_away = (top_4_iniziale['Squadra'] == away_team_name).sum()

        # Soglia base fissa
        BASE_RISK_THRESHOLD = 0.40 
        
        # Modulazione della soglia basata sull'aggressività di squadra (max 0.15 di riduzione)
        # Se aggression_abs_diff è 0.5 (netta differenza), la soglia si riduce di 0.15*0.5=0.075
        RISK_DIFFERENCE_THRESHOLD = BASE_RISK_THRESHOLD - (aggression_abs_diff * 0.15) 

        # Determina la distribuzione forzata (2-2 è la preferita)
        if (count_home == 4 or count_away == 4) and len(home_risks) >= 2 and len(away_risks) >= 2:
            # Se troppo sbilanciato (4-0/0-4) -> FORZA 2-2
            top_4_bilanciato.extend(home_risks.head(2).to_dict('records'))
            top_4_bilanciato.extend(away_risks.head(2).to_dict('records'))
        
        elif (count_home == 3 and count_away == 1) or (count_home == 1 and count_away == 3):
            # Distribuzione 3-1/1-3: Accetta SOLO se la differenza di rischio è netta E supportata da Aggressività.
            
            dominant_risks = home_risks if count_home == 3 else away_risks
            minor_risks = away_risks if count_home == 3 else home_risks
            
            # 1. Verifica coerenza tra Aggressività e Dominanza
            is_home_dominant_in_risks = (count_home == 3)
            is_home_dominant_in_aggression = (aggression_diff_factor > 0.0)
            
            coherence = (is_home_dominant_in_risks == is_home_dominant_in_aggression)

            if len(minor_risks) < 2:
                # Caso limite: Accettiamo 3-1 se non c'è il 2° giocatore nell'altra squadra
                top_4_bilanciato = top_4_iniziale.to_dict('records')
            else:
                # Rischio del 3° giocatore dominante vs Rischio del 2° giocatore minoritario
                risk_dominant_3rd = dominant_risks.iloc[2]['Rischio_Finale']
                risk_minor_2nd = minor_risks.iloc[1]['Rischio_Finale']
                
                # Accetta 3-1 se:
                # A) La differenza di rischio supera la soglia DINAMICA
                # E B) C'è coerenza tra dominanza nel rischio e dominanza nell'aggressività (o l'aggressività è neutrale/non significativa)
                
                risk_diff_is_significant = (risk_dominant_3rd > (risk_minor_2nd + RISK_DIFFERENCE_THRESHOLD))
                
                # Se la squadra aggressiva è quella DOMINANTE NEL RISCHIO, accettiamo 3-1 più facilmente.
                # Se l'aggressività è opposta, la soglia rimane più alta (BASE_RISK_THRESHOLD).
                
                # Logica finale: Se il rischio è significativo O se la differenza di aggressività è molto alta
                if risk_diff_is_significant:
                     # Accetta 3-1
                     top_4_bilanciato = top_4_iniziale.to_dict('records')
                else:
                     # Forza il 2-2
                     top_4_bilanciato.extend(home_risks.head(2).to_dict('records'))
                     top_4_bilanciato.extend(away_risks.head(2).to_dict('records'))

        elif count_home == 2 and count_away == 2:
            # Mantiene il 2-2 se è già presente
            top_4_bilanciato = top_4_iniziale.to_dict('records')
        
        else:
            # Ogni altro caso (incluse le situazioni non 4-0/0-4) -> FORZA 2-2 (se possibile)
            top_4_bilanciato.extend(home_risks.head(min(2, len(home_risks))).to_dict('records'))
            top_4_bilanciato.extend(away_risks.head(min(2, len(away_risks))).to_dict('records'))
            # Filtra per assicurarsi che siano esattamente 4
            top_4_bilanciato = sorted(top_4_bilanciato, key=lambda x: x['Rischio_Finale'], reverse=True)[:4]


        # Riordina il TOP 4 bilanciato in base al Rischio_Finale (per la visualizzazione)
        top_4_final_df = pd.DataFrame(top_4_bilanciato).sort_values(
            'Rischio_Finale', ascending=False
        ).to_dict('records')
        
        # 10. Info arbitro
        ref_name = referee_df['Nome'].iloc[0] if not referee_df.empty else 'N/A'
        ref_cards = referee_df['Gialli a partita'].iloc[0] if not referee_df.empty else SERIE_A_AVG_CARDS_PER_GAME
        
        if ref_cards > 5.0:
            severity = 'strict'
        elif ref_cards < 3.5:
            severity = 'permissive'
        else:
            severity = 'medium'
        
        # 11. Stima cartellini totali partita
        collective_risk_sum = all_players['Rischio_Finale'].sum()
        AVG_COLLECTIVE_RISK = 5.5
        crf_raw = collective_risk_sum / AVG_COLLECTIVE_RISK
        crf = np.clip(crf_raw, 0.9, 1.1) 
        expected_cards = np.clip(ref_cards * crf, 3.5, 5.5)
        
        # 12. Prepara output
        return {
            'all_predictions': all_players,
            'top_4_predictions': top_4_final_df, # Usa il TOP 4 bilanciato
            'match_info': {
                'home_team': home_df['Squadra'].iloc[0],
                'away_team': away_df['Squadra'].iloc[0],
                'expected_total_cards': f"{expected_cards:.1f}",
                'algorithm_confidence': 'High' if len(critical_matchups) >= 3 else 'Medium'
            },
            'referee_profile': {
                'name': ref_name,
                'cards_per_game': ref_cards,
                'strictness_factor': referee_factor,
                'severity_level': severity
            },
            'critical_matchups': [
                {
                    'risk_score': m['risk_score'],
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
                'aggression_diff_factor': aggression_diff_factor,
                'dynamic_risk_threshold_applied': RISK_DIFFERENCE_THRESHOLD
            }
        }


# =========================================================================
# FUNZIONI DI COMPATIBILITÀ
# =========================================================================

def get_field_zone(position, heatmap):
    """Wrapper per compatibilità con app.py"""
    return get_player_zone(position, heatmap)

def get_player_role_category(position):
    """Wrapper per compatibilità con app.py"""
    return get_player_role(position)

# Classe principale esportata
SuperAdvancedCardPredictionModel = OptimizedCardPredictionModel