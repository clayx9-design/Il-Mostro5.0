# ... (import e costanti invariati)

class OptimizedCardPredictionModel:
    # ... (init e metodi invariati fino a predict_match_cards)

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

        # ... (resto invariato: concat, calcoli rischi, bilanciamento, output)
        
        # Nel return, aggiungi info sul filtro
        return {
            # ... (invariato)
            'algorithm_summary': {
                'methodology': 'Modello Ottimizzato - Filtro 5 Partite + Rischio Falli Minimo 1.5',
                'weights_used': WEIGHTS,
                'min_games_filter_applied': 5,
                'aggression_factor_min': 1.5,
                'players_after_filter': {'home': len(home_df), 'away': len(away_df)}
            }
        }

# ... (resto invariato)