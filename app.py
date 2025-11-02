# =========================================================================
# FUNZIONI GESTIONE TITOLARI (FASE 1) - LOGICA OTTIMIZZATA PER TREND STAGIONALI
# =========================================================================
# NOTA: Assicurati che numpy sia importato all'inizio del file: import numpy as np

def get_fouls_suffered_metric(df):
    """
    Estrae le metriche dei falli subiti (Totale e Stagionale) 
    e calcola la metrica 'Falli_Subiti_Used' con priorità al Totale,
    ma identifica anche il TREND stagionale (Spread).
    """
    df = df.copy()
    
    # Assicura le colonne per il calcolo
    total_col = 'Media Falli Subiti 90s Totale'
    seasonal_col = 'Media Falli Subiti 90s Stagionale'
    
    has_total = total_col in df.columns
    has_seasonal = seasonal_col in df.columns
    
    # Inizializza le colonne
    df['Falli_Subiti_Totale'] = pd.to_numeric(df.get(total_col, 0.0), errors='coerce').fillna(0.0)
    df['Falli_Subiti_Stagionale'] = pd.to_numeric(df.get(seasonal_col, 0.0), errors='coerce').fillna(0.0)
    df['90s Giocati Totali'] = pd.to_numeric(df.get('90s Giocati Totali', 0.0), errors='coerce').fillna(0.0)
    
    # Logica base per 'Falli_Subiti_Used' (priorità a Totale)
    if has_total:
        df['Falli_Subiti_Used'] = df['Falli_Subiti_Totale']
        if has_seasonal:
            # Se la media totale è zero o nulla, usa la stagionale
            mask_zero = (df['Falli_Subiti_Used'] == 0) | df['Falli_Subiti_Used'].isna()
            df.loc[mask_zero, 'Falli_Subiti_Used'] = df.loc[mask_zero, 'Falli_Subiti_Stagionale']
            df['Falli_Subiti_Source'] = 'Totale'
            df.loc[mask_zero, 'Falli_Subiti_Source'] = 'Stagionale'
        else:
            df['Falli_Subiti_Source'] = 'Totale'
    elif has_seasonal:
        df['Falli_Subiti_Used'] = df['Falli_Subiti_Stagionale']
        df['Falli_Subiti_Source'] = 'Stagionale'
    else:
        df['Falli_Subiti_Used'] = 0.0
        df['Falli_Subiti_Source'] = 'N/A'
    
    df['Falli_Subiti_Used'] = df['Falli_Subiti_Used'].fillna(0.0)
    
    return df


def identify_high_risk_victims(home_df, away_df):
    """
    Identifica i giocatori che SUBISCONO molti falli con logica a 3 livelli:
    
    1. STANDARD: Falli_Subiti_Used ≥ 1.5 (soglia classica)
    2. TREND STAGIONALE: Incremento stagionale ≥ 0.5 E valore stagionale ≥ 1.5
       → Rileva giocatori come Bonny (basso storico, alto quest'anno)
    3. ASSOLUTO: Falli_Subiti_Used ≥ 2.0 (override per valori altissimi)
    """
    all_victims = []
    
    for df, team_type in [(home_df, 'Casa'), (away_df, 'Trasferta')]:
        df = get_fouls_suffered_metric(df)
        
        # Filtra giocatori con dati significativi
        df_valid = df[(df['Falli_Subiti_Used'] > 0) & (df['90s Giocati Totali'] >= 1)].copy()
        
        if df_valid.empty:
            continue
        
        # ===================================================================
        # CALCOLO SPREAD STAGIONALE (la chiave per identificare trend)
        # ===================================================================
        df_valid['Stagional_Spread'] = 0.0
        mask_both_valid = (df_valid['Falli_Subiti_Stagionale'] > 0) & (df_valid['Falli_Subiti_Totale'] > 0)
        df_valid.loc[mask_both_valid, 'Stagional_Spread'] = (
            df_valid.loc[mask_both_valid, 'Falli_Subiti_Stagionale'] - 
            df_valid.loc[mask_both_valid, 'Falli_Subiti_Totale']
        )
        
        # ===================================================================
        # SOGLIE DI RILEVAZIONE
        # ===================================================================
        SPREAD_THRESHOLD_HIGH = 0.5      # Incremento stagionale significativo
        MIN_STANDARD_FOULS = 1.5         # Soglia classica
        MIN_ABSOLUTE_FOULS = 2.0         # Override per valori altissimi
        MIN_SEASONAL_FOULS = 1.5         # Valore minimo stagionale per trend
        MIN_90S = 2.0                    # Minimo minutaggio per affidabilità
        
        # ===================================================================
        # LIVELLO 1: RILEVAZIONE STANDARD (Soglia Classica)
        # ===================================================================
        victims_standard = df_valid[
            df_valid['Falli_Subiti_Used'] >= MIN_STANDARD_FOULS
        ].copy()
        
        # ===================================================================
        # LIVELLO 2: RILEVAZIONE TREND STAGIONALE (🔥 Hot Form)
        # ===================================================================
        # Identifica giocatori con FORTE incremento stagionale
        # Esempio: Bonny con 0.8 falli/90 totali ma 2.2 stagionali
        victims_seasonal_trend = df_valid[
            (df_valid['Stagional_Spread'] >= SPREAD_THRESHOLD_HIGH) &
            (df_valid['Falli_Subiti_Stagionale'] >= MIN_SEASONAL_FOULS) &
            (df_valid['90s Giocati Totali'] >= MIN_90S)
        ].copy()
        
        # ===================================================================
        # LIVELLO 3: RILEVAZIONE ASSOLUTA (Override per valori estremi)
        # ===================================================================
        victims_absolute = df_valid[
            df_valid['Falli_Subiti_Used'] >= MIN_ABSOLUTE_FOULS
        ].copy()
        
        # ===================================================================
        # COMBINAZIONE RISULTATI (Rimuove duplicati mantenendo priorità)
        # ===================================================================
        all_victims_df = pd.concat([
            victims_absolute,      # Priorità massima
            victims_seasonal_trend, # Priorità alta (trend)
            victims_standard       # Priorità standard
        ], ignore_index=False).drop_duplicates(subset=['Player'], keep='first')
        
        # ===================================================================
        # ELABORAZIONE FINALE E ETICHETTATURA
        # ===================================================================
        for _, player in all_victims_df.iterrows():
            
            # Determina la categoria di rischio
            if player['Falli_Subiti_Used'] >= MIN_ABSOLUTE_FOULS:
                risk_label = "🔥 Assoluto"
                risk_score = player['Falli_Subiti_Used']
            
            elif player['Stagional_Spread'] >= SPREAD_THRESHOLD_HIGH:
                # CASO BONNY: Incremento stagionale marcato
                risk_label = "🔥 Hot Form"
                # Usa il valore stagionale per il ranking (più rappresentativo)
                risk_score = player['Falli_Subiti_Stagionale']
            
            elif player['Falli_Subiti_Used'] >= MIN_STANDARD_FOULS:
                risk_label = "🔴 Alto"
                risk_score = player['Falli_Subiti_Used']
            
            else:
                risk_label = "🟡 Standard"
                risk_score = player['Falli_Subiti_Used']
            
            all_victims.append({
                'Player': player['Player'],
                'Squadra': player['Squadra'],
                'Team_Type': team_type,
                'Falli_Subiti_90': player['Falli_Subiti_Used'],
                'Falli_Subiti_Stagionale': player['Falli_Subiti_Stagionale'],
                'Falli_Subiti_Totale': player['Falli_Subiti_Totale'],
                'Stagional_Spread': player['Stagional_Spread'],
                'Falli_Source': player['Falli_Subiti_Source'],
                'Posizione': player.get('Posizione_Primaria', 'N/A'),
                'Ruolo': player.get('Ruolo', 'N/A'),
                'Ranking_Metric': risk_score,  # Usa la metrica più appropriata per il ranking
                'Risk_Label': risk_label
            })
        
    # ===================================================================
    # ORDINAMENTO FINALE
    # ===================================================================
    # Ordina per Ranking_Metric (che può essere Totale o Stagionale a seconda del caso)
    all_victims.sort(key=lambda x: x['Ranking_Metric'], reverse=True)
    
    return all_victims


def display_starter_verification(high_risk_victims):
    """
    Mostra interfaccia verifica titolarità per FASE 1.
    Versione aggiornata con indicatori del trend stagionale.
    """
    st.markdown("---")
    st.markdown("### 🔍 FASE 1: Verifica Titolari")
    st.markdown("""
    <div class='verification-box'>
        <h4>⚠️ Giocatori che Subiscono Molti Falli</h4>
        <p><strong>👉 Seleziona i giocatori NON TITOLARI</strong> per escluderli dall'analisi.</p>
        <p style='font-size: 0.9em; color: #666;'>
            🔥 <strong>Hot Form</strong> = Incremento stagionale significativo (es. Bonny)<br>
            🔥 <strong>Assoluto</strong> = Valore altissimo (≥2.0 falli/90s)<br>
            🔴 <strong>Alto</strong> = Valore elevato costante (≥1.5 falli/90s)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    excluded = []
    if not high_risk_victims:
        st.info("✅ Nessun giocatore ad alto rischio identificato.")
        return excluded
    
    home_victims = [p for p in high_risk_victims if p['Team_Type'] == 'Casa']
    away_victims = [p for p in high_risk_victims if p['Team_Type'] == 'Trasferta']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if home_victims:
            st.markdown(f"#### 🏠 {home_victims[0]['Squadra']}")
            for player in home_victims:
                # Emoji basata sulla categoria
                risk_emoji = "🔥" if "🔥" in player['Risk_Label'] else ("🔴" if "Alto" in player['Risk_Label'] else "🟡")
                
                # Costruisci label informativa
                if "Hot Form" in player['Risk_Label']:
                    detail = f"({player['Falli_Subiti_Stagionale']:.1f} stagionale ↑ da {player['Falli_Subiti_Totale']:.1f})"
                else:
                    detail = f"({player['Falli_Subiti_90']:.1f} falli/90s)"
                
                is_excluded = st.checkbox(
                    f"{risk_emoji} **{player['Player']}** - {player['Ruolo']} {detail}",
                    key=f"pre_home_{player['Player']}"
                )
                if is_excluded:
                    excluded.append(player['Player'])
    
    with col2:
        if away_victims:
            st.markdown(f"#### ✈️ {away_victims[0]['Squadra']}")
            for player in away_victims:
                # Emoji basata sulla categoria
                risk_emoji = "🔥" if "🔥" in player['Risk_Label'] else ("🔴" if "Alto" in player['Risk_Label'] else "🟡")
                
                # Costruisci label informativa
                if "Hot Form" in player['Risk_Label']:
                    detail = f"({player['Falli_Subiti_Stagionale']:.1f} stagionale ↑ da {player['Falli_Subiti_Totale']:.1f})"
                else:
                    detail = f"({player['Falli_Subiti_90']:.1f} falli/90s)"
                
                is_excluded = st.checkbox(
                    f"{risk_emoji} **{player['Player']}** - {player['Ruolo']} {detail}",
                    key=f"pre_away_{player['Player']}"
                )
                if is_excluded:
                    excluded.append(player['Player'])
    
    st.markdown("---")
    
    if excluded:
        st.warning(f"⚠️ **{len(excluded)} giocatori esclusi:** {', '.join(excluded)}")
        st.session_state['excluded_pre'] = excluded
    else:
        st.success("✅ Nessuna esclusione (FASE 1).")
        st.session_state['excluded_pre'] = []
    
    return excluded