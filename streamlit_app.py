import streamlit as st
import pandas as pd

# Funzione per caricare dati (da Excel)
@st.cache_data
def load_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        arbitri_data = pd.read_excel(xls, sheet_name=sheet_names[-1], header=0)  # Ultima sheet: arbitri
        teams_data = {}
        for name in sheet_names[:-1]:
            df = pd.read_excel(xls, sheet_name=name, header=0)
            teams_data[name] = df
        return teams_data, arbitri_data
    except Exception as e:
        st.error(f"Errore caricamento: {e}")
        return {}, None

# Funzione calculate_risk (stessa del tuo script)
def calculate_risk(player_row, is_home=True, ref_yellow_per_match=3.0):
    gialli_totali = pd.to_numeric(player_row.get('Cartellini Gialli Totali', 0), errors='coerce') or 0
    nineties_total = pd.to_numeric(player_row.get('90s Giocati Totali', 1), errors='coerce') or 1
    gialli_25_26 = pd.to_numeric(player_row.get('Cartellini Gialli 25/26', 0), errors='coerce') or 0
    nineties_25_26 = pd.to_numeric(player_row.get('90s Giocati 25/26', 1), errors='coerce') or 1
    rossi = pd.to_numeric(player_row.get('Cartellini Rossi Totali', 0), errors='coerce') or 0
    falli = pd.to_numeric(player_row.get('Falli Fatti Totali', 0), errors='coerce') or 0
    pos = str(player_row.get('Pos', '')).upper()

    if nineties_total < 5.56 or 'GK' in pos:
        return 0

    base_risk = (gialli_totali / nineties_total) if nineties_total > 0 else 0
    trend_risk = (gialli_25_26 / nineties_25_26) if nineties_25_26 > 0 else 0
    risk = base_risk * 0.7 + trend_risk * 0.3
    risk += rossi * 0.5
    risk *= (1 + (falli / nineties_total) * 0.1)

    pos_factor = 1.0
    if 'DF' in pos:
        pos_factor = 1.25
    elif 'MF' in pos:
        pos_factor = 1.15

    home_factor = 1.2 if is_home else 0.8
    ref_factor = ref_yellow_per_match / 3.0
    return risk * pos_factor * home_factor * ref_factor

# Interfaccia Web
st.title("Pronostici Cartellini - Il Mostro 5.0")

# Carica Excel (opzionale: per demo, usa path locale; per deploy, carica su GitHub)
uploaded_file = st.file_uploader("Carica Il Mostro 5.0.xlsx", type="xlsx")
if uploaded_file is not None:
    # Salva temporaneamente per lettura
    with open("temp.xlsx", "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_path = "temp.xlsx"
else:
    file_path = "Il Mostro 5.0.xlsx"  # Assumi locale per test

teams_data, arbitri_data = load_data(file_path)

if arbitri_data is not None and not arbitri_data.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        home = st.selectbox("Squadra Casa:", list(teams_data.keys()))
    with col2:
        away = st.selectbox("Squadra Trasferta:", list(teams_data.keys()))
    with col3:
        ref_name = st.selectbox("Arbitro:", arbitri_data['Referee_Name'].tolist())

    if st.button("Elabora Pronostico") and home != away:
        ref_row = arbitri_data[arbitri_data['Referee_Name'] == ref_name]
        ref_yellow = pd.to_numeric(ref_row['Yellow_per_Match'].iloc[0], errors='coerce') or 3.0

        home_df = teams_data[home].copy()
        away_df = teams_data[away].copy()

        home_df['Risk'] = home_df.apply(lambda row: calculate_risk(row, True, ref_yellow), axis=1)
        away_df['Risk'] = away_df.apply(lambda row: calculate_risk(row, False, ref_yellow), axis=1)

        # Top 5
        top_home = home_df[(home_df['Pos'].isin(['DF', 'MF', 'FW'])) & (home_df['Risk'] > 0)].nlargest(5, 'Risk')[['Player', 'Risk']].round(2)
        top_away = away_df[(away_df['Pos'].isin(['DF', 'MF', 'FW'])) & (away_df['Risk'] > 0)].nlargest(5, 'Risk')[['Player', 'Risk']].round(2)

        st.subheader(f"Top 5 {home} (Casa):")
        st.dataframe(top_home)

        st.subheader(f"Top 5 {away} (Trasferta):")
        st.dataframe(top_away)

        # 4 Giocatori
        all_risks = pd.concat([home_df[['Player', 'Risk']].assign(Team=home), away_df[['Player', 'Risk']].assign(Team=away)])
        top_all = all_risks.nlargest(10, 'Risk')
        home_cands = top_all[top_all['Team'] == home].head(3)
        away_cands = top_all[top_all['Team'] == away].head(3)

        if not home_cands.empty and not away_cands.empty:
            home_mean = home_cands['Risk'].mean()
            away_mean = away_cands['Risk'].mean()
            diff = (home_mean - away_mean) / away_mean if away_mean > 0 else 0
            if abs(diff) > 0.3:
                if diff > 0:
                    selected = pd.concat([home_cands.head(3), away_cands.head(1)])
                else:
                    selected = pd.concat([home_cands.head(1), away_cands.head(3)])
            else:
                selected = pd.concat([home_cands.head(2), away_cands.head(2)])
            selected = selected.nlargest(4, 'Risk')[['Player', 'Risk', 'Team']].round(2)
            st.subheader("4 Giocatori Più Probabili (Bilanciati):")
            st.dataframe(selected)
else:
    st.warning("Carica il file Excel per procedere.")