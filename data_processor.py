import pandas as pd
import numpy as np
import streamlit as st

class DataProcessor:
    def __init__(self):
        self.required_columns = [
            'Nome', 'Squadra', 'Posizione', 'Età', 'Minuti_Giocati',
            'Cartellini_Gialli', 'Cartellini_Rossi', 'Falli_Commessi'
        ]
    
    def load_data(self, uploaded_file):
        """Carica e processa i dati dal file caricato"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Verifica colonne richieste
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                st.error(f"Colonne mancanti: {missing_cols}")
                return self.generate_sample_data()
            
            # Pulizia dati
            df = self._clean_data(df)
            return df
            
        except Exception as e:
            st.error(f"Errore nel caricamento del file: {e}")
            return self.generate_sample_data()
    
    def _clean_data(self, df):
        """Pulisce e valida i dati"""
        # Rimuovi righe con valori mancanti critici
        df = df.dropna(subset=['Nome', 'Squadra', 'Posizione'])
        
        # Riempi valori mancanti numerici con 0
        numeric_cols = ['Età', 'Minuti_Giocati', 'Cartellini_Gialli', 'Cartellini_Rossi', 'Falli_Commessi']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Validazione valori
        df['Età'] = df['Età'].clip(16, 45)
        df['Minuti_Giocati'] = df['Minuti_Giocati'].clip(0, 3500)
        df['Cartellini_Gialli'] = df['Cartellini_Gialli'].clip(0, 20)
        df['Cartellini_Rossi'] = df['Cartellini_Rossi'].clip(0, 5)
        df['Falli_Commessi'] = df['Falli_Commessi'].clip(0, 150)
        
        # Standardizza posizioni
        position_mapping = {
            'GK': 'Portiere', 'Goalkeeper': 'Portiere', 'Portiere': 'Portiere',
            'DEF': 'Difensore', 'Defender': 'Difensore', 'Difensore': 'Difensore',
            'MID': 'Centrocampista', 'Midfielder': 'Centrocampista', 'Centrocampista': 'Centrocampista',
            'FWD': 'Attaccante', 'Forward': 'Attaccante', 'Attaccante': 'Attaccante'
        }
        
        df['Posizione'] = df['Posizione'].map(position_mapping).fillna('Centrocampista')
        
        return df
    
    def generate_sample_data(self):
        """Genera dati di esempio per la dimostrazione"""
        np.random.seed(42)
        
        # Nomi di giocatori italiani
        nomi = [
            "Marco Rossi", "Luca Bianchi", "Andrea Verdi", "Matteo Neri", "Davide Bruno",
            "Francesco Romano", "Alessandro Greco", "Simone Ricci", "Fabio Marino", "Paolo Conti",
            "Giuseppe Ferrari", "Antonio Lombardi", "Stefano Gallo", "Roberto Fontana", "Michele Villa",
            "Daniele Caruso", "Emanuele Rizzo", "Vincenzo Barbieri", "Salvatore Ferrara", "Nicola Giordano",
            "Giovanni Martino", "Claudio Santoro", "Maurizio Leone", "Alberto Longo", "Raffaele Fiore",
            "Cristiano Pellegrino", "Domenico Mariani", "Sergio Parisi", "Massimo De Santis", "Gianluca Vitale"
        ]
        
        squadre = [
            "Juventus", "Inter", "Milan", "Napoli", "Roma", "Lazio", 
            "Atalanta", "Fiorentina", "Bologna", "Torino"
        ]
        
        posizioni = ["Portiere", "Difensore", "Centrocampista", "Attaccante"]
        
        n_players = len(nomi)
        
        data = {
            'Nome': nomi,
            'Squadra': np.random.choice(squadre, n_players),
            'Posizione': np.random.choice(posizioni, n_players, p=[0.1, 0.4, 0.35, 0.15]),
            'Età': np.random.randint(18, 38, n_players),
            'Minuti_Giocati': np.random.randint(500, 3200, n_players),
        }
        
        df = pd.DataFrame(data)
        
        # Genera statistiche realistiche basate sulla posizione
        cartellini_gialli = []
        cartellini_rossi = []
        falli_commessi = []
        
        for _, row in df.iterrows():
            # Fattori basati sulla posizione
            if row['Posizione'] == 'Portiere':
                base_yellow = np.random.poisson(1)
                base_red = np.random.poisson(0.1)
                base_fouls = np.random.poisson(8)
            elif row['Posizione'] == 'Difensore':
                base_yellow = np.random.poisson(6)
                base_red = np.random.poisson(0.3)
                base_fouls = np.random.poisson(45)
            elif row['Posizione'] == 'Centrocampista':
                base_yellow = np.random.poisson(5)
                base_red = np.random.poisson(0.2)
                base_fouls = np.random.poisson(35)
            else:  # Attaccante
                base_yellow = np.random.poisson(3)
                base_red = np.random.poisson(0.1)
                base_fouls = np.random.poisson(25)
            
            # Fattore età (giovani più impulsivi)
            age_factor = 1.3 if row['Età'] < 23 else 1.1 if row['Età'] > 32 else 1.0
            
            # Fattore minuti (più minuti = più opportunità per cartellini)
            minutes_factor = row['Minuti_Giocati'] / 2500
            
            final_yellow = int(base_yellow * age_factor * minutes_factor)
            final_red = int(base_red * age_factor * minutes_factor)
            final_fouls = int(base_fouls * age_factor * minutes_factor)
            
            cartellini_gialli.append(max(0, final_yellow))
            cartellini_rossi.append(max(0, final_red))
            falli_commessi.append(max(0, final_fouls))
        
        df['Cartellini_Gialli'] = cartellini_gialli
        df['Cartellini_Rossi'] = cartellini_rossi
        df['Falli_Commessi'] = falli_commessi
        
        return df
    
    def export_predictions(self, df, predictions):
        """Esporta le predizioni in formato CSV"""
        export_df = df.copy()
        export_df = pd.concat([export_df, predictions], axis=1)
        
        # Riordina colonne
        column_order = [
            'Nome', 'Squadra', 'Posizione', 'Età', 'Minuti_Giocati',
            'Cartellini_Gialli', 'Cartellini_Rossi', 'Falli_Commessi',
            'Rischio_Giallo', 'Rischio_Rosso'
        ]
        
        export_df = export_df[column_order]
        return export_df.to_csv(index=False)
    
    def get_data_summary(self, df):
        """Restituisce un riassunto dei dati"""
        summary = {
            'total_players': len(df),
            'teams': df['Squadra'].nunique(),
            'avg_age': df['Età'].mean(),
            'total_yellow_cards': df['Cartellini_Gialli'].sum(),
            'total_red_cards': df['Cartellini_Rossi'].sum(),
            'total_fouls': df['Falli_Commessi'].sum(),
            'position_distribution': df['Posizione'].value_counts().to_dict()
        }
        return summary