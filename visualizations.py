import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_prediction_charts(df):
    """Crea grafici per l'analisi delle predizioni"""
    charts = {}
    
    # 1. Distribuzione del rischio
    fig_dist = go.Figure()
    
    fig_dist.add_trace(go.Histogram(
        x=df['Rischio_Giallo'],
        name='Rischio Giallo',
        opacity=0.7,
        marker_color='#FFD700',
        nbinsx=20
    ))
    
    fig_dist.add_trace(go.Histogram(
        x=df['Rischio_Rosso'],
        name='Rischio Rosso',
        opacity=0.7,
        marker_color='#FF6B6B',
        nbinsx=20
    ))
    
    fig_dist.update_layout(
        title='📊 Distribuzione del Rischio Cartellini',
        xaxis_title='Percentuale di Rischio',
        yaxis_title='Numero di Giocatori',
        barmode='overlay',
        template='plotly_white',
        height=400
    )
    
    charts['risk_distribution'] = fig_dist
    
    # 2. Analisi per posizione
    position_stats = df.groupby('Posizione').agg({
        'Rischio_Giallo': 'mean',
        'Rischio_Rosso': 'mean',
        'Cartellini_Gialli': 'mean',
        'Falli_Commessi': 'mean'
    }).round(2)
    
    fig_pos = go.Figure()
    
    fig_pos.add_trace(go.Bar(
        name='Rischio Giallo Medio',
        x=position_stats.index,
        y=position_stats['Rischio_Giallo'],
        marker_color='#FFD700',
        yaxis='y'
    ))
    
    fig_pos.add_trace(go.Bar(
        name='Rischio Rosso Medio',
        x=position_stats.index,
        y=position_stats['Rischio_Rosso'],
        marker_color='#FF6B6B',
        yaxis='y2'
    ))
    
    fig_pos.update_layout(
        title='⚽ Rischio per Posizione',
        xaxis_title='Posizione',
        yaxis=dict(title='Rischio Giallo (%)', side='left'),
        yaxis2=dict(title='Rischio Rosso (%)', side='right', overlaying='y'),
        template='plotly_white',
        height=400
    )
    
    charts['position_analysis'] = fig_pos
    
    # 3. Confronto squadre
    team_stats = df.groupby('Squadra').agg({
        'Rischio_Giallo': 'mean',
        'Rischio_Rosso': 'mean'
    }).round(2).head(10)  # Top 10 squadre
    
    fig_team = go.Figure()
    
    fig_team.add_trace(go.Scatter(
        x=team_stats['Rischio_Giallo'],
        y=team_stats['Rischio_Rosso'],
        mode='markers+text',
        text=team_stats.index,
        textposition='top center',
        marker=dict(
            size=15,
            color=team_stats['Rischio_Giallo'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Rischio Giallo")
        )
    ))
    
    fig_team.update_layout(
        title='🏆 Confronto Rischio tra Squadre',
        xaxis_title='Rischio Giallo Medio (%)',
        yaxis_title='Rischio Rosso Medio (%)',
        template='plotly_white',
        height=400
    )
    
    charts['team_comparison'] = fig_team
    
    # 4. Matrice di correlazione
    corr_data = df[['Età', 'Cartellini_Gialli', 'Cartellini_Rossi', 
                   'Falli_Commessi', 'Rischio_Giallo', 'Rischio_Rosso']].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_data.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig_corr.update_layout(
        title='🔗 Matrice di Correlazione',
        template='plotly_white',
        height=400
    )
    
    charts['correlation_matrix'] = fig_corr
    
    return charts

def create_player_dashboard(player_data):
    """Crea dashboard per singolo giocatore"""
    # Radar chart delle statistiche del giocatore
    categories = ['Cartellini Gialli', 'Cartellini Rossi', 'Falli Commessi', 
                 'Rischio Giallo', 'Rischio Rosso']
    
    # Normalizza i valori per il radar chart
    values = [
        min(player_data['Cartellini_Gialli'] * 10, 100),  # Scala i cartellini
        min(player_data['Cartellini_Rossi'] * 20, 100),   # Scala i rossi
        min(player_data['Falli_Commessi'] * 2, 100),      # Scala i falli
        player_data['Rischio_Giallo'],
        player_data['Rischio_Rosso'] * 2  # Scala il rischio rosso
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=player_data['Nome'],
        line_color='#FF6B6B',
        fillcolor='rgba(255, 107, 107, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title=f"📊 Profilo Rischio - {player_data['Nome']}",
        template='plotly_white',
        height=400
    )
    
    return fig

def create_risk_gauge(risk_value, title, color_scheme='yellow'):
    """Crea un gauge per visualizzare il rischio"""
    colors = {
        'yellow': ['#90EE90', '#FFD700', '#FF6B6B'],  # Verde, Giallo, Rosso
        'red': ['#90EE90', '#FFA500', '#FF0000']      # Verde, Arancione, Rosso
    }
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': colors[color_scheme][1]},
            'steps': [
                {'range': [0, 30], 'color': colors[color_scheme][0]},
                {'range': [30, 70], 'color': colors[color_scheme][1]},
                {'range': [70, 100], 'color': colors[color_scheme][2]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_timeline_chart(df):
    """Crea un grafico temporale delle predizioni (simulato)"""
    # Simula dati temporali
    dates = pd.date_range('2024-01-01', periods=10, freq='W')
    
    # Calcola rischio medio per settimana (simulato)
    avg_yellow_risk = []
    avg_red_risk = []
    
    np.random.seed(42)
    base_yellow = df['Rischio_Giallo'].mean()
    base_red = df['Rischio_Rosso'].mean()
    
    for i in range(10):
        # Simula variazione settimanale
        yellow_var = base_yellow + np.random.normal(0, 5)
        red_var = base_red + np.random.normal(0, 2)
        
        avg_yellow_risk.append(max(0, min(100, yellow_var)))
        avg_red_risk.append(max(0, min(50, red_var)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=avg_yellow_risk,
        mode='lines+markers',
        name='Rischio Giallo',
        line=dict(color='#FFD700', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=avg_red_risk,
        mode='lines+markers',
        name='Rischio Rosso',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='📈 Trend Rischio Cartellini nel Tempo',
        xaxis_title='Data',
        yaxis_title='Rischio Medio (%)',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_comparison_chart(df, player1, player2):
    """Confronta due giocatori"""
    p1_data = df[df['Nome'] == player1].iloc[0]
    p2_data = df[df['Nome'] == player2].iloc[0]
    
    metrics = ['Cartellini_Gialli', 'Cartellini_Rossi', 'Falli_Commessi', 
              'Rischio_Giallo', 'Rischio_Rosso']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=player1,
        x=metrics,
        y=[p1_data[m] for m in metrics],
        marker_color='#4ECDC4'
    ))
    
    fig.add_trace(go.Bar(
        name=player2,
        x=metrics,
        y=[p2_data[m] for m in metrics],
        marker_color='#FF6B6B'
    ))
    
    fig.update_layout(
        title=f'⚖️ Confronto: {player1} vs {player2}',
        xaxis_title='Metriche',
        yaxis_title='Valori',
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    return fig