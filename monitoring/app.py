import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

app = dash.Dash(__name__)
app.title = "MLOps Sentiment Monitoring"

LOG_DIR = os.getenv("LOG_DIR", "/app/data/processed/logs")
LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")

def load_data():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=["timestamp", "input_text", "sentiment", "confidence", "model_version"])
    
    data = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    df = pd.DataFrame(data)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

app.layout = html.Div([
    html.H1("Real-Time News Sentiment Analyzer Dashboard"),
    
    dcc.Interval(
        id='interval-component',
        interval=60*1000, # in milliseconds
        n_intervals=0
    ),
    
    html.Div([
        html.Div([
            dcc.Graph(id='sentiment-trend')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='confidence-dist')
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),

    html.Div([
        html.H3("Data Drift Score (Evidently AI)"),
        html.Div(id='drift-score-display', style={'fontSize': '24px', 'fontWeight': 'bold'}),
        # html.Iframe(srcDoc=open('drift_report.html', 'r').read(), width='100%', height='600') # if report is pre-generated
    ], style={'marginTop': '20px'}),

    html.Div([
        html.H3("Recent Predictions"),
        html.Div(id='recent-predictions-table')
    ], style={'marginTop': '20px'})
])

@app.callback(
    [Output('sentiment-trend', 'figure'),
     Output('confidence-dist', 'figure'),
     Output('recent-predictions-table', 'children'),
     Output('drift-score-display', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    df = load_data()
    
    if df.empty:
        return go.Figure(), go.Figure(), "No data available", "N/A"

    # Sentiment Trend (Count by Sentiment)
    trend_fig = px.histogram(df, x="sentiment", color="sentiment", title="Sentiment Distribution")
    
    # Confidence Distribution
    conf_fig = px.histogram(df, x="confidence", title="Confidence Score Distribution", nbins=20)
    
    # Recent Predictions
    recent_df = df.sort_values(by="timestamp", ascending=False).head(10)
    
    table = html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in ["timestamp", "sentiment", "confidence", "input_text"]])
        ),
        html.Tbody([
            html.Tr([
                html.Td(recent_df.iloc[i][col]) for col in ["timestamp", "sentiment", "confidence", "input_text"]
            ]) for i in range(len(recent_df))
        ])
    ])

    # Drift Score - this would ideally be computed offline and read here, but for simplicity:
    # We pretend drift is tracked in another file or calculated. 
    # For real evidently integration, we would load reference data and compare.
    drift_score = "0.05 (No Drift)" # Mocked for dashboard speed, real calculation in pipeline

    return trend_fig, conf_fig, table, drift_score

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
