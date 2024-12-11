from flask import Flask, request, send_from_directory, send_file, abort, jsonify
import os
import pandas as pd
from datetime import datetime
import json
from user_agents import parse
import plotly.express as px
from pathlib import Path
import sqlite3
from collections import defaultdict
import hashlib

class PortfolioAnalytics:
    def __init__(self, db_path='analytics.db'):
        self.db_path = db_path
        self.initialize_database()
        
    def initialize_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS visits (
                    timestamp DATETIME,
                    path TEXT,
                    user_agent TEXT,
                    ip_hash TEXT,
                    referrer TEXT,
                    session_id TEXT,
                    interaction_type TEXT,
                    section_viewed TEXT,
                    time_spent INTEGER,
                    device_type TEXT,
                    country TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    timestamp DATETIME,
                    session_id TEXT,
                    interaction_type TEXT,
                    element_id TEXT,
                    additional_data TEXT
                )
            ''')
            
    def _get_device_type(self, user_agent):
        """
        Determina el tipo de dispositivo basado en el user agent.
        
        Args:
            user_agent: Objeto user_agents.parsers.UserAgent
            
        Returns:
            str: Tipo de dispositivo ('mobile', 'tablet', o 'desktop')
        """
        if user_agent.is_mobile:
            return 'mobile'
        elif user_agent.is_tablet:
            return 'tablet'
        else:
            return 'desktop'

    def log_visit(self, request_data):
        user_agent = parse(request_data.user_agent.string)
        ip_hash = hashlib.sha256(request_data.remote_addr.encode()).hexdigest()
        
        visit_data = {
            'timestamp': datetime.now(),
            'path': request_data.path,
            'user_agent': str(user_agent),
            'ip_hash': ip_hash,
            'referrer': request_data.referrer or '',
            'session_id': request_data.cookies.get('session_id', ''),
            'device_type': self._get_device_type(user_agent),
            'country': request_data.headers.get('CF-IPCountry', 'Unknown')
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO visits 
                (timestamp, path, user_agent, ip_hash, referrer, session_id, device_type, country)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', tuple(visit_data.values()))

    def log_interaction(self, session_id, interaction_type, element_id, additional_data=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO interactions 
                (timestamp, session_id, interaction_type, element_id, additional_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now(), session_id, interaction_type, element_id, 
                 json.dumps(additional_data) if additional_data else None))

    def generate_analytics_report(self):
        with sqlite3.connect(self.db_path) as conn:
            df_visits = pd.read_sql_query("SELECT * FROM visits", conn)
            df_interactions = pd.read_sql_query("SELECT * FROM interactions", conn)

        report = {
            'general_stats': self._calculate_general_stats(df_visits),
            'traffic_patterns': self._analyze_traffic_patterns(df_visits),
            'engagement_metrics': self._calculate_engagement_metrics(df_visits, df_interactions),
            'geographical_distribution': self._analyze_geographical_distribution(df_visits),
            'technical_metrics': self._analyze_technical_metrics(df_visits)
        }
        
        self._generate_visualization_reports(report)
        return report

    def _calculate_general_stats(self, df):
        return {
            'total_visits': len(df),
            'unique_visitors': df['ip_hash'].nunique(),
            'average_time_spent': df['time_spent'].mean() if 'time_spent' in df else None,
            'bounce_rate': self._calculate_bounce_rate(df)
        }

    def _analyze_traffic_patterns(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        hourly_traffic = df.groupby(df['timestamp'].dt.hour).size()
        daily_traffic = df.groupby(df['timestamp'].dt.date).size()
        
        return {
            'peak_hours': hourly_traffic.idxmax(),
            'daily_average': daily_traffic.mean(),
            'busiest_day': daily_traffic.idxmax(),
            'traffic_growth': self._calculate_traffic_growth(daily_traffic)
        }

    def _generate_visualization_reports(self, report_data):
        output_dir = Path('analytics_reports')
        output_dir.mkdir(exist_ok=True)
        
        # Visualizaciones de tráfico
        traffic_fig = px.line(report_data['traffic_patterns']['daily_traffic'])
        traffic_fig.write_html(output_dir / 'traffic_patterns.html')
        
        # Mapa de calor de visitas
        if 'geographical_distribution' in report_data:
            geo_fig = px.choropleth(
                report_data['geographical_distribution'],
                locations='country',
                color='visits'
            )
            geo_fig.write_html(output_dir / 'geographical_distribution.html')

app = Flask(__name__, static_folder=None)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
analytics = PortfolioAnalytics()

@app.before_request
def log_request():
    if not request.path.startswith(('/static/', '/analytics/', '/favicon.ico')):
        analytics.log_visit(request)

@app.route('/analytics/report')
def get_analytics_report():
    report = analytics.generate_analytics_report()
    return jsonify(report)

@app.route('/analytics/interaction', methods=['POST'])
def log_interaction():
    data = request.get_json()
    analytics.log_interaction(
        session_id=request.cookies.get('session_id', ''),
        interaction_type=data.get('type'),
        element_id=data.get('element_id'),
        additional_data=data.get('data')
    )
    return jsonify({'status': 'success'})

# Función auxiliar para verificar si un archivo existe
def file_exists(path):
    return os.path.exists(path) and os.path.isfile(path)

@app.route('/')
def serve_index():
    index_path = os.path.join(BASE_DIR, 'src', 'index.html')
    if file_exists(index_path):
        return send_file(index_path)
    abort(404)

@app.route('/<path:filename>')
def serve_root_files(filename):
    # Primero buscar en src/
    src_path = os.path.join(BASE_DIR, 'src', filename)
    if file_exists(src_path):
        return send_file(src_path)
    
    # Si no está en src/, buscar en la raíz
    root_path = os.path.join(BASE_DIR, filename)
    if file_exists(root_path):
        return send_file(root_path)
    
    abort(404)

@app.route('/src/<path:filename>')
def serve_src_files(filename):
    return send_from_directory('src', filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    assets_path = os.path.join(BASE_DIR, 'assets', filename)
    if file_exists(assets_path):
        return send_file(assets_path)
    abort(404)

@app.route('/css/<path:filename>')
def serve_css(filename):
    css_path = os.path.join(BASE_DIR, 'src', 'css', filename)
    if file_exists(css_path):
        return send_file(css_path)
    abort(404)

@app.route('/js/<path:filename>')
def serve_js(filename):
    js_path = os.path.join(BASE_DIR, 'src', 'js', filename)
    if file_exists(js_path):
        return send_file(js_path)
    abort(404)

@app.after_request
def add_header(response):
    # Prevenir caché durante desarrollo
    response.headers['Cache-Control'] = 'no-store'
    # Configurar CORS para desarrollo local
    response.headers['Access-Control-Allow-Origin'] = '*'
    # Configurar tipos MIME correctos
    if response.mimetype == 'text/html':
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
    elif response.mimetype == 'text/css':
        response.headers['Content-Type'] = 'text/css; charset=utf-8'
    elif response.mimetype == 'application/javascript':
        response.headers['Content-Type'] = 'application/javascript; charset=utf-8'
    return response

@app.errorhandler(404)
def not_found(e):
    return f"Archivo no encontrado: {request.path}", 404

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True, use_reloader=True)