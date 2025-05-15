from flask import Flask, request, send_from_directory, send_file, abort, jsonify, render_template_string
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
import markdown
import glob
import re
from datetime import datetime

app = Flask(__name__, static_folder=None)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Blog posts directory
BLOG_DIR = os.path.join(BASE_DIR, 'blog_posts')

# Ensure blog directory exists
os.makedirs(BLOG_DIR, exist_ok=True)

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

# --- Blog functionality ---
def parse_markdown_metadata(content):
    """
    Parse markdown content to extract metadata and content.
    Metadata should be in YAML-like format at the top of the file:
    
    ---
    title: Post Title
    date: 2025-05-01
    description: Short description of the post
    tags: tag1, tag2, tag3
    ---
    
    Content starts here...
    """
    metadata = {
        'title': 'Untitled Post',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'description': '',
        'tags': []
    }
    
    # Check if content starts with metadata section
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    
    if match:
        metadata_text = match.group(1)
        content = content[match.end():]
        
        # Parse each metadata line
        for line in metadata_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'tags':
                    metadata[key] = [tag.strip() for tag in value.split(',')]
                else:
                    metadata[key] = value
    else:
        # If no metadata block, try to extract title from first heading
        title_match = re.match(r'^#\s+(.+)$', content.split('\n')[0])
        if title_match:
            metadata['title'] = title_match.group(1).strip()
    
    return metadata, content

def format_date(date_str):
    """Format date in a readable format"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%B %d, %Y')
    except ValueError:
        return date_str

def get_blog_posts():
    """Get all blog posts with metadata"""
    posts = []
    
    if not os.path.exists(BLOG_DIR):
        return posts
        
    for post_path in glob.glob(os.path.join(BLOG_DIR, '*.md')):
        filename = os.path.basename(post_path)
        slug = os.path.splitext(filename)[0]
        
        try:
            with open(post_path, 'r', encoding='utf-8') as f:
                content = f.read()
                metadata, post_content = parse_markdown_metadata(content)
                
                # If no date in metadata, use file creation time
                if metadata.get('date') == datetime.now().strftime('%Y-%m-%d'):
                    file_date = datetime.fromtimestamp(os.path.getctime(post_path))
                    metadata['date'] = file_date.strftime('%Y-%m-%d')
                
                posts.append({
                    'slug': slug,
                    'title': metadata.get('title', 'Untitled'),
                    'date': metadata.get('date'),
                    'formatted_date': format_date(metadata.get('date')),
                    'description': metadata.get('description', ''),
                    'tags': metadata.get('tags', []),
                    'content': post_content
                })
        except Exception as e:
            print(f"Error processing {post_path}: {str(e)}")
    
    # Sort posts by date (newest first)
    posts.sort(key=lambda x: x['date'], reverse=True)
    return posts

@app.route('/blog')
def blog_index():
    """Show list of blog posts"""
    posts = get_blog_posts()
    theme = request.cookies.get('portfolio-theme', 'light')
    
    # HTML template for the blog index page
    html_template = '''
    <!DOCTYPE html>
    <html lang="en" data-theme="{{ theme }}">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Blog | Juan Lara</title>
        
        <!-- Favicons -->
        <link rel="apple-touch-icon" sizes="180x180" href="/assets/icons/apple-touch-icon.png">
        <link rel="icon" type="image/png" sizes="32x32" href="/assets/icons/favicon-32x32.png">
        <link rel="icon" type="image/png" sizes="16x16" href="/assets/icons/favicon-16x16.png">
        <link rel="manifest" href="/assets/icons/site.webmanifest">
        
        <!-- Font Awesome para íconos -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        
        <!-- Hojas de estilo -->
        <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
        <link href="/styles.css" rel="stylesheet">
        
        <!-- Fuentes -->
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500&display=swap" rel="stylesheet">
    </head>
    <body>
        <!-- Header -->
        <header>
            <nav class="container">
                <div class="nav-content">
                    <h1>Juan Lara</h1>
                    
                    <button class="hamburger">
                        <span></span>
                        <span></span>
                        <span></span>
                    </button>
                
                    <nav class="nav-links">
                        <a href="/#about">About</a>
                        <a href="/#experience">Experience</a>
                        <a href="/#education">Education</a>
                        <a href="/#projects">Projects</a>
                        <a href="/blog" class="active">Blog</a>
                        <a href="/#contact">Contact</a>
                    </nav>
                </div>
            </nav>
        </header>

        <!-- Blog Header -->
        <div class="hero">
            <div class="container">
                <div class="hero-content">
                    <h1>My Blog</h1>
                    <p>Thoughts, insights, and technical deep-dives on AI, mathematics, and programming</p>
                </div>
            </div>
        </div>
        
        <!-- Blog List -->
        <section id="blog-posts">
            <div class="container">
                <div class="blog-list">
                    {% if posts %}
                        {% for post in posts %}
                        <div class="blog-item">
                            <h2><a href="/blog/{{ post.slug }}">{{ post.title }}</a></h2>
                            <div class="post-meta">
                                <span class="date">{{ post.formatted_date }}</span>
                                {% if post.tags %}
                                <span class="tags">
                                    {% for tag in post.tags %}
                                    <span class="tag">{{ tag }}</span>
                                    {% endfor %}
                                </span>
                                {% endif %}
                            </div>
                            {% if post.description %}
                            <p class="description">{{ post.description }}</p>
                            {% endif %}
                            <a href="/blog/{{ post.slug }}" class="read-more">Read more <i class="fas fa-arrow-right"></i></a>
                        </div>
                        {% endfor %}
                    {% else %}
                        <div class="empty-state">
                            <p>No blog posts found. Check back soon!</p>
                        </div>
                    {% endif %}
                </div>
            </div>
        </section>
        
        <!-- Footer -->
        <footer>
            <div class="container">
                <div class="footer-content">
                    <p class="copyright">© 2025 Juan Lara. All rights reserved.</p>
                </div>
            </div>
        </footer>
        
        <!-- Theme switcher -->
        <div class="theme-switcher">
            <button class="theme-toggle" aria-label="Toggle dark mode">
                <svg width="" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="5"></circle>
                    <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"></path>
                </svg>
            </button>
        </div>
        
        <!-- Scripts -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/feather-icons/4.28.0/feather.min.js"></script>
        <script>
            feather.replace();
            
            // Theme toggle functionality
            document.addEventListener('DOMContentLoaded', function() {
                const themeToggle = document.querySelector('.theme-toggle');
                
                if (themeToggle) {
                    themeToggle.addEventListener('click', function() {
                        const currentTheme = document.documentElement.getAttribute('data-theme');
                        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                        
                        document.documentElement.setAttribute('data-theme', newTheme);
                        localStorage.setItem('portfolio-theme', newTheme);
                        
                        // Update icon
                        const themeIcon = document.querySelector('.theme-toggle svg');
                        if (themeIcon) {
                            if (newTheme === 'dark') {
                                themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
                            } else {
                                themeIcon.innerHTML = '<circle cx="12" cy="12" r="5"></circle><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"></path>';
                            }
                        }
                    });
                    
                    // Set initial icon
                    const currentTheme = document.documentElement.getAttribute('data-theme');
                    const themeIcon = document.querySelector('.theme-toggle svg');
                    if (themeIcon && currentTheme === 'dark') {
                        themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
                    }
                }
                
                // Mobile menu
                const hamburger = document.querySelector('.hamburger');
                const navLinks = document.querySelector('.nav-links');
                
                if (hamburger && navLinks) {
                    hamburger.addEventListener('click', function() {
                        hamburger.classList.toggle('active');
                        navLinks.classList.toggle('active');
                        document.body.style.overflow = navLinks.classList.contains('active') ? 'hidden' : '';
                    });
                    
                    navLinks.querySelectorAll('a').forEach(link => {
                        link.addEventListener('click', function() {
                            if (navLinks.classList.contains('active')) {
                                hamburger.classList.remove('active');
                                navLinks.classList.remove('active');
                                document.body.style.overflow = '';
                            }
                        });
                    });
                }
            });
        </script>
    </body>
    </html>
    '''
    
    # Replace template variables
    return render_template_string(
        html_template,
        posts=posts,
        theme=theme
    )

@app.route('/blog/<slug>')
def blog_post(slug):
    """Display a single blog post"""
    posts = get_blog_posts()
    post = next((p for p in posts if p['slug'] == slug), None)
    theme = request.cookies.get('portfolio-theme', 'light')
    
    if not post:
        abort(404)
    
    # Convert Markdown to HTML with extras for code highlighting, tables, etc.
    post_html = markdown.markdown(
        post['content'], 
        extensions=[
            'markdown.extensions.extra',
            'markdown.extensions.codehilite',
            'markdown.extensions.toc'
        ]
    )
    
    # Get related posts based on tags
    related_posts = []
    if post['tags']:
        for p in posts:
            if p['slug'] != post['slug'] and any(tag in p['tags'] for tag in post['tags']):
                related_posts.append(p)
                if len(related_posts) >= 3:  # Limit to 3 related posts
                    break
    
    # Get next and previous posts
    post_index = next((i for i, p in enumerate(posts) if p['slug'] == slug), None)
    prev_post = posts[post_index + 1] if post_index < len(posts) - 1 else None
    next_post = posts[post_index - 1] if post_index > 0 else None
    
    # HTML template for the blog post page
    html_template = '''
    <!DOCTYPE html>
    <html lang="en" data-theme="{{ theme }}">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ post.title }} | Juan Lara</title>
        
        <!-- Favicons -->
        <link rel="apple-touch-icon" sizes="180x180" href="/assets/icons/apple-touch-icon.png">
        <link rel="icon" type="image/png" sizes="32x32" href="/assets/icons/favicon-32x32.png">
        <link rel="icon" type="image/png" sizes="16x16" href="/assets/icons/favicon-16x16.png">
        <link rel="manifest" href="/assets/icons/site.webmanifest">
        
        <!-- Font Awesome para íconos -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        
        <!-- Hojas de estilo -->
        <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
        <link href="/styles.css" rel="stylesheet">
        
        <!-- Code highlighting -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.0/styles/github.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.0/highlight.min.js"></script>
        
        <!-- Fuentes -->
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500&display=swap" rel="stylesheet">
    </head>
    <body>
        <!-- Header -->
        <header>
            <nav class="container">
                <div class="nav-content">
                    <h1>Juan Lara</h1>
                    
                    <button class="hamburger">
                        <span></span>
                        <span></span>
                        <span></span>
                    </button>
                
                    <nav class="nav-links">
                        <a href="/#about">About</a>
                        <a href="/#experience">Experience</a>
                        <a href="/#education">Education</a>
                        <a href="/#projects">Projects</a>
                        <a href="/blog" class="active">Blog</a>
                        <a href="/#contact">Contact</a>
                    </nav>
                </div>
            </nav>
        </header>
        
        <!-- Blog Post -->
        <section id="blog-post">
            <div class="container">
                <div class="blog-navigation">
                    <a href="/blog" class="back-link"><i class="fas fa-arrow-left"></i> Back to Blog</a>
                </div>
                
                <article class="blog-post">
                    <header class="post-header">
                        <h1>{{ post.title }}</h1>
                        <div class="post-meta">
                            <span class="date">{{ post.formatted_date }}</span>
                            {% if post.tags %}
                            <span class="tags">
                                {% for tag in post.tags %}
                                <span class="tag">{{ tag }}</span>
                                {% endfor %}
                            </span>
                            {% endif %}
                        </div>
                    </header>
                    
                    <div class="post-content">
                        {{ post_html|safe }}
                    </div>
                </article>
                
                <!-- Post Navigation -->
                <div class="post-navigation">
                    {% if prev_post %}
                    <a href="/blog/{{ prev_post.slug }}" class="prev-post">
                        <span class="nav-label"><i class="fas fa-arrow-left"></i> Previous</span>
                        <span class="post-title">{{ prev_post.title }}</span>
                    </a>
                    {% else %}
                    <div class="prev-post empty"></div>
                    {% endif %}
                    
                    {% if next_post %}
                    <a href="/blog/{{ next_post.slug }}" class="next-post">
                        <span class="nav-label">Next <i class="fas fa-arrow-right"></i></span>
                        <span class="post-title">{{ next_post.title }}</span>
                    </a>
                    {% else %}
                    <div class="next-post empty"></div>
                    {% endif %}
                </div>
                
                <!-- Related Posts -->
                {% if related_posts %}
                <div class="related-posts">
                    <h3>Related Posts</h3>
                    <div class="related-posts-grid">
                        {% for related in related_posts %}
                        <a href="/blog/{{ related.slug }}" class="related-post">
                            <h4>{{ related.title }}</h4>
                            <span class="date">{{ related.formatted_date }}</span>
                        </a>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </div>
        </section>
        
        <!-- Footer -->
        <footer>
            <div class="container">
                <div class="footer-content">
                    <p class="copyright">© 2025 Juan Lara. All rights reserved.</p>
                </div>
            </div>
        </footer>
        
        <!-- Theme switcher -->
        <div class="theme-switcher">
            <button class="theme-toggle" aria-label="Toggle dark mode">
                <svg width="" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="5"></circle>
                    <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"></path>
                </svg>
            </button>
        </div>
        
        <!-- Scripts -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/feather-icons/4.28.0/feather.min.js"></script>
        <script>
            feather.replace();
            
            // Code highlighting
            document.addEventListener('DOMContentLoaded', function() {
                document.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightBlock(block);
                });
                
                // Theme toggle functionality
                const themeToggle = document.querySelector('.theme-toggle');
                
                if (themeToggle) {
                    themeToggle.addEventListener('click', function() {
                        const currentTheme = document.documentElement.getAttribute('data-theme');
                        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                        
                        document.documentElement.setAttribute('data-theme', newTheme);
                        localStorage.setItem('portfolio-theme', newTheme);
                        
                        // Update icon
                        const themeIcon = document.querySelector('.theme-toggle svg');
                        if (themeIcon) {
                            if (newTheme === 'dark') {
                                themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
                            } else {
                                themeIcon.innerHTML = '<circle cx="12" cy="12" r="5"></circle><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"></path>';
                            }
                        }
                    });
                    
                    // Set initial icon
                    const currentTheme = document.documentElement.getAttribute('data-theme');
                    const themeIcon = document.querySelector('.theme-toggle svg');
                    if (themeIcon && currentTheme === 'dark') {
                        themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
                    }
                }
                
                // Mobile menu
                const hamburger = document.querySelector('.hamburger');
                const navLinks = document.querySelector('.nav-links');
                
                if (hamburger && navLinks) {
                    hamburger.addEventListener('click', function() {
                        hamburger.classList.toggle('active');
                        navLinks.classList.toggle('active');
                        document.body.style.overflow = navLinks.classList.contains('active') ? 'hidden' : '';
                    });
                    
                    navLinks.querySelectorAll('a').forEach(link => {
                        link.addEventListener('click', function() {
                            if (navLinks.classList.contains('active')) {
                                hamburger.classList.remove('active');
                                navLinks.classList.remove('active');
                                document.body.style.overflow = '';
                            }
                        });
                    });
                }
            });
        </script>
    </body>
    </html>
    '''
    
    # Replace template variables
    return render_template_string(
        html_template,
        post=post,
        post_html=post_html,
        related_posts=related_posts,
        prev_post=prev_post,
        next_post=next_post,
        theme=theme
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
