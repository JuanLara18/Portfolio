from flask import Flask, request, send_from_directory, send_file, abort
import os

app = Flask(__name__, static_folder=None)

# Obtener la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    app.run(
        host='127.0.0.1',
        port=8000,
        debug=True,
        use_reloader=True
    )