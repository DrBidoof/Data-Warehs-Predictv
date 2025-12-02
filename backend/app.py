from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import sys
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = {
    'exploration': str(REPO_ROOT / 'Data exploration.py'),
    'modelling': str(REPO_ROOT / 'Data modelling.py'),
    'predictive': str(REPO_ROOT / 'Predictive model building.py'),
}


def run_script(script_path):
    # Use the same Python interpreter running this app
    python_exe = sys.executable or 'python'
    # Run script unbuffered so output appears promptly
    proc = subprocess.Popen([python_exe, '-u', script_path], cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


@app.get('/run/<script>')
def run(script: str):
    path = SCRIPTS.get(script)
    if not path:
        return jsonify(error='Unknown script'), 404
    if not os.path.exists(path):
        return jsonify(error=f'Script file not found: {path}'), 404

    code, out, err = run_script(path)

    # After running, gather any images saved in outputs/<script>/
    outputs_dir = REPO_ROOT / 'outputs' / script
    images = []
    if outputs_dir.exists():
        for p in sorted(outputs_dir.iterdir()):
            if p.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # expose via /outputs/<script>/<filename>
                images.append(f"/outputs/{script}/{p.name}")

    return jsonify(code=code, stdout=out, stderr=err, images=images)


@app.get('/outputs/<script>/<path:filename>')
def serve_output(script, filename):
    outputs_root = REPO_ROOT / 'outputs' / script
    if not outputs_root.exists():
        return jsonify(error='Not found'), 404
    return send_from_directory(str(outputs_root), filename)

if __name__ == '__main__':
    # listen on 5000 to match previous backend
    app.run(host='0.0.0.0', port=5000)
