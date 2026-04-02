from flask import Flask, jsonify, send_from_directory
from pathlib import Path
import pandas as pd
import config

app = Flask(__name__, static_folder='gui')

@app.route('/')
def index():
    return send_from_directory('gui', 'index.html')

@app.route('/<path:path>')
def send_gui_files(path):
    return send_from_directory('gui', path)

@app.route('/api/kpi/<run_id>/<kpi_name>')
def get_kpi_data(run_id, kpi_name):
    kpi_file = Path(config.KPI_OUTPUT_DIR) / run_id / f'{kpi_name}.csv'
    if not kpi_file.exists():
        return jsonify({'error': 'KPI not found'}), 404

    df = pd.read_csv(kpi_file)
    return jsonify(df.to_dict(orient='list'))

if __name__ == '__main__':
    app.run(debug=True)
