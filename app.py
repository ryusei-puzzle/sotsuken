from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import DBSCAN
import io

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask API! Use /dbscan endpoint to process data.", 200

@app.route('/dbscan', methods=['GET', 'POST'])
def dbscan_outliers():
    if request.method == 'GET':
        return "Please use POST method to submit data.", 200

    if request.method == 'POST':
        try:
            data = request.json
            if not data or "csvData" not in data:
                return jsonify({"error": "Invalid input data"}), 400

            csv_data = data.get("csvData")
            df = pd.read_csv(io.StringIO(csv_data))
            
            # DBSCAN処理
            clustering = DBSCAN(eps=5, min_samples=5).fit(df[['Text']])
            df['Cluster'] = clustering.labels_
            outliers = df[df['Cluster'] == -1]

            return jsonify(outliers.to_dict(orient="records"))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
