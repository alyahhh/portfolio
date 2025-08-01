from flask import Flask, jsonify, render_template
import os
import requests
import json
import logging
from flask_cors import CORS
import model1
from flask_mysqldb import MySQL
from io import BytesIO
from collections import Counter
import tensorflow as tf
import xgboost as xgb
from xgboost import XGBClassifier
import logging
import time  # Import the time module

app = Flask(__name__)
CORS(app)

app.config['MYSQL_HOST'] = 'detech.mysql.pythonanywhere-services.com'
app.config['MYSQL_USER'] = 'detech'
app.config['MYSQL_PASSWORD'] = 'dbdetechpassword'
app.config['MYSQL_DB'] = 'detech$dbdetech'

mysql = MySQL(app)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# MongoDB Data API setup
MONGO_DATA_API_URL = "https://ap-southeast-1.aws.data.mongodb-api.com/app/data-tfwys/endpoint/data/v1"
MONGO_DATA_API_KEY = os.environ.get('MONGO_DATA_API_KEY')  # MongoDB Data API Key from environment variable
DATA_SOURCE_NAME = os.environ.get('DATA_SOURCE_NAME')  # Data source name from environment variable

# Define global variables to store processed CSV data
csv_data_detect = None
csv_data_classify = None

# Define a global variable to store the CSV data
raw_csv_data = None

@app.route('/')
def index():
    return render_template('index.html')

def checkUnprocessedMongoDBFilesPerUser():
    try:
        with app.app_context():
            cur = mysql.connection.cursor()
            # Execute the SELECT query
            cur.execute("SELECT id, filename, user_id FROM tblmongodbfiles WHERE id IN (SELECT MAX(id) FROM tblmongodbfiles WHERE processedstatus = 0 GROUP BY user_id)")

            # Fetch the result (assuming you are interested in the result)
            for record in cur:
                id, filename, user_id = record

                result = retrieveData(str(filename), str(user_id))
                if result == "success":
                    # Wait for analysis to complete
                    time.sleep(1)  # Wait for 1 seconds before proceeding to the next file

                    # Mark the file as processed
                    curUpdateProcessedStatus = mysql.connection.cursor()
                    curUpdateProcessedStatus.execute("UPDATE tblmongodbfiles SET processedstatus = 1 WHERE id = %s", (id,))
                    mysql.connection.commit()
                    curUpdateProcessedStatus.close()
                else:
                    logger.error("Failed to retrieve and process data for file: %s", filename)

            cur.close()
            return 'success'
    except Exception as err:
        logger.exception("Error occurred while processing MongoDB files: %s", err)
        return 'error'

@app.route('/updateProcessedStatus', methods=['GET'])
def updateProcessedStatus():
    state = checkUnprocessedMongoDBFilesPerUser()
    response = {
        'status': str(state)
    }
    return jsonify(response)

def retrieveData(document_id, user_id):
    try:
        headers = {
            "Content-Type": "application/json",
            "Access-Control-Request-Headers": "*",
            "api-key": MONGO_DATA_API_KEY
        }

        payload = json.dumps({
            "collection": "Raw CSV",
            "database": "De-TECH",
            "dataSource": DATA_SOURCE_NAME,
            "filter": {"_id": {"$oid": document_id}}
        })

        response = requests.post(f"{MONGO_DATA_API_URL}/action/findOne", headers=headers, data=payload)

        if response.status_code == 200:
            data = response.json()
            if 'document' in data and 'csv_data' in data['document']:
                csv_list = data['document']['csv_data']
                csv_list = [json.dumps(item) for item in csv_list]
                csv_string = '\n'.join(csv_list)
                csv_data = csv_string.encode('utf-8')
                preprocess_result = preprocessData(str(document_id), str(user_id), csv_data)
                return preprocess_result
            else:
                return "Document or csv_data field not found in the database."
        else:
            logger.error("Error retrieving document: %s", response.text)
            return f"Error retrieving document: {response.text}"
    except Exception as e:
        logger.exception("Error in retrieveData route: %s", e)
        return f"Error in retrieving CSV data: {e}"

def preprocessData(document_id, user_id, csv_data):
    try:
        csv_prep = model1.prepare_data_for_prediction(csv_data)
        csv_data_detect = model1.normalize_for_detection(csv_prep)
        model1.analyzeCsvFile(str(document_id), str(user_id), csv_prep, csv_data_detect)
        model1.dashboardResults(str(document_id), str(user_id), csv_prep, csv_data_detect)
        return "success"
    except Exception as e:
        logger.exception("Error in preprocessing data: %s", e)
        return "error"

if __name__ == "__main__":
    app.run(debug=True)