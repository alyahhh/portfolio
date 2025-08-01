import pandas as pd
import joblib
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re
import json
import os
from flask import Flask, render_template, request, session, redirect, jsonify
from flask_mysqldb import MySQL
from io import BytesIO
from collections import Counter
app = Flask(__name__)

app.config['MYSQL_HOST'] = 'detech.mysql.pythonanywhere-services.com'
app.config['MYSQL_USER'] = 'detech'
app.config['MYSQL_PASSWORD'] = 'dbdetechpassword'
app.config['MYSQL_DB'] = 'detech$dbdetech'

mysql = MySQL(app)


def clean_value(value):
    if pd.notnull(value):
        cleaned_value = re.sub(r'[a-zA-Z"{}]', '', value)
        return cleaned_value.strip()
    return value

def remove_first_symbol(value):
    if pd.notnull(value):
        return value[1:] if len(value) > 0 else value
    return value

def prepare_data_for_prediction(raw_csv_data):

    csv_data_io = BytesIO(raw_csv_data)
    df = pd.read_csv(csv_data_io)

    # Add headers
    your_headers = ['src ip', 'dst ip', 'src port', 'dst port', 'protocol',
                'timestamp', 'flow duration', 'flow byts s', 'flow pkts s',
                'fwd pkts s', 'bwd pkts s', 'tot fwd pkts', 'tot bwd pkts',
                'totlen fwd pkts', 'totlen bwd pkts', 'fwd pkt len max',
                'fwd pkt len min', 'fwd pkt len mean', 'fwd pkt len std',
                'bwd pkt len max', 'bwd pkt len min', 'bwd pkt len mean',
                'bwd pkt len std', 'pkt len max', 'pkt len min', 'pkt len mean',
                'pkt len std', 'pkt len var', 'fwd header len', 'bwd header len',
                'fwd seg size min', 'fwd act data pkts', 'flow iat mean',
                'flow iat max', 'flow iat min', 'flow iat std', 'fwd iat tot',
                'fwd iat max', 'fwd iat min', 'fwd iat mean', 'fwd iat std',
                'bwd iat tot', 'bwd iat max', 'bwd iat min', 'bwd iat mean',
                'bwd iat std', 'fwd psh flags', 'bwd psh flags', 'fwd urg flags',
                'bwd urg flags', 'fin flag cnt', 'syn flag cnt', 'rst flag cnt',
                'psh flag cnt', 'ack flag cnt', 'urg flag cnt', 'ece flag cnt',
                'down up ratio', 'pkt size avg', 'init fwd win byts', 'init bwd win byts',
                'active max', 'active min', 'active mean', 'active std', 'idle max',
                'idle min', 'idle mean', 'idle std', 'fwd byts b avg', 'fwd pkts b avg',
                'bwd byts b avg', 'bwd pkts b avg', 'fwd blk rate avg', 'bwd blk rate avg',
                'fwd seg size avg', 'bwd seg size avg', 'cwe flag count', 'subflow fwd pkts',
                'subflow bwd pkts', 'subflow fwd byts', 'subflow bwd byts'
                ]

    if len(your_headers) == len(df.columns):
        df.columns = your_headers
    else:
        print("Number of headers does not match the number of columns in the CSV.")

    # cleans values
    df = df.applymap(clean_value)

    # fix the format of values
    df = df.applymap(remove_first_symbol)

    # removes slash
    df.columns = [col.lower().replace('/', ' ') for col in df.columns]

    # Check for NaN values
    nan_counts = df.isnull().sum().sum()

    if nan_counts > 0:
        print(f"The CSV data has {nan_counts} NaN values. Filling with 0...")
        df = df.fillna(0)
        print("NaN values filled.")
    else:
        print("The CSV data has no NaN values.")

    # Remove duplicate rows
    prepped_csv = df.drop_duplicates(keep='first')
    print("Duplicate rows removed.")

    return prepped_csv

def normalize_for_detection(prepped_csv):

    df = prepped_csv

    important_features_list = ['init fwd win byts', 'dst port', 'fwd seg size min', 'fwd header len',
                                'src port', 'flow duration', 'fwd pkt len max', 'fwd pkts s',
                                'fwd seg size avg', 'fwd iat mean', 'subflow fwd byts', 'flow iat min',
                                'flow iat max', 'fwd iat max', 'fwd iat min', 'fwd pkt len mean',
                                'bwd pkt len std', 'flow iat mean', 'fwd iat tot', 'subflow fwd pkts',
                                'bwd pkt len max', 'bwd seg size avg', 'bwd header len', 'totlen bwd pkts',
                                'fwd pkt len std', 'pkt len max', 'init bwd win byts', 'subflow bwd byts',
                                'pkt len std', 'pkt len mean', 'tot bwd pkts', 'bwd pkt len mean',
                                'subflow bwd pkts', 'tot fwd pkts']

    important_features_list = [ 'dst port', 'src port', 'fwd act data pkts', 'init fwd win byts', 'subflow fwd byts',
             'totlen fwd pkts', 'fwd header len', 'tot bwd pkts', 'bwd header len', 'flow iat mean',
            'flow duration', 'subflow fwd pkts', 'flow iat min', 'bwd seg size avg', 'fwd seg size min',
             'flow iat max', 'fwd seg size avg', 'fwd pkts s', 'fwd iat max', 'fwd iat min', 'fwd iat tot',
             'subflow bwd byts', 'fwd iat mean', 'subflow bwd pkts', 'bwd pkts s', 'fwd pkt len max',
            'totlen bwd pkts', 'fwd pkt len mean', 'pkt len max', 'pkt len mean', 'bwd pkt len mean',
             'rst flag cnt', 'pkt len var', 'bwd pkt len std', 'init bwd win byts', 'flow iat std',
            'tot fwd pkts']
                                        important_features_list = [ 'src port',
                                    'dst port',
                                    'fwd header len',
                                    'subflow fwd byts',
                                    'fwd pkts s',
                                    'flow iat max',
                                    'flow iat mean',
                                    'totlen fwd pkts',
                                    'subflow fwd pkts',
                                    'flow duration',
                                    'init fwd win byts',
                                    'bwd header len',
                                    'subflow bwd pkts',
                                    'fwd seg size min',
                                    'fwd act data pkts',
                                    'tot fwd pkts',
                                    'flow iat min',
                                    'fwd iat tot',
                                    'bwd seg size avg',
                                    'bwd pkt len mean',
                                    'bwd pkts s',
                                    'fwd seg size avg',
                                    'fwd pkt len mean',
                                    'fwd iat max',
                                    'fwd iat min',
                                    'pkt len var',
                                    'fwd iat mean',
                                    'tot bwd pkts',
                                    'ece flag cnt',
                                    'rst flag cnt',
                                    'pkt size avg',
                                    'fwd pkt len max',
                                    'subflow bwd byts',
                                    'init bwd win byts',
                                    'pkt len mean',
                                    'flow iat std',
                                    'pkt len std']

    csv_data = df[important_features_list]

    csv_data = csv_data.apply(pd.to_numeric, errors='coerce')

    categorical_cols = ['src port', 'dst port']
    df_categorical = csv_data[categorical_cols]
    df_numerical = csv_data.drop(columns=categorical_cols)

    std_scaler = StandardScaler()
    df_numerical_scaled = pd.DataFrame(std_scaler.fit_transform(df_numerical), columns=df_numerical.columns)

    df_normalized = pd.concat([df_numerical_scaled, df_categorical.reset_index(drop=True)], axis=1)
    df_normalized = df_normalized[important_features_list]
    return df_normalized



def normalize_for_classification(prepped_csv):

    df = prepped_csv

    important_features_list = ['fwd seg size min', 'bwd pkts s', 'pkt len var', 'pkt len mean',
       'totlen fwd pkts', 'subflow bwd byts', 'flow duration', 'flow iat max',
       'fwd header len', 'fwd pkt len mean', 'fwd pkt len std',
       'bwd pkt len mean', 'tot bwd pkts', 'flow iat mean', 'subflow fwd pkts',
       'bwd header len', 'subflow fwd byts', 'pkt len std', 'bwd seg size avg',
       'init bwd win byts', 'pkt size avg', 'pkt len max', 'bwd pkt len std',
       'fwd iat mean', 'flow iat min', 'fwd seg size avg', 'fwd iat max',
       'init fwd win byts', 'fwd pkts s', 'fwd pkt len max', 'fwd iat tot',
       'totlen bwd pkts', 'fwd iat min', 'src port', 'dst port']

    new_df = df[important_features_list]

    categorical_cols = ['src port', 'dst port']
    df_categorical = new_df[categorical_cols]
    df_numerical = new_df.drop(columns=categorical_cols)
    scaler = MinMaxScaler()
    df_numerical_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), columns=df_numerical.columns)

    df_normalized = pd.concat([df_numerical_scaled, df_categorical.reset_index(drop=True)], axis=1)
    return df_normalized

def svm_predict(csv_data_detect):
    try:
        data = csv_data_detect

        current_directory = os.path.dirname(os.path.abspath(__file__))
        svm_model_path = os.path.join(current_directory, "binary_svmmodel.h5")
        svm_model = tf.keras.models.load_model(svm_model_path)

        svm_predictions_raw = svm_model.predict(data)
        svm_threshold = 0.5  # Set your desired threshold
        svm_predictions = np.where(svm_predictions_raw >= svm_threshold, 1, 0)

        logger.info("SVM prediction completed successfully.")
        return svm_predictions
    except Exception as e:
        logger.error(f"Error making SVM prediction: {e}")
        raise

def nb_predict(csv_data_detect):
    try:
        data = csv_data_detect

        current_directory = os.path.dirname(os.path.abspath(__file__))
        nb_model_path = os.path.join(current_directory, "binary_nbmodel.pkl")
        nb_model = joblib.load(nb_model_path)

        nb_predictions = nb_model.predict(data)

        label_encoder = LabelEncoder()
        nb_predictions_numeric = label_encoder.fit_transform(nb_predictions)

        logger.info("Naive Bayes prediction completed successfully.")
        return nb_predictions_numeric
    except Exception as e:
        logger.error(f"Error making Naive Bayes prediction: {e}")
        raise

def dt_predict(csv_data_detect):
    try:
        data = csv_data_detect

        current_directory = os.path.dirname(os.path.abspath(__file__))
        dt_model_path = os.path.join(current_directory, "binary_dtmodel.pkl")
        dt_model = joblib.load(dt_model_path)

        dt_predictions = dt_model.predict(data)

        logger.info("Decision Tree prediction completed successfully.")
        return dt_predictions
    except Exception as e:
        logger.error(f"Error making Decision Tree prediction: {e}")
        raise

def ensemble_predict(svm_predict, nb_predict, dt_predict):
    try:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        ensemble_model_path = os.path.join(current_directory, "retrainensemble_model.pkl")

        logger.info(f"Loading XGBoost model from: {ensemble_model_path}")

        # Load the xgboost model
        xgb_model = xgb.Booster()
        xgb_model.load_model(ensemble_model_path)

        logger.info("XGBoost model loaded successfully.")

        stacked_predictions = np.column_stack((svm_predict, nb_predict, dt_predict))
        stacked_df = pd.DataFrame(data=stacked_predictions, columns=['SVM', 'NaiveBayes', 'DecisionTree'])

        # Predict using xgboost model
        predictions = xgb_model.predict(xgb.DMatrix(stacked_df))

        most_common_prediction = Counter(predictions).most_common(1)[0][0]

        label = 'Malicious' if most_common_prediction == 1 else 'Normal'

        logger.info("Ensemble prediction completed successfully.")
        return label
    except Exception as e:
        logger.error(f"Error making Ensemble prediction: {e}")
        raise

def detect_traffic(csv_data_detect):

    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path1 = os.path.join(current_directory, "dt_model_34.pkl")
    detect_model = joblib.load(file_path1)

    data = csv_data_detect

    predictions = detect_model.predict(data)

    # Find the most common prediction (mode)
    most_common_prediction = Counter(predictions).most_common(1)[0][0]

    # Determine the label for the most common prediction
    label = 'Malicious' if most_common_prediction == 1 else 'Normal'

    return label


def classify_traffic(csv_data_classify):

    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path1 = os.path.join(current_directory, "mc_dt_latest(50_50).pkl")
    classify_model = load_models(file_path1)

    data = csv_data_classify

    # Predict using the loaded model
    classification = classify_model.predict(data)

    result_counts = pd.Series(classification).value_counts(normalize=True) * 100

    # Mapping of labels to terms
    label_to_term = {
        4: 'Normal',
        2: 'Denial of Service',
        0: 'Botnet',
        3: 'Infiltration',
        1: 'Bruteforce',
        5: 'SQL Injection'
    }

    # Initialize classification results with all label names set to 0%
    classification_results = {label: "0.00%" for label in label_to_term.values()}

    # Update the classification results with actual percentages
    for label, percentage in result_counts.items():
        term = label_to_term.get(label, 'Unknown')  # Get the corresponding term or 'Unknown' if not found
        classification_results[term] = f"{percentage:.2f}%"

    # Convert the dictionary to JSON format
    classification_results_json = json.dumps(classification_results)

    return classification_results


def analyzeCsvFile(document_id, userid, csv_prep, csv_data_detect, csv_data_classify):
    # Convert the 'Timestamp' column to datetime format and separate into 'Date' and 'Time'
    csv_prep['timestamp'] = pd.to_datetime(csv_prep['timestamp'])
    csv_prep['date'] = csv_prep['timestamp'].dt.date
    csv_prep['time'] = csv_prep['timestamp'].dt.time

    # Extract time component of the first row's timestamp and format as string
    start_time = csv_prep['time'].iloc[0].strftime('%H:%M:%S')

    # Extract date component of the first row's timestamp
    # file_date = csv_prep['date'].iloc[0].date()

    #FLOW DURATION
    # Convert 'flow duration' column to numeric, coercing errors to NaN
    csv_prep['flow duration'] = pd.to_numeric(csv_prep['flow duration'], errors='coerce')

    # Filter out NaN values
    capturing_duration_microseconds = csv_prep['flow duration'].dropna()

    # Calculate total capturing duration in minutes
    total_capturing_duration_minutes = capturing_duration_microseconds.sum() / 60000000


    # Analyze most frequent values
    most_frequent_protocol = csv_prep['protocol'].mode().values[0]
    most_frequent_src_ip = csv_prep['src ip'].mode().values[0]
    most_frequent_dst_ip = csv_prep['dst ip'].mode().values[0]
    most_frequent_dst_port = csv_prep['dst port'].mode().values[0]


    traffic_detection = detect_traffic(csv_data_detect)

    traffic_classification = classify_traffic(csv_data_classify) # Output is in json format
    normal = traffic_classification["Normal"]
    dos = traffic_classification["Denial of Service"]
    bruteforce = traffic_classification["Bruteforce"]
    botnet = traffic_classification["Botnet"]
    sqlinjection = traffic_classification["SQL Injection"]
    infiltration = traffic_classification["Infiltration"]

          if traffic_detection == "Malicious":
            # Proceed with traffic classification only if traffic is detected as malicious
            traffic_classification = classify_traffic(csv_data_classify) # Output is in json format

            # Extract individual threat types from the classification result
            normal = traffic_classification.get("Normal", 0) # Using .get to avoid KeyError if the key doesn't exist
            dos = traffic_classification.get("Denial of Service", 0)
            bruteforce = traffic_classification.get("Bruteforce", 0)
            botnet = traffic_classification.get("Botnet", 0)
            sqlinjection = traffic_classification.get("SQL Injection", 0)
            infiltration = traffic_classification.get("Infiltration", 0)


        else:
            # Skip the classification if traffic is not malicious
            # You can also initialize the variables to default values here if needed
            normal = dos = bruteforce = botnet = sqlinjection = infiltration = "0.00%"

    #CODE TO INSERT DATA to tblanalysis
    curInsert = mysql.connection.cursor()

    curInsert.execute("INSERT INTO tblanalysis (duration, protocol, sourceip, destip, destport, user_id, starttime, traffic, normal, dos, brutef, botnet, sqlinj, inf) VALUES('" + str(total_capturing_duration_minutes) + "','" + most_frequent_protocol + "','" + most_frequent_src_ip + "','" + most_frequent_dst_ip + "','" + most_frequent_dst_port + "','" + str(userid) + "','" + str(start_time) + "','" + traffic_detection + "','" + str(normal) + "','" + str(dos) + "','" + str(bruteforce) + "','" + str(botnet) + "','" + str(sqlinjection) + "','" + str(infiltration) + "')")

    mysql.connection.commit()

    curInsert.close()


# ===========================DASHBOARD================================

#def dashboardCsvFile(document_id, user_id, csv_prep):
#    try:
#        # Extract the correct flow bytes column
#        flow_bytes_column = csv_prep['flow byts s']
#        byts_five_rows = flow_pkts_column.head(5)
#
#        # Extract the correct flow bytes column
#        flow_bytes_column = csv_prep['flow pkts s']
#        pkts_five_rows = flow_pkts_column.head(5)
#
#        # CODE TO INSERT DATA to tbldashboard
#        curInsert = mysql.connection.cursor()

#        curInsert.execute(
#            "INSERT INTO tbldashboard ( aveflowbytsrate, user_id, aveflowpktsrate) VALUES('" +
#            str(byts_five_rows) + "','" + str(user_id) + "','" +
#            str(pkts_five_rows) + "')")

#        mysql.connection.commit()

#        curInsert.close()

#        logger.info("dashboard completed successfully.")
#    except Exception as e:
#        logger.error(f"Error analyzing CSV file: {e}")
#        raise

