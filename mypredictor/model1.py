import pandas as pd
import joblib
from joblib import load
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import numpy as np
import re
import json
import os
from flask import Flask
from flask_mysqldb import MySQL
from io import BytesIO
from collections import Counter
import tensorflow as tf
import xgboost as xgb
from xgboost import XGBClassifier
import logging
import pytz
from datetime import datetime

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'detech.mysql.pythonanywhere-services.com'
app.config['MYSQL_USER'] = 'detech'
app.config['MYSQL_PASSWORD'] = 'dbdetechpassword'
app.config['MYSQL_DB'] = 'detech$dbdetech'

mysql = MySQL(app)

# Load the machine learning models
def load_models(model_paths):
    models = [load(model_path) for model_path in model_paths]
    logger.info("Models loaded successfully.")
    return models

# Data Preparation
def clean_value(value):
    try:
        if pd.notnull(value):
            cleaned_value = re.sub(r'[a-zA-Z"{}]', '', value)
            return cleaned_value.strip()
        return value
    except Exception as e:
        logger.error(f"Error cleaning value: {e}")
        raise

def remove_first_symbol(value):
    try:
        if pd.notnull(value):
            return value[1:] if len(value) > 0 else value
        return value
    except Exception as e:
        logger.error(f"Error removing first symbol: {e}")
        raise

def prepare_data_for_prediction(raw_csv_data):
    try:
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
            logger.error("Number of headers does not match the number of columns in the CSV.")
            raise ValueError("Number of headers does not match the number of columns in the CSV.")

        # Cleans values
        df = df.applymap(clean_value)

        # Fix the format of values
        df = df.applymap(remove_first_symbol)

        # Check for NaN values
        nan_counts = df.isnull().sum().sum()

        if nan_counts > 0:
            logger.warning(f"The CSV data has {nan_counts} NaN values. Filling with 0...")
            df = df.fillna(0)
            logger.info("NaN values filled.")
        else:
            logger.info("The CSV data has no NaN values.")

        # Remove duplicate rows
        prepped_csv = df.drop_duplicates(keep='first')
        logger.info("Duplicate rows removed.")

        return prepped_csv
    except Exception as e:
        logger.error(f"Error preparing data for prediction: {e}")
        raise

def normalize_for_detection(prepped_csv):
    try:
        df = prepped_csv
        important_features_list = ['dst port','tot fwd pkts','tot bwd pkts','totlen fwd pkts','totlen bwd pkts','fwd pkt len mean','bwd pkt len max','fwd iat mean','flow iat max','flow iat min','fwd iat mean','fwd iat std','bwd iat mean','bwd iat std','bwd iat max','bwd iat min','fwd header len','bwd header len','fwd pkts s','bwd pkts s','pkt len mean','pkt len var','fwd seg size avg','bwd seg size avg','init fwd win byts','init bwd win byts']

        csv_data = df[important_features_list]
        csv_data = csv_data.apply(pd.to_numeric, errors='coerce')

        categorical_cols = ['dst port']
        df_categorical = csv_data[categorical_cols]
        df_numerical = csv_data.drop(columns=categorical_cols)

        std_scaler = StandardScaler()
        df_numerical_scaled = pd.DataFrame(std_scaler.fit_transform(df_numerical), columns=df_numerical.columns)

        df_normalized = pd.concat([df_numerical_scaled, df_categorical.reset_index(drop=True)], axis=1)
        df_normalized = df_normalized[important_features_list]
        logger.info("Data normalized for detection.")
        return df_normalized
    except Exception as e:
        logger.error(f"Error normalizing data for detection: {e}")
        raise

def svm_predict(csv_data_detect):
    try:
        data = csv_data_detect

        current_directory = os.path.dirname(os.path.abspath(__file__))
        svm_model_path = os.path.join(current_directory, "26-multi_class_svm_model.h5")
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
        nb_model_path = os.path.join(current_directory, "26-NB_model.pkl")
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
        dt_model_path = os.path.join(current_directory, "26-DT_model.pkl")
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
        ensemble_model_path = os.path.join(current_directory, "26-Ensemble_model_multi_class.pkl")
        xgb_model = joblib.load(ensemble_model_path)

        stacked_predictions = np.column_stack((svm_predict, nb_predict, dt_predict))

        classifications = xgb_model.predict(stacked_predictions)

        result_counts = pd.Series(classifications).value_counts(normalize=True) * 100

        # Mapping of labels to terms
        label_to_term = {
            0: 'Normal',
            1: 'Denial of Service',
            3: 'Botnet',
            4: 'Infiltration',
            2: 'Bruteforce'
        }

        # Initialize classification results with all label names set to 0%
        classification_results = {label: "0.00%" for label in label_to_term.values()}

        # Update the classification results with actual percentages
        for label, percentage in result_counts.items():
            term = label_to_term.get(label, 'Unknown')  # Get the corresponding term or 'Unknown' if not found
            classification_results[term] = f"{percentage:.2f}%"

        return classification_results

    except Exception as e:
        print(f"Error: {e}")

def detect_traffic(csv_data_detect):
    try:
        svm_data = svm_predict(csv_data_detect)
        nb_data = nb_predict(csv_data_detect)
        dt_data = dt_predict(csv_data_detect)
        ensemble_result = ensemble_predict(svm_data, nb_data, dt_data)

        logger.info("Traffic detection completed successfully.")
        return ensemble_result
    except Exception as e:
        logger.error(f"Error detecting traffic: {e}")
        raise

def top_src_ip(raw_csv_data):
    df = raw_csv_data
    df['fwd pkts s'] = pd.to_numeric(df['fwd pkts s'], errors='coerce')
    df['fwd pkts s'] = df['fwd pkts s'].fillna(0)
    sum_fwd_pkts_per_ip = df.groupby('src ip')['fwd pkts s'].sum()
    sum_fwd_pkts_per_ip_df = sum_fwd_pkts_per_ip.reset_index(name='sum of fwd pkts s')
    sum_fwd_pkts_per_ip_df = sum_fwd_pkts_per_ip_df.sort_values(by='sum of fwd pkts s', ascending=False)
    json_src_ip = sum_fwd_pkts_per_ip_df.set_index('src ip')['sum of fwd pkts s'].to_dict()

    return json_src_ip

def top_dst_ip(raw_csv_data):
    df = raw_csv_data
    df['bwd pkts s'] = pd.to_numeric(df['bwd pkts s'], errors='coerce')
    df['bwd pkts s'] = df['bwd pkts s'].fillna(0)
    sum_fwd_pkts_per_ip = df.groupby('dst ip')['bwd pkts s'].sum()
    sum_bwd_pkts_per_ip_df = sum_fwd_pkts_per_ip.reset_index(name='sum of bwd pkts s')
    sum_bwd_pkts_per_ip_df = sum_bwd_pkts_per_ip_df.sort_values(by='sum of bwd pkts s', ascending=False)
    json_dst_ip = sum_bwd_pkts_per_ip_df.set_index('dst ip')['sum of bwd pkts s'].to_dict()

    return json_dst_ip

def get_service_name(row, service_df):
    port_number = row['dst port']
    if port_number.isdigit() and int(port_number) <= 49151:
        matching_service = service_df.loc[service_df['Port Number'] == port_number, 'Service Name']
        if not matching_service.empty:
            return matching_service.iloc[0]
        else:
            return 'Reserved'
    else:
        return 'Dynamically Assigned Port' if int(port_number) > 49151 else 'Unassigned'

def top_ports(raw_csv_data):

    same_directory = os.path.dirname(os.path.abspath(__file__))
    service_csv_file = os.path.join(same_directory, 'ServiceNamePortNumber.csv')
    service_df = pd.read_csv(service_csv_file)

    # Remove leading and trailing whitespaces from 'dst port' column in raw_csv_data
    raw_csv_data['dst port'] = raw_csv_data['dst port'].str.strip()

    # Convert the 'Port Number' column in service_df to string and remove leading/trailing whitespaces
    service_df['Port Number'] = service_df['Port Number'].astype(str).str.strip()

    # Merge the raw data DataFrame with the service DataFrame based on port number
    merged_df = pd.merge(raw_csv_data, service_df, left_on='dst port', right_on='Port Number', how='left')

    # Create a new column 'Service Name' based on the criteria using the get_service_name() function
    merged_df['Service Name'] = merged_df.apply(lambda row: get_service_name(row, service_df), axis=1)

    # Sum 'flow byts s' for each 'Service Name'
    sum_flow_byts = merged_df.groupby('Service Name')['flow byts s'].sum()

    # Convert the Series to a DataFrame and reset the index
    sum_flow_byts_df = sum_flow_byts.reset_index(name='sum of flow byts s')

    # Sort the DataFrame by the summed 'flow byts s' in descending order
    sum_flow_byts_df = sum_flow_byts_df.sort_values(by='sum of flow byts s', ascending=False)

    # Convert the DataFrame to JSON format
    json_ports = sum_flow_byts_df.set_index('Service Name')['sum of flow byts s'].to_dict()

    return json_ports

def protocol_distribution(raw_csv_data):
    # Define protocol values
    protocol_values = {
        0: 'HOPOPT',
        1: 'ICMP',
        2: 'IGMP',
        6: 'TCP',
        8: 'EGP',
        17: 'UDP',
        27: 'RDP',
        36: 'XTP',
        88: 'EIGRP',
        89: 'OSPF',
        121: 'SMP'
    }

    df = raw_csv_data.copy()

    # Convert 'protocol' column to integer type
    df['protocol'] = df['protocol'].astype(int)

    # Map protocol numbers to protocol names
    df['protocol'] = df['protocol'].map(protocol_values).fillna('Others')

    # Count occurrences of each protocol
    protocol_counts = df['protocol'].value_counts()

    return protocol_counts

def analyzeCsvFile(document_id, user_id, csv_prep, csv_data_detect):
    try:
        # Concatenate 'date' and 'time' columns to form a new 'datetime' column
        csv_prep['datetime'] = pd.to_datetime(csv_prep['timestamp'])

        # Extract time component of the first row's timestamp and format as string
        start_time = csv_prep['datetime'].iloc[0].strftime('%H:%M:%S')

        # Extract date component of the first row's timestamp
        file_date = csv_prep['datetime'].dt.date.iloc[0]

        # FLOW DURATION
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

        traffic_classification = detect_traffic(csv_data_detect)

        normal = traffic_classification["Normal"]
        dos = traffic_classification["Denial of Service"]
        bruteforce = traffic_classification["Bruteforce"]
        botnet = traffic_classification["Botnet"]
        infiltration = traffic_classification["Infiltration"]

        # Convert percentage strings to floats and find the max
        max_category = max(traffic_classification, key=lambda k: float(traffic_classification[k].rstrip('%')))

        # Classify the result as 'Normal' or 'Malicious'
        traffic_detection = 'Normal' if max_category == 'Normal' else 'Malicious'

        # Check if traffic_detection is 'Malicious'
        alerts = 1 if traffic_detection == 'Malicious' else 0

        logger.info("Analysis completed successfully.")
    except Exception as e:
        logger.error(f"Error analyzing CSV file: {e}")
        raise

    curInsertAnalysis = mysql.connection.cursor()
    curInsertHistory = mysql.connection.cursor()

    last_inserted_id_analysis = 0

    try:
        # Your existing tblanalysis insertion code
        curInsertAnalysis.execute(
            "INSERT INTO tblanalysis (dur, protocol, sourceip, destip, destport, user_id, starttime, traffic, normal, dos, brutef, botnet, inf, classify, alerts) VALUES('" +
            str(total_capturing_duration_minutes) + "','" + most_frequent_protocol + "','" + most_frequent_src_ip +
            "','" + most_frequent_dst_ip + "','" + most_frequent_dst_port + "','" + str(user_id) + "','" +
            str(start_time) + "','" + traffic_detection + "','" + str(normal) + "','" + str(dos) + "','" + str(bruteforce) + "','" +
            str(botnet) + "','" + str(infiltration) + "','" + str(max_category) + "','" + str(alerts) + "')")

        mysql.connection.commit()

        last_inserted_id_analysis = curInsertAnalysis.lastrowid

        logger.info("Analysis data inserted successfully.")
    except Exception as analysis_error:
        mysql.connection.rollback()  # Rollback changes if an error occurs during tblanalysis insertion
        raise analysis_error  # Re-raise the exception
    finally:
        curInsertAnalysis.close()

    try:
        # Your existing tblhistory insertion code
        curInsertHistory.execute(
            "INSERT INTO tblhistory (user_id, date, time, traffic, analysisid, classification, sourceip, destip) VALUES('" + str(user_id) + "','" + str(file_date) + "','" + str(start_time) + "','" + traffic_detection + "','" + str(last_inserted_id_analysis) + "','" + str(max_category) + "','" + most_frequent_src_ip + "','" + most_frequent_dst_ip + "')")

        mysql.connection.commit()
        logger.info("History data inserted successfully.")
    except Exception as history_error:
        mysql.connection.rollback()  # Rollback changes if an error occurs during tblhistory insertion
        raise history_error  # Re-raise the exception
    finally:
        curInsertHistory.close()

    # After your INSERT statements
    mysql.connection.commit()

def dashboardResults(document_id, user_id, csv_prep, csv_data_detect):
    df = csv_prep
    try:
        # PACKET CAPTURE
        df['flow pkts s'] = pd.to_numeric(df['flow pkts s'], errors='coerce')
        ave_flowpacket_rate = df['flow pkts s'].mean()

        # TRAFFIC PATTERNS
        df['flow byts s'] = pd.to_numeric(df['flow byts s'], errors='coerce')
        ave_flowbytes_rate = df['flow byts s'].mean()

        # PROTOCOL DISTRIBUTION
        protocol_counts = protocol_distribution(csv_prep)

        HOPOPT_count = protocol_counts.get("HOPOPT", 0)
        ICMP_count = protocol_counts.get("ICMP", 0)
        IGMP_count = protocol_counts.get("IGMP", 0)
        TCP_count = protocol_counts.get("TCP", 0)
        EGP_count = protocol_counts.get("EGP", 0)
        UDP_count = protocol_counts.get("UDP", 0)
        RDP_count = protocol_counts.get("RDP", 0)
        XTP_count = protocol_counts.get("XTP", 0)
        EIGRP_count = protocol_counts.get("EIGRP", 0)
        OSPF_count = protocol_counts.get("OSPF", 0)
        SMP_count = protocol_counts.get("SMP", 0)
        Others_count = protocol_counts.get("Others", 0)

        # Load your raw CSV data into a DataFrame (replace this with your actual data loading code)

        # Call each function to get the JSON outputs
        json_src_ip = top_src_ip(csv_prep)
        json_dst_ip = top_dst_ip(csv_prep)
        json_ports = top_ports(csv_prep)

        # Combine JSON outputs into a single dictionary
        combined_json = {}
        for i in range(min(len(json_src_ip), len(json_dst_ip), len(json_ports))):
            combined_json[f"Rank {i+1}"] = {
                "top_src_ip": {k: v for k, v in json_src_ip.items() if v == sorted(json_src_ip.values(), reverse=True)[i]},
                "top_dst_ip": {k: v for k, v in json_dst_ip.items() if v == sorted(json_dst_ip.values(), reverse=True)[i]},
                "top_ports": {k: v for k, v in json_ports.items() if v == sorted(json_ports.values(), reverse=True)[i]}
            }
        json_ip_ports = json.dumps(combined_json, indent=4)

        # Analyze most frequent values
        most_frequent_protocol = csv_prep['protocol'].mode().values[0]
        most_frequent_src_ip = csv_prep['src ip'].mode().values[0]
        most_frequent_dst_ip = csv_prep['dst ip'].mode().values[0]
        most_frequent_dst_port = csv_prep['dst port'].mode().values[0]

        traffic_classification = detect_traffic(csv_data_detect)

        normal = traffic_classification["Normal"]
        dos = traffic_classification["Denial of Service"]
        bruteforce = traffic_classification["Bruteforce"]
        botnet = traffic_classification["Botnet"]
        infiltration = traffic_classification["Infiltration"]

        # Convert percentage strings to floats and find the max
        max_category = max(traffic_classification, key=lambda k: float(traffic_classification[k].rstrip('%')))

        # Classify the result as 'Normal' or 'Malicious'
        traffic_detection = 'Normal' if max_category == 'Normal' else 'Malicious'

        # Check if traffic_detection is 'Malicious'
        alerts = 1 if traffic_detection == 'Malicious' else 0

        # Update numalerts based on the current value and the new alerts value
        cur = mysql.connection.cursor()
        cur.execute("SELECT numalerts FROM tbldashboardmetrics WHERE user_id = %s", (user_id,))
        result = cur.fetchone()
        if result is not None:
            numalerts = result[0]
        else:
            # If no records found for the user, set numalerts to 0
            numalerts = 0

        # Update numalerts in the database
        new_numalerts = numalerts + alerts
        cur.execute("UPDATE tbldashboardmetrics SET numalerts = %s WHERE user_id = %s", (new_numalerts, user_id))
        mysql.connection.commit()
        cur.close()

        try:
            # Check if the cursor is None or if it's still connected
            if cur is None or not cur.connection:
                # Reconnect if the cursor is None or not connected
                conn = mysql.connection
                cur = conn.cursor()

            # Fetch the current value of TCP_count with a default value of 0
            cur.execute("SELECT IFNULL(TCP_count, 0) FROM tbldashboardmetrics WHERE user_id = %s", (user_id,))
            result = cur.fetchone()
            if result is not None:
                current_TCP_count = result[0]
            else:
                current_TCP_count = 0

            # Update TCP_count in the database
            new_TCP_count = current_TCP_count + TCP_count  # Add the new TCP count to the current count
            cur.execute("UPDATE tbldashboardmetrics SET TCP_count = %s WHERE user_id = %s", (new_TCP_count, user_id))

            # Update total_TCP_count with the new total
            cur.execute("UPDATE tbldashboardmetrics SET total_TCP_count = total_TCP_count + %s WHERE user_id = %s", (TCP_count, user_id))

            # Commit the changes to the database
            mysql.connection.commit()

        except Exception as e:
            logger.error(f"Error retrieving Dashboard Results: {e}")
            raise

        finally:
            # Close the cursor
            if cur:
                cur.close()

        try:
            # Check if the cursor is None or if it's still connected
            if cur is None or not cur.connection:
                # Reconnect if the cursor is None or not connected
                conn = mysql.connection
                cur = conn.cursor()

            # Fetch the current value of UDP_count with a default value of 0
            cur.execute("SELECT IFNULL(UDP_count, 0) FROM tbldashboardmetrics WHERE user_id = %s", (user_id,))
            result = cur.fetchone()
            if result is not None:
                current_UDP_count = result[0]
            else:
                current_UDP_count = 0

            # Update UDP_count in the database
            new_UDP_count = current_UDP_count + UDP_count  # Add the new UDP count to the current count
            cur.execute("UPDATE tbldashboardmetrics SET UDP_count = %s WHERE user_id = %s", (new_UDP_count, user_id))

            # Update total_UDP_count with the new total
            cur.execute("UPDATE tbldashboardmetrics SET total_UDP_count = total_UDP_count + %s WHERE user_id = %s", (UDP_count, user_id))

            # Commit the changes to the database
            mysql.connection.commit()

        except Exception as e:
            logger.error(f"Error retrieving Dashboard Results: {e}")
            raise

        finally:
            # Close the cursor
            if cur:
                cur.close()

        logger.info("Dashboard completed successfully.")
    except Exception as e:
        logger.error(f"Error retrieving Dashboard Results: {e}")
        raise

    curInsertDashboard = mysql.connection.cursor()

    try:
        # Insert calculated metrics along with protocol frequency into MySQL
        curInsertDashboard.execute(
            "INSERT INTO tbldashboardmetrics (user_id, aveflowpktsrate, aveflowbytsrate, alerts, numalerts, TCP_count, total_TCP_count, UDP_count, total_UDP_count) VALUES('" + str(user_id) + "','" + str(ave_flowpacket_rate) + "','" + str(ave_flowbytes_rate) + "','" + str(alerts) + "','" + str(new_numalerts) + "','" + str(TCP_count) + "','" + str(new_TCP_count) + "','" + str(UDP_count) + "','" + str(new_UDP_count) + "')")

        mysql.connection.commit()

        logger.info("Dashboard data inserted successfully.")
    except Exception as dashboard_error:
        mysql.connection.rollback()
        raise dashboard_error
    finally:
        curInsertDashboard.close()


    curInsertRankings = mysql.connection.cursor()

    try:
        # Insert rankings
        curInsertRankings.execute(
            "INSERT INTO tblrankings (ranking, user_id) VALUES('" + str(json_ip_ports) + "', '" + str(user_id) + "')")

        mysql.connection.commit()

        logger.info("Dashboard data inserted successfully.")
    except Exception as dashboard_error:
        mysql.connection.rollback()
        raise dashboard_error
    finally:
        curInsertRankings.close()

    # After your INSERT statements
    mysql.connection.commit()

