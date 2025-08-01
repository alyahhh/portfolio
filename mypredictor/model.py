import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re
import json
import os
from flask import Flask, render_template, request, session, redirect, jsonify
from flask_mysqldb import MySQL
from collections import Counter
from io import BytesIO

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'detech.mysql.pythonanywhere-services.com'
app.config['MYSQL_USER'] = 'detech'
app.config['MYSQL_PASSWORD'] = 'dbdetechpassword'
app.config['MYSQL_DB'] = 'detech$dbdetech'

mysql = MySQL(app)


# Data Preparation------------------------------------------
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

    df = pd.read_csv(raw_csv_data)

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

#def normalize_for_detection(prepped_csv):

    df = prepped_csv

    important_features_list = ['init fwd win byts', 'dst port', 'fwd seg size min',
                     'flow duration', 'src port', 'fwd header len',
                     'flow iat min', 'fwd pkts s', 'flow iat max', 'fwd iat tot',
                     'fwd iat max', 'fwd iat min', 'fwd iat mean', 'fwd pkt len max',
                     'totlen fwd pkts', 'fwd iat mean', 'subflow fwd byts',
                     'fwd seg size avg', 'fwd pkt len mean', 'bwd pkt len std',
                     'init bwd win byts', 'bwd seg size avg', 'fwd pkt len std',
                     'bwd pkt len max', 'subflow bwd byts', 'bwd pkts s',
                     'subflow fwd pkts', 'tot fwd pkts', 'bwd header len',
                     'pkt len max', 'totlen bwd pkts', 'bwd pkt len mean',
                     'pkt len var', 'tot bwd pkts', 'pkt len std', 'flow iat std',
                     'pkt size avg']

    csv_data = df[important_features_list]

    csv_data = csv_data.apply(pd.to_numeric, errors='coerce')

    categorical_cols = ['src port', 'dst port']
    df_categorical = csv_data[categorical_cols]
    df_numerical = csv_data.drop(columns=categorical_cols)

    std_scaler = StandardScaler()
    df_numerical_scaled = pd.DataFrame(std_scaler.fit_transform(df_numerical), columns=df_numerical.columns)

    df_normalized = pd.concat([df_numerical_scaled, df_categorical.reset_index(drop=True)], axis=1)
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


def detect_traffic(csv_data_detect):

    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path2 = os.path.join(current_directory, "8317_dt_model_1.pkl")
    detect_model = joblib.load(file_path2)

    data = csv_data_detect

    predictions = detect_model.predict(data)

    # Find the most common prediction (mode)
    most_common_prediction = Counter(predictions).most_common(1)[0][0]

    # Determine the label for the most common prediction
    label = 'Malicious' if most_common_prediction == 1 else 'Normal'

    # Print the predictions and the most common one with its label
    print("\nThe most frequent prediction (label: count):")
    print(label)

    return label



def classify_traffic(csv_data_classify):

    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path1 = os.path.join(current_directory, "mc_dt_latest(50_50)_1.pkl")
    classify_model = joblib.load(file_path1)

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

    # Display the classification results and their percentage frequencies
    print("Classification Results:")
    for label, percentage in result_counts.items():
        term = label_to_term.get(label, 'Unknown')  # Get the corresponding term or 'Unknown' if not found
        print(f"{term}: {percentage:.2f}%")

    classification_results = {}

    for label, percentage in result_counts.items():
        term = label_to_term.get(label, 'Unknown')  # Get the corresponding term or 'Unknown' if not found
        classification_results[term] = f"{percentage:.2f}%"

    # Convert the dictionary to JSON format
    classification_results_json = json.dumps(classification_results)

    # Display the classification results in JSON format
    print("Classification Results in JSON Format:")
    print(classification_results_json)

    return classification_results





def analyzeCsvFile(csv_prep,csv_data_classify,csv_data_detect):
    # Assuming csv_prep is your DataFrame
    # csv_prep = pd.read_csv(str(document_id))  # Replace 'your_csv_file.csv' with the actual file path
    # csv_prep = prepare_data_for_prediction(raw_csv_data)

    # Convert the 'Timestamp' column to datetime format and separate into 'Date' and 'Time'
    csv_prep['timestamp'] = pd.to_datetime(csv_prep['timestamp'])

    # Extract date and time into separate columns
    csv_prep['date'] = csv_prep['timestamp'].dt.date
    csv_prep['time'] = csv_prep['timestamp'].dt.time

    #TIMESTAMP (TIME)
    # Convert 'timestamp' column to datetime
    #csv_prep['timestamp'] = pd.to_datetime(csv_prep['timestamp'])

    # Extract time component of the first row's timestamp and format as string
    start_time = csv_prep['time'].iloc[0].strftime('%H:%M:%S')



    #TIMESTAMP (DATE)
    # Convert 'timestamp' column to datetime
    #csv_prep['timestamp'] = pd.to_datetime(csv_prep['timestamp'])

    # Extract date component of the first row's timestamp
    file_date = csv_prep['date'].iloc[0]



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

    #traffic_detection = detect_traffic(csv_data_detect) # Binary Classification

    # Frequency Percentage of Traffic Classification
    #traffic_classification = classify_traffic(csv_data_classify) # Output is in json format

    # Print the results
    print(file_date)
    print(start_time)
    print(total_capturing_duration_minutes)
    print(most_frequent_protocol)
    print(most_frequent_src_ip)
    print(most_frequent_dst_ip)
    print(most_frequent_dst_port)
    #print(traffic_detection)
    # print results for traffic classification

    #duration, protocol, sourceipaddress, destinationipaddress, destinationport


    #CODE TO INSERT DATA to tblanalysis
    curInsert = mysql.connection.cursor()

    # curInsert.execute("INSERT INTO tblanalysis (duration, protocol, sourceipaddress, destinationipaddress, destinationport, user_id) VALUES('" + str(total_capturing_duration_minutes) + "','" + most_frequent_protocol + "','" + most_frequent_src_ip + "','" + most_frequent_dst_ip + "','" + most_frequent_dst_port + "','" + str(userid) + "')")

    curInsert.execute("INSERT INTO tblAnalysisTab (time, date, duration, protocol, sourceip, destip, destport, traffic, user_id) VALUES('" + str(start_time) + "','" + str(file_date) + "','" + str(total_capturing_duration_minutes) + "','" + most_frequent_protocol + "','" + most_frequent_src_ip + "','" + most_frequent_dst_ip + "','" + most_frequent_dst_port + "','" + str(userid) + "','" + str(traffic_detection) + "')")


    mysql.connection.commit()

    curInsert.close()


# Data Preparation--------------------------------------

#CODE TO INSERT DATA to tbldashboard
#    curInsert = mysql.connection.cursor()

#    curInsert.execute("INSERT INTO tblanalysis (duration, protocol, sourceipaddress, destinationipaddress, destinationport, user_id) VALUES('" + str(total_capturing_duration_minutes) + "','" + most_frequent_protocol + "','" + most_frequent_src_ip + "','" + most_frequent_dst_ip + "','" + most_frequent_dst_port + "','" + str(userid) + "')")

#    curInsert.execute("INSERT INTO tblDashboardTab (time, aveflowpktsrate, aveflowbytsrate, protcol, sourceipadd, destipadd, alerts, targetports, user_id) VALUES('" + str(start_time) + "','" + str(file_date) + "','" + str(total_capturing_duration_minutes) + "','" + most_frequent_protocol + "','" + most_frequent_src_ip + "','" + most_frequent_dst_ip + "','" + most_frequent_dst_port + "','" + str(userid) + "','" + str(traffic_detection) + "')")


#    mysql.connection.commit()

#    curInsert.close()