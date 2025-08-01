# Handling model

import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re
import json


# Load the machine learning models----------------------------------
def load_models(model_paths):
    models = [load(model_path) for model_path in model_paths]
    return models

model_paths = ["./mc_svm.pkl", "./bi_nb.pkl"]
models = load_models(model_paths)

detect_model = models[1]
classify_model = models[0]
# Load the machine learning models----------------------------------


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

# Data Preparation--------------------------------------



# Predict------------------------------------
#def make_detection_prediction(data):
    # Perform prediction using detect_model
    #return detect_model.predict(data)


def make_classification_prediction(data):
    # Perform prediction using classify_model
    #return classify_model.predict(data)
# Predict-------------------------------------------



# Analysis Tab ----------------------------------------------------------
 def perform_analysis(csv_data):

    # Capture start time
    start_time = csv_data['Time'].min()

    # Calculate capturing duration in different units
    capturing_duration_microseconds = csv_data['Flow Duration'].sum()
    capturing_duration_seconds = capturing_duration_microseconds / 1e6
    capturing_duration_minutes = capturing_duration_seconds / 60
    capturing_duration_hours = capturing_duration_minutes / 60

    # Combine durations into a single string
    capturing_duration_combined = f"{int(capturing_duration_hours)}h {int(capturing_duration_minutes % 60)}m {int(capturing_duration_seconds % 60)}s {int(capturing_duration_microseconds % 1e6)}μs"

    # Analyze most frequent values
    most_frequent_protocol = csv_data['Protocol'].mode().values[0]
    most_frequent_src_ip = csv_data['Src Ip'].mode().values[0]
    most_frequent_dst_ip = csv_data['Dst Ip'].mode().values[0]
    most_frequent_dst_port = csv_data['Dst Port'].mode().values[0]
    most_frequent_detection_result = csv_data['Detection Result'].mode().values[0]
    most_frequent_classification_result = csv_data['Classification Result'].mode().values[0]

    # Print the results
    print("Start Time:", start_time)
    print("Capturing Duration (microseconds):", capturing_duration_microseconds)
    print("Capturing Duration (seconds):", capturing_duration_seconds)
    print("Capturing Duration (minutes):", capturing_duration_minutes)
    print("Capturing Duration (hours):", capturing_duration_hours)
    print("Most Frequent Protocol:", most_frequent_protocol)
    print("Most Frequent Source IP:", most_frequent_src_ip)
    print("Most Frequent Destination IP:", most_frequent_dst_ip)
    print("Most Frequent Destination Port:", most_frequent_dst_port)
    print("Most Frequent Detection Result:", most_frequent_detection_result)
    print("Most Frequent Classification Result:", most_frequent_classification_result)

    # Return the results in a JSON format
    result_json = {
        "Start Time": start_time,
        "Capturing Duration (microseconds)": capturing_duration_microseconds,
        "Capturing Duration (seconds)": capturing_duration_seconds,
        "Capturing Duration (minutes)": capturing_duration_minutes,
        "Capturing Duration (hours)": capturing_duration_hours,
        "Most Frequent Protocol": most_frequent_protocol,
        "Most Frequent Source IP": most_frequent_src_ip,
        "Most Frequent Destination IP": most_frequent_dst_ip,
        "Most Frequent Destination Port": most_frequent_dst_port,
        "Most Frequent Detection Result": most_frequent_detection_result,
        "Most Frequent Classification Result": most_frequent_classification_result
    }

    # Save the JSON data to a file named analysis_tab.json
    with open('analysis_tab.json', 'w') as json_file:
        json.dump(result_json, json_file, indent=4)

    return result_json

    # output must in json format  [name: analysis_tab.json]
# Analysis Tab ----------------------------------------------------------



# History Tab ----------------------------------------------------
def history_tab(csv_data):
    # Extract the first "Date" and "Time" values from the analysis
    analysis_result = perform_analysis(csv_data)
    date = analysis_result["Start Time"].split()[0]  # Extracting the date part
    time = analysis_result["Start Time"].split()[1]  # Extracting the time part

    # Determine the traffic status based on the value of 'Most Frequent Detection Result'
    traffic_result = analysis_result["Most Frequent Detection Result"]
    traffic = 'Normal' if traffic_result == 'Normal' else 'Malicious'

    # Get the classification from the 'Most Frequent Classification Result'
    classification = analysis_result["Most Frequent Classification Result"]

    # Return the extracted information as a dictionary
    result_dict = {
        "Date": date,
        "Time": time,
        "Traffic": traffic,
        "Classification": classification
    }

    # Save the extracted information to a file named history_tab.json
    with open('history_tab.json', 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)

    return result_dict
# History Tab ---------------------------------------------------
