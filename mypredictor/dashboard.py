import pandas as pd
import re
import json
from flask_mysqldb import MySQL
from flask import Flask
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'detech.mysql.pythonanywhere-services.com'
app.config['MYSQL_USER'] = 'detech'
app.config['MYSQL_PASSWORD'] = 'dbdetechpassword'
app.config['MYSQL_DB'] = 'detech$dbdetech'

mysql = MySQL(app)


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
        df = pd.read_csv(raw_csv_data)

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

        df = df.applymap(clean_value)
        df = df.applymap(remove_first_symbol)

        nan_counts = df.isnull().sum().sum()

        if nan_counts > 0:
            logger.info(f"The CSV data has {nan_counts} NaN values. Filling with 0...")
            df = df.fillna(0)
            logger.info("NaN values filled.")
        else:
            logger.info("The CSV data has no NaN values.")

        prepped_csv = df.drop_duplicates(keep='first')
        logger.info("Duplicate rows removed.")

        return prepped_csv
    except Exception as e:
        logger.error(f"Error preparing data for prediction: {e}")
        raise


#============================================= Dashboard code starts here ==============================================================

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

def top_ports(raw_csv_data):
    df = raw_csv_data
    sum_flow_byts = df.groupby('dst port')['flow byts s'].sum()
    sum_flow_byts_df = sum_flow_byts.reset_index(name='sum of flow byts s')
    sum_flow_byts_df = sum_flow_byts_df.sort_values(by='sum of flow byts s', ascending=False)
    json_ports = sum_flow_byts_df.set_index('dst port')['sum of flow byts s'].to_dict()

    return json_ports


def dashboard_results(csv_prep): #csv_data_detect):

    df = csv_prep

    # PACKET CAPTURE
    df['flow pkts s'] = pd.to_numeric(df['flow pkts s'], errors='coerce')
    ave_flowpacket_rate = df['flow pkts s'].mean()

    # TRAFFIC PATTERNS
    df['flow byts s'] = pd.to_numeric(df['flow byts s'], errors='coerce')
    ave_flowbytes_rate = df['flow byts s'].mean()

     # ALERTS
    ##alert = mal_alerts(csv_data_detect)

    # PROTOCOL DISTRIBUTION
    value_counts = df['protocol'].value_counts().to_dict()
    json_protocol = json.dumps(value_counts, indent=4) #json format output

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

    #RESULTS:
    print("ave_flowpacket_rate", ave_flowpacket_rate)
    print("ave_flowbytes_rate", ave_flowbytes_rate)
    print("json_protocol", json_protocol)
    print(json_ip_ports)


#============================================= Dashboard code starts here ==============================================================



def dashboard_results(document_id, user_id, csv_prep):
    try:
        df = csv_prep
        df['flow pkts s'] = pd.to_numeric(df['flow pkts s'], errors='coerce')
        ave_flowpacket_rate = df['flow pkts s'].mean()
        df['flow byts s'] = pd.to_numeric(df['flow byts s'], errors='coerce')
        ave_flowbytes_rate = df['flow byts s'].mean()

        # Query MySQL for protocol and p_freq data
        cur = mysql.connection.cursor()
        cur.execute("SELECT protocol, p_freq FROM tbldashboardmetrics WHERE document_id = %s AND user_id = %s", (document_id, user_id))
        data = cur.fetchall()
        cur.close()

        # Process the fetched data
        protocol_freq = {protocol: p_freq for protocol, p_freq in data}

        src_ip = top_src_ip(df)
        dst_ip = top_dst_ip(df)
        ports = top_ports(df)

        logger.info("Dashboard completed successfully.")
    except Exception as e:
        logger.error(f"Error analyzing CSV file: {e}")
        raise

    curInsertDashboard = mysql.connection.cursor()

    try:
        # Convert protocol_freq dictionary to JSON string
        protocol_freq_json = json.dumps(protocol_freq)

        # Insert calculated metrics along with protocol frequency into MySQL
        curInsertDashboard.execute(
            "INSERT INTO tbldashboardmetrics (aveflowpktsrate, aveflowbytsrate, protocol, p_freq) VALUES(%s, %s, %s, %s)",
            (ave_flowpacket_rate, ave_flowbytes_rate, protocol_freq_json, protocol_freq_json))
        mysql.connection.commit()

        logger.info("Dashboard data inserted successfully.")
    except Exception as dashboard_error:
        mysql.connection.rollback()
        raise dashboard_error
    finally:
        curInsertDashboard.close()