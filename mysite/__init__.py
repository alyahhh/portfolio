import os
from flask import Flask, render_template, request, session, redirect, jsonify
from flask_mysqldb import MySQL
from datetime import datetime, timedelta
import pytz  # Import the pytz module
import json

try:
    from flask_cors import CORS
except ImportError:
    # If flask_cors is not installed, provide a warning
    print("Warning: flask_cors module not found. Cross-origin resource sharing (CORS) may not work.")
    CORS = lambda app: app  # Define a dummy CORS function that does nothing

app = Flask(__name__)
CORS(app)

app.config['MYSQL_HOST'] = 'detech.mysql.pythonanywhere-services.com'
app.config['MYSQL_USER'] = 'detech'
app.config['MYSQL_PASSWORD'] = 'dbdetechpassword'
app.config['MYSQL_DB'] = 'detech$dbdetech'


secret_key = os.urandom(24)
app.secret_key = secret_key

mysql = MySQL(app)

# Set the timezone to Asia/Manila (Philippines timezone)
philippines_timezone = pytz.timezone('Asia/Manila')

@app.route('/')
@app.route('/', methods=['GET','POST'])

def index():

    if request.method == 'POST':

        username = request.form['loginPageUsername']
        password = request.form['loginPagePassword']

        cur = mysql.connection.cursor()

        try:
            # Execute the SELECT query
            cur.execute("SELECT user_id, username, password, email FROM users WHERE username = %s AND password = %s", (username, password))

            # Fetch the result (assuming you are interested in the result)
            user = cur.fetchone()

            if user:
                # User with the given username and password exists
                session["userLoggedIn"] = username
                session["userIdLoggedIn"] = user[0]

                # Get the current UTC time
                utc_now = datetime.utcnow()

                # Convert UTC time to Philippines timezone
                philippines_time = utc_now.replace(tzinfo=pytz.utc).astimezone(philippines_timezone)

                # Print or use the Philippines time as needed
                print("Philippines Time:", philippines_time)

                # Use the stored procedure to log the login attempt with the Philippines time
                cur.callproc('LogLoginAttempt', (username, password, 0, 0))

                # Commit the changes
                mysql.connection.commit()

                return redirect('/home')
            else:
                session["userLoggedIn"] = ""
                # User does not exist, handle accordingly (e.g., return an error message)
                alert_message = "Login failed. Please check your username and password."
                return render_template('index.html', alert_message=alert_message)

        except Exception as e:
            # Handle exceptions and log them for debugging
            print("Error:", str(e))
            mysql.connection.rollback()
            # Handle the error response as needed

        finally:
            # Close the cursor after executing the queries
            cur.close()

    session["userLoggedIn"] = ""
    return render_template(
        'index.html',
        title='Home Page',
    )



@app.route('/register', methods=['GET','POST'])

def register():

    if request.method == 'POST':
        username = request.form['registerPageUsername']
        email = request.form['registerPageEmail']
        password = request.form['registerPagePassword']

        cur = mysql.connection.cursor()

        cur.execute("SELECT * FROM users WHERE username = '"  + username + "'")

        user = cur.fetchone()

        cur.close()


        if user:
            alert_message = "Username already exists!"
            return render_template('register.html', alert_message=alert_message)
        else:
            # INSERT NEW USER
            curInsert = mysql.connection.cursor()

            curInsert.execute("INSERT INTO users (username, password, email) VALUES('" + username + "','" + password + "','" + email + "')")

            mysql.connection.commit()

            curInsert.close()

            # INSERT ENTRY in tblSwitch once user is created
            curInsertTblSwitch = mysql.connection.cursor()

            curInsertTblSwitch.execute("INSERT INTO tblSwitch VALUES (0, NULL, (SELECT MAX(user_id) FROM users))")

            mysql.connection.commit()

            curInsertTblSwitch.close()

            alert_message = "Account Successfully Registered!"
            return render_template('register.html', alert_message=alert_message)

    return render_template(
        'register.html',
        title='Register',
    )



@app.route('/home')
def home():
    userLoggedIn = session["userLoggedIn"];

    return render_template(
        'home.html',
        title='Home',
        userLoggedIn=userLoggedIn
    )



@app.route('/about')
def about():
    userLoggedIn = session["userLoggedIn"];

    return render_template(
        'about.html',
        title='About',
        userLoggedIn=userLoggedIn
    )



@app.route('/detect')
def detect():
    userLoggedIn = session["userLoggedIn"];
    userIdLoggedIn = session["userIdLoggedIn"];

    if userLoggedIn == "":
        return redirect('/')
    else:
        return render_template(
            'detect.html',
            title='Detect',
            userLoggedIn=userLoggedIn,
            userIdLoggedIn=userIdLoggedIn
        )



@app.route('/abstract')
def abstract():
    userLoggedIn = session["userLoggedIn"];

    return render_template(
        'abstract.html',
        title='Abstract',
        userLoggedIn=userLoggedIn
    )



@app.route('/members')
def members():
    userLoggedIn = session["userLoggedIn"];

    return render_template(
        'members.html',
        title='Members',
        userLoggedIn=userLoggedIn
    )



@app.route('/test123')
def test123():
    userLoggedIn = session["userLoggedIn"];

    return render_template(
        'test123.html',
        title='Bootstrap Grid',
        userLoggedIn=userLoggedIn
    )



@app.route('/analysis')
def analysis():
    userLoggedIn = session["userLoggedIn"];
    userIdLoggedIn = session["userIdLoggedIn"];

    cur = mysql.connection.cursor()
    # Execute the SELECT query
    cur.execute("SELECT dur, protocol, sourceip, destip, destport, starttime, traffic, normal, dos, brutef, botnet, inf, classify FROM tblanalysis WHERE user_id = " + str(userIdLoggedIn) + " ORDER BY id DESC LIMIT 1")

    analysisRecord = cur.fetchone()

    if analysisRecord:
        # User with the given username and password exists
        dur = analysisRecord[0]
        protocol = analysisRecord[1]
        sourceip = analysisRecord[2]
        destip = analysisRecord[3]
        destport = analysisRecord[4]
        starttime = analysisRecord[5]
        traffic = analysisRecord[6]
        normal = analysisRecord[7]
        dos = analysisRecord[8]
        brutef = analysisRecord[9]
        botnet = analysisRecord[10]
        inf = analysisRecord[11]
        classify = analysisRecord[12]

        return render_template(
            'analysis.html',
            title='Analysis',
            userLoggedIn=userLoggedIn,
            userIdLoggedIn=userIdLoggedIn,
            recordExists='1',
            dur=dur,
            protocol=protocol,
            sourceip=sourceip,
            destip=destip,
            destport=destport,
            starttime=starttime,
            traffic=traffic,
            normal=normal,
            dos=dos,
            brutef=brutef,
            botnet=botnet,
            inf=inf,
            classify=classify
        )
    else:
        return render_template(
            'analysis.html',
            title='Analysis',
            userLoggedIn=userLoggedIn,
            userIdLoggedIn=userIdLoggedIn,
            recordExists='0',

        )



@app.route('/statistics')
def statistics():
    userLoggedIn = session["userLoggedIn"];

    return render_template(
        'statistics.html',
        title='Statistics',
        userLoggedIn=userLoggedIn
    )



@app.route('/history')
def history():
    userLoggedIn = session["userLoggedIn"]
    userIdLoggedIn = session["userIdLoggedIn"]

    cur = mysql.connection.cursor()
    # Execute the SELECT query
    cur.execute("SELECT `date`, `time`, `sourceip`, `destip`, `traffic`, `classification` FROM tblhistory WHERE user_id = " + str(userIdLoggedIn) + " ORDER BY id DESC ")

    historyRecord = cur.fetchall()

    historyJSONdata = []


    for row in historyRecord:
        tblHistoryDate = str(row[0])
        tblHistoryTime = str(row[1])
        tblHistorySourceip = str(row[2])
        tblHistoryDestip = str(row[3])
        tblHistoryTraffic = str(row[4])
        tblHistoryClassification = str(row[5])

        row_data = {
            'date': tblHistoryDate,
            'time': tblHistoryTime,
            'sourceip': tblHistorySourceip,
            'destip': tblHistoryDestip,
            'traffic': tblHistoryTraffic,
            'classification': tblHistoryClassification
        }

        historyJSONdata.append(row_data)

    cur.close()

    return render_template(
        'history.html',
        title='History',
        userLoggedIn=userLoggedIn,
        userIdLoggedIn=userIdLoggedIn,
        historyJSONdata=json.dumps(historyJSONdata),
    )


@app.route('/dashboard')
def dashboard():
    userLoggedIn = session.get("userLoggedIn")
    userIdLoggedIn = session.get("userIdLoggedIn")

    cur = mysql.connection.cursor()
    # Execute the SELECT query to fetch the latest numalerts
    cur.execute("SELECT numalerts FROM tbldashboardmetrics WHERE user_id = %s ORDER BY id DESC LIMIT 1", (userIdLoggedIn,))
    numalertsRecord = cur.fetchone()

    # Extracting numalerts from fetched record
    numalerts = numalertsRecord[0]

    cur.execute("SELECT total_TCP_count, total_UDP_count FROM tbldashboardmetrics WHERE user_id = %s ORDER BY id DESC LIMIT 1", (userIdLoggedIn,))
    tcp_udp_count_record = cur.fetchone()
    total_tcp_count, total_udp_count = tcp_udp_count_record if tcp_udp_count_record else (0, 0)


    # Execute the SELECT query to fetch the latest 15 records in descending order
    cur.execute("SELECT aveflowpktsrate, aveflowbytsrate, alerts, numalerts FROM tbldashboardmetrics WHERE user_id = %s ORDER BY id DESC LIMIT 15", (userIdLoggedIn,))
    dashboardRecords = cur.fetchall()

    # Get Latest ranking from tblrankings
    cur.execute("SELECT ranking FROM tblrankings WHERE user_id = %s ORDER BY id DESC", (userIdLoggedIn,))
    latestranking = cur.fetchall()

    rankingJSONString = '[';

    for row in latestranking:
        rankingJSONString = rankingJSONString + str(row[0])

    rankingJSONString = rankingJSONString + ']'

    # Extracting data from fetched records
    aveflowpktsrates = [record[0] for record in dashboardRecords]
    aveflowbytsrates = [record[1] for record in dashboardRecords]
    alerts = [record[2] for record in dashboardRecords]

    # Initialize chart data with zeros
    chartDay = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    chartPackets = [0] * len(chartDay)

    chartHours = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    chartTraffic = [0] * len(chartHours)

    chartWeek = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    chartNPackets = [0] * len(chartWeek)

    chartHUT = ["TCP", "UDP"]
    chartDt = [0] * len(chartHUT)

    # Update chart data with actual values if available
    if aveflowbytsrates:
        chartTraffic[:len(aveflowbytsrates)] = aveflowbytsrates
    if aveflowpktsrates:
        chartNPackets[:len(aveflowpktsrates)] = aveflowpktsrates
    if alerts:
        chartPackets[:len(alerts)] = alerts

    return render_template(
        'dashboard.html',
        title='Dashboard',
        userLoggedIn=userLoggedIn,
        userIdLoggedIn=userIdLoggedIn,
        resultExists='1',
        numalerts=numalerts,
        total_tcp_count=total_tcp_count,
        total_udp_count=total_udp_count,
        chartDay=chartDay,
        chartPackets=chartPackets,
        chartHours=chartHours,
        chartTraffic=chartTraffic,
        chartWeek=chartWeek,
        chartNPackets=chartNPackets,
        chartHUT=chartHUT,
        chartDt=chartDt,
        latestranking=rankingJSONString,
    )


@app.route('/api/startCapture/<int:user_id>', methods=['GET', 'POST', 'OPTIONS'])
def startCapture(user_id):

    curUpdate = mysql.connection.cursor()

    curUpdate.execute("UPDATE tblSwitch SET switchid = 1, switchtime = NOW() WHERE user_id = " + str(user_id))

    mysql.connection.commit()

    curUpdate.close()

    response = {
        'message': 'Capture started.',
        'status': 'Success.'
    }
    return jsonify(response)



@app.route('/api/endCapture/<int:user_id>', methods=['GET', 'POST', 'OPTIONS'])
def endCapture(user_id):

    curUpdate = mysql.connection.cursor()

    curUpdate.execute("UPDATE tblSwitch SET switchid = 0 WHERE user_id = " + str(user_id))

    mysql.connection.commit()

    curUpdate.close()

    response = {
        'message': 'Capture stopped.',
        'status': 'Success.'
    }
    return jsonify(response)



@app.route('/api/getCaptureState/<int:user_id>', methods=['GET', 'POST', 'OPTIONS'])
def getCaptureState(user_id):

    try:
        cur = mysql.connection.cursor()

        cur.execute("SELECT switchid, DATE_FORMAT(switchtime, '%Y-%m-%d %H:%i:%s'), DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s') FROM tblSwitch WHERE user_id = " + str(user_id)  + " LIMIT 1")

        result = cur.fetchone()

        if result is not None:
            switchid = result[0]
            switchtime = result[1]
            serverdatetime = result[2]

            cur.close()

            response = {
                'switchid': switchid,
                'switchtime': switchtime,
                'serverdatetime': serverdatetime,
                'result': 'success'
            }
            return jsonify(response), 200, {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            }
        else:
            response = {
                'result': 'no result',
            }
            return jsonify(response), 200, {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            }
    except Exception as err:
        return jsonify({'error': f"Error: {err}"})



@app.route('/api/saveLatestFileName/<string:filename>/<int:user_id>', methods=['GET', 'POST', 'OPTIONS'])
def saveLatestFileName(filename, user_id):

    curUpdate = mysql.connection.cursor()

    curUpdate.execute("INSERT INTO tblmongodbfiles (filename, user_id) VALUES  ('" + str(filename) + "', '" + str(user_id) + "')")

    mysql.connection.commit()

    curUpdate.close()

    response = {
        'status': 'Success'
    }
    return jsonify(response)