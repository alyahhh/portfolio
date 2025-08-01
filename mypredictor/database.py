import pandas as pd
from pymongo import MongoClient
import base64
from bson import ObjectId
import io
#import time

# MongoDB Atlas connection details
mongo_uri = "mongodb+srv://detech322:mamamg2023@de-tech.tuz0ow4.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(mongo_uri)
db = client['De-TECH2']
collection = db['Raw CSV']



def connect():
    global client, db, collection
    try:
        client = MongoClient(mongo_uri)
        db = client['De-TECH2']
        collection = db['Raw CSV']
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        


def binary_to_csv(binary_data):
    # Decode base64-encoded data to bytes
    decoded_data = base64.b64decode(binary_data)

    # If the content is a CSV text, decode it to a string
    try:
        csv_string = decoded_data.decode('utf-8')
    except UnicodeDecodeError:
        # Handle the case where decoding as UTF-8 is not appropriate
        # For instance, you might want to save the data as a binary file
        # or handle it differently depending on your use case.
        raise

    csv_data = pd.read_csv(io.StringIO(csv_string))
    return csv_data



def retrieve_and_get_csv_from_mongodb(start_object_id_str="000000000000000000000000"):
    current_object_id = ObjectId(start_object_id_str)
    check_interval = 60  # You can adjust this interval as needed

    while True:
        document = collection.find_one({"_id": current_object_id})

        if document:
            binary_data = document.get('csv_file')
            if binary_data:
                csv_data = base64.b64decode(binary_data)
                csv_df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))

                # Convert 'Timestamp' column to datetime and separate into 'Date' and 'Time'
                csv_df['Timestamp'] = pd.to_datetime(csv_df['Timestamp'])
                csv_df['Date'] = csv_df['Timestamp'].dt.date
                csv_df['Time'] = csv_df['Timestamp'].dt.time

                # Process your csv_df here...

                # Check for deletion condition and delete if necessary
                if current_object_id >= ObjectId("00000000000000000000ffff"):
                    delete_old_files_from_mongodb(current_object_id)

                # Increment the ObjectId for the next iteration
                current_object_id = ObjectId(str(current_object_id + 1))

            else:
                print(f"Document with _id {current_object_id} is missing 'csv_file'.")
                current_object_id = ObjectId(str(current_object_id + 1))
                continue
        else:
            print(f"No new data found for ObjectId: {current_object_id}.")
            break

        # Uncomment this for a continuous script
        # time.sleep(check_interval)




def delete_data_from_mongodb(file_id):
    result = collection.delete_one({"_id": ObjectId(file_id)})
    if result.deleted_count == 1:
        print(f"Document with _id {file_id} deleted successfully.")
    else:
        print(f"Document with _id {file_id} not found.")



def delete_old_files_from_mongodb(current_object_id):
    delete_threshold = ObjectId("00000000000000000000ffff")
    while current_object_id >= delete_threshold:
        delete_object_id = current_object_id - 0xffff
        delete_data_from_mongodb(str(delete_object_id))
        current_object_id -= 0xffff
