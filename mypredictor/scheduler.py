import time
import schedule
from __init__ import checkUnprocessedMongoDBFilesPerUser

if __name__ == "__main__":
    schedule.every(5).seconds.do(checkUnprocessedMongoDBFilesPerUser)

    while True:
        schedule.run_pending()
        time.sleep(60)
