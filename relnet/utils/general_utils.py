import numpy as np
import time
from datetime import datetime

date_format = "%Y-%m-%d-%H-%M-%S"


def find_latest_file(files_dir):
    filenames = [f.name for f in list(files_dir.glob("*.pickle"))]
    latest_file_date = datetime.utcfromtimestamp(0)
    latest_file = None
    for f in filenames:
        datestring = "".join(f.split(".")[:-1])
        file_date = datetime.strptime(datestring, date_format)
        if file_date > latest_file_date:
            latest_file = f
            latest_file_date = file_date
    if latest_file is not None:
        return files_dir / latest_file
    else:
        return None


def is_time_expired(time_started, time_allowed):
    return get_current_time_millis() - time_started > time_allowed


def get_current_time_millis():
    return int(time.time() * 1000)
