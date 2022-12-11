import contextlib
import csv
import logging
import multiprocess
import numpy as np
import os
import platform
import sys
import xxhash
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

date_format = "%Y-%m-%d-%H-%M-%S"
hash_instance = xxhash.xxh64()


def get_hash(obj_list):
    for obj in obj_list:
        hash_instance.update(obj)
    result = hash_instance.intdigest()
    hash_instance.reset()
    return result


def get_multiprocessing_context():
    if platform.system() != 'Windows':
        ctx = multiprocess.get_context("forkserver")
    else:
        ctx = multiprocess.get_context()
    return ctx


def get_device_placement():
    return os.getenv("RELNET_DEVICE_PLACEMENT", "CPU")


@contextlib.contextmanager
def local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_logger_instance(filename):
    root_logger = logging.getLogger('')
    root_logger.setLevel(logging.INFO)

    has_stdout = False
    has_file = False
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            has_file = True
        if isinstance(handler, StreamHandler):
            has_stdout = True

    if not has_stdout:
        sh = StreamHandler(sys.stdout)
        sh.addFilter(HostnameFilter())
        root_logger.addHandler(sh)

    if filename is not None and not has_file:
        fh = RotatingFileHandler(filename,
                                 maxBytes=32 * 1024 * 1024,
                                 backupCount=10)
        fh.addFilter(HostnameFilter())
        formatter = logging.Formatter(fmt='%(hostname)s %(asctime)s - PID%(process)d %(message)s', datefmt=date_format)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    return root_logger


def setup_metrics_to_csv(filename, metric_list):
    """ Write line of data or header to metric data csv """
    org_filename = filename
    i = 1
    while os.path.exists(filename):
        filename = f'{org_filename[:-4]}_{i}.csv'
        i += 1

    with open(filename, 'w') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(metric_list)
        f_object.close()

    return filename


class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True
