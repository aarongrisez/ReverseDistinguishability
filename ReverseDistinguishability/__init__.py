import os
import pathlib

__PATH_ENV_VAR = os.environ.get('RD_ROOT_PATH')
if not __PATH_ENV_VAR:
    ROOT_PATH = pathlib.Path('.')
else:
    ROOT_PATH = pathlib.Path(__PATH_ENV_VAR)

paths = {
    'data': ROOT_PATH / 'data',
    'logs': ROOT_PATH / 'logs',
    'params': ROOT_PATH / 'data' / 'params',
}

LOG_FILE_PATH = paths['logs'] / 'execution_log.log'

logging.basicConfig(filename=logfile,
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
