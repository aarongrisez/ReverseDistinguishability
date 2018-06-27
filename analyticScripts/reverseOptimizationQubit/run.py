import worker
import glob
from argparse import ArgumentParser

DATA_PATH = './Data/'
PARAM_PATH = './Params/'

def checkForParamsFiles():
    """
    Checks if there are remaining parameter files
    """
    files = glob.glob(PARAM_PATH + 'param_chunk*.npy')
    if len(files) >= 1:
        return True, files[0]
    else:
        return False

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', help='Depth of computation, int', default=4, type=int)
    parser.add_argument('-dp', help='Data directory path, str', default=DATA_PATH, type=str)
    parser.add_argument('-sn', help='Whether to calculate for only one copy of n, bool', default=False, type=bool)
    parser.add_argument('-wn', help='Worker number, int', default=1, type=int)
    a = parser.parse_args()
    params_left, first_file = checkForParamsFiles()
    worker_instance = worker.Worker(depth=a.d, data_path=a.dp, worker_number=a.wn, single_n=a.sn)
    while params_left:
        worker_instance.setParamFile(first_file)
        worker_instance.workFunction()
        params_left = checkForParamsFiles()
