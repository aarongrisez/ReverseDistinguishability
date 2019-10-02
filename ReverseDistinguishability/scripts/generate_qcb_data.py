import ReverseDistinguishability.quantities as q
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import shgo
import pandas as pd
import datetime
import os
import logging
import uuid

DATA_PATH = r'C:\Users\Aaron\Documents\CurrentProjects\ReverseDistinguishability\data'
logging.basicConfig(filename=os.path.join(DATA_PATH, f'process_{str(uuid.uuid4())}_data.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s')

df = pd.DataFrame(columns=['theta','r1','r2','result'])

NUM_SEQUENCES = 25
NUM_RUNS = 15
NUM_RSAMPLES = 25
NUM_TSAMPLES = 15

logging.info(f'Params: sequences: {NUM_SEQUENCES}, runs: {NUM_RUNS}, rsamples: {NUM_RSAMPLES}, tsamples: {NUM_TSAMPLES}')

for j in range(NUM_SEQUENCES):
    logging.info(f'Starting sequence')
    for i in range(NUM_RUNS):
        logging.info(f'Starting run {i} in sequence.')
        rsamples = NUM_RSAMPLES
        tsamples = NUM_TSAMPLES
        thetas = np.random.uniform(0, np.pi, tsamples)
        r1vals = np.random.uniform(0, 1, rsamples)
        r2vals = np.random.uniform(0, 1, rsamples)
        results = np.zeros(shape=(len(thetas), len(r1vals), len(r2vals)))
        for k, t in enumerate(thetas):
            logging.info(f'Starting theta = {t}')
            for i, r1 in enumerate(r1vals):
                for j, r2 in enumerate(r2vals):
                    df = df.append({
                            'theta': t, 
                            'r1': r1, 
                            'r2': r2, 
                            'result': q.qcb(r1, r2, t).fun
                        },
                        ignore_index=True)
    logging.info('Dumping to file')
    df.to_hdf(os.path.join(DATA_PATH, 'qcb.h5'), key='s' + datetime.datetime.now().strftime('%d%m%Y%H%M%S'))