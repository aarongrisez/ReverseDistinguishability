import numpy as np
import hashlib
import os
import logging
import problem
import json

class Worker():

    def __init__(self, depth=0, data_path='', worker_number=0, single_n=False):
        self.depth = depth
        self.data_path = data_path
        self.wnum = worker_number
        self.single_n = single_n

    def setParamFile(self, filename):
        self.filename = filename

    def workFunction(self):
        """
        Creates a worker 
        """
        logging.info('Worker ' + str(self.wnum) + ' started on ' + self.filename)
        self.loadParameters()
        if self.params_list.size == 0:
            logging.info('Encountered empty param file, ignoring...') 
        else:
            self.process()
        os.remove(self.filename)
        print('Finished param chunk')
        logging.info('Worker ' + str(self.wnum) + ' Done, ready for new param file')

    def loadParameters(self):
        self.params_list = np.load(self.filename)

    def process(self):
        if self.single_n:
            self.chunk = np.zeros(4)
            self.matrices = {}
        else:
            self.chunk = np.zeros(self.depth+3)
            self.matrices = {}
        for j in self.params_list:
            self.chunk[0] = j[0]
            self.chunk[1] = j[0]
            self.chunk[2] = j[1]
            self.matrices['r1']=j[0]
            self.matrices['r2']=j[0]
            self.matrices['theta']=j[1]
            if self.single_n:
                self.chunk[4] = problem.calculateSingleN(j, self.depth)
            else:
                soln = problem.calculate(j, self.depth)
                self.chunk[3:self.depth+3] = soln[0]
                self.matrices['sequence'] = soln[1]
            self.saveData()

    def saveData(self):
        """
        Saves data in chunks
        """
        #Write to log "saving Chunk", time, and system spec
        #Save to file
        ID = hashlib.sha1()
        ID.update(np.array2string(self.chunk).encode('utf-8'))
        chunk_file = self.data_path + 'chunk' + '_' + ID.hexdigest()
        np.save(chunk_file + '.npy', self.chunk)
        with open(chunk_file + '.json', 'w') as jsonfile:
            json.dump(self.matrices, jsonfile)
