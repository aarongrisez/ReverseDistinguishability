import numpy as np
import glob
import os

DATA_PATH = './Data/'

def collectChunks():
    """
    Collects all file names for the chunks in Data/
    """
    return glob.glob(DATA_PATH + 'chunk*.npy')

def loadChunks(chunk_files):
    """
    Loads all chunks into a list of np arrays, passed a list of chunk_files
    """
    temp = []
    for i in chunk_files:
        temp.append(np.load(i))
    return temp

def maxChunkLength(chunks):
    """
    Takes list of chunk arrays, returns the maximum length of any chunk
    """
    maximum = 0
    for i in chunks:
        if i.shape[0] > maximum:
            maximum = i.shape[0]
    return maximum

def makeArray(num_chunks, max_chunk_length):
    """
    Creates consolidating array
    """
    return np.zeros((num_chunks, max_chunk_length))

def fillArray(chunks, array):
    for (i, j) in enumerate(chunks):
        array[i,:] = j
    return array

if __name__ == "__main__":
    chunk_files = collectChunks()
    chunk_arrays = loadChunks(chunk_files) 
    max_chunk_length = maxChunkLength(chunk_arrays)
    array = makeArray(len(chunk_arrays), max_chunk_length)
    filled = fillArray(chunk_arrays, array)
    np.save(DATA_PATH + 'consolidated.npy', filled)
