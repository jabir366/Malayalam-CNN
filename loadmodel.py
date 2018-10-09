from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np

from keras.models import load_model
data=np.load('/home/workstation/Desktop/jab/MHCR/datasetfull.npy')
model = load_model('model_MHCR_1.h5')

