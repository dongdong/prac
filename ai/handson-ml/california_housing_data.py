from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np
import ssl


def get_housing_data():
    ssl._create_default_https_context = ssl._create_unverified_context
    housing = fetch_california_housing()
    return housing

def get_scaled_housing_data():
    housing = get_housing_data()
    mean = np.mean(housing.data, axis=0)
    std = np.std(housing.data, axis=0)
    housing.data = (housing.data - mean) / std
    return housing

def get_standard_scaler_housing_data():
    housing = get_housing_data()
    scaler = StandardScaler()
    housing.data = scaler.fit_transform(housing.data)
    return housing

class HousingData():
    def __init__(self):
        self.housing = get_housing_data()
        m, n = self.housing.data.shape
        self.housing_data = np.c_[np.ones((m, 1)), self.housing.data]
    def get_data(self):
        return self.housing
    def get_data_shape(self):
        return self.housing.data.shape
    def fetch_all(self):
        m, n = self.housing.data.shape
        X_all = self.housing_data.reshape((m, n + 1))
        y_all = self.housing.target.reshape((m, 1))
        return X_all, y_all
    def fetch_batch(self, index, size):
        m, n = self.housing.data.shape
        start = index * size;
        end = start + size;
        len = size
        if end > m:
            end = m
        #print('range', start, end, end - start)
        X_batch = self.housing_data[start:end].reshape((end - start, n + 1))
        y_batch = self.housing.target[start:end].reshape((end - start, 1))
        return X_batch,y_batch

class HousingDataScaled(HousingData):
    def __init__(self):
        self.housing = get_standard_scaler_housing_data()
        m, n = self.housing.data.shape
        self.housing_data = np.c_[np.ones((m, 1)), self.housing.data]
        
