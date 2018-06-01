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
