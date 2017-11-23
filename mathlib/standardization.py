import numpy as np
import pickle


class pyStandardScaler():
    """
    Class pyStandardScaler provides methods
    for classical standardization of data set:
    x_std = (x - E(x)) / sigma(x)
    The result was proved by comparison with
    result of StandardScaler from sklearn.preprocessing.
    """
    def __init__(self):
        self.__mean = 0
        self.__sigma = 1

    def fit_std(self, data):
        self.__mean = data.mean(axis=0)
        self.__sigma = data.std(axis=0)

    def standardize(self, data):
        if self.__sigma.sum() != 0:
            data_std = (data - self.__mean) / self.__sigma
        else:
            data_std = (data - self.__mean)
        return data_std

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        # print("Load standard scaler...")
        try:
            with open(path, 'rb') as f:
                sc = pickle.load(f)
                self.__mean = sc.__mean
                self.__sigma = sc.__sigma
        except:
            sc = None

    def get_param(self):
        return self.__mean, self.__sigma

    # def set_param(self, ):


    # def normalize(self, data):
    #     if data.min() < 0:
    #
    #     else:
    #     data_norm = data