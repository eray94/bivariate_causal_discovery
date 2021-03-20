import numpy as np
import torch
from torch.autograd import Variable
from typing import Tuple

from src.constant import ModelConstants, Random, Paths

class DataPreProcess:
    @staticmethod
    def read_dataset(dataset_name: str) -> np.ndarray:
        """Read and convert txt files.

                Args:
                    dataset_name (str): Name of the dataset file.
                Returns:
                    Array: Regression data.
        """
        dataset = []

        file_num = float(dataset_name.split("pair")[1].split(".txt")[0])
        data = open(Paths.RAW_DATA_RELATIVE_PATH + str(dataset_name), 'r')
        for line in data:
            k = line.split(" ")
            if file_num % 2 == 1:
                dataset.append([float(k[0]), float(k[1])])  # causal relation from x to y
            else:
                dataset.append([float(k[1]), float(k[0])])  # causal relation from y to x
        data.close()

        return np.array(dataset)

    @staticmethod
    def get_test_train_split(dataset: np.ndarray) -> Tuple:
        """Split raw data into train and test splits.

                Args:
                    dataset (np.ndarray): Data array.
                Returns:
                    Tuple: Train and test splits.
        """

        test_size = int(len(dataset) * ModelConstants.TEST_RATIO)

        np.random.seed(Random.SEED)
        np.random.shuffle(dataset)

        test = dataset[:test_size]
        train = dataset[test_size:]

        x_train = Variable(torch.FloatTensor(train[:, 0]).unsqueeze(dim=-1))
        y_train = Variable(torch.FloatTensor(train[:, 1]).unsqueeze(dim=-1))
        x_test = Variable(torch.FloatTensor(test[:, 0]).unsqueeze(dim=-1))
        y_test = Variable(torch.FloatTensor(test[:, 1]).unsqueeze(dim=-1))

        return x_train, y_train, x_test, y_test
