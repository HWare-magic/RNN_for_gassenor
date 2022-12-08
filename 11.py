import pandas as pd
import numpy as np
import torch
import torch.nn as nn
data_path = r'C:\Users\86136\PycharmProjects\pythonProject\AI -learn from zero\data\model_test\test.csv'
data_test = pd.read_csv(data_path,header=None)
print(type(data_test))
data = np.array(data_test)
print(data_test)
print(data)
# data_test_t =pd.DataFrame(data_test.values.T)
# print(data_test_t)
tor_data = torch.Tensor(data)
print(type(tor_data))
print(tor_data)