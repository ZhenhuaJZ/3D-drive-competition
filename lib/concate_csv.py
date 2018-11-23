import pandas as pd
import os
from functools import reduce
from tqdm import tqdm

file_path = "./TrainSet"
# file_path = "./TestSet"

def concate_csv(file_path):

    with open(os.path.join("pts.txt")) as f: #for TrainSet 
    # with open(os.path.join("test_pts.txt")) as f: #for TestSet
        txt = f.readlines()
        txt = list(map(lambda s: s.strip(), txt))

    for file in tqdm(txt):
        df1 = pd.read_csv(os.path.join(file_path, 'pts', file))
        df2 = pd.read_csv(os.path.join(file_path, 'intensity', file))
        df3 = pd.read_csv(os.path.join(file_path, 'category', file)) #fro TrainSet
        dfs = [df1, df2, df3] #for TrainSet
        # dfs = [df1, df2] #for TestSet
        dff = pd.concat(dfs, axis =1)
        dff.to_csv(os.path.join(file_path, 'all_data', file), header = ["x","y","z","i","c"], index = None)
        # dff.to_csv(os.path.join(file_path, 'all_data', file), header = ["x","y","z","i"], index = None)

concate_csv(file_path)
