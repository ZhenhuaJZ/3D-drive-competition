import numpy as np
import pandas as pd

txt_path = ""
result_path = ""
sample_path = ""

with open(txt_path) as f:
    txt = f.readlines()
    txt = list(map(lambda s: s.strip(), txt))

for file in txt:
    result = pd.read_csv(os.path.join(result_path, file))
    sample = pd.read_csv(os.path.join(sample_path, file))
    counter = 0
    if len(result) != len(sample):
        print('filename ' + file + " contains sample {}, and result {}".format(len(sample), len(result)))
        counter = counter + 1

print("total {} files having unequal rows".format(counter))
