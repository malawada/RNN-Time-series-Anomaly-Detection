import sys, os, pdb
import pandas as pd

bearing_sets = []

for root, dir, files in os.walk('numenta_out'):
    for file in files:
        if("combinedfiles" in file and ".csv" in file):
            bearing_sets.append(root + '\\' + file)

os.makedirs('numenta_dataset', exist_ok=True)
for path in bearing_sets:
    result = pd.read_csv(path)
    path = path.split("\\")
    os.makedirs('numenta_dataset\\'+ path[1], exist_ok=True)
    result.to_csv('numenta_dataset\\'+ path[1] + "\\"+ path[-1], columns=['timestamp', 'raw_score'], header=['timestamp','value'], index=False)