import json
import numpy as np

with open('002191333_kps.json') as f:
    kps = json.load(f)

src_kps = np.array(kps['src_kps']).reshape(19,2)
tgt_kps = np.array(kps['tgt_kps']).reshape(19,2)
print(src_kps)