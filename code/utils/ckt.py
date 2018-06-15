import data_utils
from config.Deconfig import Deconfig
import pickle

ll = data_utils.get_some_captions(5000)
max = 0
for l in ll:
    h = l.split()
    if max < len(h):
        max = len(h)

print(max)


