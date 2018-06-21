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


s="A man wearing a helmet , red pants with white stripes going down the sides and a white and red shirt is on a small bicycle using only his hands while his legs are up in the air , while another man wearing a light blue shirt with dark blue trim and black pants with red stripes going up the sides is standing nearby , gesturing toward the first man and holding a small figurine of one of the seven dwarves ."
print(len(s))