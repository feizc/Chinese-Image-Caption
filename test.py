from PIL import Image
from cv2 import line
import requests
from thinc import numpy 
from dataset import filter
from random import sample 

# generate test set 
def case_selection(data_path, number): 
    with open(data_path, 'r', encoding='utf-8') as f: 
        lines = filter(f.readlines(), threshold=0.9, min_len=15) 
    return sample(lines, number) 

if __name__ == '__main__': 
    data_path = './data/part-00044'  
    case_selection(data_path, 50)