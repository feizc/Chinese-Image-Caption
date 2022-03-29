import os 
import numpy as np 


def filter(lines, threshold=0.5, key_word='dp.cmt'): 
    refine_lines = [] 
    for line in lines: 
        if key_word not in line: 
            continue 
        line = line.split('\t') 
        p = float(line[0]) 
        if p < threshold: 
            continue 
        refine_lines.append([line[1], line[2]]) 
    return refine_lines 


def data_read(data_path): 
    file_list = os.listdir(data_path) 
    data = [] 
    for file in file_list:
        file_path = os.path.join(data_path, file) 
        with open(file_path, 'r') as f: 
            lines = filter(f.readlines()) 
        data += lines 
    return data 


def data_statics(data): 
    len_list = [] 
    for iterm in data: 
        len_list.append(len(iterm[0])) 
    print('dataset size: ', len(data)) 
    print('average_length: ', np.mean(len_list))


if __name__ == '__main__': 
    data_path = 'data' 
    data = data_read(data_path) 
    data_statics(data)