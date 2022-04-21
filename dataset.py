import os
import pickle 
import numpy as np 
from transformers import BertTokenizer 
from torch.utils.data import Dataset 
import torch 
from PIL import Image 
import requests 
import io 


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import mt_convert_url 

GPU_FLAG = False 
SPECIAL_TOKENS = ["[bos]", "[eos]",] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]"}


def filter(lines, min_len=10, threshold=0.5, key_word='dp.cmt'): 
    refine_lines = [] 
    for line in lines: 
        if key_word not in line: 
            continue 
        line = line.split('\t') 
        p = float(line[0]) 
        if p < threshold: 
            continue 
        if len(line[1]) < min_len:
            continue
        refine_lines.append([line[1], line[2]]) 
    return refine_lines 


def data_read(data_path, threshold=0.9, min_len=15): 
    file_list = os.listdir(data_path) 
    data = [] 
    for file in file_list: 
        if 'part' not in file:
            continue 
        file_path = os.path.join(data_path, file) 
        with open(file_path, 'r') as f: 
            lines = filter(f.readlines(), min_len, threshold)  
        data += lines 
    return data 


def data_statics(data): 
    len_list = [] 
    for iterm in data: 
        len_list.append(len(iterm[0])) 
    print('dataset size: ', len(data)) 
    print('average_length: ', np.mean(len_list))



class CommentDataset(Dataset): 
    def __init__(self, data, tokenizer, image_preprocess, image_encoder, args, device):
        self.data = data 
        self.tokenizer = tokenizer 
        self.image_preprocess = image_preprocess 
        self.image_encoder = image_encoder 
        self.max_length = args.max_length 
        self.prefix_length = args.prefix_length 
        self.device = device 
        self.flag = args.clip_flag
        self.bos, self.eos = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    def __len__(self): 
        return len(self.data) 

    def pad_tokens(self, text_ids): 
        padding = self.max_length - text_ids.shape[0] 
        if padding > 0: 
            text_ids = torch.cat((text_ids, torch.zeros(padding, dtype=torch.int64)-1)) 
        elif padding < 0: 
            text_ids = text_ids[:self.max_length] 
        mask = text_ids.ge(0) 
        text_ids[~mask] = 0 
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return text_ids, mask

    def __getitem__(self, index): 
        img_txt_pair = self.data[index] 
        url = img_txt_pair[1] 
        if GPU_FLAG == True: 
            url = mt_convert_url(url)
        txt = img_txt_pair[0] 
        image = Image.open(requests.get(url, stream=True).raw)
        image = self.image_preprocess(image).unsqueeze(0).to(self.device) 
        if self.flag == True:
            image_features = self.image_encoder.encode_image(image).squeeze(0)
        else:
            image_features = self.image_encoder.extract_features(image).view(-1)
        # image_features = torch.zeros((1, 512)).float().squeeze(0)
        txt_ids = torch.Tensor([self.bos] + tokenize(txt, self.tokenizer) + [self.eos]).long() 
        txt_ids, mask = self.pad_tokens(txt_ids)
        return image_features, txt_ids, mask 
    


class LabelDataset(Dataset): 
    def __init__(self, data, tokenizer, image_preprocess, image_encoder, args, device):
        self.data = data 
        self.tokenizer = tokenizer 
        self.image_preprocess = image_preprocess 
        self.image_encoder = image_encoder 
        self.max_length = args.max_length 
        self.prefix_length = args.prefix_length 
        self.device = device 
        self.flag = args.clip_flag
        self.bos, self.eos = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        self.left, self.right = tokenizer.convert_tokens_to_ids(['「', '」'])

    def __len__(self): 
        return len(self.data) 

    def pad_tokens(self, text_ids): 
        padding = self.max_length - text_ids.shape[0] 
        if padding > 0: 
            text_ids = torch.cat((text_ids, torch.zeros(padding, dtype=torch.int64)-1)) 
        elif padding < 0: 
            text_ids = text_ids[:self.max_length] 
        mask = text_ids.ge(0) 
        text_ids[~mask] = 0 
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return text_ids, mask

    def __getitem__(self, index): 
        img_txt_pair = self.data[index] 
        url = img_txt_pair[1] 
        if GPU_FLAG == True: 
            url = mt_convert_url(url)
        txt = img_txt_pair[0] 
        try:
            image = Image.open(requests.get(url, stream=True).raw)
            image = self.image_preprocess(image).unsqueeze(0).to(self.device) 
            if self.flag == True:
                image_features = self.image_encoder.encode_image(image).squeeze(0)
            else:
                image_features = self.image_encoder.extract_features(image).view(-1)
        except:
            print(url)
            image_features = torch.zeros((1, 216832)).float().squeeze(0).to(self.device)
        # image_features = torch.zeros((1, 512)).float().squeeze(0)
        txt_ids = torch.Tensor([self.bos, self.left] + tokenize(txt, self.tokenizer) + [self.right, self.eos]).long() 
        txt_ids, mask = self.pad_tokens(txt_ids)
        return image_features, txt_ids, mask 




class FastLabelDataset(Dataset): 
    def __init__(self, tokenizer, image_preprocess, image_encoder, args, device):
        self.data_path = args.data_path 
        self.file_name_list = os.listdir(self.data_path) 
        self.file_name_list = [x for x in self.file_name_list if '.pkl' in x]  
        self.file_idx = 0 
        self.file_path = os.path.join(self.data_path, self.file_name_list[self.file_idx]) 
        self.file = open(self.file_path, 'rb')

        self.tokenizer = tokenizer 
        self.image_preprocess = image_preprocess 
        self.image_encoder = image_encoder 
        self.max_length = args.max_length 
        self.prefix_length = args.prefix_length 
        self.device = device 
        self.flag = args.clip_flag
        self.bos, self.eos = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        self.left, self.right = tokenizer.convert_tokens_to_ids(['「', '」'])

    def __len__(self): 
        return 10

    def pad_tokens(self, text_ids): 
        padding = self.max_length - text_ids.shape[0] 
        if padding > 0: 
            text_ids = torch.cat((text_ids, torch.zeros(padding, dtype=torch.int64)-1)) 
        elif padding < 0: 
            text_ids = text_ids[:self.max_length] 
        mask = text_ids.ge(0) 
        text_ids[~mask] = 0 
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return text_ids, mask

    def __getitem__(self, index): 
        try:
            data = pickle.load(self.file) 
            image = io.BytesIO(data[2]) 
            image = Image.open(image) 
        except: 
            self.file.close() 
            self.file_idx += 1 
            self.file_path = os.path.join(self.data_path, self.file_name_list[self.file_idx]) 
            self.file = open(self.file_path, 'rb') 

            data = pickle.load(self.file) 
            image = io.BytesIO(data[2]) 
            image = Image.open(image) 

        
        txt = data[3].strip().split('\t')[1]
        try:
            image = self.image_preprocess(image).unsqueeze(0).to(self.device) 
            if self.flag == True:
                image_features = self.image_encoder.encode_image(image).squeeze(0)
            else:
                image_features = self.image_encoder.extract_features(image).view(-1)
        except:
            print(self.file_name_list[self.file_idx])
            image_features = torch.zeros((1, 216832)).float().squeeze(0).to(self.device)
        # image_features = torch.zeros((1, 512)).float().squeeze(0)
        txt_ids = torch.Tensor([self.bos, self.left] + tokenize(txt, self.tokenizer) + [self.right, self.eos]).long() 
        txt_ids, mask = self.pad_tokens(txt_ids)
        return image_features, txt_ids, mask 





class FastCommentDataset(Dataset): 
    def __init__(self, data_path, tokenizer, args, device): 
        self.tokenizer = tokenizer 
        with open(data_path, 'rb') as f: 
            all_data = pickle.load(f) 
        self.image_embeddings = all_data['image_embedding'] 
        self.captions = all_data['captions']
        self.max_length = args.max_length 
        self.prefix_length = args.prefix_length 
        self.device = device 
        self.bos, self.eos = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS) 
    
    def __len__(self): 
        return len(self.captions) 

    def pad_tokens(self, text_ids): 
        padding = self.max_length - text_ids.shape[0] 
        if padding > 0: 
            text_ids = torch.cat((text_ids, torch.zeros(padding, dtype=torch.int64)-1)) 
        elif padding < 0: 
            text_ids = text_ids[:self.max_length] 
        mask = text_ids.ge(0) 
        text_ids[~mask] = 0 
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return text_ids, mask
    
    def __getitem__(self, index):
        # image_features = self.image_embeddings[self.captions[index]['image_embedding']] 
        image_features = self.image_embeddings[index] 
        txt = self.captions[index]['caption'] 
        txt_ids = torch.Tensor([self.bos] + tokenize(txt, self.tokenizer) + [self.eos]).long() 
        txt_ids, mask = self.pad_tokens(txt_ids)
        return image_features, txt_ids, mask 



def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj) 



def collate_fn(batch, pad_token=0): 
    def padding(seq, pad_token): 
        max_len = max([i.size(0) for i in seq]) 
        if len(seq[0].size()) == 1: 
            result = torch.ones((len(seq), max_len)).long() * pad_token 
        else: 
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float() * pad_token 
        for i in range(len(seq)): 
            result[i, :seq[i].size(0)] = seq[i] 
        return result 
    
    img_list, txt_list = [], [] 
    for i in batch: 
        img_list.append(i[0]) 
        txt_list.append(i[1]) 
    txt_list = padding(txt_list, pad_token) 
    img_list = torch.stack(img_list)
    return img_list, txt_list 



