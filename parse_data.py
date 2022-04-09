# store the image embedding for training acceleration 
from sys import flags
import torch 
from dataset import data_read, data_statics 
from utils import mt_convert_url 
import pickle 
import clip 
import os 
from tqdm import tqdm 
from PIL import Image 
import requests 


DATA_STATIC = True 
GPU_FLAG = False 
device = "cuda:0" if torch.cuda.is_available() else "cpu" 


def main(): 
    data_path = './data' 
    output_path = './data/train.pkl'

    data = data_read(data_path, 0.9) 
    if DATA_STATIC:
        data_statics(data)  

    clip_model, preprocess = clip.load("ckpt/clip/ViT-B-32.pt", device=device) 
    all_embeddings = [] 
    all_captions = [] 
    for i in tqdm(range(len(data))): 
        d = {'caption': data[i][0]} 
        url = data[i][1] 
        if GPU_FLAG == True: 
            url = mt_convert_url(url) 
        try: 
            image = Image.open(requests.get(url, stream=True).raw) 
        except: 
            print(url) 
            continue 
        image = preprocess(image).unsqueeze(0).to(device) 
        with torch.no_grad():
            image_feature = clip_model.encode_image(image).cpu() 
        d['image_embedding'] = i 
        all_captions.append(d) 
        all_embeddings.append(image_feature) 

        if (i+1) % 10000 ==0: 
            with open(output_path, 'wb') as f: 
                pickle.dump({'image_embedding': torch.cat(all_embeddings, dim=0), 'captions': all_captions}, f) 
        if i == 5: 
            break 
    
    with open(output_path, 'wb') as f: 
        pickle.dump({'image_embedding': torch.cat(all_embeddings, dim=0), 'captions': all_captions}, f)


# combine the pickle from different sources 
def pickle_data_combine(data_path='./data'): 
    name_list = os.listdir(data_path) 

    image_embedding = [] 
    captions = []
    for name in name_list: 
        if '.pkl' not in name:
            continue 
        if 'combine' in name:
            continue
        name_path = os.path.join(data_path, name)
        with open(name_path, 'rb') as f: 
            all_data = pickle.load(f) 

        image_embedding.append(all_data['image_embedding'])
        captions += all_data['captions'] 
    
    print('combine size: ')
    print(torch.cat(image_embedding, dim=0).size())
    print(len(captions)) 

    output_path = os.path.join(data_path, 'train_combine.pkl') 
    with open(output_path, 'wb') as f: 
        pickle.dump({'image_embedding': torch.cat(image_embedding, dim=0), 'captions': captions}, f)



if __name__ == '__main__':
    # main() 
    pickle_data_combine()