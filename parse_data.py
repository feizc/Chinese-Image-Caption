# store the image embedding for training acceleration 
import torch 
from dataset import data_read, data_statics 
from utils import mt_convert_url 
import pickle 
import clip 
import os 
from tqdm import tqdm 
from PIL import Image 
import requests 


from efficientnet_pytorch import EfficientNet  
from utils import get_image_trans 

DATA_STATIC = True 
GPU_FLAG = False 
CLIP_FLAG = False 
device = "cuda:0" if torch.cuda.is_available() else "cpu" 


def main(): 
    data_path = './data' 
    output_path = './data/train.pkl'

    data = data_read(data_path, 0.9) 
    if DATA_STATIC:
        data_statics(data)  

    if CLIP_FLAG == False:
        model = EfficientNet.from_name('efficientnet-b4') 
        model_path = './ckpt/efficientnet/0-gs1110000-checkpoint.pth.tar' 
        param_data = torch.load(model_path, map_location=device) 
        model.load_state_dict(param_data['state_dict'], strict=False)
        preprocess = get_image_trans(train=False) 
        model = model.to(device) 

    else: 
        model, preprocess = clip.load("ckpt/clip/ViT-B-32.pt", device=device) 
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
            if CLIP_FLAG == False:
                image_feature = model.extract_features(image).view(1, -1).cpu()
            else:
                image_feature = model.encode_image(image).cpu() 
        print(image_feature.size())
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


def label_combine(data_path='./data/label'): 
    file_list = os.listdir(data_path) 
    data_list = []
    for file_name in file_list: 
        if '.pkl' not in file_name: 
            continue 
        file_path = os.path.join(data_path, file_name) 
        with open(file_path, 'rb') as f: 
            while True: 
                try:
                    data = pickle.load(f) 
                except:
                    break 
                url, txt = data[3].strip().split('\t')
                data_list.append([txt, url]) 
    print('total number:', len(data_list)) 
    print(data_list[:100])
    return data_list 





if __name__ == '__main__':
    #main() 
    # pickle_data_combine()
    label_combine()