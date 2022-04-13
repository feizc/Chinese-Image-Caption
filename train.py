import torch 
import clip 
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline, AdamW, get_linear_schedule_with_warmup
import argparse
import os 
from tqdm import tqdm 
from torch.nn import functional as nnf
import random 
import numpy as np 
from shutil import copyfile 

from dataset import CommentDataset, data_read, FastCommentDataset
from model import CaptionModel, CaptionPrefix
from torch.utils.data import Dataset, DataLoader
from parse_data import CLIP_FLAG 
from utils import accuracy_compute 
from efficientnet_pytorch import EfficientNet  
from utils import get_image_trans 

FAST_TRAIN = False 
CLIP_FLAG = False 

device = "cuda" if torch.cuda.is_available() else "cpu" 
SPECIAL_TOKENS = ["[bos]", "[eos]",] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]"}
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


import warnings 
warnings.filterwarnings('ignore')


def train(train_dataloader, model, args):  
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 
    
    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_dataloader)
    ) 

    if args.resume_last: 
        fname = os.path.join(args.output_dir, "latest.pt") 
        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            # torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optimizer.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler']) 
            print('load last ckpt!')


    model.train() 
    for epoch in range(args.epochs): 
        print(f">>> Training epoch {epoch}") 
        running_loss = .0 
        running_acc = .0 
        best_acc = .0 
        progress = tqdm(total=len(train_dataloader), desc='image captioning') 
        for idx, (img_features, tokens, mask) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, img_features = tokens.to(device), mask.to(device), img_features.to(device, dtype=torch.float32) 
            outputs = model(tokens, img_features, mask) 
            logits = outputs.logits[:, args.prefix_length - 1: -1] 
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad() 
            running_loss += loss.item() 
            running_acc += accuracy_compute(logits, tokens)
            progress.set_postfix({"loss": running_loss / (idx + 1), "acc": running_acc / (idx + 1)})
            progress.update()
            if idx % 10000 == 0:
                torch.save({
                    'torch_rng_state': torch.get_rng_state(),
                    # 'cuda_rng_state': torch.cuda.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    'epoch': epoch,
                    "acc": running_acc / (idx + 1),
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(args.output_dir, "latest.pt"),
                )
            if idx == 3:
                break 
        progress.close() 
        if running_acc > best_acc: 
            best_acc = running_acc 
            copyfile(os.path.join(args.output_dir, "latest.pt"), os.path.join(args.output_dir, "best.pt"))
        break 



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--output_dir', default='./ckpt/caption') 
    parser.add_argument('--fast_data_path', default='./data/train_combine.pkl')
    parser.add_argument('--prefix_length', type=int, default=10) 
    parser.add_argument('--max_length', type=int, default=30) 
    parser.add_argument('--num_layers', type=int, default=8) 
    parser.add_argument('--batch_size', type=int, default=5) 
    parser.add_argument('--prefix_length_clip', type=int, default=10) 
    parser.add_argument('--lr', type=float, default=2e-5) 
    parser.add_argument('--epochs', type=int, default=10) 
    parser.add_argument('--warmup_steps', type=int, default=5000) 
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer') 
    parser.add_argument('--resume_last', type=bool, default=False)  
    parser.add_argument('--clip_flag', type=bool, default=False)  
    args = parser.parse_args()
    gpt2_path = 'ckpt/gpt2' 
    tokenizer = BertTokenizer.from_pretrained(gpt2_path) 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
    if CLIP_FLAG == True: 
        image_encoder, preprocess = clip.load("ckpt/clip/ViT-B-32.pt", device=device) 
    else: 
        image_encoder = EfficientNet.from_name('efficientnet-b4') 
        model_path = './ckpt/efficientnet/0-gs1110000-checkpoint.pth.tar' 
        param_data = torch.load(model_path, map_location=device) 
        image_encoder.load_state_dict(param_data['state_dict'], strict=False)
        preprocess = get_image_trans(train=False) 
        image_encoder = image_encoder.to(device) 
    
    if FAST_TRAIN == True: 
        dataset = FastCommentDataset(args.fast_data_path, tokenizer, args, device) 
    else:
        data = data_read(args.data_path) 
        dataset = CommentDataset(data, tokenizer, preprocess, image_encoder, args, device) 
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    prefix_dim = 512
    model =  CaptionPrefix(args.prefix_length, tokenizer=tokenizer, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                            num_layers=args.num_layers, mapping_type=args.mapping_type)
    model = model.to(device) 

    train(train_dataloader, model, args)


if __name__ == '__main__':
    main()


