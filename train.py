import torch 
import clip 
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline 
import argparse

from dataset import CommentDataset, data_read


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--out_dir', default='./ckpt/caption') 
    parser.add_argument('--prefix_length', type=int, default=10) 
    parser.add_argument('--max_length', type=int, default=30)
    args = parser.parse_args()
    gpt2_path = 'ckpt/gpt2' 
    tokenizer = BertTokenizer.from_pretrained(gpt2_path) 

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    clip_model, preprocess = clip.load("ViT-B/32", device=device) 

    data = data_read(args.data_path) 
    dataset = CommentDataset(data, tokenizer, preprocess, clip_model, args) 
    image_features, txt_ids, mask  = dataset[0] 
    print(image_features.size(), txt_ids, mask)



    # text_generator = TextGenerationPipeline(model, tokenizer)  
    # print(text_generator("这是很久之前的事情了", max_length=100, do_sample=True)) 

if __name__ == '__main__':
    main()
