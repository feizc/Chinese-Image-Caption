import torch 
import clip 
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline, AdamW, get_linear_schedule_with_warmup
import argparse
import os 
from tqdm import tqdm 
from torch.nn import functional as nnf


from dataset import CommentDataset, data_read
from model import CaptionModel 
from torch.utils.data import Dataset, DataLoader 

device = "cuda" if torch.cuda.is_available() else "cpu" 

def train(train_dataloader, model, args):  
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 
    
    model.train() 
    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_dataloader)
    )

    for epoch in range(args.epochs): 
        print(f">>> Training epoch {epoch}") 
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
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if idx % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "latest.pt"),
                )
            progress.close()
            break 
        break 



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--output_dir', default='./ckpt/caption') 
    parser.add_argument('--prefix_length', type=int, default=10) 
    parser.add_argument('--max_length', type=int, default=30) 
    parser.add_argument('--num_layers', type=int, default=8) 
    parser.add_argument('--batch_size', type=int, default=5) 
    parser.add_argument('--prefix_length_clip', type=int, default=10) 
    parser.add_argument('--lr', type=float, default=2e-5) 
    parser.add_argument('--epochs', type=int, default=10) 
    parser.add_argument('--warmup_steps', type=int, default=5000) 
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    args = parser.parse_args()
    gpt2_path = 'ckpt/gpt2' 
    tokenizer = BertTokenizer.from_pretrained(gpt2_path) 

    clip_model, preprocess = clip.load("ViT-B/32", device=device) 

    data = data_read(args.data_path) 
    dataset = CommentDataset(data, tokenizer, preprocess, clip_model, args) 
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    prefix_dim = 512
    model =  CaptionModel(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                            num_layers=args.num_layers, mapping_type=args.mapping_type)
    model = model.to(device) 

    train(train_dataloader, model, args)


    # text_generator = TextGenerationPipeline(model, tokenizer)  
    # print(text_generator("这是很久之前的事情了", max_length=100, do_sample=True)) 

if __name__ == '__main__':
    main()
