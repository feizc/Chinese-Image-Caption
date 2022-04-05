import torch 
import clip 
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline, AdamW, get_linear_schedule_with_warmup
import argparse
import os 
from tqdm import tqdm 
from torch.nn import functional as nnf


from dataset import CommentDataset, data_read, FastCommentDataset
from model import CaptionModel 
from torch.utils.data import Dataset, DataLoader 
from utils import accuracy_compute 

FAST_TRAIN = True 

device = "cuda" if torch.cuda.is_available() else "cpu" 
SPECIAL_TOKENS = ["[bos]", "[eos]",] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]"}


def train(train_dataloader, model, args):  
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 
    
    model.train() 
    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_dataloader)
    )

    running_loss = .0 
    runing_acc = .0 
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
            running_loss += loss.item() 
            runing_acc += accuracy_compute(logits, tokens)
            progress.set_postfix({"loss": running_loss / (idx + 1), "acc": runing_acc / (idx + 1)})
            progress.update()
            if idx % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "latest.pt"),
                )
            if idx == 3:
                break 
        progress.close() 
        break 



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--output_dir', default='./ckpt/caption') 
    parser.add_argument('--fast_data_path', default='./data/train.pkl')
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
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 

    clip_model, preprocess = clip.load("ckpt/clip/ViT-B-32.pt", device=device) 

    if FAST_TRAIN == True: 
        dataset = FastCommentDataset(args.fast_data_path, tokenizer, args, device) 
    else:
        data = data_read(args.data_path) 
        dataset = CommentDataset(data, tokenizer, preprocess, clip_model, args, device) 
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    prefix_dim = 512
    model =  CaptionModel(args.prefix_length, tokenizer=tokenizer, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                            num_layers=args.num_layers, mapping_type=args.mapping_type)
    model = model.to(device) 

    train(train_dataloader, model, args)


if __name__ == '__main__':
    main()


