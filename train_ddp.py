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
import torch.distributed as dist

from dataset import CommentDataset, data_read, FastCommentDataset
from model import CaptionModel 
from torch.utils.data import Dataset, DataLoader 
from utils import accuracy_compute
from torch.nn.parallel import DistributedDataParallel 
#from apex import amp
#from apex.parallel import convert_syncbn_model
#from apex.parallel import DistributedDataParallel 



FAST_TRAIN = True 
 
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
    
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    if args.local_rank != -1:
        if args.fp16:
            model = DistributedDataParallel(model, delay_allreduce=True)
        else: 
            print('no fp16.')
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)


    if args.resume_last: 
        fname = os.path.join(args.output_dir, "latest.pt") 
        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            #model.load_state_dict(data['state_dict'], strict=False)
            model.load(data['state_dict'], map_location=cuda)
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
            tokens, mask, img_features = tokens.to(args.device), mask.to(args.device), img_features.to(args.device, dtype=torch.float32) 
            outputs = model(tokens, img_features, mask) 
            logits = outputs.logits[:, args.prefix_length - 1: -1] 
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward() 
            
            if idx % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
            running_loss += loss.item() 
            running_acc += accuracy_compute(logits, tokens)
            progress.set_postfix({"loss": running_loss / (idx + 1), "acc": running_acc / (idx + 1)})
            progress.update()
            if idx % 10000 == 0 and args.local_rank == 0:
                torch.save({
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    'epoch': epoch,
                    "acc": running_acc / (idx + 1),
                    #'state_dict': model.state_dict(),
                    'state_dict': getattr(model, 'module', model),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(args.output_dir, "latest.pt"),
                )
            #if idx == 3:
            #    break 
        progress.close() 
        if running_acc > best_acc: 
            best_acc = running_acc 
            copyfile(os.path.join(args.output_dir, "latest.pt"), os.path.join(args.output_dir, "best.pt"))
        # break 



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--output_dir', default='./ddp') 
    parser.add_argument('--fast_data_path', default='./data/train_combine.pkl')
    parser.add_argument('--prefix_length', type=int, default=10) 
    parser.add_argument('--max_length', type=int, default=40) 
    parser.add_argument('--num_layers', type=int, default=8) 
    parser.add_argument('--batch_size', type=int, default=5) 
    parser.add_argument('--prefix_length_clip', type=int, default=10) 
    parser.add_argument('--lr', type=float, default=2e-5) 
    parser.add_argument('--epochs', type=int, default=30) 
    parser.add_argument('--warmup_steps', type=int, default=5000) 
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer') 
    parser.add_argument('--resume_last', type=bool, default=True) 
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)") 
    parser.add_argument("--fp16", type=int, default=0,
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args() 
    
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)

    gpt2_path = 'ckpt/gpt2' 
    tokenizer = BertTokenizer.from_pretrained(gpt2_path) 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 

    clip_model, preprocess = clip.load("ckpt/clip/ViT-B-32.pt", device=args.device) 

    if FAST_TRAIN == True: 
        dataset = FastCommentDataset(args.fast_data_path, tokenizer, args, args.device) 
    else:
        data = data_read(args.data_path) 
        dataset = CommentDataset(data, tokenizer, preprocess, clip_model, args, device) 

    train_sampler = torch.utils.data.sampler.RandomSampler(dataset) if args.local_rank == -1 else torch.utils.data.distributed.DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
    
    prefix_dim = 512
    model =  CaptionModel(args.prefix_length, tokenizer=tokenizer, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                            num_layers=args.num_layers, mapping_type=args.mapping_type) 
    if args.fp16:
        model = convert_syncbn_model(model)
        
    model = model.to(args.device) 

    train(train_dataloader, model, args)


if __name__ == '__main__':
    main()



