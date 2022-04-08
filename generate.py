from ast import arg
import torch 
from transformers import BertTokenizer 
import argparse 
import os 
from PIL import Image 
import numpy as np 
import torch.nn.functional as F

from model import CaptionModel 
import clip 


SPECIAL_TOKENS = ["[bos]", "[eos]",] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]"} 
device = "cuda" if torch.cuda.is_available() else "cpu"  
# 'greedy' 
DECODE_STRATEGY = 'sample'



def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits



def sample_sequence(image_features, model, tokenizer, args): 
    bos, eos = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS) 
    ys = [bos] 
    for i in range(args.max_length): 
        tokens = torch.Tensor(ys).long().unsqueeze(0).to(device)
        outputs = model(tokens, image_features) 
        logits = outputs.logits[:, -1][0] / args.temperature 
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1) 

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1) 
        if i < args.min_length and prev.item() == eos:
            while prev.item() == eos:
                prev = torch.multinomial(probs, num_samples=1) 
        
        ys.append(prev.item()) 
        
        if  prev.item() == eos: 
            break 
    return ys 




def greedy_decode(image_features, model, tokenizer, args): 
    bos, eos = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS) 
    ys = [bos] 
    for i in range(args.max_length): 
        tokens = torch.Tensor(ys).long().unsqueeze(0).to(device)
        outputs = model(tokens, image_features) 
        logits = outputs.logits[:, -1]
        logits = logits.cpu().data.numpy() 
        next_word = np.argsort(logits[0])[-1]
        ys.append(next_word) 
        if next_word == eos: 
            break 
    return ys 
        


def test_data_read(path): 
    img_name_list = os.listdir(path) 
    return img_name_list 



def main(): 

    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', default='./test')
    parser.add_argument('--model_path', default='./ckpt/latest.pt') 
    parser.add_argument('--token_path', default='./ckpt/gpt2') 
    parser.add_argument('--prefix_length', type=int, default=10)  
    parser.add_argument('--prefix_length_clip', type=int, default=10) 
    parser.add_argument('--num_layers', type=int, default=8) 
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--max_length', type=int, default=30) 
    parser.add_argument("--no_sample", type=bool, default=False, help="Set to use greedy decoding instead of sampling")
    parser.add_argument('--min_length', type=int, default=5) 
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.token_path) 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
    
    clip_model, preprocess = clip.load("ckpt/clip/ViT-B-32.pt", device=device) 
    prefix_dim = 512
    model =  CaptionModel(args.prefix_length, tokenizer=tokenizer, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                            num_layers=args.num_layers, mapping_type=args.mapping_type) 
    
    weight_dict = torch.load(args.model_path, map_location=None) 
    model.load_state_dict(weight_dict['state_dict'], strict=False) 
    model = model.to(device) 
    model.eval() 

    img_name_list = test_data_read(args.data_path) 
    for img_name in img_name_list: 
        img_path = os.path.join(args.data_path, img_name) 
        image = Image.open(img_path) 
        image = preprocess(image).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image).float()
        if DECODE_STRATEGY == 'greedy': 
            caps = greedy_decode(image_features, model, tokenizer, args) 
        elif DECODE_STRATEGY == 'sample': 
            caps = sample_sequence(image_features, model, tokenizer, args)
            print(tokenizer.batch_decode(caps))




if __name__ =="__main__": 
    main()
