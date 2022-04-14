import torch 
from transformers import BertTokenizer 
import argparse 
import os 
from PIL import Image 
import numpy as np 
import torch.nn.functional as F
from dataset import filter
from random import sample 
import requests 
from tqdm import tqdm
import copy 


from model import CaptionModel 
import clip 
from utils import mt_convert_url 
from efficientnet_pytorch import EfficientNet  
from utils import get_image_trans 



SPECIAL_TOKENS = ["[bos]", "[eos]",] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]"} 
device = "cuda:0" if torch.cuda.is_available() else "cpu"  
# {'greedy', 'sample', 'beam_search'} 
DECODE_STRATEGY = 'sample'
FROM_URL = False 
GPU_FLAG = False  
CLIP_FLAG = False 



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
    prob_list = []
    for i in range(args.max_length): 
        tokens = torch.Tensor(ys).long().unsqueeze(0).to(device)
        outputs = model(tokens, image_features) 
        logits = outputs.logits[:, -1][0] / args.temperature 
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1) 

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1) 

        if (i < args.min_length and prev.item() == eos) or prev.item() == tokenizer.unk_token_id: 
            number = 1 
            while prev.item() == eos or prev.item() == tokenizer.unk_token_id: 
                prev = torch.multinomial(probs, num_samples=1) 
                number += 1
                if number > 100: 
                    break 
        
        if  prev.item() == eos: 
            break 
        
        ys.append(prev.item()) 
        prob_list.append(torch.topk(probs, 1)[0].item())
    return ys, np.mean(prob_list)


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
        



def beam_search(image_features, model, tokenizer, args): 
    bos, eos = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)  
    current_output = [bos] 
    hyplist = [([], 0., current_output)] 
    comp_hyplist = [] 
    best_state = None

    for i in range(args.max_length): 
        new_hyplist = []
        argmin = 0 
        for out, lp, st in hyplist: 
            tokens = torch.Tensor(out).long().unsqueeze(0).to(device)
            outputs = model(tokens, image_features) 
            logp = F.log_softmax(outputs.logits[:,-1], dim=-1) 
            lp_vec = logp.cpu().data.numpy() + lp 
            lp_vec = np.squeeze(lp_vec) 
            if i >= args.min_length: 
                new_lp = lp_vec[eos] + args.penalty * (len(out) + 1) 
                comp_hyplist.append((out, new_lp)) 
                if best_state is None or best_state < new_lp:
                    best_state = new_lp 
            count = 1
            for o in np.argsort(lp_vec)[::-1]: 
                if o == tokenizer.unk_token_id or o == eos:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == args.beam_size:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = copy.deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = copy.deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == args.beam_size:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                count += 1
        hyplist = new_hyplist 
    
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
        return maxhyps
    else:
        return [([], 0)]



def test_data_read(path): 
    img_name_list = os.listdir(path) 
    return img_name_list 


def case_selection(data_path, number): 
    with open(data_path, 'r', encoding='utf-8') as f: 
        lines = filter(f.readlines(), threshold=0.9, min_len=15) 
    return sample(lines, number) 



def list2str(sentence):
    res = ''
    for s in sentence: 
        res += s 
    return res 



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', default='./test')
    parser.add_argument('--model_path', default='./ckpt/caption/latest.pt') 
    parser.add_argument('--token_path', default='./ckpt/gpt2') 
    parser.add_argument('--prefix_length', type=int, default=10)  
    parser.add_argument('--prefix_length_clip', type=int, default=10) 
    parser.add_argument('--num_layers', type=int, default=8) 
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--max_length', type=int, default=30) 
    parser.add_argument("--no_sample", type=bool, default=False, help="Set to use greedy decoding instead of sampling")
    parser.add_argument('--min_length', type=int, default=5) 
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--penalty", type=float, default=0.3, help="elngth penalty")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.token_path) 
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
    
    prefix_dim = 512
    model =  CaptionModel(args.prefix_length, tokenizer=tokenizer, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                            num_layers=args.num_layers, mapping_type=args.mapping_type) 
    
    weight_dict = torch.load(args.model_path, map_location=None) 
    model.load_state_dict(weight_dict['state_dict'], strict=False) 
    model = model.to(device) 
    model.eval() 
    

    if FROM_URL == False: 
        print('test from case')
        img_name_list = test_data_read(args.data_path) 
        for img_name in img_name_list: 
            if '.jpeg' not in img_name: 
                continue 
            img_path = os.path.join(args.data_path, img_name) 
            image = Image.open(img_path) 
            image = preprocess(image).unsqueeze(0).to(device) 
            if CLIP_FLAG == True:
                image_features = image_encoder.encode_image(image).float()
            else:
                image_features = image_encoder.extract_features(image).view(1, -1) 
            if DECODE_STRATEGY == 'greedy': 
                caps = greedy_decode(image_features, model, tokenizer, args) 
            elif DECODE_STRATEGY == 'sample': 
                caps, probs = sample_sequence(image_features, model, tokenizer, args)
                print(tokenizer.batch_decode(caps)) 
            elif DECODE_STRATEGY == 'beam_search':
                caps = beam_search(image_features, model, tokenizer, args) 
                print(caps)
                print(tokenizer.decode(caps[0][0][1:]))
    else: 
        print('test from data') 
        # data_path = './data/part-00044' 
        # samples = case_selection(data_path, 2) 
        

        data_path = 'test/meishi_image_500K.txt' 
        with open(data_path, 'r', encoding='utf-8') as f: 
            lines = f.readlines() 
        
        lines = lines[1:3]
        results = []
        progress = tqdm(total=len(lines), desc='image captioning') 
        with torch.no_grad():
            for line in lines: 
                url = line.split('\t')[0] 
                if GPU_FLAG == True: 
                   url = mt_convert_url(url) 
                image = Image.open(requests.get(url, stream=True).raw)
                image = preprocess(image).unsqueeze(0).to(device)
                image_features = image_encoder.encode_image(image).float()
                if DECODE_STRATEGY == 'greedy': 
                    caps = greedy_decode(image_features, model, tokenizer, args)
                    probs = .0 # not implemented  
                elif DECODE_STRATEGY == 'sample': 
                    caps, probs = sample_sequence(image_features, model, tokenizer, args) 
                line = line.strip() + '\t' + list2str(tokenizer.batch_decode(caps[1:])) + '\t' + str(round(probs, 3)) 
                results.append(line) 
                progress.update()
            with open('result.txt', 'w', encoding='utf-8') as f: 
                for line in results: 
                    f.write(line + '\n')
        

if __name__ =="__main__": 
    main()
