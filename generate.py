from transformers import BertTokenizer 
import argparse 

from model import CaptionModel 


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', default='./test')
    parser.add_argument('--output_dir', default='./ckpt/caption') 
    args = parser.parse_args()

    



if __name__ =="__main__": 
    main()