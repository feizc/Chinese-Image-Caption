from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline 


gpt2_path = 'ckpt/gpt2'
tokenizer = BertTokenizer.from_pretrained(gpt2_path) 
model = GPT2LMHeadModel.from_pretrained(gpt2_path) 

text_generator = TextGenerationPipeline(model, tokenizer)  

print(text_generator("这是很久之前的事情了", max_length=100, do_sample=True)) 

