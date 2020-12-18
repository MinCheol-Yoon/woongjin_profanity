from run_purifier import PuriProcessor , convert_single_example_to_feature
from mask_tokenizer import BertTokenizer , BasicTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from modeling_purifier import BertConfig , BertForSequenceClassification
import tensorflow as tf
import numpy as np


vocab_file = 'vocab_korea.txt'
tokenizer = BertTokenizer(vocab_file)
task_name = 'Puri'
processors = {"puri": PuriProcessor}
task_name = task_name.lower()
processor = processors[task_name]()
 
output_modes = {"puri": "classification"}
output_mode = output_modes[task_name]

config = BertConfig(vocab_size_or_config_json_file= 105879, 
                    hidden_size = 768,
                    num_hidden_layers = 12, 
                    num_attention_heads = 12 , 
                    intermediate_size = 3072)

model = BertForSequenceClassification(config, 2)
model.bert.from_pretrained('bert-base-multilingual-cased')
model.load_state_dict(torch.load('bert_filtering.pt'))

model.cuda()

device_name = tf.test.gpu_device_name()
if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

    

def split_token_get(text,tokenizer,MAX):
  token = tokenizer.tokenize(text,size = False)
  if len(token) > MAX-2:
    token = token[:(MAX-2)]
  token = ["[CLS]"]+ token + ["[SEP]"]
  return token

def get_input_ids(token , tokenizer , MAX):
  segment_ids = [0] * len(token)
  input_ids = tokenizer.convert_tokens_to_ids(token)

  input_mask = [1] * (len(input_ids) -1 )
  input_mask += [0]
  
  padding = [0] * (MAX -len(input_ids) )

  input_ids += padding
  input_mask += padding
  segment_ids += padding

  assert len(input_ids) == MAX
  return input_ids

def get_segment_ids(token, tokenizer, MAX):
  segment_ids = [0] * len(token)
  input_ids = tokenizer.convert_tokens_to_ids(token)
  padding = [0] * (MAX -len(input_ids) )
  
  segment_ids += padding
  assert len(segment_ids) == MAX
  return segment_ids

def get_input_mask(token, tokenizer, MAX):
  
  input_ids = tokenizer.convert_tokens_to_ids(token)

  input_mask = [1]* (len(input_ids) -1 )
  input_mask += [0]

  padding = [0] * (MAX -len(input_ids) )

  input_mask += padding

  assert len(input_mask) == MAX
  return input_mask

def flat_accuracy(preds, labels):
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))

def convert_input_data(sentences):
  tokenized_texts = [split_token_get(ss, tokenizer , 128) for ss in sentences]
  input_ids = [get_input_ids(ss,tokenizer , 128) for ss in tokenized_texts]
  segment_ids = [get_segment_ids(ss,tokenizer, 128 ) for ss in tokenized_texts]
  input_mask = [get_input_mask(ss,tokenizer,128) for ss in tokenized_texts]
  

  return input_ids, input_mask

def test_sentences(sentences):
  

  input, mask = convert_input_data(sentences)
  input = torch.tensor(input)
  mask = torch.tensor(mask)
  gpu_input = input.to(device)
  gpu_mask = mask.to(device)

  model.eval()
  with torch.no_grad():
    outputs = model(input_ids =gpu_input,
                    token_type_ids = None,
                    attention_mask =gpu_mask,
                    labels = None)
  logits = outputs[0]

  logits = logits.detach().cpu().numpy()

  return logits

def softmax(a) :
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y





def filtering_text( a ):
    logit = test_sentences([a])
    a = round( softmax(logit)[0][1] , 3)
    print('욕일 확률: ', a)
    b = softmax(logit)
    result = np.argmax(softmax(logit), axis=1)
    if result == 1:
        print('욕설이 포함되었습니다')
    else:
        print('clean')
    return b, result
