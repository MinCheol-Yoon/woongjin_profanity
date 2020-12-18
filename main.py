from flask import Flask, render_template, request,jsonify
from modeling_purifier import BertConfig , BertForSequenceClassification
import torch
from mask_tokenizer import BertTokenizer
import numpy as np
from flask_cors import CORS, cross_origin
app = Flask(__name__)


config = BertConfig(vocab_size_or_config_json_file= 105879, 
                    hidden_size = 768,
                    num_hidden_layers = 12, 
                    num_attention_heads = 12 , 
                    intermediate_size = 3072)
model = BertForSequenceClassification(config, 2)
a = torch.load('/home/isgood/bert_filtering.pt',map_location=torch.device('cpu'))
model.load_state_dict(a)
tokenizer = BertTokenizer(vocab_file = '/home/isgood/vocab_korea.txt')


def test_sentences(sentences, model , tokenizer):
    device = torch.device('cpu')
    tokenized_texts = [split_token_get(ss,tokenizer,128) for ss in sentences]
    gpu_input = [get_input_ids(ss,tokenizer, 128) for ss in tokenized_texts]
    gpu_mask = [get_input_mask(sentences,tokenizer, 128) for sentences in tokenized_texts]
    gpu_input = torch.tensor(gpu_input)
    gpu_mask = torch.tensor(gpu_mask)

    model.eval()
    torch.no_grad
    outputs = model(input_ids =gpu_input,token_type_ids = None,attention_mask =gpu_mask,labels = None)
    logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    return logits

def softmax(a) :
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y


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



@app.route('/test', methods = ['POST'])
@cross_origin()
def test():
    
    cont = request.json['name']
    cont = str(cont)
    
    tokenized_texts = [split_token_get(ss,tokenizer,128) for ss in [cont]]
    
    logits = test_sentences( sentences = [cont], model = model, tokenizer = tokenizer)
    probit = round( softmax(logits)[0][1], 3)
    pro = probit
    
    return jsonify({"Result": float(pro)})

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        cont = request.form['context']
        tokenized_texts = [split_token_get(ss,tokenizer,128) for ss in [cont]]
        logits = test_sentences( sentences = [cont], model = model , tokenizer = tokenizer)
        probit = round( softmax(logits)[0][1] , 3)
        pro = probit
        return render_template('index.html', print='욕설일 확률은 {}'.format(pro))
 

if __name__ == '__main__':
    
    app.run()
