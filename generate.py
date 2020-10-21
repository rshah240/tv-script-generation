import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import sys
import hyperparameters
from utils import RNN, token_lookup, get_data, create_lookup_tables
import argparse


SPECIAL_WORDS = {'PADDING':'<PAD>'}
train_on_gpu = torch.cuda.is_available()

def preprocess_data(text,prime_id):
    """
    :param text: raw text
    :return:  tokenized data
    """
    token_dict = token_lookup()
    for key,token in token_dict.items():
        text = text.replace(key,' {} '.format(token))
    text = text.lower()
    text = text.split()
    prime_id = prime_id.lower()
    vocab_to_int,int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
    int_text = [vocab_to_int[prime_id]]
    return int_text,vocab_to_int,int_to_vocab,token_dict


def generate(model,prime_id,int_to_vocab,token_dict,pad_value,predict_len=100):
    """
    Generate text using the neural network
    :param model:
    :param prime_id:
    :param int_to_vocab:
    :param token_dict:
    :param pad_value:
    :param predict_len:
    :return: The generated_text
    """
    if train_on_gpu:
        model.cuda()

    model.eval()

    #create a sequence (batch_size = 1) with the prime_id
    current_seq = np.full((1,hyperparameters.seq_length),pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]

    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)

        #Initialize the hidden state
        hidden = model.init_hidden(current_seq.size(0))
        #get the output of the model
        output, _ = model(current_seq,hidden)

        #get the next world probabilities
        p = F.softmax(output,dim = 1).data
        if train_on_gpu:
            p = p.cpu() #moving to cpu
        #use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()

        #select the likely next word index with some element of random
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i,p=p/p.sum())

        #retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)

        if train_on_gpu:
            current_seq = current_seq.cpu()
        current_seq = np.roll(current_seq,-1,1)
        current_seq[-1][-1] = word_i

    gen_sentences = ' '.join(predicted)

    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')

    #return all the sentences
    return gen_sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prime', type=str, required=True, help='Starting word of the Tv Script')
    parser.add_argument('--model', type=str, required=True, help='The model for generation of the text')
    parser.add_argument('--length', type=int, required=True, help='Prediction length(Words)')
    parser.add_argument('--data',type=str,required=True, help='Data for Tokenization of the Text')

    args = parser.parse_args()
    prime_word = str(args.prime)
    predict_len = int(args.length)
    text = get_data(args.data)

    #prime_word = 'Jerry'

    prime_word = prime_word.lower()
    int_text, vocab_to_int, int_to_vocab, token_dict = preprocess_data(text,prime_word)
    path_model = str(args.model)
    with open(path_model,'rb') as f:
        model_dict = torch.load(f)
    model = RNN(vocab_size=model_dict['vocab_size'],output_size=model_dict['output_size'],embedding_dim=model_dict['embedding_dim'],
                hidden_dim=model_dict['hidden_dim'],n_layers = model_dict['n_layers'],batch_size=hyperparameters.batch_size)
    model.load_state_dict(model_dict['state_dict'])
    pad_value = vocab_to_int[SPECIAL_WORDS['PADDING']]
    gen_sentences = generate(model,prime_id = vocab_to_int[prime_word + ':'],pad_value = pad_value,predict_len=predict_len,int_to_vocab=int_to_vocab,token_dict=token_dict)
    print(gen_sentences)




