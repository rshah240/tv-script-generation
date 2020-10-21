import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
import hyperparameters
from utils import RNN, token_lookup, create_lookup_tables, get_data
import argparse


SPECIAL_WORDS = {'PADDING':'<PAD>'}
train_on_gpu = torch.cuda.is_available()


def preprocess_data(text):
    """
    :param text: raw text
    :return:  tokenized data
    """
    token_dict = token_lookup()
    for key,token in token_dict.items():
        text = text.replace(key,' {} '.format(token))
    text = text.lower()
    text = text.split()
    vocab_to_int,int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
    int_text = [vocab_to_int[word] for word in text]
    return int_text,vocab_to_int,int_to_vocab,token_dict

def batch_data(int_text):
    """
    :param int_text: words to int data
    :return: Dataloader of pytorch
    """
    sequence_length = hyperparameters.seq_length
    X = []
    Y = []
    for i in range(len(int_text) - sequence_length):
        x_intermediate = int_text[i:i+sequence_length]
        y_intermediate = int_text[i+sequence_length]
        X.append(x_intermediate)
        Y.append(y_intermediate)
    X = torch.from_numpy(np.array(X))
    Y = torch.from_numpy(np.array(Y))
    data = TensorDataset(X,Y)
    data_loader = DataLoader(data,batch_size=hyperparameters.batch_size)
    return data_loader



def train(data_loader,vocab_size):
    """
    :param model:
    :param data_loader:
    :return:
    """
    model = RNN(vocab_size=vocab_size,output_size=vocab_size,n_layers=hyperparameters.n_layers,embedding_dim=hyperparameters.embedding_dim,
                hidden_dim=hyperparameters.hidden_dim,batch_size=hyperparameters.batch_size)
    if train_on_gpu:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=hyperparameters.lr)
    model.train()
    epochs = hyperparameters.epochs
    training_loss = []
    clip = 5
    print_every = 500
    for e in range(epochs):
        hidden = model.init_hidden(hyperparameters.batch_size)
        for batch_i,(input,target) in enumerate(data_loader,1):
            h = tuple([each.data for each in hidden])
            if train_on_gpu:
                input,target = input.cuda(),target.cuda()
            #Zero grading the model to remove the accumulated Graidents
            n_batches = len(data_loader.dataset) // hyperparameters.batch_size
            #Precuationary step against shape error.
            if(batch_i>n_batches):
                break
            optimizer.zero_grad()
            x,hidden = model(input,h)
            loss = criterion(x,target)
            loss.backward()
            #clipping the gradients as a precaution to exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
            training_loss.append(loss.item())
            if batch_i % print_every == 0:
                print("Epochs:{}/{} ....".format(e+1,epochs),
                      "Loss: {:.6f}".format(np.mean(np.array(training_loss))))

    #Saving the model
    model_dict = {'vocab_size':model.vocab_size,'hidden_dim':model.hidden_dim,'state_dict':model.state_dict(),
                  'n_layers':model.n_layers,'output_size':model.output_size,'embedding_dim':model.embedding_dim}
    #Saving the model
    with open('Model_Friends.pt','wb') as f:
        torch.save(model_dict,f)

def main():
    """
    Main function to define the parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,required=True, help='Data for Tokenization of the Text')
    args = parser.parse_args()
    text = get_data(args.data)
    int_text,vocab_to_int,int_to_vocab,token_dict = preprocess_data(text)

    vocab_size = len(vocab_to_int)

    data_loader = batch_data(int_text)
    torch.cuda.empty_cache()
    train(data_loader,vocab_size)

if __name__ == "__main__":
    main()


    
