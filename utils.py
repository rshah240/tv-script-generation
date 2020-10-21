import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import hyperparameters

SPECIAL_WORDS = {'PADDING':'<PAD>'}
train_on_gpu = torch.cuda.is_available()

def get_data(path):
    """
    :return: raw text
    """
    with open(path,'r') as f:
        text = f.read()
        return text

def create_lookup_tables(text):
    """
    :param text: Raw text
    :return: A tuple of dicts (vocab_to_int,int_to_vocab)
    """
    counts = Counter(text)
    vocab = sorted(counts,key = counts.get)
    vocab_to_int = {word:ii for ii,word in enumerate(vocab)}
    int_to_vocab = {vocab_to_int[i]:i for i in vocab_to_int}

    #returning tuple
    return  vocab_to_int,int_to_vocab
def token_lookup():
    """
    :return: dictionary containing key is punctuation and the value is token
    """
    punctuation_to_token = {".": "||Period||", ",": "||Comma||", ";": "||Semi_colan||", '"': "||Quotation_Mark||",
                            "?": "||Question_Mark||", "!": "||Exclamation_Mark||", "(": "||Left_Parentheses||",
                            ")": "||Right_Parentheses||", "-": "||Dash||", '\n': '||Return||'}
    return punctuation_to_token

class RNN(nn.Module):

    def __init__(self,vocab_size,output_size,embedding_dim,hidden_dim,n_layers,batch_size):
        """
        :param vocab_size: The number of input dimensions of neural network
        :param output_size: The number of output dimensions of neural network
        :param embedding_dim: The number of embedding dimensions of embedding layers
        :param hidden_dim: The number of hidden dimensions
        :param n_layers: The number of lstm layers
        """

        super().__init__()
        #Defining the parameters
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        #Defining the layers
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim,num_layers=self.n_layers,batch_first=True)
        self.fc = nn.Linear(self.hidden_dim,self.output_size)
        self.dropout = nn.Dropout(hyperparameters.dropout_layer)

    def forward(self,x,hidden):
        """
        :param x: input to neural network
        :param hidden: hidden state to lstm
        :return: x,hidden
        """
        batch_size = x.size(0)
        x = x.long()
        x = self.embedding(x)
        x,hidden = self.lstm(x,hidden)
        #stacking lstm outputs
        x = x.contiguous().view(-1,self.hidden_dim)

        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(batch_size,-1,self.output_size)
        x = x[:,-1]
        #return one batch of output words scores and the hidden state
        return x,hidden

    def init_hidden(self,batch_size):
        """
        :param batch_size: batch_size to the neural networks
        :return:
        """
        weights = next(self.parameters())
        if train_on_gpu:
            hidden = ((weights.new(self.n_layers,batch_size,self.hidden_dim).zero_().cuda()),
                      (weights.new(self.n_layers,batch_size,self.hidden_dim).zero_().cuda()))
        else:
            hidden = ((weights.new(self.n_layers,batch_size,self.hidden_dim).zero_()),
                      (weights.new(self.n_layers,batch_size,self.hidden_dim).zero_()))

        return hidden