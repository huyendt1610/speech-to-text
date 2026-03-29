# pip install transformers jiwer ipywidgets
import torch 
import torch.nn as nn 
from transformers import Wav2Vec2CTCTokenizer 


class MaskedConvd2d(nn.Conv2d): # inherit from nn.Conv2d
    def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding=0, 
            bias=True, 
            **kwargs): 
        super(MaskedConvd2d, self).__init__(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias,
                                            **kwargs)
    
    def forward(self, x, seq_lens): # define how data go through the model - the computing process of nn, seq_lens: valid lengths that come out of it 
        batch_size , channels, height, width = x.shape # standard form for a convolution (like for an image)
        output_seq_lens = self._compute_out_seq_len(seq_lens)

        conv_out = super().forward(x) # using the original forward method of the original convolution on this data

        mask = torch.zeros(batch_size, output_seq_lens.max(), device=x.device) # device list: CPU/GPU => x, mask need to be on the same device to avoid errors

        for i, length in enumerate(output_seq_lens): 
            mask[i, :length] = 1

        mask = mask.unsqueeze(1).unsqueeze(1)

        conv_out = conv_out * mask

        return conv_out, output_seq_lens

    def _compute_out_seq_len(self, seq_lens): 
        return torch.floor((seq_lens + (2 * self.padding[1]) - (self.kernel_size[1] -1) -1) // self.stride[1]) + 1

# ### Convolutional Feature Extractor 
# 
# Create a stack of two convolutions with BatchNorm2d and the HardTanh Activation function 
# 
# The output of the convolutions will give the shape (Batch x Channels X Mel_features x time). To give this to the future RNN, we need to reshape it to (Batch x time x Channels * Mel_features)
# 
# from (B, C, H, W) => (B, C, H*W) ~ (B, embedding, tokens)
# => need to this format: (B, tokens, embedding)
# 
class ConvolutionFeatureExtractor(nn.Module): 
    def __init__(self, in_channels = 1, out_channels = 32): # default from Nvidia implementation
        super(ConvolutionFeatureExtractor, self).__init__()

        self.conv1 = MaskedConvd2d(in_channels, out_channels, kernel_size=(11, 41), stride=(2,2), padding=(5, 20), bias=False) # co BatchNorm roi => no need bias
        self.bn1 = nn.BatchNorm2d(out_channels) # normalize the batch 

        self.conv2 = MaskedConvd2d(out_channels, out_channels, kernel_size=(11, 21), stride=(2,1), padding=(5, 10), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) # normalize the batch 

        self.output_feature_dim = 20 
        self.conv_output_features = self.output_feature_dim * out_channels

    def forward(self, x, seq_lens): 
        
        #print("Before conv1: ", x.shape)
        #print(seq_lens)

        x, seq_lens = self.conv1(x, seq_lens)
        x = self.bn1(x)
        x = torch.nn.functional.hardtanh(x)

       # print("\nAfter Convolution 1 layer: ", x.shape)
        #print(seq_lens)

        x, seq_lens = self.conv2(x, seq_lens)
        x = self.bn2(x)
        x = torch.nn.functional.hardtanh(x)

        #x = x.permute(0, 3, 1, 2) # change dimension orders from (0, 1, 2, 3) => (0, 3, 1, 2)
        #print("\nAfter Convolution 2 layer: ", x.shape)
        #print(seq_lens)

        x = x.permute(0, 3, 1, 2) # change dimension orders from (0, 1, 2, 3) => (0, 3, 1, 2)
        #print("\nAfter permute: ", x.shape)
        x = x.flatten(2) # flatten from the dimension with index = 2
        #print("\nAfter flatten: ", x.shape) 

        return x, seq_lens
    
class RNNLayer(nn.Module): 
    def __init__(self, input_size, 
                 hidden_size = 512, 
                 dropout=0.3): # what the invading implementaion used 
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.rnn = nn.LSTM( # Long Short-Term Memory
            input_size = input_size, # no of features at each timestep 
            hidden_size=hidden_size, # no of neuron at hidden state 
            batch_first=True, 
            bidirectional=True, # when training the rnn, future steps can look in the past steps and vice versa as the entire input is passed at once
            dropout=dropout
        )

        self.layernorm = nn.LayerNorm(2 * hidden_size) # need *2 as for both forward and backward direction with bidirectional=True

    def forward(self, x, seq_lens):
        batch, seq_len, embed_dim = x.shape 

        # packing to be used in rnn to save time processing 
        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True)

        out, _ = self.rnn(packed_x)

        # unpacking 
        x, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=seq_len, batch_first=True )

        x = self.layernorm(x) # normalize data

        return x 

# ### Put it together 
# 
# Put it all together to create our DeepSpeech2 model 

class DeepSpeech2(nn.Module): 
    def __init__(self, 
                conv_in_channels = 1, 
                conv_out_channels = 32, 
                rnn_hidden_size = 512, # reduce model size
                rnn_depth = 5, # reduce model size, but in docs [5-7]
                tokenizer = None
                ):
        
        super().__init__()

        self.feature_extractor = ConvolutionFeatureExtractor(conv_in_channels, conv_out_channels)

        self.output_hidden_features = self.feature_extractor.conv_output_features # 640 features 

        self.rnns = nn.ModuleList(  # a bunch of RNN 
            [
                # first layer: input_size = 640 
                # after that: input_size = 512 * 2 = 1024
                RNNLayer(input_size=self.output_hidden_features if i == 0 else 2 * rnn_hidden_size, hidden_size=rnn_hidden_size)
                for i in range(rnn_depth)
            ]
        )

        if tokenizer is None: 
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")


        self.head = nn.Sequential( # output: probabilities for each vocab
            nn.Linear(2 * rnn_hidden_size, rnn_hidden_size), # input, output
            nn.Hardtanh(),
            nn.Linear(rnn_hidden_size, tokenizer.vocab_size) # fully connected layer to predict what the letter is said at this timestep?
        )


    def forward(self, x, seq_lens): 
        x, final_seq_lens = self.feature_extractor(x, seq_lens)

        for rnn in self.rnns: 
            x = rnn(x, final_seq_lens) # after convolution layer => final_seq_lens will never change

        x = self.head(x)
        
        return x, final_seq_lens

