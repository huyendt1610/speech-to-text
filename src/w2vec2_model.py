import torch 
from torch import nn 
import torch.nn.functional as F 

from .utils import (
    W2Vec2Config,
    compute_sub_attention_mask
)

class W2Vec2LayerNormConvLayer(nn.Module): 
    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_size,
                 stride,
                 bias):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias
        )

        self.layernorm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x): 
        # Conv expects the data in the shape (B,C,L)
        x = self.conv(x)

        # layerNorm expects (B, L, C)
        x = x.transpose(-1, -2) # swap the two last dimensions
        x = self.layernorm(x) # normalize the dimension: channels ~ features after conv1d
        x = x.transpose(-1, -2)

        # activation 
        x = self.activation(x)

        return x

# go through all convd layers 
class W2Vec2FeatureEncoder(nn.Module): 
    def __init__(self, config):
        super().__init__()

        self.config = config 

        assert len(config.conv_dim) == len(config.conv_stride) == len(config.conv_kernel), \
            "Check config for the same number of convolution components"

        num_conv_blocks = len(config.conv_kernel)
        conv_channels = (1,) + tuple(config.conv_dim) # starting point: just 1 channel
        self.conv_layers = nn.ModuleList()

        for conv_idx in range(num_conv_blocks): 
            self.conv_layers.append(
                W2Vec2LayerNormConvLayer(
                    in_channels=conv_channels[conv_idx],
                    out_channels=conv_channels[conv_idx + 1],
                    kernel_size=config.conv_kernel[conv_idx],
                    stride=config.conv_stride[conv_idx],
                    bias=config.conv_bias
                )
            )
    
    def forward(self, x): 
        for layer in self.conv_layers: 
            x = layer(x)
        return x
    
# add positional info before passing in to transfomers: positional info is very important in seq data
# output of Conv1D is calculated based on neighbor frames => its output has the info about relative position 
# then add to embedding: x = x + pos = content + context => Transfomer understands context
class W2Vec2PositionalConvEmbedding(nn.Module): 
    def __init__(self, config: W2Vec2Config): 
        super().__init__()

        self.config = config 

        self.conv = nn.Conv1d(
            in_channels=config.embedding_dimension, 
            out_channels=config.embedding_dimension, 
            kernel_size=config.conv_positional_emb_kernel_size, 
            padding=config.conv_positional_emb_kernel_size//2, # the shape does not change 
            groups=config.conv_positional_emb_groups
        )

        self.activation = nn.GELU() # Nhưng smooth hơn ReLU, không đột ngột cắt x < 0

    def forward(self, x): 
        batch_size, seq_len, embed = x.shape

        # pass in the shape: (batch, seq lens, embeddings)
        # convd expects: (batch, embeddings, seq lens) => transpose 
        x = x.transpose(1,2) # or (-1,-2)
        positional_embeddings = self.conv(x)

        positional_embeddings = positional_embeddings[:, :, :seq_len] # trim if longer than seq_len
        positional_embeddings= self.activation(positional_embeddings)

        positional_embeddings = positional_embeddings.transpose(1,2) # 

        return positional_embeddings

# Mapping dimension & norm
# Projection layer = Linear layer: map feature vector D → D_model (for transfomers)
# The outputs of this model will be put to:
# 1. normed outputs: to quantizer to do vector quantization 
# 2. projected outputs: to the transformers
class W2Vec2FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config 

        # out of conv has 512 dims => transfomers need embedding_dimension: int = 768 
        self.projection = nn.Linear(config.conv_dim[-1], config.embedding_dimension)

        # normalize the output of conv before passing it to projection 
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1])
        self.dropout = nn.Dropout(config.feature_projection_dropout_p)
    
    def forward(self, x): 
        normed_x = self.layer_norm(x)
        projected_x = self.projection(normed_x)
        projected_x = self.dropout(projected_x)

        return projected_x, normed_x


class W2Vec2Attention(nn.Module): 
    def __init__(self, config):
        super().__init__()

        self.config = config

        # 512 dimension vector is divided equally among num_attention_heads
        # each head calculates attention sperately: Q_i K_i V_i
        # and concat them up 
        self.head_dim = config.embedding_dimension//config.num_attention_heads

        self.q_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.k_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.v_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)

        self.out_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)

    def forward(self, x, attention_mask): 
        batch, seq_len, embed_dim = x.shape 

        q = self.q_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2) # swap (B, S, head_dim, num_heads)
        k = self.k_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2)

        # multiple heads
        attention_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=self.config.attention_dropout_p if self.training else 0.0
        )

        # swap back to (B, S, num_heads, head_dim) => (B, S, embedding_dimension)
        attention_out = attention_out.transpose(1,2).flatten(2)
        attention_out = self.out_proj(attention_out) # get all the heads to know each other, weights for each head 

        return attention_out

# attention learns relationships between frames: inter-frame relationship, context (output of cnn: a 512-dim vector): 
# FeedForward layer learns relationships between features in each vector [f0, f1, .... f511] (
# f1 = low-frequency energy, f2 = high-frequency energy... how f1 relevance f2...
class W2Vec2FeedForward(nn.Module):
    """
    Regular MLP module after the attention computation (Multi-Layer Perceptron)
    MLP (Linear → GELU → Linear)
        768 → 768*4 → GELU → 768*4 → 768
    """
    def __init__(self, config  ):
        super().__init__()

        hidden_size = config.embedding_dimension * config.mlp_ratio 
        self.intermediate_dense = nn.Linear(config.embedding_dimension, hidden_size)
        self.activation = nn.GELU() 
        self.intermediate_dropout = nn.Dropout(config.mlp_dropout_p)

        self.output_dense = nn.Linear(hidden_size, config.embedding_dimension)
        self.output_dropout = nn.Dropout(config.mlp_dropout_p)

    def forward(self, x): 
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x
    
# attention encoder 
class W2Vec2EncoderLayer(nn.Module): 
    def __init__(self, config):
        super().__init__()
    
        self.attention = W2Vec2Attention(config)
        self.dropout = nn.Dropout(config.transformer_encoder_dropout_p)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)

        self.feedforward = W2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)
    
    def forward(self, x, attention_mask=None): 
        # x + ...: Residual Connection
        x = x + self.dropout(self.attention(x, attention_mask))
        x = self.layer_norm(x)

        x = x + self.feedforward(x)
        x = self.final_layer_norm(x)

        return x

class W2Vec2Encoder(nn.Module): 
    def __init__(self, config): 
        super().__init__()

        self.config = config

        # convs
        self.pos_conv_embed = W2Vec2PositionalConvEmbedding(config)
        self.layernorm = nn.LayerNorm(config.embedding_dimension)
        self.dropout = nn.Dropout(config.conv_positional_emb_drop_p)

        # pass in next to transformer blocks
        self.layers = nn.ModuleList([
            W2Vec2EncoderLayer(config) for _ in range(config.num_transformer_layers)
        ]) 

    def forward(self, x, attention_mask=None): # attention_mask: sub_attention_mask
        batch, seq_len, embed_dim = x.shape

        # attention mask: (B, S), and Flask Attention expects: (B, 1, S, S)
        if attention_mask is not None: 
            attention_mask = attention_mask.bool() 
            x[~attention_mask] = 0 # after conv, paddings maybe has values => reset, so no need to pay attention to 
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, S)
            attention_mask = attention_mask.repeat(1, 1, seq_len, 1) # repeat seq_len times at the dim no 2 # (B, 1, S, S) 

        position_embeddings = self.pos_conv_embed(x) 
        x = x + position_embeddings # add context 
        x = self.layernorm(x)
        x = self.dropout(x)

        for layer in self.layers: 
            x = layer(x, attention_mask)

        return x

class W2Vec2Model(nn.Module): 
    def __init__(self, config): 
        super().__init__()

        self.feature_extractor = W2Vec2FeatureEncoder(config)
        self.feature_projection = W2Vec2FeatureProjection(config) 

        if config.masking_probability > 0.0: 
            self.masked_spec_embed = nn.Parameter(# learnable parameter # to create a mask 
                torch.FloatTensor(config.embedding_dimension) # initialization value
            )
            torch.nn.init.uniform_(self.masked_spec_embed)

        self.encoder = W2Vec2Encoder(config)

    def forward(self,
                input_values, 
                attention_mask = None, 
                sub_attention_mask = None, 
                mask_time_indices = None, # time steps selected to be masked 
                return_features_to_quantize = False): 
        
        extract_features = self.feature_extractor(input_values)
        # print(extract_features.shape) # (Batch, embeddings, seq_len)
        extract_features = extract_features.transpose(1,2) # (Batch, seq_len, embeddings)

        if (sub_attention_mask is None and attention_mask is not None): 
            sub_attention_mask = compute_sub_attention_mask(self.config, attention_mask).to(input_values.device)

        # hidden_states: for transformers
        # extract_features: for quantization 
        hidden_states, extract_features = self.feature_projection(extract_features)

        if mask_time_indices is not None: 
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype) # to mask by a mask
        
        # transformers
        encoder_outputs = self.encoder(hidden_states, attention_mask=sub_attention_mask)

        # if also need data to quantize
        if return_features_to_quantize: 
            return encoder_outputs, extract_features
        else: 
            return encoder_outputs

class W2Vec2GumbleVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_groups = config.num_codebook_groups # how many codebooks => 2: 1 frame => (code_i, code_j)
        self.num_vars = config.num_codevectors_per_group # num of codes/quantizer: each codebook has 320 vectors to be selected

        # config.encodevector_dim//self.num_groups
        # choose from each codebook a vector => then concatenate them to have a vector with dim: config.encodevector_dim
        # each sub vector has the dim config.encodevector_dim//self.num_groups
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.encodevector_dim//self.num_groups)
        )

        # map from 512 to 640 (640 possible codes we can pick from)
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)
        
        self.temperature = 2

    def _compute_perplexity(self, probs, mask=None): 
        # only want to compute the perplexity for the masked tokens 
        # to make sure that when quantizing the embeddings for those masked positions, those quantized tokens are actually using a good amount of the codebook
        if mask is not None: 
            marginal_probs = probs[mask.flatten()] # filter to get only masked tokens
            # print(marginal_probs) # [249, 2, 320] all matched positions, each has 2 codebook group, each codebook has 320 probs of selecting vector in codebook
            # if a codebook is being well utilized => when avg the probs together (for each code in each codebook) should be close to uniform = 1/num of masked tokens
            print(marginal_probs.sum(dim=0).shape) # [2, 320]
            # print(mask)

            print(mask.sum())
            marginal_probs = marginal_probs.sum(dim=0) / mask.sum()
        else: 
            marginal_probs = probs.mean(0)

    def forward(self, hidden_states, mask_time_indices=None):
        batch_size, seq_len, hidden_size = hidden_states.shape #

        hidden_states = self.weight_proj(hidden_states) 
        # print(hidden_states.shape) #[B, seq, 640]
        hidden_states = hidden_states.reshape(batch_size * seq_len * self.num_groups, -1)

        # print(hidden_states.shape) # [B*seq * 2, 320] ~ [B*seq * 2 codebooks, each codebook has 320 things]
        if self.training:  
            # hard=True: to return one-hot vector: [000 .... 1....0] => 1: for the selected index
            codevector_probs = nn.functional.gumbel_softmax(hidden_states.float(), tau=self.temperature, hard=True) 

            # compute perplexity => try to avoid using just a small portion of coedebook
            hidden_states = hidden_states.reshape(batch_size * seq_len, self.num_groups, -1)
            codevector_soft_dist = hidden_states.softmax(axis=-1) # result how many % to choose each vector in each codebook
            
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)

class W2Vec2ForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.w2vec2 = W2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.pre_quatizer_dropout)

        self.quantizer = W2Vec2GumbleVectorQuantizer(config)

    def forward(self, 
                input_values, 
                attention_mask = None, 
                sub_attention_mask = None, 
                mask_time_indices = None,
                sampled_negative_indices = None): 
        
        if mask_time_indices is not None: 
            mask_time_indices = mask_time_indices.bool() #.to(torch.bool)

        # compute transformer outputs 
        tranformer_outputs, features_to_quantize = self.w2vec2(input_values, 
                                                                attention_mask, 
                                                                sub_attention_mask, 
                                                                mask_time_indices, 
                                                                return_features_to_quantize=True
                                                            )

        # quantization is done by Gumble-softmax
        self.quantizer(features_to_quantize, mask_time_indices)

if __name__ == "__main__":
    from .utils import W2Vec2Config
    from .dataset import W2Vec2LibriDataset, W2Vec2CollateFunctionForPreTraining
    from torch.utils.data import DataLoader 

    path_to_data_root = "C:/Users/HuyenDT/Downloads/LibriSpeech"

    w2v2_config = W2Vec2Config(num_transformer_layers=2)
    # model = W2Vec2Model(config=w2v2_config)
    model = W2Vec2ForPreTraining(config=w2v2_config)
    dataset = W2Vec2LibriDataset(path_to_data_root=path_to_data_root, 
                                 include_splits = ["dev-clean"])
    collate_fn = W2Vec2CollateFunctionForPreTraining(config=w2v2_config)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    batch = next(iter(loader))
    #print(batch["input_values"].shape)
    out = model(**batch)
        