import numpy as np 
from dataclasses import dataclass, asdict
from typing import Literal, Optional
import torch

import warnings
warnings.filterwarnings('ignore')

@dataclass # automatic create __init__(), __repr__(), __eq__()
class Wave2Vec2ForPreTrainingOutput: 
    loss: Optional[torch.FloatTensor] = None
    projected_states: torch.FloatTensor = None
    projected_quantitized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None


@dataclass 
class W2Vec2Config: 
    # Feature encoder convolution config 
    conv_dim: tuple = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: tuple = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = True 
    feature_projection_dropout_p: float = 0.0

    # positional convolutional embeddings 
    conv_positional_emb_drop_p: float = 0.0 
    conv_positional_emb_groups: int = 16 
    conv_positional_emb_kernel_size: int = 128 

    # Transformer config 
    num_transformer_layers: int = 12
    num_attention_heads: int = 12 # Number of attention heads run parallel: each head learns a different relationship in the data.
    embedding_dimension: int = 768 
    mlp_ratio: int = 4
    mlp_dropout_p: float = 0.0 
    attention_dropout_p: float = 0.0
    transformer_encoder_dropout_p: float = 0.0
    layer_dropout_p: float = 0.0 
    initializer_range: float = 0.02 

    # Gumbel softmax config 
    num_codebook_groups: int = 2 
    num_codevectors_per_group: int = 320 
    encodevector_dim: int = 256 
    pre_quatizer_dropout: float = 0.0 

    # masking config 
    masking_probability: float = 0.065 
    masking_span_length: int = 10 
    minimum_spans: int = 2 

    # loss config 
    contrastive_logits_temperature: float = 0.1
    diversity_loss_weight: float = 0.1

    # training config 
    num_negatives: int = 100

    # layernorm config 
    layer_norm_eps: float = 1e-5

    # ctc config 
    asr_head_dropout_b: float = 0.1
    blank_token_idx: int = 0 
    vocab_size: int = 32 

    # hugging face interface conifg 
    hf_model_name: str = "facebook/wave2vec2-base"

    # pretrain backbone config 
    path_to_pretrained_weights: str = None 

    # backbone config 
    pre_trained_backbone: Literal["pretrained", "pretrained_huggingface", "random"] = "pretrained"

    # added in to_dict() method so this config is compatible with Huggingface trainer 
    def to_dict(self):
        return asdict(self)
    

def compute_concoded_lengths(lengths, conv_kernels, conv_strides): 
    if not isinstance(lengths, torch.Tensor): 
        lengths = torch.Tensor(lengths)    
    
    def _compute_conv_out(lengths, kernel_size, stride): 
        return torch.floor((lengths - (kernel_size -1) -1) / stride) + 1 

    for k, s in zip(conv_kernels, conv_strides): 
        # print(lengths)
        lengths = _compute_conv_out(lengths, k, s)
        
    # print(lengths)
    lengths = lengths.type(torch.int)

    return lengths 

# after all conv
def compute_sub_attention_mask(config, attention_mask):
    batch_size = attention_mask.shape[0]
    raw_lengths = attention_mask.sum(axis=-1) 

    encoded_lengths = compute_concoded_lengths(raw_lengths, config.conv_kernel, config.conv_stride)
    sub_attention_mask = torch.zeros((batch_size, max(encoded_lengths)))

    for idx, length in enumerate(encoded_lengths): 
        sub_attention_mask[idx, :length] = 1
    
    return sub_attention_mask
    
def compute_span_mask(shape,
                      mask_prob=0.065,
                      mask_length=10,
                      min_masks=2,
                      attention_mask=None): 
    batch_size, total_seq_length = shape 
    if(attention_mask is None): 
        sequence_lengths = [total_seq_length] * batch_size
    else: 
        sequence_lengths = attention_mask.sum(axis=-1).to(torch.int).tolist()

    sequence_masks = []

    for length in sequence_lengths: 
        mask = torch.zeros(total_seq_length).bool() 
        # generate random numbers [0,1] => if (values < mask_prob) => True else False 
        # from True index => mask mask_length forward
        # .nonzero() get where is True 
        sample_starting_idx = (torch.rand(length) < mask_prob).nonzero()

        if len(sample_starting_idx) < min_masks:  # in case num of True index < min_mask 
            sample_starting_idx = torch.randint(low=0, high=length, size=(min_masks, 1))

        span_offsets = torch.arange(mask_length) # arr from [0, 9]
        # print(sample_starting_idx)
        spans = sample_starting_idx + span_offsets # sample_starting_idx.shape = [min_masks, 1] x [10,] => broadcast [min_masks, 1] x [1, 10] = [min_masks, 10] (double sample_starting_idx to 10 cols => then add element-wise)

        # print(spans)
        spans = spans.flatten() 
        spans = spans[spans <= length -1 ] # remove indexes > length as they are padded

        mask[spans.flatten()] = True

        # print(mask, spans, mask.shape )
        sequence_masks.append(mask.unsqueeze(0))
    
    sequence_masks = torch.concatenate(sequence_masks)
    return sequence_masks # which tokens to be masked, and what not

def sample_negative_indices(features_shape, num_negatives, mask_time_indices):
    batch_size, sequence_length = features_shape 
    sequence_index = np.arange(sequence_length)

    sampled_negatives = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)
    
    if mask_time_indices is None: 
        mask_time_indices = np.ones(shape=(batch_size, sequence_length), dtype=bool)

    for idx in range(batch_size): 
        batch_span_mask = mask_time_indices[idx]
        masked_indexes = sequence_index[batch_span_mask]

        num_masked = batch_span_mask.sum() 
        # print(num_masked)
        feature_index = np.expand_dims(np.arange(num_masked), -1)
        feature_index = np.repeat(feature_index, num_negatives, axis = -1)

        sample_index = np.random.randint(0, num_masked - 1, size=(num_masked, num_negatives))
        sample_index[(sample_index == feature_index)] += 1 # if the index is the correct answer => increase index to avoid that
        sampled_negatives[idx][batch_span_mask] = masked_indexes[sample_index]

        sampled_negatives[idx] += idx * sequence_length # after flattened => still get data by indexes (offset everything by 1, 2, 3... * sequence_length)

        
    sampled_negatives = torch.tensor(sampled_negatives, dtype=torch.long)
    return sampled_negatives


if __name__ == "__main__":

    seq_len = [25000, 32000] # num_samples 

    data = [torch.rand(l) for l in seq_len]
    attention_mask = [torch.ones(l) for l in seq_len]

    # padd to the length of the longest audio 
    data = torch.nn.utils.rnn.pad_sequence(data, padding_value=0.0, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0.0, batch_first=True)

    # print(data)
    print(attention_mask)

    config = W2Vec2Config()
    sub_attention_mask = compute_sub_attention_mask(config, attention_mask)
    span_mask = compute_span_mask(shape=tuple(sub_attention_mask.shape),
                                    attention_mask=sub_attention_mask)
    negatives = sample_negative_indices(features_shape=tuple(sub_attention_mask.shape),
                                        num_negatives=5, 
                                        mask_time_indices=span_mask)
    