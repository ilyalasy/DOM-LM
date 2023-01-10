# coding=utf-8

""" DOM-LM model configuration"""

from transformers.utils import logging

from transformers.configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)



class DOMLMConfig(PretrainedConfig):
    r"""
    ...

    Configuration objects inherit from [`BertConfig`] and can be used to control the model outputs. Read the
    documentation from [`BertConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the MarkupLM model. Defines the different tokens that can be represented by the
            *inputs_ids* passed to the forward method of [`MarkupLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.        
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed into [`MarkupLMModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        
        max_node_embeddings (`int`, *optional*, defaults to 2048):
            The maximum DOM elements amount this model might ever be used with.
        max_sibling_embeddings (`int`, *optional*, defaults to 128):
            The maximum DOM siblings amount this model might ever be used with.
        max_depth_embeddings (`int`, *optional*, defaults to 128):
            The maximum depth of DOM tree.
        max_tag_embeddings (`int`, *optional*, defaults to 128):
            The maximum value that the tag unit embedding might ever use.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).

    """
    model_type = "domlm"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,        
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=2,
        gradient_checkpointing=False,        
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        # should it be global for whole DOM or only for subtrees?
        max_node_embeddings=2048, # calculate mean elements count in swde dataset
        
        max_sibling_embeddings=128, # should be fine
        max_depth_embeddings=128,
        max_tag_embeddings = 128,
        max_position_embeddings=512,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        # additional properties
        self.max_node_embeddings = max_node_embeddings
        self.max_sibling_embeddings = max_sibling_embeddings
        self.max_depth_embeddings = max_depth_embeddings
        self.max_tag_embeddings = max_tag_embeddings
        self.max_position_embeddings = max_position_embeddings

        self.node_pad_id = max_node_embeddings  - 1 
        self.sibling_pad_id = max_sibling_embeddings - 1 
        self.depth_pad_id = max_depth_embeddings - 1
        self.tag_pad_id = max_tag_embeddings - 1
        