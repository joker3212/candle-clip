use candle_nn::Activation;

pub struct CLIPTextConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    projection_dim: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    hidden_act: Activation,
    initializer_range: f64,
    initializer_factor: f64,
    attention_dropout: f64,
    pad_token_id: usize,
    bos_token_id: usize,
    eos_token_id: usize
}

pub struct CLIPVisionConfig {
        hidden_size: usize,
        intermediate_size: usize,
        projection_dim: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_channels: usize,
        patch_size: usize,
        image_size: usize,
        initializer_range: f64,
        initializer_factor: f64,
        attention_dropout: f64,
        layer_norm_eps: f64,
        hidden_act: Activation
}

pub struct CLIPConfig {
    clip_text_config: CLIPTextConfig,
    clip_vision_config: CLIPVisionConfig,
    projection_dim: usize,
    logit_scale_init_value: f64,
    initializer_factor: f64
}