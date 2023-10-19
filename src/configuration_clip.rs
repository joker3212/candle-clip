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

impl Default for CLIPTextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 490408,
            hidden_size: 512,
            intermediate_size: 2048,
            projection_dim: 512,
            num_hidden_layers: 12,
            num_attention_heads: 8,
            max_position_embeddings: 77,
            hidden_act: Activation::NewGelu,
            layer_norm_eps: 1e-5,
            attention_dropout: 0.0,
            initializer_range: 0.02,
            initializer_factor: 1.0,
            pad_token_id: 1,
            bos_token_id: 49406,
            eos_token_id: 49407 
        }
    }
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

impl Default for CLIPVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            projection_dim: 512,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            patch_size: 32,
            image_size: 224,
            initializer_range: 0.02,
            initializer_factor: 1.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            hidden_act: Activation::NewGelu
        }
    }
}


pub struct CLIPConfig {
    clip_text_config: CLIPTextConfig,
    clip_vision_config: CLIPVisionConfig,
    projection_dim: usize,
    logit_scale_init_value: f64,
    initializer_factor: f64
}