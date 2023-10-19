use candle_core::{Tensor, Module};
use candle_nn::{Activation, VarBuilder};
use crate::configuration_clip::{CLIPTextConfig, CLIPVisionConfig, CLIPConfig};
use crate::with_tracing::{Embedding};

struct CLIPVisionModelOutput {
    image_embeds: Option<Tensor>,
    last_hidden_state: Tensor,
    hidden_states: Option<Vec<Tensor>>,
    attentions: Option<Vec<Tensor>> 
}

struct CLIPTextModelOutput {
    text_embeds: Option<Tensor>,
    last_hidden_state: Tensor,
    hidden_states: Option<Vec<Tensor>>,
    attentions: Option<Vec<Tensor>>
}

struct CLIPOutput {
    loss: Option<Tensor>,
    logits_per_image: Tensor,
    logits_per_text: Tensor, 
    text_embeds: Tensor,
    image_embeds: Tensor, 
    text_model_output: CLIPTextModelOutput,
    vision_model_output: CLIPVisionModelOutput
}


struct CLIPVisionEmbeddings {
    config: CLIPVisionConfig,
    embed_dim: usize,
    image_size: usize,
    patch_size: usize,
    class_embedding: Embedding,
    patch_embedding: Embedding,
    num_patches: usize,
    num_positions: usize,
    position_embedding: Embedding
    // TODO: Fill in this type 
    // register_buffer: ,
}

struct CLIPTextEmbeddings {
    config: CLIPTextConfig,
    token_embedding: Embedding,
    position_embedding: Embedding
    // TODO: Fill in this type 
    // register_buffer: 
}

