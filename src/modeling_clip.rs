use candle_core::{Tensor, Module, Result};
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

impl Module for CLIPVisionEmbeddings {
    fn forward(&self, input_ids: Option<&Tensor>, position_ids: Option<&Tensor>, input_embeds: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let seq_length = biz;
        let position_ids = position_ids.unwrap_or_else(|| self.position_ids[: ,seq_length]);
        let input_embeds = input_embeds.unwrap_or_else(|| self.token_embedding(input_ids));
        embeddings = (input_embeds + position_embedding)?;
        Ok(embeddings)
    }
}