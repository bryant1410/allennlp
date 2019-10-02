local pretrained_model = 'bert-base-cased';

{
  dataset_reader: {
    type: 'text_token_probability',
    tokenizer: {
      word_splitter: 'bert-basic'
    },
    token_indexers: {
      bert: {
        type: 'bert-pretrained',
        pretrained_model: pretrained_model,
        do_lowercase: false
      }
    }
  },
  train_data_path: 'het.txt',
  model: {
    type: 'text_token_probability',
    text_field_embedder: {
      allow_unmatched_keys: true,
      embedder_to_indexer_map: {
        bert: ['bert', 'bert-offsets'],
      },
      token_embedders: {
        bert: {
          type: 'bert-pretrained',
          pretrained_model: pretrained_model,
          top_layer_only: true,
          requires_grad: false
        }
      }
    },
    language_model_head: {
      type: 'bert',
      model_name: pretrained_model
    }
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['masked_sentences_tokens', 'num_tokens']],
    batch_size: 32,
  },
  trainer: {
    type: 'no_op'
  }
}
