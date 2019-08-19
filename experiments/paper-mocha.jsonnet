local BATCH_SIZE = 64;
local FRAME_RATE = 3;
local ENCODER_HIDDEN_SIZE = 512;
local DECODER_HIDDEN_SIZE = 512;
{
  "random_seed": 13370,
  "numpy_seed": 1337,
  "pytorch_seed": 133,
  "dataset_reader": {
    "type": "stt",
    "lazy": true,
    "shard_size": BATCH_SIZE,
    "input_stack_rate": FRAME_RATE,
    "model_stack_rate": 2,
    "target_add_start_end_token": true,
    "target_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "bert-basic"
      }
    },
    "target_token_indexers": {
      "tokens": {
        "type": "single_id",        
        "namespace": "target_tokens"
      }
    }
  },
  "vocabulary": {
    "directory_path": "runs/vocabulary/word"
  },
  "train_data_path": "/home/nlpmaster/ssd-1t/tsm-single-npy/train",
  "validation_data_path": "/home/nlpmaster/ssd-1t/tsm-single-npy/val",
  "model": {
    "type": "seq2seq_mocha",
    "input_size": 80 * FRAME_RATE,
    "cmvn": false,
    "encoder": {
      "type": "awd-rnn",
      "input_size": 80 * FRAME_RATE,
      "hidden_size": ENCODER_HIDDEN_SIZE,
      "num_layers": 4,
      "dropout": 0.25,
      "dropouth": 0.25,
      "dropouti": 0.25,
      "wdrop": 0.1,
      "stack_rates": [1, 1, 2, 1],
    },
    "max_decoding_steps": 30,
    "target_embedding_dim": DECODER_HIDDEN_SIZE,
    "beam_size": 5,
    "attention": {
      "type": "mocha",
      "chunk_size": 6,
      "enc_dim": ENCODER_HIDDEN_SIZE,
      "dec_dim": DECODER_HIDDEN_SIZE,
      "att_dim": DECODER_HIDDEN_SIZE
    },
    "target_namespace": "target_tokens",
    "initializer": [
      [".*linear.*weight", {"type": "xavier_uniform"}],
      [".*linear.*bias", {"type": "zero"}],
      [".*weight_ih.*", {"type": "xavier_uniform"}],
      [".*weight_hh.*", {"type": "orthogonal"}],
      [".*bias_ih.*", {"type": "zero"}],
      [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
      # ["_target_embedder.weight", {"type": "uniform", "a": -1, "b": 1}],
    ]
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : BATCH_SIZE,
    "sorting_keys": [["source_features", "dimension_0"],
                     ["target_tokens", "num_tokens"]]
  },
  "trainer": {
    "num_epochs": 300,
    "patience": 20,
    "grad_clipping": 10,
    "cuda_device": 0,
    "validation_metric": "-WER",
    "num_serialized_models_to_keep": 1,
    "should_log_learning_rate": true,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.8,
      "mode": "min",
      "patience": 10
    },
    "optimizer": {
      "type": "dense_sparse_adam",
      "lr": 0.0003
    }
  }
}
