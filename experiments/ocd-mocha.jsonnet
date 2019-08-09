local BATCH_SIZE = 32;
local FRAME_RATE = 1;
local ENCODER_HIDDEN_SIZE = 512;
local DECODER_HIDDEN_SIZE = 512;
local VOCAB_PATH = "runs/vocabulary/word";
local BASE_READER = {
  "type": "mao-stt",
  "lazy": true,
  "shard_size": BATCH_SIZE,
  "input_stack_rate": FRAME_RATE,
  "model_stack_rate": 8,
  #"curriculum": [[0, 100], [8, 200],[16, 300], [32, 400]],
  "bucket": true,
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
};
{
  "random_seed": 13370,
  "numpy_seed": 1337,
  "pytorch_seed": 133,
  "dataset_reader": BASE_READER,
  "validation_dataset_reader": BASE_READER,
  "vocabulary": {
    "directory_path": VOCAB_PATH
  },
  "train_data_path": "/home/nlpmaster/ssd-1t/corpus/TSM/train_outs",
  "validation_data_path": "/home/nlpmaster/ssd-1t/corpus/TSM/valid_outs",
  "model": {
    "type": "seq2seq_mocha",
    "input_size": 80 * FRAME_RATE,
    "cmvn": true,
    #"layerwise_pretraining": [[0, 2], [16, 3], [, 4]],
    "encoder": {
      "type": "awd-rnn",
      "input_size": 80 * FRAME_RATE,
      "hidden_size": ENCODER_HIDDEN_SIZE,
      "num_layers": 4,
      "dropout": 0.5,
      "dropouth": 0.5,
      "dropouti": 0.5,
      "wdrop": 0.0,
      "stack_rates": [1, 2, 2, 2],
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
      #"dirac_at_first_step": false
    },
    "loss_type": "ocd",
    "target_namespace": "target_tokens",
    "initializer": [
      [".*linear.*weight", {"type": "xavier_uniform"}],
      [".*linear.*bias", {"type": "zero"}],
      [".*weight_ih.*", {"type": "xavier_uniform"}],
      [".*weight_hh.*", {"type": "orthogonal"}],
      [".*bias_ih.*", {"type": "zero"}],
      [".*bias_hh.*", {"type": "lstm_hidden_bias"}],
      ["_target_embedder.weight", {"type": "uniform", "a": -1, "b": 1}],
    ]
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : BATCH_SIZE,
    "sorting_keys": [["source_features", "dimension_0"],
                     ["target_tokens", "num_tokens"]],
    "track_epoch": true
  },
  "trainer": {
    "num_epochs": 300,
    "patience": 20,
    "grad_clipping": 4.0,
    "cuda_device": 0,
    "validation_metric": "+UR",
    "num_serialized_models_to_keep": 1,
    "should_log_learning_rate": true,
    // "learning_rate_scheduler": {
    //   "type": "reduce_on_plateau",
    //   "factor": 0.8,
    //   "mode": "min",
    //   "patience": 10
    // },
    // "learning_rate_scheduler": {
    //   "type": "multi_step",
    //   "milestones": [54, 68, 84],
    //   "gamma": 0.5,
    // },
    "learning_rate_scheduler": {
      "type": "noam",
      "model_size": ENCODER_HIDDEN_SIZE,
      "warmup_steps": 10000
    },
    "optimizer": {
      "type": "bert_adam",
      "lr": 0.0001,
      "betas": [0.9, 0.98],
      "weight_decay": 0.01
    }
  }
}
