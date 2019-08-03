local BATCH_SIZE = 64;
local FRAME_RATE = 1;
local ENCODER_HIDDEN_SIZE = 512;
local DECODER_HIDDEN_SIZE = 512;
local NUM_THREADS = 1;
local NUM_GPUS = 1;
local VOCAB_PATH = "runs/vocabulary/phoneme";
local TARGET_NAMESPACE = "target_tokens";

local BASE_READER = {
    "type": "kaldi-stt",
    "lazy": true,
    "shard_size": BATCH_SIZE,
    "input_stack_rate": FRAME_RATE,
    "model_stack_rate": 8,
    "lexicon_path": "/home/nlpmaster/lexicon.txt",
    "transcript_path": "/home/nlpmaster/Works/egs/pts/s5/data/all/text",
    "target_add_start_end_token": true,
    "target_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "target_token_indexers": {
      "tokens": {
        "type": "single_id",        
        "namespace": TARGET_NAMESPACE
      }
    }
};
local BASE_ITERATOR = {
  "type": "bucket",
  "max_instances_in_memory": 64 * NUM_GPUS,
  "batch_size": BATCH_SIZE,
  "sorting_keys": [["source_features", "dimension_0"],
                   [TARGET_NAMESPACE, "num_tokens"]],
  "maximum_samples_per_batch": ["dimension_0", 6400]
};

{
  "dataset_reader": {
    "type": "multiprocess",
    "base_reader": BASE_READER,
    "num_workers": NUM_THREADS,
    "output_queue_size": 1024
  },
  "vocabulary": {
    "directory_path": VOCAB_PATH,
  },
  "train_data_path": "/home/nlpmaster/Works/egs/pts/s5/fbank/*_train.*.scp",
  "validation_data_path": "/home/nlpmaster/Works/egs/pts/s5/fbank/*_test.*.scp",
  #"test_data_path": "/home/nlpmaster/Works/egs/pts/s5/fbank/*_test.*.scp",
  "model": {
    "type": "seq2seq_mocha",
    "encoder": {
      "type": "awd-rnn",
      "input_size": 80 * FRAME_RATE,
      "hidden_size": ENCODER_HIDDEN_SIZE,
      "num_layers": 4,
      "dropout": 0.25,
      "dropouth": 0.25,
      "dropouti": 0.25,
      "wdrop": 0.1,
      "stack_rates": [2, 2, 2, 1],
    },
    "max_decoding_steps": 30,
    "target_embedding_dim": DECODER_HIDDEN_SIZE,
    "beam_size": 5,
    "attention": {
      "type": "mocha",
      "chunk_size": 2,
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
      [".*bias_hh.*", {"type": "lstm_hidden_bias"}],
      ["_target_embedder.weight", {"type": "uniform", "a": -1, "b": 1}],
    ]
  },
  "iterator": {
    "type": "multiprocess",
    "base_iterator": BASE_ITERATOR,
    "num_workers": NUM_THREADS,
    "output_queue_size": 1024
  },
  "trainer": {
    "num_epochs": 300,
    "patience": 10,
    "grad_clipping": 5.0,
    "cuda_device": 0,
    "validation_metric": "-WER",
    "num_serialized_models_to_keep": 1,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.8,
      "mode": "min",
      "patience": 5
    },
    "optimizer": {
      "type": "dense_sparse_adam",
      "lr": 0.0003
    }
  }
}
