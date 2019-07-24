local BATCH_SIZE = 32;
local FRAME_RATE = 3;
local NUM_THREADS = 1;
local NUM_GPUS = 1;

local BASE_READER = {
    "type": "kaldi-stt",
    "lazy": true,
    "shard_size": BATCH_SIZE,
    "input_stack_rate": FRAME_RATE,
    "model_stack_rate": 2,
    "lexicon_path": "/home/nlpmaster/lexicon.txt",
    "transcript_path": "/home/nlpmaster/Works/egs/aidatatang_200zh/s5/data/*/text",
    "target_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "target_token_indexers": {
      "tokens": {
        "type": "single_id",        
        "namespace": "target_tokens"
      }
    }
};
local BASE_ITERATOR = {
  "type": "bucket",
  "max_instances_in_memory": 64 * NUM_GPUS,
  "batch_size": BATCH_SIZE,
  "sorting_keys": [["source_features", "dimension_0"]],
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
    "directory_path": "data/vocabulary"
  },
  "train_data_path": "/home/nlpmaster/Works/egs/aidatatang_200zh/s5/fbank/*_train.*.scp",
  "validation_data_path": "/home/nlpmaster/Works/egs/aidatatang_200zh/s5/fbank/*_dev.*.scp",
  "test_data_path": "/home/nlpmaster/Works/egs/aidatatang_200zh/s5/fbank/*_test.*.scp",
  "model": {
    "type": "ctc",
    "loss_type": "ctc",
    "encoder": {
      "type": "awd-rnn",
      "input_size": 83 * FRAME_RATE,
      "hidden_size": 512,
      "num_layers": 4,
      "dropout": 0.5,
      "dropouth": 0.5,
      "dropouti": 0.5,
      "wdrop": 0.1,
      "stack_rates": [1, 1, 2, 1],
    },
    "target_namespace": "target_tokens",
  },
  "iterator": {
    "type": "multiprocess",
    "base_iterator": BASE_ITERATOR,
    "num_workers": NUM_THREADS,
    "output_queue_size": 1024
  },
  "trainer": {
    "num_epochs": 150,
    "patience": 10,
    "grad_clipping": 5.0,
    "cuda_device": 0,
    "optimizer": {
      "type": "dense_sparse_adam"
    }
  }
}
