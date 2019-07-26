local BATCH_SIZE = 32;
local FRAME_RATE = 3;
local NUM_THREADS = 1;
local NUM_GPUS = 1;
local VOCAB_PATH = std.extVar('VOCAB_PATH') + "/vocabulary";
local TARGET_NAMESPACE = "target_tokens";

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
      "num_layers": 1,
      "dropout": 0.0,
      "dropouth": 0.0,
      "dropouti": 0.0,
      "wdrop": 0.0,
      "stack_rates": [2],
    },
    "vocab_path": VOCAB_PATH,
    "target_namespace": TARGET_NAMESPACE,
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
