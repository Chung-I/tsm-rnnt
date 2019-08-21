local BATCH_SIZE = 8;
local FRAME_RATE = 1;
local VOCAB_PATH = "data/vocabulary/ipa";
local TARGET_NAMESPACE = "target_tokens";
local NUM_THREADS = 4;
local NUM_GPUS = 1;

local BASE_READER = {
  "type": "online",
  "lazy": true,
  "input_stack_rate": FRAME_RATE,
  "model_stack_rate": 4,
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
  "maximum_samples_per_batch": ["dimension_0", 16384],
  "track_epoch": true
};

{
  "dataset_reader": {
    "type": "my-multiprocess",
    "base_reader": BASE_READER,
    "num_workers": NUM_THREADS,
    "output_queue_size": 64
    },
  "vocabulary": {
    "directory_path": VOCAB_PATH,
  },
  "train_data_path": "/home/nlpmaster/ssd-1t/corpus/TaiBible/PKL_wav",
  "validation_data_path": "/home/nlpmaster/ssd-1t/corpus/TaiBible/dev_noise_augmented",
  "model": {
    "type": "ctc",
    "loss_type": "ctc",
    "cmvn": true,
    "input_size": 81 * FRAME_RATE,
    "encoder": {
      "type": "awd-rnn",
      "input_size": 81 * FRAME_RATE,
      "hidden_size": 512,
      "num_layers": 4,
      "dropout": 0.5,
      "dropouth": 0.5,
      "dropouti": 0.5,
      "wdrop": 0.0,
      "stack_rates": [1, 2, 2, 1],
    },
    // "encoder": {
    //   "type": "lstm",
    //   "input_size": 80 * FRAME_RATE,
    //   "hidden_size": 512,
    //   "num_layers": 4
    // },
    "vocab_path": VOCAB_PATH,
    "target_namespace": TARGET_NAMESPACE,
  },
  "iterator": {
    "type": "multiprocess",
    "base_iterator": BASE_ITERATOR,
    "num_workers": NUM_THREADS,
    "output_queue_size": 64
  },
  //"iterator": {
  //  "type": "bucket",
  //  "padding_noise": 0.0,
  //  "batch_size" : BATCH_SIZE,
  //  "sorting_keys": [["source_features", "dimension_0"],
  //                   [TARGET_NAMESPACE, "num_tokens"]],
  //  "track_epoch": true,
  //  "maximum_samples_per_batch": ["dimension_0", 6400],
  //},
  "trainer": {
    "num_epochs": 600,
    "grad_clipping": 4.0,
    "cuda_device": 0,
    "validation_metric": "-WER",
    "num_serialized_models_to_keep": 1,
    "learning_rate_scheduler": {
      "type": "multi_step",
      "milestones": [48, 60, 72],
      "gamma": 0.5,
    },
    "optimizer": {
      "type": "adamw",
      "lr": 0.0003,
      "amsgrad": true,
      "weight_decay": 0.01
    }
  }
}
