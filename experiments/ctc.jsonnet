local BATCH_SIZE = 32;
local FRAME_RATE = 1;
local VOCAB_PATH = "data/phoneme/vocabulary";
local TARGET_NAMESPACE = "target_tokens";
local NUM_THREADS = 2;
local NUM_GPUS = 1;

local BASE_ITERATOR = {
  "type": "bucket",
  "max_instances_in_memory": 64 * NUM_GPUS,
  "batch_size": BATCH_SIZE,
  "sorting_keys": [["source_features", "dimension_0"],
                   [TARGET_NAMESPACE, "num_tokens"]],
  "maximum_samples_per_batch": ["dimension_0", 6400],
  "track_epoch": true
};

{
  "dataset_reader": {
    "type": "mao-stt",
    "lazy": true,
    "shard_size": BATCH_SIZE,
    "input_stack_rate": FRAME_RATE,
    "model_stack_rate": 4,
    "lexicon_path": "/home/nlpmaster/lexicon.txt",
    "curriculum": [[0, 100], [1, 200],[32, 300], [48, 400]],
    #"bucket": true,
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
  },
  "vocabulary": {
    "directory_path": VOCAB_PATH,    
  },
  "train_data_path": "/home/nlpmaster/ssd-1t/corpus/PTS-MSub-Vol1/fbank/train",
  "validation_data_path": "/home/nlpmaster/ssd-1t/corpus/PTS-MSub-Vol1/fbank/dev",
  "model": {
    "type": "ctc",
    #"loss_type": "ctc",
    "loss_type": "warp_ctc",
    "cmvn": true,
    "input_size": 80 * FRAME_RATE,
    "layerwise_pretraining": [[0, 2], [60, 3], [72, 4]],
    "encoder": {
      "type": "awd-rnn",
      "input_size": 80 * FRAME_RATE,
      "hidden_size": 512,
      "num_layers": 4,
      "dropout": 0.5,
      "dropouth": 0.5,
      "dropouti": 0.5,
      "wdrop": 0.0,
      "stack_rates": [2, 2, 1, 1],
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
  // "iterator": {
  //   "type": "multiprocess",
  //   "base_iterator": BASE_ITERATOR,
  //   "num_workers": NUM_THREADS,
  //   "output_queue_size": 1024
  // },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : BATCH_SIZE,
    "sorting_keys": [["source_features", "dimension_0"],
                     [TARGET_NAMESPACE, "num_tokens"]],
    "track_epoch": true    
  },
  "trainer": {
    "num_epochs": 600,
    # "patience": 10,
    "grad_clipping": 5.0,
    "cuda_device": -1,
    "validation_metric": "-WER",
    "num_serialized_models_to_keep": 1,
    // "learning_rate_scheduler": {
    //   "type": "reduce_on_plateau",
    //   "factor": 0.8,
    //   "mode": "min",
    //   "patience": 50,
    // },
    "learning_rate_scheduler": {
      "type": "multi_step",
      "milestones": [10, 20, 30],
      "gamma": 0.5,
    },
    "optimizer": {
      "type": "dense_sparse_adam",
      "lr": 0.0001
    }
  }
}
