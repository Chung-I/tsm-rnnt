local BATCH_SIZE = 32;
local FRAME_RATE = 1;
local ENCODER_HIDDEN_SIZE = 512;
local DECODER_HIDDEN_SIZE = 512;
local VOCAB_PATH = "data/vocabulary/phn_level";
local NUM_GPUS = 1;
local TARGET_NAMESPACE = "target_tokens";
local WORD_TARGET_NAMESPACE = "word_target_tokens";
local PHN_TARGET_NAMESPACE = "phn_target_tokens";
local NUM_THREADS = 8;
local VGG = true;
local VGG_LAYERS = 2;
local OUT_CHANNEL = 32;
local STACK_RATE = if VGG then std.pow(2, VGG_LAYERS)  else 1;
local NUM_MELS = 80;
local VGG_OUTPUT_SIZE = NUM_MELS * (if VGG then (OUT_CHANNEL / STACK_RATE) else 1) * FRAME_RATE;
local DIRECTIONS = 1;
local SUMMARY_INTERVAL = 100;
local ENCODER_OUTPUT_SIZE = ENCODER_HIDDEN_SIZE * DIRECTIONS;


local PARSER = {
      "type": "biaffine_parser",
      "text_field_embedder": {
        "tokens": {
          "type": "pass_through",
          "hidden_dim": ENCODER_OUTPUT_SIZE
        }
      },
      "encoder": {
        "type": "pass_through",
        "input_dim": ENCODER_OUTPUT_SIZE
      },
      "use_mst_decoding_for_validation": true,
      "arc_representation_dim": 100,
      "tag_representation_dim": 100,
      "dropout": 0.3,
      "input_dropout": 0.3,
      "initializer": [
        #[".*projection.*weight", {"type": "xavier_uniform"}],
        #[".*projection.*bias", {"type": "zero"}],
        [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
        [".*tag_bilinear.*bias", {"type": "zero"}]
      ]
};

local TAGGER = {
      "type": "crf_tagger",
      "text_field_embedder": {
        "tokens": {
          "type": "pass_through",
          "hidden_dim": ENCODER_OUTPUT_SIZE
        }
      },
      "encoder": {
        "type": "pass_through",
        "input_dim": ENCODER_OUTPUT_SIZE
      },
      "label_namespace": "pos",
      "dropout": 0.3,
      "initializer": [
        #[".*projection.*weight", {"type": "xavier_uniform"}],
        #[".*projection.*bias", {"type": "zero"}],
        [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
        [".*tag_bilinear.*bias", {"type": "zero"}]
      ]
};


local BASE_ITERATOR = {
  "type": "bucket",
  "padding_noise": 0.0,
  "batch_size" : BATCH_SIZE,
  "sorting_keys": [["source_features", "dimension_0"],
                    [TARGET_NAMESPACE, "num_tokens"]],
  "max_instances_in_memory": BATCH_SIZE,
  #"maximum_samples_per_batch": ["dimension_0", 36000],
  "track_epoch": true
};

local TSM_READER = {
  "type": "mao-stt",
  "lazy": true,
  "mmap": true,
  "online": false,
  "discard_energy_dim": true,
  "dep": true,
  "lexicon_path": "/home/nlpmaster/lexicon.txt",
  "shard_size": BATCH_SIZE,
  "input_stack_rate": FRAME_RATE,
  "model_stack_rate": STACK_RATE,
  "bucket": true,
  "is_phone": false,
  "num_mel_bins": NUM_MELS,
  "max_frames": 1200,
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
  },
  "word_target_token_indexers": {
    "words": {
      "type": "single_id",        
      "namespace": WORD_TARGET_NAMESPACE
    }
  }
};

local VALID_TSM_READER = TSM_READER + {
    "noskip": true
};

local PTS_READER = {
  "type": "mao-stt",
  "lazy": true,
  "mmap": true,
  "shard_size": BATCH_SIZE,
  "input_stack_rate": FRAME_RATE,
  "model_stack_rate": STACK_RATE,
  "bucket": true,
  "is_phone": false,
  "target_add_start_end_token": false,
  "target_tokenizer": {
    "type": "word",
    "word_splitter": {
      "type": "just_spaces"
    }
  },
  "target_token_indexers": {
    "tokens": {
      "type": "single_id",        
      "namespace": PHN_TARGET_NAMESPACE
    }
  }
};

{
  "random_seed": 13370,
  "numpy_seed": 1337,
  "pytorch_seed": 133,
  "dataset_reader": {
      "type": "interleaving",
      "readers": {
          [TARGET_NAMESPACE]: TSM_READER
      },
      "lazy": true
  },
  "validation_dataset_reader": {
      "type": "interleaving",
      "readers": {
          [TARGET_NAMESPACE]: VALID_TSM_READER
      },
      "lazy": true
  },
  "vocabulary": {
    "directory_path": VOCAB_PATH
  },
  "train_data_path": |||
    {
        "target_tokens": "/home/nlpmaster/ssd-1t/corpus/TSM/trains"
    }
  |||,
  "validation_data_path": |||
    {
        "target_tokens": "/home/nlpmaster/ssd-1t/corpus/TSM/valids"
    }
  |||,
  "model": {
    "type": "phn_mocha",
    "input_size": NUM_MELS * FRAME_RATE,
    "cmvn": 'none',
    "from_candidates": false,
    "sampling_strategy": "max",
    "joint_ctc_ratio": 1.0,
    "dep_ratio": 0.0,
    "pos_ratio": 0.0,
    "ctc_keep_eos": true,
    "time_mask_width": 0,
    "freq_mask_width": 0,
    "time_mask_max_ratio": 0.0,
    #"dep_parser": PARSER,
    #"pos_tagger": TAGGER,
    // "encoder": {
    //   "type": "awd-rnn",
    //   "input_size": NUM_MELS * FRAME_RATE,
    //   "hidden_size": ENCODER_HIDDEN_SIZE,
    //   "num_layers": 2,
    //   "dropout": 0.5,
    //   "dropouth": 0.5,
    //   "dropouti": 0.5,
    //   "wdrop": 0.0,
    //   "stack_rates": [2, 2],
    // },
    "encoder": {
      "type": "lstm",
      "input_size": VGG_OUTPUT_SIZE,
      "hidden_size": ENCODER_HIDDEN_SIZE,
      "num_layers": 4,
      "bidirectional": (DIRECTIONS == 2)
    },
    // "encoder": {
    //   "type": "residual_bidirectional_lstm",
    //   "input_size": VGG_OUTPUT_SIZE,
    //   "hidden_size": ENCODER_HIDDEN_SIZE,
    //   "num_layers": 5,
    //   "layer_dropout_probability": 0.0,
    //   "use_residual": true
    // },
    "dec_layers": 1,
    "cnn": {
      "type": "cnn",
      "num_layers": VGG_LAYERS,
      "in_channel": 1,
      "hidden_channel": OUT_CHANNEL,
    },
    "max_decoding_steps": 30,
    "target_embedding_dim": DECODER_HIDDEN_SIZE,
    "beam_size": 5,
    // "attention": {
    //   "type": "mocha",
    //   "chunk_size": 3,
    //   "enc_dim": ENCODER_OUTPUT_SIZE,
    //   "dec_dim": DECODER_HIDDEN_SIZE,
    //   "att_dim": DECODER_HIDDEN_SIZE
    //   #"dirac_at_first_step": false
    // },
    "attention": {
      "type": "stateful",
      "vector_dim": DECODER_HIDDEN_SIZE,
      "matrix_dim": ENCODER_OUTPUT_SIZE,
      "attention_dim": ENCODER_HIDDEN_SIZE,
      "values_dim": ENCODER_HIDDEN_SIZE,
      "num_heads" : 1
    },
    "n_pretrain_ctc_epochs": 0,
    "target_namespace": TARGET_NAMESPACE,
    "phoneme_target_namespace": PHN_TARGET_NAMESPACE,
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
  "iterator": BASE_ITERATOR,
  // "iterator": {
  //   "type": "multiprocess",
  //   "base_iterator": BASE_ITERATOR,
  //   "num_workers": NUM_THREADS,
  //   "output_queue_size": 1024
  // },
  "trainer": {
    "type": "ignore_nan",
    "num_epochs": 300,
    "patience": 20,
    "grad_norm": 4.0,
    "cuda_device": 0,
    "validation_metric": "+BLEU",
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
      "type": "multi_step",
      "milestones": [60, 72, 84],
      "gamma": 0.5,
    },
    // "optimizer": {
    //   "type": "adamw",
    //   "lr": 0.0003,
    //   "amsgrad": true,
    //   "weight_decay": 1e-6
    // }
    "optimizer": {
      "type": "adadelta",
      "lr": 1.0,
      "eps": 1e-8
    },
    "summary_interval": SUMMARY_INTERVAL
  }
}