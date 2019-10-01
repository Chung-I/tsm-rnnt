local BATCH_SIZE = 16;
local FRAME_RATE = 1;
local CORPUS = "fisher";
local CHAR = false;
local WORD_LEVEL = CORPUS != "tsm" && !CHAR;
local TARGET_EMBEDDING_DIM = if CORPUS == "fisher" && CHAR then 64 else 512;
local ENCODER_HIDDEN_SIZE = 512;
local DECODER_HIDDEN_SIZE = 512;
local DEP = false;
local POS = false;
local TSM_VOCAB_PATH = std.extVar("TSM_VOCAB_PATH");
local FISHER_VOCAB_PATH = std.extVar("FISHER_VOCAB_PATH");
local NUM_GPUS = 1;
local CHAR_TARGET_NAMESPACE = "characters";
local WORD_TARGET_NAMESPACE = "tokens";
local PHN_TARGET_NAMESPACE = "phonemes";
local TARGET_NAMESPACE = if WORD_LEVEL then WORD_TARGET_NAMESPACE else CHAR_TARGET_NAMESPACE;
local NUM_THREADS = 1;
local VGG = true;
local VGG_LAYERS = 2;
local OUT_CHANNEL = 32;
local STACK_RATE = if VGG then std.pow(2, VGG_LAYERS)  else 1;
local NUM_MELS = 80;
local VGG_OUTPUT_SIZE = NUM_MELS * (if VGG then (OUT_CHANNEL / STACK_RATE) else 1) * FRAME_RATE;
local DIRECTIONS = 2;
local SUMMARY_INTERVAL = 100;
local ENCODER_OUTPUT_SIZE = ENCODER_HIDDEN_SIZE * DIRECTIONS;
local OCD = false;
local DELTA = 2;

local PARSER = {
      "type": "biaffine_parser",
      "text_field_embedder": {
        [WORD_TARGET_NAMESPACE]: {
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
        [".*projection.*weight", {"type": "xavier_uniform"}],
        [".*projection.*bias", {"type": "zero"}],
        [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
        [".*tag_bilinear.*bias", {"type": "zero"}]
      ]
};

local TAGGER = {
      "type": "simple_tagger",
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
        [".*projection.*weight", {"type": "xavier_uniform"}],
        [".*projection.*bias", {"type": "zero"}],
        [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
        [".*tag_bilinear.*bias", {"type": "zero"}]
      ]
};


local BASE_ITERATOR = {
  "type": "bucket",
  "padding_noise": 0.0,
  "batch_size" : BATCH_SIZE,
  "sorting_keys": [["source_features", "dimension_0"],
                    ["target_tokens", "num_tokens"]],
  "max_instances_in_memory": 1024 * BATCH_SIZE,
  #"maximum_samples_per_batch": ["dimension_0", 36000],
  "track_epoch": true,
};

local ITERATOR = if NUM_THREADS < 1 then BASE_ITERATOR else
{
  "type": "multiprocess",
  "base_iterator": BASE_ITERATOR, # + {"instances_per_epoch": 1280000},
  "num_workers": NUM_THREADS,
  "output_queue_size": 1024
};

local TSM_READER = {
  "type": "mao-stt",
  "word_level": WORD_LEVEL,
  "lazy": true,
  "mmap": true,
  "online": false,
  "discard_energy_dim": true,
  "dep": DEP,
  "lexicon_path": "/home/nlpmaster/lexicon.txt",
  "shard_size": BATCH_SIZE,
  "input_stack_rate": FRAME_RATE,
  "model_stack_rate": STACK_RATE,
  "bucket": true,
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
      "namespace": WORD_TARGET_NAMESPACE,
      "start_tokens": ["@start@"],
      "end_tokens": ["@end@"],
      "lowercase_tokens": true,
    },
    "characters": {
      "type": "just-characters",
      "namespace": CHAR_TARGET_NAMESPACE,
      "start_tokens": ["@start@"],
      "end_tokens": ["@end@"]
    }
  }
};

local FISHER_READER = {
  "type": "mao-stt",
  "lazy": true,
  "mmap": true,
  "online": false,
  "word_level": WORD_LEVEL,
  "discard_energy_dim": true,
  "dep": true,
  #"lexicon_path": std.extVar("LEXICON_PATH"), 
  "fisher_ch": [std.extVar("FISHER_PATH"), "fisher", "train"],
  "shard_size": BATCH_SIZE,
  "input_stack_rate": FRAME_RATE,
  "model_stack_rate": STACK_RATE,
  "bucket": true,
  "num_mel_bins": NUM_MELS,
  "max_frames": 1200,
  "target_add_start_end_token": true,
  "target_tokenizer": {
    "type": "word",
    "word_splitter": {
      "type": "just_spaces"
    },
    // "word_filter": {
    //   "type": "regex",
    //   "patterns": ["^\\W+$"],
    // },
  },
  "target_token_indexers": {
    "tokens": {
      "type": "single_id",        
      "namespace": TARGET_NAMESPACE,
      "start_tokens": ["@start@"],
      "end_tokens": ["@end@"],
      "lowercase_tokens": true,
    },
    "characters": {
      "type": "just-characters",
      "namespace": CHAR_TARGET_NAMESPACE,
      "start_tokens": ["@start@"],
      "end_tokens": ["@end@"],
      "lowercase_tokens": true
    }
  }
};

local VALID_TSM_READER = TSM_READER + {
    "noskip": true
};

local VALID_FISHER_READER = FISHER_READER + {
    "noskip": true,
    "fisher_ch": [std.extVar("FISHER_PATH"), "fisher", "dev"]
};

local PTS_READER = {
  "type": "mao-stt",
  "lazy": true,
  "mmap": true,
  "shard_size": BATCH_SIZE,
  "input_stack_rate": FRAME_RATE,
  "model_stack_rate": STACK_RATE,
  "bucket": true,
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

local BASE_READER = if CORPUS == "tsm" then TSM_READER else FISHER_READER;
local VAL_BASE_READER = if CORPUS == "tsm" then VALID_TSM_READER else VALID_FISHER_READER;

{
  // "random_seed": 13370,
  // "numpy_seed": 1337,
  // "pytorch_seed": 133,
  "dataset_reader": if NUM_THREADS < 1 then BASE_READER else {
    "type": "multiprocess",
    "base_reader": BASE_READER,
    "num_workers": NUM_THREADS,
    "output_queue_size": 1024,
  },
  "validation_dataset_reader": VAL_BASE_READER,
  // "dataset_reader": TSM_READER,
  // "validation_dataset_reader": VALID_TSM_READER,
  // "dataset_reader": {
  //     "type": "interleaving",
  //     "readers": {
  //         [TARGET_NAMESPACE]: TSM_READER
  //     },
  //     "lazy": true
  // },
  // "validation_dataset_reader": {
  //     "type": "interleaving",
  //     "readers": {
  //         [TARGET_NAMESPACE]: VALID_TSM_READER
  //     },
  //     "lazy": true
  // },
  "vocabulary": {
    "directory_path": if CORPUS == "tsm" then TSM_VOCAB_PATH else FISHER_VOCAB_PATH,
  },
  // "train_data_path": |||
  //   {
  //       "tokens": "/home/nlpmaster/ssd-1t/corpus/TSM/trains"
  //   }
  // |||,
  // "validation_data_path": |||
  //   {
  //       "tokens": "/home/nlpmaster/ssd-1t/corpus/TSM/valids"
  //   }
  // |||,
  // "train_data_path": ,
  // "validation_data_path": ,
  "train_data_path": if CORPUS == "tsm" then std.extVar("TSM_TRAIN_PATH") 
    else std.extVar("FISHER_TRAIN_PATH"),
  "validation_data_path": if CORPUS == "tsm" then std.extVar("TSM_VAL_PATH")
    else std.extVar("FISHER_VAL_PATH"),
  "model": {
    "type": "phn_mocha",
    "input_size": NUM_MELS * FRAME_RATE,
    "cmvn": 'utt',
    "from_candidates": false,
    "sampling_strategy": if OCD then "sample" else "max",
    "max_decoding_ratio": 1.0,
    "dep_ratio": if DEP then 0.1 else 0.0,
    "pos_ratio": 0.0,
    "time_mask_width": 0,
    "freq_mask_width": 0,
    "time_mask_max_ratio": 0.0,
    "delta": DELTA,
    "dep_parser": if DEP then PARSER else null,
    "pos_tagger": if (DEP && POS) then TAGGER else null,
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
    "att_ratio": if DEP then 0.9 else 1.0,
    "encoder": {
      "type": "lstm",
      "input_size": VGG_OUTPUT_SIZE,
      "hidden_size": ENCODER_HIDDEN_SIZE,
      "num_layers": 4,
      "bidirectional": (DIRECTIONS == 2)
    },
    // "ctc_layer": {
    //   "type": "ctc",
    //   "target_namespace": TARGET_NAMESPACE,
    //   "loss_ratio": 0.25
    // },
    // "projection_layer": {
    //   "_pretrained": {
    //     "archive_file": "runs/ctc/model.tar.gz",
    //     "module_path": "_joint_ctc_projection_layer",
    //     "freeze": false
    //   }
    // },
    // "rnnt_layer": {
    //   "type": "rnnt",
    //   "input_size": ENCODER_OUTPUT_SIZE,
    //   "hidden_size": DECODER_HIDDEN_SIZE,
    //   "target_namespace": TARGET_NAMESPACE,
    //   "loss_ratio": 0.25,
    //   "recurrency": {
    //     "_pretrained": {
    //       "archive_file": "runs/lm/model.tar.gz",
    //       "module_path": "_contextualizer._module",
    //       "freeze": false
    //     }
    //   },
    //   "target_embedder": {
    //     "_pretrained": {
    //       "archive_file": "runs/lm/model.tar.gz",
    //       "module_path": "_text_field_embedder.token_embedder_tokens",
    //       "freeze": false
    //     }
    //   }
    // },
    // "encoder": {
    //   "_pretrained": {
    //     "archive_file": "runs/ctc/model.tar.gz",
    //     "module_path": "_encoder",
    //     "freeze": false
    //   }
    // },
    // "encoder": {
    //   "type": "residual_bidirectional_lstm",
    //   "input_size": VGG_OUTPUT_SIZE,
    //   "hidden_size": ENCODER_HIDDEN_SIZE,
    //   "num_layers": 4,
    //   "layer_dropout_probability": 0.0,
    //   "use_residual": true
    // },
    // "encoder": {
    //   "type": "stacked_custom_lstm",
    //   "input_size": VGG_OUTPUT_SIZE,
    //   "hidden_size": ENCODER_HIDDEN_SIZE,
    //   "num_layers": 4,
    //   "layer_norm": true,
    //   "bidirectional": (DIRECTIONS == 2)
    // },
    "dec_layers": 1,
    "cnn": {
      "type": "cnn",
      "num_layers": VGG_LAYERS,
      "in_channel": (DELTA + 1),
      "hidden_channel": OUT_CHANNEL,
      "nonlinearity": "relu"
    },
    // "conv_lstm": {
    //   "type": "stacked_custom_lstm",
    //   "conv": true,
    //   "input_channel": OUT_CHANNEL,
    //   "hidden_channel": OUT_CHANNEL,
    //   "input_size": NUM_MELS / STACK_RATE,
    //   "hidden_size": NUM_MELS / STACK_RATE,
    //   "kernel_size": [1, 3],
    //   "num_layers": 1,
    //   "bidirectional": (DIRECTIONS == 2)
    // },
    // "cnn": {
    //   "_pretrained": {
    //     "archive_file": "runs/ctc/model.tar.gz",
    //     "module_path": "_cnn",
    //     "freeze": false
    //   }
    // },
    "max_decoding_steps": 30,
    "target_embedding_dim": TARGET_EMBEDDING_DIM,
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
    "target_namespace": TARGET_NAMESPACE,
    "phoneme_target_namespace": PHN_TARGET_NAMESPACE,
    "initializer": [
      #[".*_projection_layer.*", "prevent"],
      #[".*_cnn.*", "prevent"],
      #[".*_encoder.*", "prevent"],
      #[".*_recurrency.*", "prevent"],
      [".*linear.*weight", {"type": "xavier_uniform"}],
      [".*linear.*bias", {"type": "zero"}],
      [".*weight_ih.*", {"type": "xavier_uniform"}],
      [".*weight_hh.*", {"type": "orthogonal"}],
      [".*bias_ih.*", {"type": "zero"}],
      [".*bias_hh.*", {"type": "lstm_hidden_bias"}],
      ["_target_embedder.weight", {"type": "uniform", "a": -1, "b": 1}]
      #["_target_embedder.weight", "prevent"],
    ]
  },
  "iterator": ITERATOR,// + {"instances_per_epoch": 1280},
  "validation_iterator": ITERATOR,
  "trainer": {
    "type": "ignore_nan",
    #"mixed_precision": true,
    "num_epochs": 300,
    "patience": 20,
    "grad_norm": 2.0,
    "cuda_device": 0,
    "validation_metric": "-att_wer",
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
    // "learning_rate_scheduler": {
    //   "type": "multi_step",
    //   "milestones": [60, 72, 84],
    //   "gamma": 0.5,
    // },
    // "optimizer": {
    //   "type": "dense_sparse_adam"
    // },
    "optimizer": {
      "type": "fused_adam",
      #"weight_decay": 1e-6
    },
    "learning_rate_scheduler": {
      "type": "noam",
      "model_size": 512,
      "warmup_steps": 6000
    },
    // "optimizer": {
    //   "type": "adadelta",
    //   "lr": 1.0,
    //   "eps": 1e-8
    // },
    "summary_interval": SUMMARY_INTERVAL
  }
}
