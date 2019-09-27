local LEXICON_PATH = std.extVar("LEXICON_PATH");
local CTC_MODEL_PATH = std.extVar("CTC_MODEL_PATH");
local BATCH_SIZE = 16;
local FRAME_RATE = 1;
local ENCODER_HIDDEN_SIZE = 512;
local DECODER_HIDDEN_SIZE = 512;
local DIRECTIONS = 1;
local ENCODER_OUTPUT_SIZE = ENCODER_HIDDEN_SIZE * DIRECTIONS;
local CHAR_TARGET_NAMESPACE = "characters";
local WORD_TARGET_NAMESPACE = "tokens";
local PHN_TARGET_NAMESPACE = "phonemes";
local WORD_LEVEL = false;
local TARGET_NAMESPACE = if WORD_LEVEL then WORD_TARGET_NAMESPACE else CHAR_TARGET_NAMESPACE;
local NUM_MELS = 80;
local STACK_RATE = 1;

{
    "dataset_reader": {
        "type": "mao-stt",
        "bucket": true,
        "discard_energy_dim": true,
        "input_stack_rate": FRAME_RATE,
        "lazy": true,
        "lexicon_path": LEXICON_PATH,
        "max_frames": 1200,
        "mmap": true,
        "model_stack_rate": 4,
        "num_mel_bins": NUM_MELS,
        "online": false,
        "shard_size": BATCH_SIZE,
        "target_add_start_end_token": true,
        "target_token_indexers": {
            "characters": {
                "type": "just-characters",
                "end_tokens": [
                    "@end@"
                ],
                "namespace": CHAR_TARGET_NAMESPACE,
                "start_tokens": [
                    "@start@"
                ]
            },
            "tokens": {
                "type": "single_id",
                "end_tokens": [
                    "@end@"
                ],
                "namespace": WORD_TARGET_NAMESPACE,
                "start_tokens": [
                    "@start@"
                ]
            }
        },
        "target_tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "just_spaces"
            }
        },
        "word_level": false
    },
    "iterator": {
        "type": "bucket",
        "batch_size": BATCH_SIZE,
        "instances_per_epoch": 1280000,
        "max_instances_in_memory": 4096,
        "padding_noise": 0,
        "sorting_keys": [
            [
                "source_features",
                "dimension_0"
            ],
            [
                "target_tokens",
                "num_tokens"
            ]
        ],
        "track_epoch": true
    },
    "model": {
        "type": "phn_mocha",
        "attention": {
            "type": "stateful",
            "attention_dim": ENCODER_HIDDEN_SIZE,
            "matrix_dim":ENCODER_OUTPUT_SIZE ,
            "num_heads": 1,
            "values_dim": ENCODER_HIDDEN_SIZE,
            "vector_dim": DECODER_HIDDEN_SIZE
        },
        "beam_size": 5,
        "cmvn": "none",
        "cnn": {
            "_pretrained": {
                "archive_file": CTC_MODEL_PATH + "model.tar.gz",
                "freeze": false,
                "module_path": "_cnn"
            }
        },
        "ctc_layer": {
            "type": "ctc",
            "loss_ratio": 0.5,
            "target_namespace": "characters"
        },
        "dec_layers": 1,
        "dep_ratio": 0,
        "encoder": {
            "_pretrained": {
                "archive_file": CTC_MODEL_PATH + "model.tar.gz",
                "freeze": false,
                "module_path": "_encoder"
            }
        },
        "freq_mask_width": 0,
        "from_candidates": false,
        "initializer": [
            [
                ".*_projection_layer.*",
                "prevent"
            ],
            [
                ".*_cnn.*",
                "prevent"
            ],
            [
                ".*_encoder.*",
                "prevent"
            ],
            [
                ".*linear.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*linear.*bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*weight_ih.*",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*weight_hh.*",
                {
                    "type": "orthogonal"
                }
            ],
            [
                ".*bias_ih.*",
                {
                    "type": "zero"
                }
            ],
            [
                ".*bias_hh.*",
                {
                    "type": "lstm_hidden_bias"
                }
            ],
            [
                "_target_embedder.weight",
                {
                    "a": -1,
                    "b": 1,
                    "type": "uniform"
                }
            ]
        ],
        "input_size": NUM_MELS * FRAME_RATE,
        "max_decoding_ratio": 1,
        "max_decoding_steps": 30,
        "phoneme_target_namespace": "phonemes",
        "pos_ratio": 0,
        "projection_layer": {
            "_pretrained": {
                "archive_file": CTC_MODEL_PATH + "/model.tar.gz",
                "freeze": false,
                "module_path": "_joint_ctc_projection_layer"
            }
        },
        "rnnt_layer": {
            "type": "rnnt",
            "hidden_size": 512,
            "input_size": 512,
            "loss_ratio": 0.5,
            "num_layers": 1,
            "target_embedding_dim": 512,
            "target_namespace": "characters"
        },
        "sampling_strategy": "max",
        "target_embedding_dim": 512,
        "target_namespace": "characters",
        "time_mask_max_ratio": 0,
        "time_mask_width": 0
    },
    "train_data_path": std.extVar("TRAIN_DATA_DIR"),
    "validation_data_path": std.extVar("VAL_DATA_DIR"),
    "trainer": {
        "type": "ignore_nan",
        "cuda_device": 0,
        "grad_norm": 2,
        "num_epochs": 300,
        "num_serialized_models_to_keep": 1,
        "optimizer": {
            "type": "adamw",
            "amsgrad": true,
            "lr": 5e-05
        },
        "patience": 20,
        "should_log_learning_rate": true,
        "summary_interval": 100,
        "validation_metric": "-rnnt_wer"
    },
    "vocabulary": {
        "directory_path": std.extVar("VOCAB_PATH")
    },
    "validation_dataset_reader": {
        "type": "mao-stt",
        "bucket": true,
        "discard_energy_dim": true,
        "input_stack_rate": FRAME_RATE,
        "lazy": true,
        "lexicon_path": LEXICON_PATH,
        "max_frames": 1200,
        "mmap": true,
        "model_stack_rate": 4,
        "noskip": true,
        "num_mel_bins": 80,
        "online": false,
        "shard_size": BATCH_SIZE,
        "target_add_start_end_token": true,
        "target_token_indexers": {
            "characters": {
                "type": "just-characters",
                "end_tokens": [
                    "@end@"
                ],
                "namespace": "characters",
                "start_tokens": [
                    "@start@"
                ]
            },
            "tokens": {
                "type": "single_id",
                "end_tokens": [
                    "@end@"
                ],
                "namespace": "tokens",
                "start_tokens": [
                    "@start@"
                ]
            }
        },
        "target_tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "just_spaces"
            }
        },
        "word_level": false
    },
    "validation_iterator": {
        "type": "bucket",
        "batch_size": BATCH_SIZE,
        "max_instances_in_memory": 4096,
        "padding_noise": 0,
        "sorting_keys": [
            [
                "source_features",
                "dimension_0"
            ],
            [
                "target_tokens",
                "num_tokens"
            ]
        ],
        "track_epoch": true
    }
}
