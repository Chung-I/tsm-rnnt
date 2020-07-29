{
    "dataset_reader": {
        "type": "mao-stt",
        "bucket": true,
        "discard_energy_dim": true,
        "input_stack_rate": 1,
        "lazy": true,
        "lexicon_path": "data/lexicon.txt",
        "max_frames": 1200,
        "mmap": true,
        "model_stack_rate": 4,
        "num_mel_bins": 80,
        "online": false,
        "shard_size": 16,
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
    "iterator": {
        "type": "bucket",
        "batch_size": 32,
        "max_instances_in_memory": 32,
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
        "type": "old_phn_mocha",
        "attention": {
            "type": "stateful",
            "attention_dim": 512,
            "matrix_dim": 1024,
            "num_heads": 1,
            "values_dim": 512,
            "vector_dim": 512
        },
        "beam_size": 5,
        "cmvn": "utt",
        "cnn": {
            "type": "cnn",
            "hidden_channel": 32,
            "in_channel": 1,
            "num_layers": 2
        },
        "ctc_keep_eos": true,
        "dec_layers": 1,
        "dep_ratio": 0,
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "hidden_size": 512,
            "input_size": 640,
            "num_layers": 4
        },
        "freq_mask_width": 0,
        "from_candidates": false,
        "initializer": [
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
        "input_size": 80,
        "joint_ctc_ratio": 0,
        "max_decoding_steps": 30,
        "n_pretrain_ctc_epochs": 0,
        "phoneme_target_namespace": "phonemes",
        "pos_ratio": 0,
        "sampling_strategy": "max",
        "target_embedding_dim": 512,
        "target_namespace": "characters",
        "time_mask_max_ratio": 0,
        "time_mask_width": 0
    },
    "train_data_path": std.extVar("DATA_ROOT") + "TSM/trains",
    "validation_data_path": std.extVar("DATA_ROOT") + "TSM/valids",
    "trainer": {
        "type": "ignore_nan",
        "cuda_device": 0,
        "grad_norm": 4,
        "learning_rate_scheduler": {
            "type": "multi_step",
            "gamma": 0.5,
            "milestones": [
                60,
                72,
                84
            ]
        },
        "num_epochs": 6,
        "num_serialized_models_to_keep": 1,
        "optimizer": {
            "type": "adadelta",
            "eps": 1e-08,
            "lr": 1
        },
        "patience": 20,
        "should_log_learning_rate": true,
        "summary_interval": 100,
        "validation_metric": "+BLEU"
    },
    "vocabulary": {
        "directory_path": "data/vocabulary/phn_level"
    },
    "validation_dataset_reader": {
        "type": "mao-stt",
        "bucket": true,
        "discard_energy_dim": true,
        "input_stack_rate": 1,
        "lazy": true,
        "lexicon_path": "data/lexicon.txt",
        "max_frames": 1200,
        "mmap": true,
        "model_stack_rate": 4,
        "noskip": true,
        "num_mel_bins": 80,
        "online": false,
        "shard_size": 16,
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
    }
}
