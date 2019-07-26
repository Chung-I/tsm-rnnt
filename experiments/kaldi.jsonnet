local BATCH_SIZE = 32;
local FRAME_RATE = 3;
local VOCAB_PATH = std.extVar('MODEL_PATH') + "/vocabulary";
local TARGET_NAMESPACE = "target_tokens";

{
  "dataset_reader": {
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
  },
  "train_data_path": "/home/nlpmaster/Works/egs/aidatatang_200zh/s5/fbank/raw_fbank_pitch_train.1.scp",
  "validation_data_path": "/home/nlpmaster/Works/egs/aidatatang_200zh/s5/fbank/raw_fbank_pitch_dev.1.scp",
  "test_data_path": "/home/nlpmaster/Works/egs/aidatatang_200zh/s5/fbank/raw_fbank_pitch_test.1.scp",
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
    "vocab_path": VOCAB_PATH,
    "target_namespace": TARGET_NAMESPACE,
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : BATCH_SIZE,
    "sorting_keys": [["source_features", "dimension_0"],
                     [TARGET_NAMESPACE, "num_tokens"]]
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
