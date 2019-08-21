local BATCH_SIZE = 128;
local EMBEDDING_DIM = 256;
local ENCODER_HIDDEN_SIZE = 512;
local DECODER_HIDDEN_SIZE = 512;
local TARGET_NAMESPACE = "target_tokens";
local NUM_THREADS = 8;
local NUM_GPUS = 1;

local BASE_READER = {
  "type": "mt",
  "source_affix": "de",
  "target_affix": "en",
  "source_tokenizer": {
    "type": "word",
    "word_splitter": {
      "type": "just_spaces"
    }
  },
  "target_tokenizer": {
    "type": "word",
    "word_splitter": {
      "type": "just_spaces"
    }
  },
  "source_token_indexers": {
    "tokens": {
      "type": "single_id",
      "namespace": "source_tokens"
    }
  },
  "target_token_indexers": {
    "tokens": {
      "type": "single_id",
      "namespace": "target_tokens"
    }
  },
  "lazy": true
};

local BASE_ITERATOR = {
  "type": "bucket",
  "max_instances_in_memory": 1024 * NUM_GPUS,
  "batch_size": BATCH_SIZE,
  "sorting_keys": [["source_tokens", "num_tokens"],
                   ["target_tokens", "num_tokens"]],
  "maximum_samples_per_batch": ["num_tokens", 16384],
  "track_epoch": true
};

{
  "dataset_reader": {
    "type": "multiprocess",
    "base_reader": BASE_READER,
    "num_workers": 3,
    "output_queue_size": 1024 * 3 * NUM_GPUS
  },
  "train_data_path": "data/translation/tok-wps/train/*.de-en.en",
  "validation_data_path": "data/translation/tok-wps/validation/*.en",
  "test_data_path": "data/translation/tok-wps/test/*..en",
  "vocabulary": {
    "directory_path": "data/translation/tok-wps/vocab"
  },
  "model": {
    "type": "mt_seq2seq_mocha",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "embedding_dim": EMBEDDING_DIM,
          "trainable": true
        },
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": EMBEDDING_DIM,
      "hidden_size": ENCODER_HIDDEN_SIZE,
      "num_layers": 2
    },
    "max_decoding_steps": 100,
    "target_embedding_dim": EMBEDDING_DIM,
    "decoder_hidden_dim": DECODER_HIDDEN_SIZE,
    "target_namespace": "target_tokens",
    "attention": {
      "type": "bilinear",
      "matrix_dim": ENCODER_HIDDEN_SIZE,
      "vector_dim": DECODER_HIDDEN_SIZE,
    },
    "splits": [1000, 10000],
    "beam_size": 10
  },
  "iterator": {
    "type": "multiprocess",
    "base_iterator": BASE_ITERATOR,
    "num_workers": NUM_THREADS,
    "output_queue_size": 1024 * 3 * NUM_GPUS
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 10,
    "cuda_device": 0,
    "validation_metric": "+BLEU",
    "optimizer": {
      "type": "adamw",
      "lr": 0.0003,
      "amsgrad": true,
      "weight_decay": 0.01
    }
  }
}
