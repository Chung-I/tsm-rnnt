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

local NUM_GPUS = 1;
local NUM_THREADS = 2;
local BATCH_SIZE = 4096;
local BASE_ITERATOR = {
  "type": "bucket",
  "max_instances_in_memory": BATCH_SIZE * NUM_GPUS,
  "batch_size": BATCH_SIZE,
  "sorting_keys": [["source_tokens", "num_tokens"],
                   ["target_tokens", "num_tokens"]],
  "maximum_samples_per_batch": ["num_tokens", 65536]
};

{
  "dataset_reader": {
    "type": "multiprocess",
    "base_reader": BASE_READER,
    "num_workers": NUM_THREADS,
    "output_queue_size": BATCH_SIZE * 2
  },
  "train_data_path": "data/translation/tok-wps/train/*.de-en.en",
  "validation_data_path": "data/translation/tok-wps/validation/*.de-en.en",
  "test_data_path": "data/translation/tok-wps/test/*.de-en.en",
  "vocabulary": {
    "directory_path": "data/translation/tok-wps/vocab"
  },
  "model": {
    "type": "simple_seq2seq",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "embedding_dim": 512,
          "trainable": true
        },
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 512,
      "hidden_size": 512,
      "num_layers": 4
    },
    "max_decoding_steps": 100,
    "target_embedding_dim": 512,
    "target_namespace": "target_tokens",
    "attention": {
      "type": "dot_product"
    },
    "beam_size": 10
  },
  "iterator": {
    "type": "multiprocess",
    "base_iterator": BASE_ITERATOR,
    "num_workers": NUM_THREADS,
    "output_queue_size": BATCH_SIZE * 2
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 10,
    "cuda_device": -1,
    "optimizer": {
      "type": "adamw",
      "lr": 0.0003,
      "amsgrad": true,
      "weight_decay": 0.01
    }
  }
}
