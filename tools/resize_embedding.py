from itertools import chain
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('embedding_weight_key')
    parser.add_argument('new_vocab_file')
    parser.add_argument('output_model_file')
    args = parser.parse_args()

    with open(args.new_vocab_file) as fp:
        new_tokens = fp.read().splitlines()

    num_tokens = len(new_tokens) + 1 # plus one padding token

    model = torch.load(args.model_file)
    weight = model[args.embedding_weight_key]
    if len(weight.shape) == 2:
        _, embedding_dim = weight.shape
        weight = torch.FloatTensor(num_tokens, embedding_dim)
    elif len(weight.shape) == 1:
        weight = torch.FloatTensor(num_tokens)
    else:
        raise NotImplementedError

    torch.nn.init.zeros_(weight)
    model[args.embedding_weight_key] = weight
    torch.save(model, args.output_model_file)
