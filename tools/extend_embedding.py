from itertools import chain
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('embedding_weight_key')
    parser.add_argument('old_vocab_file')
    parser.add_argument('new_vocab_file')
    parser.add_argument('output_model_file')
    args = parser.parse_args()

    with open(args.old_vocab_file) as fp:
        old_tokens = fp.read().splitlines()

    with open(args.new_vocab_file) as fp:
        new_tokens = fp.read().splitlines()

    num_tokens_to_add = len(new_tokens) - len(old_tokens)

    model = torch.load(args.model_file)
    weight = model[args.embedding_weight_key]
    if len(weight.shape) == 2:
        _, embedding_dim = weight.shape
        extra_weight = torch.FloatTensor(num_tokens_to_add, embedding_dim)
    elif len(weight.shape) == 1:
        extra_weight = torch.FloatTensor(num_tokens_to_add)
    else:
        raise NotImplementedError
    torch.nn.init.zeros_(extra_weight)
    device = weight.data.device
    extended_weight = torch.cat([weight.data, extra_weight.to(device)], dim=0)
    model[args.embedding_weight_key] = extended_weight
    torch.save(model, args.output_model_file)
