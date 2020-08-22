import json
import torch


def lstm_hidden_bias(tensor: torch.Tensor) -> None:
    """
    Initialize the biases of the forget gate to 1, and all other gates to 0,
    following Jozefowicz et al., An Empirical Exploration of Recurrent Network Architectures
    """
    # gates are (b_hi|b_hf|b_hg|b_ho) of shape (4*hidden_size)
    tensor.data.zero_()
    hidden_size = tensor.shape[0] // 4
    tensor.data[hidden_size:(2 * hidden_size)] = 1.0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('weight_init_json')
    parser.add_argument('output_model_file')
    args = parser.parse_args()

    model = torch.load(args.model_file)
    with open(args.weight_init_json) as fp:
        weight_init_dict = json.load(fp)
    for key, method in weight_init_dict.items():
        weight = model[key]
        if isinstance(method, dict):
            getattr(torch.nn.init, f"{method['type']}_")(weight, **method["params"])
        elif method == "lstm_hidden_bias":
            lstm_hidden_bias(weight)
        else:
            getattr(torch.nn.init, f"{method}_")(weight)
        device = weight.data.device
        model[key] = weight
    torch.save(model, args.output_model_file)
