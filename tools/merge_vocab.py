from itertools import chain
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('primary_file')
    parser.add_argument('secondary_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    with open(args.primary_file) as fp:
        primary_tokens = fp.read().splitlines()

    with open(args.secondary_file) as fp:
        secondary_tokens = fp.read().splitlines()

    tokens_to_add = list(set(secondary_tokens) - set(primary_tokens))
    with open(args.output_file, "w") as fp:
        for token in chain(primary_tokens, tokens_to_add):
            fp.write(f"{token}\n")
