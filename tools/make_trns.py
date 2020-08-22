from pathlib import Path
import argparse
import re
import json
from phn_tokenizer import tokenize_to_phn


def char_tokenize(utt):
    utt = re.sub("\W+", "", utt)
    return list(utt)

def get_text(audio_path: Path, txt_dir: Path, field: str):
    wav_dir, speaker, utt = audio_path.parts
    prefix = re.match("(.*)\-\d+\.wav", utt).group(1)
    json_filename = Path(txt_dir).joinpath(speaker).joinpath(f"{prefix}.json")
    with open(json_filename) as json_file:
        txt_obj = json.load(json_file)
    return txt_obj[field]

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--tokenizer', default='char')
    parser.add_argument('--field', default="漢羅台文")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    with open(input_dir.joinpath("refs.txt")) as fp:
        paths = map(Path, fp.read().splitlines())

    if args.tokenizer == "char":
        tokenize = char_tokenize
    elif args.tokenizer == "phn":
        tokenize = tokenize_to_phn
    else:
        raise NotImplementedError

    with open(output_dir.joinpath(f"{args.field}.txt"), "w") as fp:
        for path in paths:
            text = get_text(path, input_dir.joinpath('json'), args.field)
            tokenized_text = tokenize(text)
            fp.write(f"{' '.join(tokenized_text)}\n")

