from pathlib import Path
import argparse
import re
import json
from collections import Counter

def get_text(audio_path: Path, txt_dir: Path, field: str):
    wav_dir, speaker, utt = audio_path.parts
    prefix = re.match("(.*)\-\d+\.wav", utt).group(1)
    json_filename = Path(txt_dir).joinpath(speaker).joinpath(f"{prefix}.json")
    with open(json_filename) as json_file:
        txt_obj = json.load(json_file)
    return txt_obj[field]

def flatten(l):
    return [item for sublist in l for item in sublist]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('--splitter', default='space')
    parser.add_argument('output_file')
    args = parser.parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    if args.splitter == 'space':
        split = lambda utt: utt.split()
    else:
        raise NotImplementedError

    with open(input_file) as fp:
        utts = fp.read().splitlines()

    utts = [split(utt) for utt in utts]
    flat_utts = flatten(utts)
    tokens = [word for word, count in Counter(flat_utts).most_common()]
    tokens = ["@@UNKNOWN@@", "@start@", "@end@"] + tokens

    with open(output_file, "w") as fp:
        fp.write("\n".join(tokens))

