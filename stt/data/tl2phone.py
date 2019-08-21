# translate every single word transcribed with Tai-Lo (Taiwan Romanization System) to phoneme-separated ones in a file.
# Author: Chung-Yi Li

import re
import argparse
import yaml
import pdb


def maximal_matching(phones):
    initials = phones['initials']
    finals = phones['finals']
    seq = []

    def _fn(gram):
        i = 0
        while i <= len(gram):
            i += 1
            if gram[:i] not in phones['initials']:
                break
        i -= 1
        if i > 0:
            ini = initials.get(gram[:i])
            if gram[:i] in ['tsi', 'ji', 'tshi', 'si']:
                # to tackle with tsi ji tshi si problem
                gram = 'i' + gram
            seq.append(ini)
        while i < len(gram):
            rest = ''.join(gram[i:])
            maybe = finals.get(rest)
            if maybe is None:
                seq.append(finals[gram[i]])
            else:
                seq.append(finals[rest])
                break
            i += 1
        return seq
    return _fn


def split_phoneme(phones, TONE):
    def _fn(gram):
        tone = re.findall('\d', gram)
        if tone:
            tone = tone[0]
        gram = re.sub('\d', '', gram)
        parse = maximal_matching(phones)
        seq = parse(gram)
        seq = ' '.join(seq)
        return seq
    return _fn


def split_phoneme_line(initials, tone):
    split = split_phoneme(initials, tone)

    def _fn(line):
        line = line.lower()
        line = re.sub("\[.*\]", "", line)
        line = re.sub("\/[^\s]+", "", line)
        line = re.sub("\(.*\)", "", line)
        line = re.sub("\".*\"", "", line)
        line = re.sub("\*\*[^\s]+", "", line)
        line = re.sub("κ", "k", line)
        line = re.sub("[;,\?\:\"\(\)\[\]\!.\n]+", "", line)
        line = re.sub("→", "", line)
        #line = re.sub("[\W^\-_]+", "", line)
        if line == '':
            return ""
        grams = re.split('[\-\s]+', line)
        phonemes = []
        for gram in grams:
            if len(gram) == 0:
                continue
            try:
                splits = split(gram)
                phonemes.append(splits)
            except:
                return ""
        return ' '.join(phonemes)

    return _fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-it', help='input file', default='data.tsm')
    parser.add_argument('-ot', help='output file', default='phoneme.tsm')
    parser.add_argument('-inif', default='phonemes.yaml',
                        help='path for initials.txt which stores initials for the language.')
    parser.add_argument('-t', default='IGNORE',
                        choices=['KEEP', 'IGNORE', 'SPLIT'])
    args = parser.parse_args()
    with open(args.it) as f:
        tlines = f.readlines()
    with open(args.inif) as f:
        phones = yaml.load(f)
    split_line = split_phoneme_line(phones, args.t)
    processed = []
    for idx, tline in enumerate(tlines):
        processed.append(split_line(tline))
    #processed = [split_line(tline) for tline in tlines]
    with open(args.ot, 'w') as ft:
        for tline in processed:
            ft.write(tline + '\n')
