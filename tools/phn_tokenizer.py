import re
import numpy as np

vowels = ['a', 'e', 'i', 'o', 'oo', 'u', 'm', 'n', 'ng']
vowel_suffixes = ['h', 'nn']
consonants = [
    'p',   'ph',   'm',  'b',  # bilabial
    't',   'th',   'n',  'l',  # alveolar
    'ts',  'tsh',  's',  'j',  # alveolar affricate
    'tsi', 'tshi', 'si', 'ji', # alveolo-palatal affricate
    'k',   'kh',   'h',  'g', 'ng', 
]

onsets = vowels + consonants
codas = ['p', 't', 'k', 'h', 'nn', 'm', 'n', 'ng']
nucleuses = vowels


def tokenize_to_phn(utt):
    return [phn_tokenize(syl) for syl in re.split("\W", utt)]

def phn_tokenize(syllable_with_tone):
    import pdb
    pdb.set_trace()
    assert syllable_with_tone[-1].isdigit()
    syllable, tone = syllable_with_tone[:-1], syllable_with_tone[-1]
    remainders = syllable
    phns = []
    has_tone = False

    # onset
    possible_onsets = list(filter(lambda onset: remainders.startswith(onset), onsets))
    if possible_onsets:
        idx = np.argmax(list(map(len, possible_onsets)))
        onset = possible_onsets[idx]
        remainders = remainders[len(onset):]
        # append tone to vowel
        if onset in vowels and not has_tone:
            onset += tone
            has_tone = True
        phns.append(onset)

    if len(remainders) > 0:
        possible_nucleuses = list(filter(lambda nucleus: remainders.startswith(nucleus), nucleuses))
        if possible_nucleuses:
            idx = np.argmax(list(map(len, possible_nucleuses)))
            nucleus = possible_nucleuses[idx]
            remainders = remainders[len(nucleus):]
            if not has_tone:
                nucleus += tone
            phns.append(nucleus)

    if len(remainders) > 0:
        possible_codas = list(filter(lambda coda: remainders.startswith(coda), codas))
        if possible_codas:
            idx = np.argmax(list(map(len, possible_codas)))
            coda = possible_codas[idx]
            remainders = remainders[len(coda):]
            phns.append(coda)

    print(syllable_with_tone, phns)
    return phns
