from pathlib import Path
import json
from functools import partial
import torchaudio
from torchaudio.transforms import Resample
import multiprocessing as mp
from collections import deque
import psutil
import tqdm


def get_bar(*args, **kwargs):
    return tqdm.tqdm(*args, ncols=80, **kwargs)


def preprocess(mp3):
    sample_rate = 16000
    root_dir = Path("/home/nlpmaster/ssd-1t/corpus/TaiBible/PKL")
    new_dir = Path("/home/nlpmaster/ssd-1t/corpus/TaiBible/PKL_wav")
    y, sr = torchaudio.load(str(root_dir.joinpath(mp3)))
    resample = Resample(orig_freq=sr, new_freq=sample_rate)
    resampled_y = resample(y)
    wavfile = new_dir.joinpath(mp3)
    wavfile.parent.mkdir(exist_ok=True)
    torchaudio.save(str(wavfile), resampled_y, sample_rate=sample_rate)
    return str(wavfile)


def get_usable_cpu_count():
    p = psutil.Process()
    ret = 0
    with p.oneshot():
        ret = len(p.cpu_affinity())
    return ret


USABLE_CPU_COUNT = get_usable_cpu_count()


def mp_progress_map(func, arg_iter, num_workers=USABLE_CPU_COUNT):
    rs = []
    pool = mp.Pool(processes=num_workers)
    for args in arg_iter:
        rs.append(pool.apply_async(func, args))
    pool.close()

    rets = []
    for r in get_bar(rs):
        rets.append(r.get())
    pool.join()
    return rets


def tailo2phone_factory():
    import yaml
    from stt.data.POJ_TL import poj_tl
    from stt.data.tl2phone import split_phoneme_line
    with open('stt/data/phonemes-cl.yaml') as fp:
        phoneme_mapping = yaml.load(fp)
    split_line = split_phoneme_line(phoneme_mapping, 'IGNORE')

    def func(tailo):
        pojt = poj_tl(tailo)
        tls = pojt.pojt_pojs().pojs_tls()
        text = split_line(tls)
        return text

    return func


def pkl(root):

    #root_dir = Path("/home/nlpmaster/ssd-1t/corpus/TaiBible/PKL")
    root_dir = Path(root)
    tailo2phone = tailo2phone_factory()
    # new_dir.mkdir(exist_ok=True)

    utts_file = root_dir.joinpath('data.json')

    with open(utts_file) as fp:
        utts = json.load(fp)
    utt_dict = {utt["mp3"]: utt["tailo"] for utt in utts}
    for chap in root_dir.iterdir():
        for wavfile in chap.glob("*.mp3"):
            tailo = utt_dict[str(wavfile.relative_to(root_dir))]
            text = tailo2phone(tailo)
            yield str(wavfile), text

    # for utt in utts:
    #    text = tailo2phone(utt["tailo"])
    #    wavfile = root_dir.joinpath(utt["mp3"])
    #    if not wavfile.exists():
    #        print("wavfile {} doesn't exist; dropping ...".format(wavfile))
    #        continue
    #    yield wavfile, text
    # if resample:
    #    a = mp_progress_map(preprocess, ((o,) for o in mp3s))


CORPORA = {
    "/home/nlpmaster/ssd-1t/corpus/TaiBible/PKL_wav": pkl,
    "/home/nlpmaster/ssd-1t/corpus/TaiBible/train_noise_augmented": pkl,
    "/home/nlpmaster/ssd-1t/corpus/TaiBible/dev_noise_augmented": pkl
}
