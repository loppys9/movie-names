import logging
import os
import pickle
import random

import requests
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)

LABLE_LENGTH = 96
#EXTS = [".mkv", ".mp4", ".divx", ".iso"]
EXTS = [".mkv"]


def _get_name_and_year(attrs: list):
    name = None
    year = None

    for attr in attrs:
        if attr["@attributes"]["name"] == "imdbtitle":
            name = attr["@attributes"]["value"]
        if attr["@attributes"]["name"] == "imdbyear":
            year = attr["@attributes"]["value"]

    if name is None or year is None:
        raise ValueError("Missing attribute information.")

    log.info(name, year)

    return name, year


def _create_downloaded_filename(title, ext):
    return title + ext


def _create_desired_filename(name, year, ext):
    return f"{name} ({year}){ext}"


def _get_data(api_key, offset, cat=None, query=None):
    url = "https://api.nzbplanet.net/api"
    if cat:
        params = {"apikey": api_key, "t": "movie", "extended": 1, "o": "json", "offset": offset, "cat": cat}
    if query:
        params = {"apikey": api_key, "t": "movie", "extended": 1, "o": "json", "offset": offset, "q": query}
    r = requests.get(url, params=params)
    log.info(r.url)

    return r.json()


def _parse_data(data, mapping):
    total = 0
    try:
        channel = data["channel"]
        item = channel["item"]
        total = int(channel["response"]["@attributes"]["total"])
    except:
        print(data)
        return

    for entry in item:
        title = entry["title"]
        attr = entry["attr"]

        try:
            name, year = _get_name_and_year(attr)
        except:
            continue

        ext = random.choice(EXTS)
        mapping.add((_create_downloaded_filename(title, ext), _create_desired_filename(name, year, ext)))

    return total


def get_movie_list(api_key) -> tuple:
    cats = [(2040, 10000), (2030, 10000), (2010, 6000), (2070, 10000)]
    have = []
    want = []

    for cat in cats:
        for offset in tqdm(range(0, cat[1] // 100)):
            data = _get_data(api_key, offset * 100, cat[0])
            _parse_data(data, have, want)

    print(len(have), len(want))

    return have, want


def _search_terms(api_key, mapping):
    terms = ["w", "west", "wedding", "war", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "u", "v", "x", "y", "z"]

    for term in terms:
        data = _get_data(api_key, 0, query=term)
        results = _parse_data(data, mapping)
        for offset in tqdm(range(1, results // 100)):
            data = _get_data(api_key, offset * 100, query=term)
            _parse_data(data, mapping)


def append_movie_list(api_key, filename):
    have = []
    want = []
    try:
        have, want = read_data(filename)
    except:
        pass

    mapping = set()
    for a, b in zip(have, want):
        mapping.add((a, b))

    cats = [(2040, 10000), (2030, 10000), (2010, 6000), (2070, 10000)]
    """
    for cat in cats:
        for offset in tqdm(range(0, cat[1] // 100)):
            data = _get_data(api_key, offset * 100, cat[0])
            _parse_data(data, mapping)
    """

    _search_terms(api_key, mapping)

    have = []
    want = []
    for val in mapping:
        have.append(val[0])
        want.append(val[1])

    print(len(have), len(want))

    write_data(have, want, filename)


def write_more_data(have, want, filename):
    already_have, already_want = read_data(filename)

    have += already_have
    want += already_want

    write_data(have, want, filename)


def write_data(have, want, filename, tensor=False):
    have_file = filename + ".have"
    want_file = filename + ".want"

    with open(have_file, "wb") as f:
        if tensor:
            torch.save(have, f)
        else:
            pickle.dump(have, f)

    with open(want_file, "wb") as f:
        if tensor:
            torch.save(want, f)
        else:
            pickle.dump(want, f)


def read_data(filename, tensor=False, device=None) -> tuple:
    have_file = filename + ".have"
    want_file = filename + ".want"

    with open(have_file, "rb") as f:
        if tensor:
            have = torch.load(f, map_location=device)
        else:
            have = pickle.load(f)

    with open(want_file, "rb") as f:
        if tensor:
            want = torch.load(f, map_location=device)
        else:
            want = pickle.load(f)

    return have, want


def to_tensor(vals, label=False):
    length = 256
    if label:
        length = LABLE_LENGTH

    tensors = []

    for val in vals:
        lst = list(bytes(val, "utf-8"))
        lst += [0] * (length - len(lst))
        tens = torch.tensor(lst, dtype=torch.float32)
        tensors.append(tens)

    tens = torch.stack(tensors, dim=0)

    return tens


def remove_long_names(have, want):
    bad_ind = []
    bad_cnt = 0
    for ind, w in enumerate(want):
        if w[0].lower() != have[ind][0].lower():
            bad_ind.append(ind)
        if len(w) > LABLE_LENGTH:
            bad_ind.append(ind)
        if ':' in w:
            want[ind] = w.replace(':', "")
                #bad_ind.append(ind)
        if "'" in w:
            want[ind] = w.replace("'", "")
        if "!" in w:
            want[ind] = w.replace("'", "")

    have = [x for i, x in enumerate(have) if i not in bad_ind]
    want = [x for i, x in enumerate(want) if i not in bad_ind]

    return have, want


def _data_augmentation(have, want):
    opts = set()

    for h in have:
        vals = h.split(".")
        b = h.split()
        if len(b) > len(vals):
            vals = b
        for val in reversed(vals[:-1]):
            if val.isnumeric():
                break
            opts.add(val)

    delims = [".", ".",  " "]
    list_opts = list(opts)
    more_have = []
    more_want = []
    multiplier = 4
    for _ in range(multiplier):
        for w in want:
            delim = random.choice(delims)
            a = w.split(".")[0]
            name = w.split(".")[0].replace("(", "").replace(")", "").replace(" ", delim)
            extras = random.randint(3, 8)
            for _ in range(extras):
                name += delim + random.choice(list_opts)

            ext = random.choice(EXTS)
            name += ext
            if len(name) < 257:
                more_have.append(name)
                more_want.append(a+ext)

    have += more_have
    want += more_want


def split_data(filename):
    have, want = read_data(filename)
    have, want = remove_long_names(have, want)
    print(len(have), len(want))
    _data_augmentation(have, want)
    print(len(have), len(want))

    test_split = 0.2

    both = list(zip(have, want))
    random.shuffle(both)
    have, want = zip(*both)

    test_len = int(len(have) * test_split)

    test_have = have[:test_len]
    test_want = want[:test_len]

    train_have = have[test_len:]
    train_want = want[test_len:]

    head, tail = os.path.split(filename)

    train_tail = "train_" + tail
    test_tail = "test_" + tail
    train_file = os.path.join(head, train_tail)
    test_file = os.path.join(head, test_tail)

    train_have = to_tensor(train_have)
    train_want = to_tensor(train_want, True)
    test_have = to_tensor(test_have)
    test_want = to_tensor(test_want, True)

    write_data(train_have, train_want, train_file, True)
    write_data(test_have, test_want, test_file, True)

    return train_file, test_file
