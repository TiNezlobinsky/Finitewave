import tqdm


def nop(iterable=None, *a, **k):
    return iterable


tqdm.tqdm = nop

conf = {}
