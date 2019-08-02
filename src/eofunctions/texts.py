from utils.text import text_begins, text_ends, text_contains, text_merge


def eo_text_begins(*args, **kwargs):
    return text_begins(*args, **kwargs)


def eo_text_ends(*args, **kwargs):
    return text_ends(*args, **kwargs)


def eo_text_contains(*args, **kwargs):
    return text_contains(*args, **kwargs)


def eo_text_merge(*args, **kwargs):
    return text_merge(*args, **kwargs)
