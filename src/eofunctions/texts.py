def text_begins(data, pattern, case_sensitive=True):
    if case_sensitive:
        return data.startswith(pattern)
    else:
        return data.lower().startswith(pattern.lower())

def text_ends(data, pattern, case_sensitive=True):
    if case_sensitive:
        return data.endswith(pattern)
    else:
        return data.lower().endswith(pattern.lower())

def text_contains(data, pattern, case_sensitive=True):
    if case_sensitive:
        return pattern in data
    else:
        return pattern.lower() in data.lower()

def text_merge(data, separator=''):
    data = [str(elem).lower() if type(elem) != str else elem for elem in data]

    return separator.join(data)