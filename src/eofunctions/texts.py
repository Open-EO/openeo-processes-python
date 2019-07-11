
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
    separator = str(separator)
    return separator.join(data)

# old function, not used anymore
# def str2time(string):
#     rfc3339_pattern = "^([0-9]+)-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])[Tt]([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9]|60)(\.[0-9]+)?(([Zz])|([\+|\-]([01][0-9]|2[0-3]):[0-5][0-9]))$"
#     rfc3339_regex = re.compile(rfc3339_pattern)
#     date_time = None
#     if re.match(rfc3339_regex, string):
#         date_time = parse_dt(string)
#         if date_time.tzinfo is None:
#             date_time = date_time.replace(tzinfo=timezone.utc)
#         rfc3339_date_pattern = "^([0-9]+)-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])$"
#         rfc3339_date_regex = re.compile(rfc3339_date_pattern)
#         if re.match(rfc3339_date_regex, string):
#             date_time_max = date_time + timedelta(hours=24) - timedelta(seconds=1)
#             date_time = (date_time, date_time_max)

#    return date_time