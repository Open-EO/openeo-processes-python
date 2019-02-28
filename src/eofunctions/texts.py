from dateutil.parser import parse as parse_dt
from datetime import timezone, timedelta


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


def str2time(string):
    date_time = None
    try:
        date_time = parse_dt(string)
        if date_time.tzinfo is None:
            date_time = date_time.replace(tzinfo=timezone.utc)
    except:
        pass

    return date_time

# old function, not used anymore
# def str2time(string):
#     # handle timezone formatting
#     if "+" in string:
#         string_parts = string.split('+')
#         string_parts[-1] = string_parts[-1].replace(':', '')
#         string = "+".join(string_parts)
#
#     if "-" in string:
#         string_parts = string.split('-')
#         string_parts[-1] = string_parts[-1].replace(':', '')
#         string = "-".join(string_parts)
#
#     used_time_formats = ["%Y%m%d%H%i%s", "%Y%m%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H%M%S", "%Y-%m-%d %H:%M:%S %z",
#                          "%Y-%m-%d %H:%M:%S%z", "%H:%M:%S %z", "%H:%M:%S%z"]
#     date_time = None
#     for used_time_format in used_time_formats:
#         try:
#             date_time = datetime.strptime(string, used_time_format)
#         except:
#             continue
#
#     return date_time