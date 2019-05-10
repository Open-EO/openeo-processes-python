from dateutil.parser import parse as parse_dt
from datetime import timezone, timedelta, datetime, time
import re


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


def str2time(string, allow_24=False):
    # handle timezone formatting
    if "+" in string:
        string_parts = string.split('+')
        string_parts[-1] = string_parts[-1].replace(':', '')
        string = "+".join(string_parts)

    if "t" in string.lower():  # special handling due to - sign in date string
        if "-" in string[10:]:
            string_parts = string[10:].split('-')
            string_parts[-1] = string_parts[-1].replace(':', '')
            string = string[:10] + "-".join(string_parts)
    else:
        if "-" in string:
            string_parts = string.split('-')
            string_parts[-1] = string_parts[-1].replace(':', '')
            string = "-".join(string_parts)

    if allow_24:
        pattern = re.compile("24:\d{2}:\d{2}")
        pattern_match = re.search(pattern, string)
        if pattern_match:
            old_sub_string = pattern_match.group()
            new_sub_string = "23" + old_sub_string[2:]
            string = string.replace(old_sub_string, new_sub_string)

    rfc3339_time_formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%Sz", "%Y-%m-%dt%H:%M:%SZ",
                            "%Y-%m-%dt%H:%M:%Sz", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dt%H:%M:%S%z",
                            "%H:%M:%SZ", "%H:%M:%S%z"]
    date_time = None
    for i, used_time_format in enumerate(rfc3339_time_formats):
        try:
            date_time = datetime.strptime(string, used_time_format)
            if date_time.tzinfo is None:
                date_time = date_time.replace(tzinfo=timezone.utc)
            if i == 0:
                date_time_max = datetime.combine(date_time.date(), time()) + timedelta(hours=24) \
                                - timedelta(seconds=1)
                date_time = (datetime.combine(date_time.date(), time()).replace(tzinfo=timezone.utc),
                             date_time_max.replace(tzinfo=timezone.utc))
            break
        except:
            continue

    if date_time and allow_24:
        date_time += timedelta(hours=1)

    return date_time