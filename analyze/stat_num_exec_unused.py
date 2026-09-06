"""Unused experiment helper preserved without rewriting in B23."""

import re


def extract_num_cores(filename):
    pattern = r'(?P<num_cores>\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group('num_cores'))
    else:
        return -1
