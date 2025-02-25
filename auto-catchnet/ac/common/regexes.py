import re


def get_number_seq(name, num_repeat_at_least: int = 3):
    pattern = r"[0-9]{" + str(num_repeat_at_least) + ",}"
    match = re.search(pattern, name)
    if match:
        return match[0]
    return -1
