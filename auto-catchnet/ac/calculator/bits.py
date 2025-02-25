def bits_to_kb(n_bits):
    return n_bits / nth_bytes(1)


def bits_to_mb(n_bits):
    return n_bits / nth_bytes(2)


def bits_to_gb(n_bits):
    return n_bits / nth_bytes(3)


def nth_bytes(nth):
    return 8 * 1024 ** nth
