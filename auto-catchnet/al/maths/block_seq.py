import math


def get_block_range(block_size, batch_size, batch_index):
    start_pos, end_pos = batch_index * batch_size, (batch_index + 1) * batch_size
    first_block_idx = int(math.floor(start_pos / block_size))

    block_global_start_pos = block_size * first_block_idx
    start_pos_in_block = start_pos - block_global_start_pos
    end_pos_in_block = start_pos_in_block + batch_size

    if block_size * (first_block_idx+1) < end_pos:
        return [first_block_idx, first_block_idx+1], start_pos_in_block, end_pos_in_block
    return [first_block_idx], start_pos_in_block, end_pos_in_block
