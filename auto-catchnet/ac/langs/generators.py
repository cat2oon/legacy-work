
def chunks(list_to_chunk, chunk_size):
    for i in (range(0, len(list_to_chunk), chunk_size)):
        yield list_to_chunk[i:i + chunk_size]


