def is_micro_search(search_for):
    return search_for.upper() == "MICRO"


def get_xy_id_and_op_id(arc_seq, cell_id):
    x_id = arc_seq[4 * cell_id + 0]
    x_op = arc_seq[4 * cell_id + 1]
    y_id = arc_seq[4 * cell_id + 2]
    y_op = arc_seq[4 * cell_id + 3]

    return x_id, x_op, y_id, y_op