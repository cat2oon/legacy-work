import argparse

import numpy as np
import pygraphviz as pgv


def construct_block(graph, num_block, ops):
    ops_name = ["conv 3x3", "conv 5x5", "avg pool", "max pool", "identity", "add", "concat"]

    for i in range(0, 2):
        graph.add_node(num_block * 10 + i + 1,
                       label="{}".format(ops_name[ops[2 * i + 1]]),
                       color='black',
                       fillcolor='yellow',
                       shape='box',
                       style='filled')

    # graph.add_subgraph([num_block*10+1, num_block*10+2], rank='same')

    graph.add_node(num_block * 10 + 3,
                   label="Add",
                   color='black',
                   fillcolor='greenyellow',
                   shape='box',
                   style='filled')

    graph.add_subgraph([num_block * 10 + 1, num_block * 10 + 2, num_block * 10 + 3],
                       name='cluster_s{}'.format(num_block))

    for i in range(0, 2):
        graph.add_edge(num_block * 10 + i + 1, num_block * 10 + 3)


def connect_block(graph, num_block, ops, output_used):
    for i in range(0, 2):
        graph.add_edge(ops[2 * i] * 10 + 3, (num_block * 10) + i + 1)
        output_used.append(ops[2 * i] * 10 + 3)


def creat_graph(cell_arc):
    G = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open', rankdir='TD')

    # creat input
    G.add_node(3, label="H[i-1]", color='black', shape='box')
    G.add_node(13, label="H[i]", color='black', shape='box')
    G.add_subgraph([3, 13], name='cluster_inputs', rank='same', rankdir='TD', color='white')

    # creat blocks
    for i in range(0, len(cell_arc)):
        construct_block(G, i + 2, cell_arc[i])

    # connect blocks to each other
    output_used = []
    for i in range(0, len(cell_arc)):
        connect_block(G, i + 2, cell_arc[i], output_used)

    # creat output
    G.add_node((len(cell_arc) + 2) * 10 + 3,
               label="Concat",
               color='black',
               fillcolor='pink',
               shape='box',
               style='filled')

    for i in range(0, len(cell_arc) + 2):
        if not (i * 10 + 3 in output_used):
            G.add_edge(i * 10 + 3, (len(cell_arc) + 2) * 10 + 3)

    return G


def str_to_arc_seq(arc_str):
    arc_seq = ""
    for pos in range(1, len(arc_str)//2 + 1):
        arc_seq += "{} ".format(arc_str[pos])
    return arc_seq


def arc_seq_to_arc_arr(arc_seq):
    arc_arr = np.array([int(x) for x in arc_seq.split(" ") if x])
    arc_arr = np.reshape(arc_arr, [-1, 4])
    return arc_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv_arc', type=str, required=True)
    parser.add_argument('--reduce_arc', type=str, required=True)
    args = parser.parse_args()

    conv_arc = args.conv_arc
    reduce_arc = args.reduce_arc

    c_cell = np.array([int(x) for x in conv_arc.split(" ") if x])
    r_cell = np.array([int(x) for x in reduce_arc.split(" ") if x])

    c_cell = np.reshape(c_cell, [-1, 4])
    r_cell = np.reshape(r_cell, [-1, 4])

    gc = creat_graph(c_cell)
    gr = creat_graph(r_cell)

    gc.write("./samples/conv-cell.dot")
    gr.write("./samples/reduce-cell.dot")

    vizGn = pgv.AGraph("./samples/conv-cell.dot")
    vizGr = pgv.AGraph("./samples/reduce-cell.dot")

    vizGn.layout(prog='dot')
    vizGr.layout(prog='dot')

    vizGn.draw("./samples/conv-cell.png")
    vizGr.draw("./samples/reduce-cell.png")
