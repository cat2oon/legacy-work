import tensorflow as tf


def lstm(x, prev_c, prev_h, w):
    ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)  # [x(1) x(2) ... ph(1) ph(2) ...]
    i, f, o, g = tf.split(ifog, 4, axis=1)
    i = tf.sigmoid(i)  # input  gate  i_t=sig(W_i*[t_t-1,x_t]+b_i) : which new values to memory
    f = tf.sigmoid(f)  # forget gate  f_t=sig(W_f*[h_t-1,x_t]+b_f) : which prev values to forget
    o = tf.sigmoid(o)  # output gate (affect next hidden state)
    g = tf.tanh(g)     # IMO) it's just 균형자 for i (i가 선별벡터라면 g는 prev와 더해지며 강화/약화)
    next_c = i * g + f * prev_c     # next cell state = i*g (what to memory) + f*c_t-1 (remain after forget)
    next_h = o * tf.tanh(next_c)    # next hidden state
    return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = lstm(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h

