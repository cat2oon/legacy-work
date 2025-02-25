import os

import tensorflow as tf


def make_writer(out_path, idx):
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    path = os.path.join(out_path, 'everyone_{:05d}.tfr'.format(idx))
    writer = tf.python_io.TFRecordWriter(path=path, options=options)
    return writer


def count_tfexamples(tfrecord_files, compression_type=2):
    def tf_iterator(file):
        opts = None
        if compression_type == 2:
            opts = tf.python_io.TFRecordOptions(compression_type=2)
        return tf.python_io.tf_record_iterator(file, opts)

    count = 0
    for tfrecord_file in tfrecord_files:
        for _ in tf_iterator(tfrecord_file):
            count += 1
    return count


def to_dataset(tfrecord_files,
               tf_example_decoder,
               num_epochs=40,
               batch_size=64,
               num_parallel=16,
               buffer_size=1024 ** 2,
               compression_type="GZIP"):
    dataset = tf.data.TFRecordDataset(tfrecord_files,
                                      buffer_size=buffer_size,
                                      num_parallel_reads=num_parallel,
                                      compression_type=compression_type)
    # dataset = dataset.map(example_deserializer)
    # TODO : experiment perf 해보고 map에서 시간 지연이 있다면 num_parallel
    dataset = dataset.map(tf_example_decoder, num_parallel_calls=64)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()

    return iterator