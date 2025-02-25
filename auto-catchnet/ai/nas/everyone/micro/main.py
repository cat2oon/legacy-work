import time
import warnings

from ac.filesystem.paths import init_directory
from ac.common.prints import *
from ai.nas.everyone.micro.configs import *
from ai.nas.core.nas import NAS, ModelInputs
from ai.nas.params.controller import ControllerParams
from ai.nas.params.model import ModelParams
from ai.nas.params.nas import NASParams
from ai.libs.tf.setup.hooks import *
from ai.libs.tf.setup.sess import get_sess_config
from keras import backend as K

warnings.filterwarnings('ignore')
FLAGS = define_configs()


def train(nas: NAS):
    model_params = ModelParams.from_flags(FLAGS, name="m1")
    ctrl_params = ControllerParams.from_flags(FLAGS, name="c1")
    inputs = ModelInputs(model_params)

    nas.build(inputs, model_params, ctrl_params)
    ops = nas.get_ops()
    controller_ops = ops["controller"]

    console_print("Starting session")
    with tf.train.SingularMonitoredSession(config=get_sess_config(),
                                           hooks=make_hooks(FLAGS.output_dir),
                                           checkpoint_dir=FLAGS.output_dir) as sess:
        K.set_session(sess)

        init_ops = nas.tf_init_ops()
        sess.run(init_ops)

        num_train_batches = nas.get_num_train_batches()
        start_time = time.time()
        console_print(">>> NAS start #train: {} at time: {}".format(num_train_batches, datetime.datetime.now()))

        for _ in range(FLAGS.num_epochs * num_train_batches):
            model_ops_val = sess.run(nas.get_model_run_ops())
            global_step = sess.run(ops["child"]["global_step"])
            epoch = global_step // nas.get_num_train_batches()
            nas.log_model_every(global_step, epoch, model_ops_val, start_time)

            if not nas.is_eval_step(global_step):
                continue

            if nas.is_ctrl_training_epoch(epoch):
                console_print("Epoch {}: Training controller".format(epoch))
                for ct_step in range(nas.get_num_ctrl_train_step()):
                    ctrl_ops_val = sess.run(nas.get_ctrl_run_ops())
                    ctrl_step = sess.run(controller_ops["train_step"])
                    nas.log_ctrl_every(ct_step, ctrl_step, ctrl_ops_val, start_time)
                nas.sample_arch(sess)

            console_print("Epoch {}: Eval".format(epoch))
            nas.run_eval(sess)


def main(_):
    console_print("*** start NAS ***")
    params = NASParams.from_flags(FLAGS)

    print_experiment(params)
    init_directory(FLAGS.output_dir, FLAGS.reset_output_dir)
    print_user_flags()
    print_time_stamp()

    nas = NAS(params)
    train(nas)

    console_print("*** complete ***")


if __name__ == "__main__":
    tf.app.run()
