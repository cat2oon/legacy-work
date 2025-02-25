import time
import numpy as np

from ac.common.prints import console_print
from ai.nas.core.validator import NASValidator
from ai.nas.everyone.micro.model_inputs import ModelInputs
from ai.nas.everyone.micro.micro_child import MicroChild
from ai.nas.everyone.micro.micro_controller import MicroController
from ai.nas.params.nas import NASParams
from ai.nas.params.model import ModelParams
from ai.nas.params.controller import ControllerParams


class NAS:
    def __init__(self, params: NASParams):
        self.params = params
        self.validator = None

        self.ops = None
        self.eval_func = None
        self.sample_ops = None

        self.model = None
        self.model_ops = None
        self.model_params = None
        self.model_inputs = None

        self.controller = None
        self.ctrl_ops = None
        self.ctrl_params = None
        self.eval_every_steps = None

    def build(self, model_inputs: ModelInputs, model_params: ModelParams, ctrl_params: ControllerParams):
        self.validator = NASValidator(model_inputs, model_params, ctrl_params)
        self.validator.validate()

        self.model_inputs = model_inputs
        self.model = MicroChild(model_inputs, model_params)
        self.model.prepare_dataset()
        self.controller = MicroController(ctrl_params)

        self.model_params = model_params
        self.ctrl_params = ctrl_params

        self.model.connect_controller(self.controller)
        self.model.build_train()
        self.model.build_valid()
        self.model.build_test()

        self.controller.build_trainer(self.model)

    """ ops """

    def tf_init_ops(self):
        init_ops = self.model_inputs.get_initializers()
        return init_ops

    def get_ops(self):
        if self.ops is not None:
            return self.ops

        self.model_ops = self.model.get_model_ops()
        self.ctrl_ops = self.controller.get_controller_ops()

        self.sample_ops = [
            self.ctrl_ops["sample_arc"],
            self.ctrl_ops["sample_val_loss"],
            self.ctrl_ops["sample_reward"],
        ]

        self.ops = {
            "child": self.model_ops,
            "controller": self.ctrl_ops,
        }

        self.eval_func = self.model.eval_once
        self.eval_every_steps = self.model.num_train_batches * self.params.eval_every_epochs

        return self.ops

    def get_model_run_ops(self):
        ops = self.model_ops
        return [
            ops["loss"],
            ops["lr"],
            ops["grad_norm"],
            ops["train_op"],
        ]

    def get_ctrl_run_ops(self):
        ops = self.ctrl_ops
        return [
            ops["loss"],
            ops["entropy"],
            ops["lr"],
            ops["grad_norm"],
            ops["sample_val_loss"],
            ops["baseline"],
            ops["skip_rate"],
            ops["train_op"],
        ]

    """ models """

    def log_model_every(self, global_step, epoch, model_ops_val, start_time):
        if global_step % self.params.model_log_every != 0:
            return

        loss, lr, gn,  _ = model_ops_val

        log_str = ""
        log_str += "epoch={:<6d}".format(epoch)
        log_str += "ch_step={:<6d}".format(global_step)
        log_str += " loss={:<8.2f}".format(loss)
        log_str += " lr={:<8.10f}".format(lr)
        log_str += " |g|={:<8.2f}".format(gn)
        # log_str += " tr_acc={:<3d}/{:>3d}".format(tr_acc, self.model_params.batch_size)
        log_str += " mins={:<10.2f}".format(float(time.time() - start_time) / 60)

        print(log_str)

    def log_ctrl_every(self, ct_step, ctrl_step, ctrl_ops_val, start_time):
        if ct_step % self.params.ctrl_log_every != 0:
            return

        loss, entropy, lr, gn, val_acc, bl, skip, _ = ctrl_ops_val

        log_string = ""
        log_string += "ctrl_step={:<6d}".format(ctrl_step)
        log_string += " loss={:<7.3f}".format(loss)
        log_string += " ent={:<5.2f}".format(entropy)
        log_string += " lr={:<6.4f}".format(lr)
        log_string += " |g|={:<8.4f}".format(gn)
        log_string += " acc={:<6.4f}".format(val_acc)
        log_string += " bl={:<5.2f}".format(bl)
        log_string += " mins={:<.2f}".format(float(time.time() - start_time) / 60)

        print(log_string)

    def is_eval_step(self, global_step):
        return global_step % self.eval_every_steps == 0 and global_step != 0

    def get_num_ctrl_train_step(self):
        return self.ctrl_params.train_steps * self.ctrl_params.get_for_train_op().num_aggregate

    def run_eval(self, sess):
        self.eval_func(sess, "test" if self.is_fixed_arc() else "valid")

    """ sampling """

    def sample_arch(self, sess, num_to_sample=64):
        console_print("sampling architectures")

        ctrl_params = self.ctrl_params
        for i in range(num_to_sample):
            arc, loss, reward = sess.run(self.sample_ops)
            console_print("sample arch[{}] val_mse_loss={:<6.4f} reward got={:6.4f}".format(i, loss, reward))

            if self.is_micro_search():
                normal_arc, reduce_arc = arc
                print(np.reshape(normal_arc, [-1]))
                print(np.reshape(reduce_arc, [-1]))
                continue

            if self.is_macro_search():
                start = 0
                for layer_id in range(ctrl_params.child_num_layers):
                    if ctrl_params.search_whole_channels:
                        end = start + 1 + layer_id
                    else:
                        end = start + 2 * ctrl_params.child_num_branches + layer_id
                    print(np.reshape(arc[start: end], [-1]))
                    start = end
                continue

    """ miscellaneous """

    def is_ctrl_training_epoch(self, epoch):
        return self.ctrl_params.training and epoch % self.ctrl_params.train_every == 0 and epoch != 0

    def is_micro_search(self):
        return self.ctrl_params.search_for.upper() == "MICRO"

    def is_macro_search(self):
        return self.ctrl_params.search_for.upper() == "MACRO"

    def is_fixed_arc(self):
        return self.model_params.fixed_arc is True

    def get_num_epochs(self):
        return self.params.num_epochs

    def get_num_train_batches(self):
        return self.model_inputs.get_num_train_batches()

    def get_num_total_steps(self):
        return self.params.num_epochs * self.get_num_train_batches()

