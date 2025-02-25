from ai.nas.everyone.micro.model_inputs import ModelInputs
from ai.nas.params.controller import ControllerParams
from ai.nas.params.model import ModelParams
from ai.nas.utils.arcs import is_micro_search


class NASValidator:
    def __init__(self,
                 model_inputs: ModelInputs,
                 model_params: ModelParams,
                 ctrl_params: ControllerParams):
        self.model_inputs = model_inputs
        self.model_params = model_params
        self.ctrl_params = ctrl_params

    def get_predicates(self):
        # TODO: list by prefix ('check')
        return [
            self.check_num_branches,
            self.check_data_path,
        ]

    def validate(self):
        return all(predicate() for predicate in self.get_predicates())

    def check_num_branches(self):
        num_branches = self.model_params.num_branches
        if is_micro_search(self.ctrl_params.search_for):
            assert num_branches == 5
        assert num_branches > 0

    def check_data_path(self):
        assert self.model_params.data_path is not None

    def check_sharable_params_consistency(self):
        # ModeParams <-> Controllerparams
        # num_cells
        # num_branches etc
        raise NotImplementedError("TODO")



