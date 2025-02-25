import torch


class GazeMSELoss(object):

    def __call__(self, input_dict, output_dict):
        def pitchyaw_to_vector(pitchyaws):
            sin = torch.sin(pitchyaws)
            cos = torch.cos(pitchyaws)
            return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)
        y = pitchyaw_to_vector(input_dict['gaze_a']).detach()
        y_hat = output_dict['gaze_a_hat']
        assert y.shape[1] == y_hat.shape[1] == 3
        return torch.mean((y - y_hat) ** 2)
