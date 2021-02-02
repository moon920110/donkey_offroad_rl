from donkeycar.parts.torch import TorchPilot
import Torch_IL_model.Imitation_Learning as IL
from Torch_IL_model.networks import ImpalaNetwork
import Torch_IL_model.const as const
import copy
import numpy as np
from PIL import Image
from utils.utils import *
import torch
import torchvision.transforms as transforms

class IL_TEST(TorchPilot):
    def __init__(self, *args, **kwargs):
        super(IL_TEST, self).__init__(*args, **kwargs)


        discrete = False

        if discrete:
            steering_n = const.DISCRETE_STEER_N
            throttle_n = const.DISCRETE_THROTTLE_N
        else:
            steering_n = 1
            throttle_n = 1

        self.transform = transforms.Compose([
        transforms.Resize((const.HEIGHT, const.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=const.MEAN, std=const.STD)
        ])

        self.model = IL.Vanilla(
            network=ImpalaNetwork(steering_n=steering_n, throttle_n=throttle_n),
            max_epoch=1,
            batch_size=1,
            save_dir="",
            log_dir="",
            lr=1,
            discrete=discrete,
            train_test_split=0.9,
            use_cuda = False
        )

    def load(self, model_path):
        self.model.load(model_path)

    def run(self, img, speed):
        #(120, 160, 3)
        img = numpy_to_pil(img)
        img = self.transform(img).float().unsqueeze(0)
        action = self.model.get_action(img, torch.Tensor([[speed]]))
        print(action)

        return 2 * action[0][0], action[0][1] # 0.4 for throttle is the best


