
import Torch_IL_model.Imitation_Learning as IL
from Torch_IL_model.networks import ImpalaNetwork
import Torch_IL_model.const as const
import copy
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

def numpy_to_pil(image_):
    image = copy.deepcopy(image_)
    
    image = (image - np.min(image))/(np.max(image) - np.min(image))

    image *= 255
    image = image.astype('uint8')

    im_obj = Image.fromarray(image)
    return im_obj

class IL_TEST:
    def __init__(self, num_action, input_shape=(120, 160, 3), batch_size=64, training=True, model_path=None, *args, **kwargs):
        super(PPO, self).__init__(*args, **kwargs)


        discrete = False
        load_dir = "./Torch_IL_model/epoch1.pth"


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
            network=ImpalaNetwork(discrete, steering_n=steering_n, throttle_n=throttle_n),
            max_epoch=1,
            batch_size=1,
            save_dir="",
            log_dir="",
            lr=1,
            discrete=discrete,
            train_test_split=0.9,
            use_cuda = False
        )
        model.load(load_dir)

    def save(self):
        pass

    def load(self, model_paths):
        pass

    def run(self, img, speed, meter, train_state):
        '''
        train_state
        0 - pendding
        1 - running
        2 - emergency stop
        3 - successed
        4 - train
        5 - driving
        '''
        if train_state > 0:
            #(120, 160, 3)
            img = numpy_to_pil(img)
            img = transform(img).float().numpy()
            img = np.expand_dims(img, 0)
            action = model.get_action(torch.Tensor(img), torch.Tensor([[speed]]))

            if train_state < 4:
                if done:
                    print("=======EPISODE DONE======")
                    print('reward sum: {}'.format(self.r_sum))

                    return 0, 0, True

                self.last_state = {
                        'img': img,
                        'speed': speed,
                        }
                self.last_actions = action
                self.train_step += 1

            elif train_state == 4:
                return 0, 0, False
            else:
                return action[0], action[1], False
        else:
            return 0, 0, False



class PPO(KerasPilot):
    def __init__(self, num_action, input_shape=(120, 160, 3), batch_size=64, training=True, model_path=None, *args, **kwargs):
        super(PPO, self).__init__(*args, **kwargs)

        self.actor = default_model(num_action, input_shape, actor_critic='actor')
        self.old_actor = default_model(num_action, input_shape, actor_critic='actor')
        self.critic = default_model(num_action, input_shape,actor_critic='critic')
        self.old_actor.set_weights(self.actor.get_weights())

        self.num_action = num_action
        self.n = 0
        self.gamma = 0.99
        self.lmbda = 0.95
        self.lr = 0.002
        self.eps = 0.1
        self.K = 2
        self.batch_size = batch_size
        self.model_path = model_path
        self.optimal = False

        self.r_sum = 0
        self.last_state = None
        self.last_action = None
        self.train_step = 0

        self.memory = MemoryPPO(img_dim=input_shape, speed_dim=(1,), act_dim=num_action, size=self.batch_size, gamma=0.99, lam=0.95)
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediciton = np.zeros((1, 2*num_action))

        self.actor.summary()
        self.critic.summary()

        if training:
            self.compile()

    def save(self):
        pass

    def load(self, model_paths):
        pass

    def run(self, img, speed, meter, train_state):
