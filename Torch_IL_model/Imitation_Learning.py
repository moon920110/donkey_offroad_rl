import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
from tensorboardX import SummaryWriter
import common.const as const
from common.utills import print_square

import cv2
import random

def tensor_imwrite(img, name):
    temp = np.transpose(img.cpu().numpy(), (1, 2, 0))
    temp = (temp * 255).astype(np.uint8)
    cv2.imwrite(name, temp)

def tensor_visualizer(img_tensor, aug_img_tensor, idx=-1):
    if idx == -1:
        idx = random.randint(0, len(img_tensor)-1)

    img_tensor_len = len(img_tensor)
    aug_len = int(len(aug_img_tensor)/len(img_tensor))
    tensor_imwrite(img_tensor[idx], "origin.png")
    for i in range(aug_len):
        tensor_imwrite(aug_img_tensor[idx + img_tensor_len*i], "aug{}.png".format(i))

class Base:
    def __init__(self,
                 network,
                 max_epoch,
                 batch_size,
                 train_test_split,
                 save_dir,
                 log_dir,
                 lr,
                 input_shape=(120, 160, 3),
                 num_action=2,
                 use_cuda=True):

        self.network = network
        print(sum(p.numel() for p in network.parameters()))
        self.input_shape = input_shape
        self.num_action = num_action
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.train_test_split = train_test_split

        self.optimizer = Adam(self.network.parameters(), lr=lr)
        self.lr = lr
        self.loss_func = nn.L1Loss()

        if len(log_dir) > 0:
            self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir

        self.use_cuda = use_cuda

        if use_cuda:
            self.network.cuda()

    def save(self, path):
        torch.save(self.network.state_dict(), path)
    
    def load(self, path):
        device = torch.device('cpu')
        self.network.load_state_dict(torch.load(path, map_location=device))

    def train_step(self, measurement):
        raise NotImplementedError("Please Implement this method")

    def eval_step(self, measurement):
        raise NotImplementedError("Please Implement this method")

    def preprocess_measurement(self, measurement):
        if 'aug_imgs' in measurement.keys():
            aug_imgs_ = measurement['aug_imgs']
            temp_size = list(aug_imgs_.size())
            aug_imgs = []
            for j in range(temp_size[1]):
                temp = []
                for i in range(temp_size[0]):
                    temp.append(torch.unsqueeze(aug_imgs_[i][j], 0))
                temp = torch.cat(temp, 0)
                aug_imgs.append(temp)

            for i, imgs in enumerate(aug_imgs):
                measurement['aug_imgs{}'.format(i)] = imgs
            measurement['aug_len'] = len(aug_imgs)
        return measurement

    def train(self, dataset):
        self.total_step = 0
        dataset_len = len(dataset)
        for epoch in range(self.max_epoch):
            for step, measurement in enumerate(dataset):
                if step == dataset_len - 1:
                    break
                self.total_step += 1

                measurement = self.preprocess_measurement(measurement)

                '''train valid split'''

                origin_measurement, eval_measurement = {}, {}
                for key in measurement.keys():
                    if type(measurement[key]) == torch.Tensor or type(measurement[key]) == np.ndarray:
                        point = int(self.train_test_split * len(measurement[key]))
                        origin_measurement[key] = measurement[key][:point]
                        eval_measurement[key] = measurement[key][point:]
                    else:
                        origin_measurement[key] = measurement[key]
                        eval_measurement[key] = measurement[key]

                '''train'''
                verbose = self.train_step(origin_measurement)

                '''valid'''
                eval_verbose = self.eval_step(eval_measurement)
                verbose.update(eval_verbose)

                '''verbose'''
                for metric in verbose.keys():
                    self.writer.add_scalar(metric, verbose[metric], self.total_step)

                verbose["epoch"] = epoch
                verbose["step/epoch"] = "{}/{}".format(step, dataset_len)
                verbose["step"] = self.total_step
                print_square(verbose)

            '''save model'''
            self.save(self.save_dir + "epoch{}.pth".format(epoch + 1))

    def train_no_valid(self, dataset):
        self.total_step = 0
        dataset_len = len(dataset)
        for epoch in range(self.max_epoch):
            for step, measurement in enumerate(dataset):
                if step == dataset_len - 1:
                    break
                self.total_step += 1

                measurement = self.preprocess_measurement(measurement)
                '''train'''
                verbose = self.train_step(measurement)

                '''verbose'''
                for metric in verbose.keys():
                    self.writer.add_scalar(metric, verbose[metric], self.total_step)

                verbose["epoch"] = epoch
                verbose["step/epoch"] = "{}/{}".format(step, dataset_len)
                verbose["step"] = self.total_step
                print_square(verbose)

            '''save model'''
            self.save(self.save_dir + "epoch{}.pth".format(epoch+1))

class Vanilla(Base):
    def __init__(self,
                 network,
                 max_epoch,
                 batch_size,
                 train_test_split,
                 save_dir,
                 log_dir,
                 lr,
                 discrete,
                 input_shape=(120, 160, 3),
                 num_action=2,
                 use_cuda=True):

        super(Vanilla, self).__init__(
            network,
            max_epoch,
            batch_size,
            train_test_split,
            save_dir,
            log_dir,
            lr,
            input_shape,
            num_action,
            use_cuda
        )

        self.discrete = discrete
        self.steering_values = torch.Tensor(const.DISCRETE_STEER)
        self.throttle_values = torch.Tensor(const.DISCRETE_THROTTLE)
        self.steering_n = const.DISCRETE_STEER_N
        self.throttle_n = const.DISCRETE_THROTTLE_N

        if self.use_cuda:
            self.steering_values = self.steering_values.cuda()
            self.throttle_values = self.throttle_values.cuda()

    def softmax_to_onehot(self, tensor):
        max_idx = torch.argmax(tensor, 1, keepdim=True)
        one_hot = torch.FloatTensor(tensor.shape)
        if self.use_cuda:
            one_hot = one_hot.cuda()
        one_hot.zero_()
        one_hot.scatter_(1, max_idx, 1)
        return one_hot

    def continuous_to_discrete(self, labels, ranges):
        '''
        change continuous values to onehot vector by ranges
        labels : torch Tensor
        ranges : torch Tensor
        '''
        ranges_len = len(ranges)
        labels_len = len(labels)
        #range_len x labels_len
        labels_mat = labels.reshape(1, labels_len).repeat(ranges_len, 1)
        ranges_mat = torch.transpose(ranges.reshape(1, ranges_len).repeat(labels_len, 1), 0, 1)

        indexs = torch.argmin(torch.abs(labels_mat - ranges_mat), dim=0)
        indexs = indexs.reshape(labels_len, 1)

        num_classes = ranges_len
        matrix = torch.arange(num_classes).reshape(1, num_classes)
        if self.use_cuda:
            matrix = matrix.cuda()
        one_hot_target = (indexs == matrix).float()
        return one_hot_target

    def discrete_to_continuous(self, softmax_output):
        data_len = softmax_output.size()[0]

        steering_prob = softmax_output[:, :self.steering_n]
        throttle_prob = softmax_output[:, self.steering_n:]
        steering_onehot = self.softmax_to_onehot(steering_prob)
        throttle_onehot = self.softmax_to_onehot(throttle_prob)

        steering_mat = self.steering_values.reshape(1, self.steering_n).repeat(data_len, 1)
        throttle_mat = self.throttle_values.reshape(1, self.throttle_n).repeat(data_len, 1)

        if self.use_cuda:
            steering_mat = steering_mat.cuda()
            throttle_mat = throttle_mat.cuda()

        steering = torch.sum(steering_mat * steering_onehot, 1)
        throttle = torch.sum(throttle_mat * throttle_onehot, 1)

        action = torch.cat((steering.reshape(data_len, 1), throttle.reshape(data_len, 1)), -1)
        return action

    def get_action(self, img_tensor, speed):
        assert list(img_tensor.size())[-3:]==[3, 120, 160], "input shape mismatch : {}".format(list(img_tensor.size()))
        with torch.no_grad():
            actions = self.network.forward(img_tensor, speed)
            if self.discrete:
                actions = self.discrete_to_continuous(actions)
        return actions

    def eval_step(self, measurement):
        with torch.no_grad():

            img_tensor = measurement['img'].float()
            speed = measurement['speed'].float()
            target_action = measurement['target_action'].float()

            if self.use_cuda:
                target_action = target_action.cuda()
                img_tensor = img_tensor.cuda()
                speed = speed.cuda()

            if self.discrete:
                steering = target_action[:, 0]
                throttle = target_action[:, 1]
                steering_discrete = self.continuous_to_discrete(steering, self.steering_values)
                throttle_discrete = self.continuous_to_discrete(throttle, self.throttle_values)
                target_action = torch.cat((steering_discrete, throttle_discrete), -1)


            origin_action, origin_feature = self.network(img_tensor, speed, get_feature=True)
            total_loss = self.loss_func(target_action, origin_action).mean()


            actions = self.get_action(img_tensor, speed)
            steering, throttle = np.transpose(actions.detach().cpu().numpy(), (1, 0))

        verbose = {
            'eval_loss/total': total_loss.item(),
            'eval_action/steering': np.mean(steering),
            'eval_action/throttle': np.mean(throttle)
        }
        return verbose

    def train_step(self, measurement):

        '''get inputs and labels'''

        img_tensor = measurement['img'].float()
        speed = measurement['speed'].float()
        target_action = measurement['target_action'].float()

        if self.use_cuda:
            target_action = target_action.cuda()
            img_tensor = img_tensor.cuda()
            speed = speed.cuda()

        if self.discrete:
            steering = target_action[:, 0]
            throttle = target_action[:, 1]
            steering_discrete = self.continuous_to_discrete(steering, self.steering_values)
            throttle_discrete = self.continuous_to_discrete(throttle, self.throttle_values)
            target_action = torch.cat((steering_discrete, throttle_discrete), -1)

        '''vanilla loss'''

        origin_action, origin_feature = self.network(img_tensor, speed, get_feature=True)
        total_loss = self.loss_func(target_action, origin_action).mean()

        '''optimize'''

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        '''verbose'''

        actions = self.get_action(img_tensor, speed)
        steering, throttle = np.transpose(actions.detach().cpu().numpy(), (1, 0))

        verbose = {
            'loss/total': total_loss.item(),
            'action/steering': np.mean(steering),
            'action/throttle': np.mean(throttle)
        }
        del total_loss
        return verbose


class Augment(Vanilla):
    def __init__(self,
                 network,
                 randomizer,
                 max_epoch,
                 batch_size,
                 train_test_split,
                 save_dir,
                 log_dir,
                 lr,
                 discrete,
                 input_shape=(120, 160, 3),
                 num_action=2,
                 use_cuda=True):

        super(Augment, self).__init__(
            network,
            max_epoch,
            batch_size,
            train_test_split,
            save_dir,
            log_dir,
            lr,
            discrete,
            input_shape,
            num_action,
            use_cuda
        )

        self.randomizer_class = randomizer
        self.randomizer = self.randomizer_class()

    def train_step(self, measurement):
        if self.total_step%1000:
            self.randomizer = self.randomizer_class()

        '''get inputs and labels'''

        img_tensor = measurement['img'].float()
        speed = measurement['speed'].float()
        target_action = measurement['target_action'].float()

        batch_size = len(img_tensor)

        if self.use_cuda:
            target_action = target_action.cuda()
            img_tensor = img_tensor.cuda()
            speed = speed.cuda()

        if self.discrete:
            steering = target_action[:, 0]
            throttle = target_action[:, 1]
            steering_discrete = self.continuous_to_discrete(steering, self.steering_values)
            throttle_discrete = self.continuous_to_discrete(throttle, self.throttle_values)
            target_action = torch.cat((steering_discrete, throttle_discrete), -1)

        '''image augmentation'''

        aug_imgs_ = []
        for i in range(measurement['aug_len']):
            temp = measurement['aug_imgs{}'.format(i)].float()
            aug_imgs_.append(temp)

        aug_imgs_ = torch.cat(aug_imgs_, 0)
        with torch.no_grad():
            aug_img_tensors = self.randomizer(aug_imgs_.cuda())

        aug_img_tensor = torch.cat(aug_img_tensors, 0)
        aug_len = int(len(aug_img_tensor)/batch_size)
        aug_speed = torch.cat([speed for _ in range(aug_len)], 0)
        aug_target_action = torch.cat([target_action for _ in range(aug_len)], 0)

        '''cuda & discrete'''

        if self.use_cuda:
            aug_img_tensor = aug_img_tensor.cuda()
            aug_speed = aug_speed.cuda()
            aug_target_action = aug_target_action.cuda()

        '''detach for memeory leak prevent'''

        img_tensor.detach()
        speed.detach()
        target_action.detach()

        aug_img_tensor.detach()
        aug_speed.detach()
        aug_target_action.detach()

        '''vanilla loss'''

        origin_action, origin_feature = self.network(img_tensor, speed, get_feature=True)
        origin_loss = self.loss_func(target_action, origin_action).mean()

        '''augment loss'''

        aug_action, aug_feature = self.network(aug_img_tensor, aug_speed, get_feature=True)
        aug_loss = self.loss_func(aug_target_action, aug_action).mean()


        '''feature loss'''

        repeated_origin_feature = torch.cat([origin_feature for _ in range(aug_len)], 0)
        feature_loss = self.loss_func(repeated_origin_feature, aug_feature).mean()

        '''optimize'''

        total_loss = origin_loss + aug_loss + feature_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        #tensor_visualizer(img_tensor, aug_img_tensor)
        #assert False

        '''verbose'''

        actions = self.get_action(img_tensor, speed)
        steering, throttle = np.transpose(actions.detach().cpu().numpy(), (1, 0))

        verbose = {
            'loss/total': total_loss.item(),
            'loss/origin': origin_loss.item(),
            'loss/augment': aug_loss.item(),
            'loss/feature': feature_loss.item(),
            'action/steering': np.mean(steering),
            'action/throttle': np.mean(throttle)
        }
        return verbose