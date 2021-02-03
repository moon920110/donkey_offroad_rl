#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Uses the data written by the donkey v2.2 tub writer,
but faster training with proper sampling of distribution over tubs. 
Has settings for continuous training that will look for new files as it trains. 
Modify on_best_model if you wish continuous training to update your pi as it builds.
You can drop this in your ~/mycar dir.
Basic usage should feel familiar: python train.py --model models/mypilot


Usage:
    train.py [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|latent|categorical|rnn|imu|behavior|3d|look_ahead|tensorrt_linear|tflite_linear|coral_tflite_linear)] [--figure_format=<figure_format>] [--continuous] [--aug]

Options:
    -h --help              Show this screen.
    -f --file=<file>       A text file containing paths to tub files, one per line. Option may be used more than once.
    --figure_format=png    The file format of the generated figure (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html), e.g. 'png', 'pdf', 'svg', ...
"""
import os
import cv2
import glob
import random
import json
import time
import zlib
from os.path import basename, join, splitext, dirname
import pickle
import datetime

from tensorflow.python import keras
from docopt import docopt
import numpy as np
from PIL import Image
from shutil import copyfile
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import donkeycar as dk
from donkeycar.parts.datastore import Tub
from donkeycar.parts.torch import TorchIL
from donkeycar.parts.augment import augment_image
from donkeycar.utils import *
from normalizer import LN

figure_format = 'png'


'''
matplotlib can be a pain to setup on a Mac. So handle the case where it is absent. When present,
use it to generate a plot of training results.
'''
try:
    import matplotlib.pyplot as plt
    do_plot = True
except:
    do_plot = False
    print("matplotlib not installed")
    

class TubDataset(Dataset):
    def __init__(self, gen_records, cfg, train):
        self.gen_records = self._process(gen_records, train)
        self.transforms = transforms.Compose([
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.gen_records)

    def __getitem__(self, idx):
        record = self.gen_records[idx]
        angle = record['angle']
        throttle = record['throttle']
        out = np.array([angle, throttle], dtype=np.float32)
        speed = np.array([record['rotaryencoder/meter_per_second']])
        img = self._pil_loader(record['image_path'])
        img = self.transforms(img)

        measurement = {
                'img': img,
                'speed': speed,
                'out': out,
                }

        return measurement

    @staticmethod
    def _pil_loader(path):
        img = Image.open(path).convert('RGB')
        return img

    @staticmethod
    def _process(gen_records, train):
        records = []
        for k, v in gen_records.items():
            if v['train'] is train:
                records.append(v)
        return records

'''
Tub management
'''


def make_key(sample):
    tub_path = sample['tub_path']
    index = sample['index']
    return tub_path + str(index)


def make_next_key(sample, index_offset):
    tub_path = sample['tub_path']
    index = sample['index'] + index_offset
    return tub_path + str(index)


def collate_records(records, gen_records, opts):
    '''
    open all the .json records from records list passed in,
    read their contents,
    add them to a list of gen_records, passed in.
    use the opts dict to specify config choices
    '''

    new_records = {}
    
    for record_path in records:

        basepath = os.path.dirname(record_path)        
        index = get_record_index(record_path)
        sample = { 'tub_path' : basepath, "index" : index }
             
        key = make_key(sample)

        if key in gen_records:
            continue

        try:
            with open(record_path, 'r') as fp:
                json_data = json.load(fp)
        except:
            continue

        image_filename = json_data["cam/image_array"]
        image_path = os.path.join(basepath, image_filename)

        sample['record_path'] = record_path
        sample["image_path"] = image_path
        sample["json_data"] = json_data        

        angle = float(json_data['user/angle'])
        throttle = float(json_data["user/throttle"])

        # if opts['categorical']:
        #     angle = dk.utils.linear_bin(angle)
        #     throttle = dk.utils.linear_bin(throttle, N=20, offset=0, R=opts['cfg'].MODEL_CATEGORICAL_MAX_THROTTLE_RANGE)

        sample['angle'] = angle
        sample['throttle'] = throttle

        try:
            speed = float(json_data['rotaryencoder/meter_per_second'])
            sample['rotaryencoder/meter_per_second'] = speed
        except:
            pass

        try:
            accl_x = float(json_data['imu/acl_x'])
            accl_y = float(json_data['imu/acl_y'])
            accl_z = float(json_data['imu/acl_z'])

            gyro_x = float(json_data['imu/gyr_x'])
            gyro_y = float(json_data['imu/gyr_y'])
            gyro_z = float(json_data['imu/gyr_z'])

            sample['imu_array'] = np.array([accl_x, accl_y, accl_z, gyro_x, gyro_y, gyro_z])
        except:
            pass

        try:
            behavior_arr = np.array(json_data['behavior/one_hot_state_array'])
            sample["behavior_arr"] = behavior_arr
        except:
            pass

        try:
            location_arr = np.array(json_data['location/one_hot_state_array'])
            sample["location"] = location_arr
        except:
            pass


        sample['img_data'] = None

        # Initialise 'train' to False
        sample['train'] = False
        
        # We need to maintain the correct train - validate ratio across the dataset, even if continous training
        # so don't add this sample to the main records list (gen_records) yet.
        new_records[key] = sample
        
    # new_records now contains all our NEW samples
    # - set a random selection to be the training samples based on the ratio in CFG file
    shufKeys = list(new_records.keys())
    random.shuffle(shufKeys)
    trainCount = 0
    #  Ratio of samples to use as training data, the remaining are used for evaluation
    targetTrainCount = int(opts['cfg'].TRAIN_TEST_SPLIT * len(shufKeys))
    for key in shufKeys:
        new_records[key]['train'] = True
        trainCount += 1
        if trainCount >= targetTrainCount:
            break
    # Finally add all the new records to the existing list
    gen_records.update(new_records)


def save_json_and_weights(model, filename):
    '''
    given a keras model and a .h5 filename, save the model file
    in the json format and the weights file in the h5 format
    '''
    if not '.h5' == filename[-3:]:
        raise Exception("Model filename should end with .h5")

    arch = model.to_json()
    json_fnm = filename[:-2] + "json"
    weights_fnm = filename[:-2] + "weights"

    with open(json_fnm, "w") as outfile:
        parsed = json.loads(arch)
        arch_pretty = json.dumps(parsed, indent=4, sort_keys=True)
        outfile.write(arch_pretty)

    model.save_weights(weights_fnm)
    return json_fnm, weights_fnm


class MyCPCallback(keras.callbacks.ModelCheckpoint):
    '''
    custom callback to interact with best val loss during continuous training
    '''

    def __init__(self, send_model_cb=None, cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_best_end_of_epoch = False
        self.send_model_cb = send_model_cb
        self.last_modified_time = None
        self.cfg = cfg

    def reset_best(self):
        self.reset_best_end_of_epoch = True

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        if self.send_model_cb:
            '''
            check whether the file changed and send to the pi
            '''
            filepath = self.filepath.format(epoch=epoch, **logs)
            if os.path.exists(filepath):
                last_modified_time = os.path.getmtime(filepath)
                if self.last_modified_time is None or self.last_modified_time < last_modified_time:
                    self.last_modified_time = last_modified_time
                    self.send_model_cb(self.cfg, self.model, filepath)

        '''
        when reset best is set, we want to make sure to run an entire epoch
        before setting our new best on the new total records
        '''        
        if self.reset_best_end_of_epoch:
            self.reset_best_end_of_epoch = False
            self.best = np.Inf
    

def train(cfg, tub_names, model_name, transfer_model, model_type, continuous, aug):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    ''' 
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    verbose = cfg.VERBOSE_TRAIN

    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE
    
    if continuous:
        print("continuous training")
    
    gen_records = {}
    opts = { 'cfg' : cfg}

    if "linear" in model_type:
        train_type = "linear"
    else:
        train_type = model_type

    kl = TorchIL(device=device)

    print('training with model type', type(kl))

    # Do not use
    if transfer_model:
        print('loading weights from model', transfer_model)
        kl.load(transfer_model)

        #when transfering models, should we freeze all but the last N layers?
        if cfg.FREEZE_LAYERS:
            num_to_freeze = len(kl.model.layers) - cfg.NUM_LAST_LAYERS_TO_TRAIN 
            print('freezing %d layers' % num_to_freeze)           
            for i in range(num_to_freeze):
                kl.model.layers[i].trainable = False        


    kl.set_optimizer('adam', cfg.LEARNING_RATE, cfg.LEARNING_RATE_DECAY)
    
    opts['torch_pilot'] = kl
    opts['continuous'] = continuous
    opts['model_type'] = model_type

    # Do not use
    extract_data_from_pickles(cfg, tub_names)

    records = gather_records(cfg, tub_names, opts, verbose=True)
    print('collating %d records ...' % (len(records)))
    collate_records(records, gen_records, opts)

    # make data loader
    train_dataset = TubDataset(gen_records, cfg, train=True)
    test_dataset = TubDataset(gen_records, cfg, train=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    model_path = os.path.expanduser(model_name)

    # checkpoint to save model after each epoch and send best to the pi.
    ln = None
    if cfg.NORM_IMAGES_ILLUMINANCE:
        ln = LN(model_name, tub_names[0])
    total_records = len(gen_records)

    num_train = 0
    num_val = 0

    for key, _record in gen_records.items():
        if _record['train'] == True:
            num_train += 1
        else:
            num_val += 1

    print("train: %d, val: %d" % (num_train, num_val))
    print('total records: %d' %(total_records))
    
    if not continuous:
        steps_per_epoch = num_train // cfg.BATCH_SIZE
    else:
        steps_per_epoch = 100
    
    val_steps = num_val // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    cfg.model_type = model_type

    kl.train(train_loader, val_loader, model_name, cfg.MAX_EPOCHS)


def extract_data_from_pickles(cfg, tubs):
    """
    Extracts record_{id}.json and image from a pickle with the same id if exists in the tub.
    Then writes extracted json/jpg along side the source pickle that tub.
    This assumes the format {id}.pickle in the tub directory.
    :param cfg: config with data location configuration. Generally the global config object.
    :param tubs: The list of tubs involved in training.
    :return: implicit None.
    """
    t_paths = gather_tub_paths(cfg, tubs)
    for tub_path in t_paths:
        file_paths = glob.glob(join(tub_path, '*.pickle'))
        print('found {} pickles writing json records and images in tub {}'.format(len(file_paths), tub_path))
        for file_path in file_paths:
            # print('loading data from {}'.format(file_paths))
            with open(file_path, 'rb') as f:
                p = zlib.decompress(f.read())
            data = pickle.loads(p)
           
            base_path = dirname(file_path)
            filename = splitext(basename(file_path))[0]
            image_path = join(base_path, filename + '.jpg')
            img = Image.fromarray(np.uint8(data['val']['cam/image_array']))
            img.save(image_path)
            
            data['val']['cam/image_array'] = filename + '.jpg'

            with open(join(base_path, 'record_{}.json'.format(filename)), 'w') as f:
                json.dump(data['val'], f)


if __name__ == "__main__":
    args = docopt(__doc__)
    cfg = dk.load_config()
    tub = args['--tub']
    model = args['--model']
    transfer = args['--transfer']
    model_type = args['--type']

    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE
        print("using default model type of", model_type)

    if args['--figure_format']:
        figure_format = args['--figure_format']
    continuous = args['--continuous']
    aug = args['--aug']
    
    dirs = preprocessFileList( args['--file'] )
    if tub is not None:
        tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
        dirs.extend( tub_paths )

    multi_train(cfg, dirs, model, transfer, model_type, continuous, aug)
