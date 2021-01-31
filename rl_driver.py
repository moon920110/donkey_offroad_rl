#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    rl_driver.py (drive) [--model=<model>] [--js] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|latent)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>]
    rl_driver.py (train) [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer)] [--continuous] [--aug] [--myconfig=<filename>] [--batch-size=<batch_size>]


Options:
    -h --help               Show this screen.
    --js                    Use physical joystick.
    -f --file=<file>        A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value>      Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
    --myconfig=filename     Specify myconfig file to use. 
                            [default: myconfig.py]
"""
import os
import time

from docopt import docopt
import numpy as np

import donkeycar as dk

#import parts
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.transform import Lambda, TriggeredCallback, DelayedTrigger
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import JoystickController
from donkeycar.parts.camera import Webcam
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.parts.controller import get_js_controller
from donkeycar.parts.encoder import RotaryEncoder
from donkeycar.utils import *
from normalizer import LN
from Torch_DDPG import DDPGAgent as DDPG
from Keras_PPO import PPO
from Keras_SAC import SAC
from Keras_TD3 import TD3
from Torch_IL import IL_TEST


class RL_Driver():
    def __init__(self, cfg, model_path=None, model_type='DDPG', meta=[], training=True, batch_size=64, transfer=None):
        # init car
        self.V = dk.vehicle.Vehicle()

        self.cfg = cfg
        self.model_path = model_path
        self.model_type = model_type
        self.weight_path = transfer
        self.meta = meta
        self.training = training
        self.model_dict = {
                'DDPG': DDPG,
                'PPO': PPO,
                'SAC': SAC,
                'TD3': TD3,
                'IL_TEST': IL_TEST
                }

        self.BATCH_SIZE = batch_size

        self.th = TubHandler(path=self.cfg.DATA_PATH)
        self.ctr = get_js_controller(self.cfg)
        self.eps = 0

        self._setup_vihecle()

    def drive(self):
        #run the vehicle for 20 seconds
        self.V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
                max_loop_count=cfg.MAX_LOOPS)

    def _reset_rotary_encoder(self):
        index = 0
        for i, entry in enumerate(self.V.parts):
            if isinstance(entry['part'], RotaryEncoder):
                entry['part'].shutdown()
                self.V.parts.pop(i)
                index = i
        # re = RotaryEncoder(mm_per_tick=5.469907)
        re = RotaryEncoder(mm_per_tick=self.cfg.ROTARY_MM_PER_TICK)
        self.V.add(re, outputs=['rotaryencoder/meter', 'rotaryencoder/meter_per_second', 'rotaryencoder/delta'], threaded=True, index=index)

        self.V.parts[index].get('thread').start()

    def _set_new_tub(self):
        # add tub to save data
        inputs=['cam/image_array',
                'angle', 'throttle',
                'user/mode', 'train_state']

        types=['image_array',
               'float', 'float',
               'str', 'int']

        # Record rotary encoder
        inputs += ['rotaryencoder/meter', 'rotaryencoder/meter_per_second', 'rotaryencoder/delta']
        types += ['float', 'float', 'float']

        tub = self.th.new_tub_writer(inputs=inputs, types=types, user_meta=self.meta)

        index = 0
        for i, part in enumerate(self.V.parts):
            if isinstance(part['part'], type(tub)):
                self.V.parts.pop(i)
                index = i
        self.V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording', index=index)

        #tell the controller about the tub
        self.ctr.set_tub(tub)

    def _setup_controller(self):
        '''
        train_state
        0 - pendding
        1 - running
        2 - emergency stop
        3 - successed
        4 - train
        5 - driving
        '''
        self.V.add(self.ctr, 
          inputs=['cam/image_array', 'kl/train_trigger'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording', 'train_state'],
          threaded=True)

        def emergency_stop():
            # Modified emergency stop
            if self.ctr.train_state == 1:
                print('E-Stop!!!')
                self.ctr.mode = 'user'
                self.ctr.constant_throttle = False
                self.ctr.estop_state = self.ctr.ES_START
                self.ctr.throttle = 0.0
                self.ctr.train_state = 2
                self.ctr.recording = False
            else:
                print("Can't activate this button")
        self.ctr.set_button_down_trigger('A', emergency_stop)

        def success_episode():
            # When the donkey reach the goal
            if self.ctr.train_state == 1: 
                print('Episode SUCESSED!!!!!')
                self.ctr.mode = 'user'
                self.ctr.constant_throttle = False
                self.ctr.estop_state = self.ctr.ES_START
                self.ctr.throttle = 0.0
                self.ctr.train_state = 3
                self.ctr.recording = False
            else:
                print("Can't activate this button")
        self.ctr.set_button_down_trigger('Y', success_episode)

        def start_rl_episode():
            # Start RL Episode
            if self.ctr.train_state == 0:
                print("New Episode (eps_{}) Start!!!".format(self.eps))
                self.eps += 1
                self._set_new_tub()
                self._reset_rotary_encoder()
                self.ctr.mode = 'rl_pilot'
                self.ctr.train_state = 1
            else:
                print("Can't activate this button")
        self.ctr.set_button_down_trigger('B', start_rl_episode)

    def _setup_vihecle(self):
        # add camera
        print("cfg.CAMERA_TYPE", self.cfg.CAMERA_TYPE)
        inputs = []
        cam = Webcam(image_w=self.cfg.IMAGE_W, image_h=self.cfg.IMAGE_H, image_d=self.cfg.IMAGE_DEPTH)
        self.V.add(cam, inputs=inputs, outputs=['cam/image_array'], threaded=True)

        self._setup_controller()
        
        #this throttle filter will allow one tap back for esc reverse
        th_filter = ThrottleFilter()
        self.V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

        #See if we should even run the pilot module.
        #This is only needed because the part run_condition only accepts boolean
        class PilotCondition:
            def run(self, mode):
                if mode == 'user':
                    return False
                else:
                    return True

        self.V.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])
        
        def get_record_alert_color(num_records):
            col = (0, 0, 0)
            for count, color in cfg.RECORD_ALERT_COLOR_ARR:
                if num_records >= count:
                    col = color
            return col

        class RecordTracker:
            def __init__(self):
                self.last_num_rec_print = 0
                self.dur_alert = 0
                self.force_alert = 0

            def run(self, num_records):
                if num_records is None:
                    return 0

                if self.last_num_rec_print != num_records or self.force_alert:
                    self.last_num_rec_print = num_records

                    if num_records % 10 == 0:
                        print("recorded", num_records, "records")

                    if num_records % cfg.REC_COUNT_ALERT == 0 or self.force_alert:
                        self.dur_alert = num_records // cfg.REC_COUNT_ALERT * cfg.REC_COUNT_ALERT_CYC
                        self.force_alert = 0

                if self.dur_alert > 0:
                    self.dur_alert -= 1

                if self.dur_alert != 0:
                    return get_record_alert_color(num_records)

                return 0

        rec_tracker_part = RecordTracker()
        self.V.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

        # Adding Rotary Encoder
        # re = RotaryEncoder(mm_per_tick=5.469907)
        re = RotaryEncoder(mm_per_tick=self.cfg.ROTARY_MM_PER_TICK)
        self.V.add(re, outputs=['rotaryencoder/meter', 'rotaryencoder/meter_per_second', 'rotaryencoder/delta'], threaded=True)
        
        class ImgPreProcess():
            '''
            preprocess camera image for inference.
            normalize and crop if needed.
            '''
            def __init__(self, cfg):
                self.cfg = cfg
                self.ln = None
                if cfg.NORM_IMAGES_ILLUMINANCE:
                    self.ln = LN(model_path, train=False)

            def run(self, img_arr):
                img = normalize_and_crop(img_arr, self.cfg)
                if self.ln is not None:
                    img = self.ln.normalize_lightness(img_arr)
                return img

        inf_input = 'cam/normalized/cropped'
        self.V.add(ImgPreProcess(self.cfg),
            inputs=['cam/image_array'],
            outputs=[inf_input],
            run_condition='run_pilot')

        inputs=[inf_input]

        def load_model(kl, model_path):
            start = time.time()
            print('loading model', model_path)
            kl.load(model_path)
            print('finished loading in %s sec.' % (str(time.time() - start)) )

        def load_weights(kl, weights_path):
            start = time.time()
            try:
                print('loading model weights', weights_path)
                kl.actor.load_weights(weights_path)
                print('finished loading in %s sec.' % (str(time.time() - start)) )
            except Exception as e:
                print(e)
                print('ERR>> problems loading weights', weights_path)

        # Set the rl network
        kl = self.model_dict[self.model_type](num_action=2, input_shape=(120, 160, 3), batch_size=self.BATCH_SIZE, model_path = self.model_path)

        if self.weight_path:
            load_weights(kl, self.weight_path)

        if not self.training:
            model_reload_cb = None
            if '.h5' in self.model_path or '.uff' in self.model_path or 'tflite' in self.model_path or '.pkl' in self.model_path:
                #when we have a .h5 extension
                #load everything from the model file
                load_model(kl, self.model_path)

                def reload_model(filename):
                    load_model(kl, filename)

                model_reload_cb = reload_model

            else:
                print("ERR>> Unknown extension type on model file!!")
                return

        #these parts will reload the model file, but only when ai is running so we don't interrupt user driving
        # self.V.add(FileWatcher(self.model_path), outputs=['modelfile/dirty'], run_condition="ai_running")
        # self.V.add(DelayedTrigger(100), inputs=['modelfile/dirty'], outputs=['modelfile/reload'], run_condition="ai_running")
        # self.V.add(TriggeredCallback(self.model_path, model_reload_cb), inputs=["modelfile/reload"], run_condition="ai_running")

        outputs=['pilot/angle', 'pilot/throttle', 'kl/train_trigger']

        self.V.add(kl, 
            inputs=[inf_input, 'rotaryencoder/meter_per_second', 'rotaryencoder/meter', 'train_state'], 
            outputs=outputs)
        
        #Choose what inputs should change the car.
        class DriveMode:
            def __init__(self, cfg):
                self.cfg = cfg

            def run(self, mode,
                        user_angle, user_throttle,
                        pilot_angle, pilot_throttle):
                if mode == 'user':
                    return user_angle, user_throttle

                elif mode == 'local_angle':
                    return pilot_angle if pilot_angle else 0.0, user_throttle

                else:
                    throttle = max(0, min(pilot_throttle * self.cfg.AI_THROTTLE_MULT, 1 * self.cfg.AI_THROTTLE_MULT)) if pilot_throttle else 0.0
                    steering = max(-1, min(pilot_angle, 1)) if pilot_angle else 0.0
                    return steering, throttle

        self.V.add(DriveMode(self.cfg),
              inputs=['user/mode', 'user/angle', 'user/throttle',
                      'pilot/angle', 'pilot/throttle'],
              outputs=['angle', 'throttle'])

        class AiRunCondition:
            '''
            A bool part to let us know when ai is running.
            '''
            def run(self, mode):
                if mode == "user":
                    return False
                return True

        self.V.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

        #Ai Recording
        class AiRecordingCondition:
            '''
            return True when ai mode, otherwize respect user mode recording flag
            '''
            def run(self, mode, recording, train_state):
                if mode == 'user':
                    if 1 < train_state < 4:
                        return True
                    return False
                return True

        self.V.add(AiRecordingCondition(), inputs=['user/mode', 'recording', 'train_state'], outputs=['recording'])

        #Drive train setup
        steering_controller = PCA9685(self.cfg.STEERING_CHANNEL, self.cfg.PCA9685_I2C_ADDR, busnum=self.cfg.PCA9685_I2C_BUSNUM)
        steering = PWMSteering(controller=steering_controller,
                                        left_pulse=self.cfg.STEERING_LEFT_PWM,
                                        right_pulse=self.cfg.STEERING_RIGHT_PWM)

        throttle_controller = PCA9685(self.cfg.THROTTLE_CHANNEL, self.cfg.PCA9685_I2C_ADDR, busnum=self.cfg.PCA9685_I2C_BUSNUM)
        throttle = PWMThrottle(controller=throttle_controller,
                                        max_pulse=self.cfg.THROTTLE_FORWARD_PWM,
                                        zero_pulse=self.cfg.THROTTLE_STOPPED_PWM,
                                        min_pulse=self.cfg.THROTTLE_REVERSE_PWM)

        self.V.add(steering, inputs=['angle'], threaded=True)
        self.V.add(throttle, inputs=['throttle'], threaded=True)

        self.ctr.print_controls()


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    if args['drive']:
        model_type = args['--type']

        drive(cfg, model_path=args['--model'],
              model_type=model_type,
              meta=args['--meta'])

    else:
        model = args['--model']
        model_type = args['--type']
        transfer = args['--transfer']
        batch_size = int(args['--batch-size'])


        rd = RL_Driver(cfg, model_path=model, model_type=model_type, batch_size=batch_size, transfer=transfer)
        rd.drive()

