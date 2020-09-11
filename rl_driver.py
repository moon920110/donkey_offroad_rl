#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    manage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|latent)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>]
    manage.py (train) [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer)] [--continuous] [--aug] [--myconfig=<filename>]


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
from donkeycar.utils import *
from normalizer import LN


class RL_Driver():
    def __init__(self, cfg, model_path=None, model_type='DDPG', meta=[], training=True):
        # init car
        self.V = dk.vehicle.Vehicle()

        self.cfg = cfg
        self.model_path = model_path
        self.model_type = model_type
        self.meta = meta
        self.training = training

        self.th = None
        self.ctr = None
        self.emergency_stopped = False
        self.goal = False

        self._setup_vihecle()
        if self.training:
            self._start_parts_thread()

    def stop(self):
        self.V.stop()

    def drive(self):
        #run the vehicle for 20 seconds
        V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
                max_loop_count=cfg.MAX_LOOPS)

    def step(self, action):
        # Get outputs and apply it to car. 
        self.V.mem.put(['pilot/angle', 'pilot/throttle'], action)
        self.V.update_parts()

        # Then return the next obs, reward, done, info
        obs = self._get_state()
        reward = self._get_reward()
        done = self.emergency_stopped or self.goal

        return obs, reward, done

    def reset(self):
        # Set new episode
        self._set_new_tube()
        self.emergency_stopped = False
        self.goal = False

        # Get Observation(img, speed)
        self.V.update_parts()
        obs = self._get_state()
    
        return obs

    def _start_parts_thread(self):
        for entry in self.V.parts:
            if entry.get('thread'):
                entry.get('thread').start()

    def _set_new_tube(self):
        #add tub to save data
        inputs=['cam/image_array',
                'user/angle', 'user/throttle',
                'user/mode']

        types=['image_array',
               'float', 'float',
               'str']

        if self.cfg.RECORD_DURING_AI:
            inputs += ['pilot/angle', 'pilot/throttle']
            types += ['float', 'float']

        if self.cfg.ENABLE_ROTARY_ENCODER:
            inputs += ['rotaryencoder/meter', 'rotaryencoder/meter_per_second', 'rotaryencoder/delta']
            types += ['float', 'float', 'float']

        self.th = TubHandler(path=self.cfg.DATA_PATH)
        tub = self.th.new_tub_writer(inputs=inputs, types=types, user_meta=self.meta)
        self.V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

        #tell the controller about the tub
        self.ctr.set_tub(tub)

    def _get_state(self):
        observation = {
                'img': self.V.mem.get(['cam/image_array']),
                'speed': self.V.mem.get(['rotaryencoder/meter_per_second']),
                }

        return observation

    def _get_reward(self):
        meter = self.V.mem.get(['rotaryencoder/meter'])[0]
        emergency_stop = -100 if self.emergency_stopped else 0
        goal = 100 if self.goal else 0

        return meter + emergency_stop + goal

    def _setup_controller(self):
        self.ctr = get_js_controller(self.cfg)
        self.V.add(self.ctr, 
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

        def emergency_stop():
            # Modified emergency stop
            print('E-Stop!!!')
            self.ctr.mode = 'user'
            self.ctr.recording = False
            self.ctr.constant_throttle = False
            self.ctr.estop_state = self.ctr.ES_STAT
            self.ctr.throttle = 0.0
            self.emergency_stopped = True
        self.ctr.set_button_down_trigger('A', emergency_stop)

        def success_episode():
            # When the donkey reach the goal
            print('Episode SUCESSED!!!!!')
            self.ctr.mode = 'user'
            self.ctr.recording = False
            self.ctr.constant_throttle = False
            self.ctr.estop_state = self.ctr.ES_STAT
            self.ctr.throttle = 0.0
            self.goal = True
        self.ctr.set_button_down_trigger('Y', success_episode)

        def start_rl_episode():
            # Start RL Episode
            print("New Episode Start!!!")
            self.ctr.mode = 'rl_pilot'
            self.ctr.recording = True
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

        #Rotary Encoder added
        if self.cfg.ENABLE_ROTARY_ENCODER:
            from donkeycar.parts.encoder import RotaryEncoder
            re = RotaryEncoder(mm_per_tick=5.469907)
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
                kl.model.load_weights(weights_path)
                print('finished loading in %s sec.' % (str(time.time() - start)) )
            except Exception as e:
                print(e)
                print('ERR>> problems loading weights', weights_path)

        if not self.training:
            # TODO: modify this part
            kl = RL()

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

            #this part will signal visual LED, if connected
            self.V.add(FileWatcher(self.model_path, verbose=True), outputs=['modelfile/modified'])

            #these parts will reload the model file, but only when ai is running so we don't interrupt user driving
            self.V.add(FileWatcher(self.model_path), outputs=['modelfile/dirty'], run_condition="ai_running")
            self.V.add(DelayedTrigger(100), inputs=['modelfile/dirty'], outputs=['modelfile/reload'], run_condition="ai_running")
            self.V.add(TriggeredCallback(self.model_path, model_reload_cb), inputs=["modelfile/reload"], run_condition="ai_running")

            outputs=['pilot/angle', 'pilot/throttle']

            self.V.add(kl, inputs=inputs, # inference input: inf_input
                outputs=outputs, # pilot angle/throttle
                run_condition='run_pilot')
        
        #Choose what inputs should change the car.
        class DriveMode:
            def run(self, mode,
                        user_angle, user_throttle,
                        pilot_angle, pilot_throttle):
                if mode == 'user':
                    return user_angle, user_throttle

                elif mode == 'local_angle':
                    return pilot_angle if pilot_angle else 0.0, user_throttle

                else:
                    return pilot_angle if pilot_angle else 0.0, pilot_throttle * self.cfg.AI_THROTTLE_MULT if pilot_throttle else 0.0

        self.V.add(DriveMode(),
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
            def run(self, mode, recording):
                if mode == 'user':
                    return recording
                return True

        if self.cfg.RECORD_DURING_AI and not self.training:
            self.V.add(AiRecordingCondition(), inputs=['user/mode', 'recording'], outputs=['recording'])

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

    # TODO: Not implemented yet
    if args['drive']:
        model_type = args['--type']

        drive(cfg, model_path=args['--model'],
              model_type=model_type,
              meta=args['--meta'])

    else:
        model = args['--model']
        model_type = args['--type']
        transfer = args['--transfer']

        rd = RL_Driver(cfg)
        obs = rd.reset()
        while True:
            time.sleep(1)
            obs, reward, done = rd.step([0, 1])
