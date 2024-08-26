import math
import numpy as np
from step_detect import StepDetect
from angle_detect import AngleDetect


def update_position(pre_position, rotation):
    step_len = 7.5
    x, y = pre_position[0], pre_position[1]
    x_new = x + step_len * math.cos(rotation * math.pi / 180)
    y_new = y + step_len * math.sin(rotation * math.pi / 180)
    return [x_new, y_new]


class PDR():
    def __init__(self):
        self.pre_position = [80, 240]
        self.rotation = 0
        self.num_beacon = 3
        self.step_detector = StepDetect()
        self.angle_detector = AngleDetect()
        self.all_step = 0
        self.init_compass = 0

        self.stastic_beacon = []

    def pdr_position(self, imuLog):
        self.cur_step = 0
        for data in imuLog:
            if data['logType'] == "compass":
                angle = data['angle']
                angle_timestap = data['timestamp'] / 1e9
                self.angle_detector.update_angle_and_timestamp(angle, angle_timestap)
                self.init_compass = angle
                self.rotation = self.angle_detector.smooth_angle()

            elif data['logType'] == "accelerometer":
                acc_x, acc_y, acc_z = data['x'], data['y'], data['z']
                acc_timestamp = data['timestamp'] / 1e9
                acc_norm = np.linalg.norm([acc_z, acc_y, acc_z])
                self.step_detector.update_acc_and_timestamp(acc_norm, acc_timestamp)
                step_flag = self.step_detector.step_detect()

                if step_flag:
                    new_position = update_position(self.pre_position, self.rotation)
                    if new_position[0] <= 75: new_position[0] = 75
                    if new_position[0] >= 145: new_position[0] = 145
                    if new_position[1] <= 235: new_position[1] = 235
                    if new_position[1] >= 285: new_position[1] = 285

                    self.pre_position = new_position

                    self.cur_step += 1
        self.all_step += self.cur_step

        return self.pre_position
