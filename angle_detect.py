"""方向检测。
主要思路是对过去一段时间的角度数据进行平滑拟合， 得到当前的角度作为方向。 """
import numpy as np


class AngleDetect():

    def __init__(self):
        self.sliding_window_length = 64
        self.angle_sliding_window = []
        self.timestamp_sliding_window = []

    def update_angle_and_timestamp(self, new_angle, new_timestamp):
        if len(self.angle_sliding_window) > 0:  # process the -180 and 180
            if abs(new_angle - self.angle_sliding_window[-1]) > 300:
                new_angle = self.angle_sliding_window[-1]

        # new_angle = self.find_nearest_angle(new_angle)

        if len(self.angle_sliding_window) >= self.sliding_window_length:
            self.angle_sliding_window = self.angle_sliding_window[1:]
            self.timestamp_sliding_window = self.timestamp_sliding_window[1:]

        self.angle_sliding_window.append(new_angle)
        self.timestamp_sliding_window.append(new_timestamp)

    def smooth_fft_angle(self, init_angle, timestamp_list):
        # 对信号进行时间域平滑
        sample_rate = int(len(timestamp_list) / (timestamp_list[-1] - timestamp_list[0]))
        padding_num = 8
        padding_angle = [0] * padding_num + init_angle + [0] * padding_num
        num_point = len(padding_angle)
        fhat = np.fft.fft(padding_angle, num_point)
        freq = sample_rate / num_point * np.arange(num_point)
        indices = np.zeros_like(freq)
        for i in range(len(freq)):
            # if (1 <= freq[i] <=3) or (7 <= freq[i] <= 10):
            if (freq[i] <= 1) or (sample_rate - 1 < freq[i]):
                indices[i] = 1
        fhat = indices * fhat
        angle_smooth = np.fft.ifft(fhat)
        angle_smooth = np.real(angle_smooth)

        angle_smooth = angle_smooth[padding_num:num_point - padding_num]

        return angle_smooth

    def find_nearest_angle(self, angle):
        angle_ref_prior = [-270, -180, -90, 0, 90, 180, 270]
        angle_reference_second = [-225, -135, -45, 45, 135, 225]

        diff_prior = [abs(angle - v) for v in angle_ref_prior]
        min_pos_prior = np.array(diff_prior).argmin()

        diff_second = [abs(angle - v) for v in angle_reference_second]
        min_pos_second = np.array(diff_second).argmin()

        if (diff_prior[min_pos_prior] - diff_second[min_pos_second]) > 5:  # 15
            return angle_reference_second[min_pos_second]
        else:
            return angle_ref_prior[min_pos_prior]

        return angle_reference[min_pos]

    def smooth_angle(self):
        # return self.angle_sliding_window[-1]
        if len(self.angle_sliding_window) < 4:
            return self.angle_sliding_window[-1]

        angle_smooth = self.smooth_fft_angle(init_angle=self.angle_sliding_window,
                                             timestamp_list=self.timestamp_sliding_window)
        mean_angle = np.mean(angle_smooth[-5:])
        return self.find_nearest_angle(mean_angle)
        # return mean_angle


if __name__ == "__main__":

    angle_detector = AngleDetect()

    angle_list = []  ## 实时获取到的角度数据， 通常可以通过指南针得到
    timestamp_list = []  ## 数据时间戳， 与上面的角度值一一对应
    for angle, timestamp in enumerate(zip(angle_list, timestamp_list)):
        angle_detector.update_angle_and_timestamp(angle, timestamp)
        smooth_angle = angle_detector.smooth_angle()
