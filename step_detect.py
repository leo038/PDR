"""步数检测。
主要原理是利用加速度的波峰和波谷特征确定步数。
实现了实时计算和离线计算2种方式。 离线计算拿到所有时间步的数据再进行计算；实时计算无法一次性拿到所有数据， 获得一个更新数据就计算一次。"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


class StepDetect():
    def __init__(self):
        self.max_peak_valley_interval = 5  # the max interval between neighboring peak and valley (or valley and peak)
        self.min_peak_peak_interval = 2  # the min interval between neighboring peaks (or valley)
        self.acc_threshold = 0.8  # the threshold to be a peak or a valley
        self.sliding_window_length = 128  # use last 64 data
        self.acc_sliding_window = [9.8] * self.sliding_window_length
        self.timestamp_sliding_window = [-1] * self.sliding_window_length
        self.acc_sliding_window_after_smooth = None  # the acc after smooth
        self.last_peak_index = None  # used by sliding window
        self.last_valley_index = None
        self.single_left = None
        self.sample_rate_default = 5
        # self.step_length = 7.5

    def update_acc_and_timestamp(self, acc_new, timestamp_new):
        # print("acc_new, timestamp:", acc_new, timestamp_new)
        if len(self.timestamp_sliding_window) > 0 and timestamp_new == self.timestamp_sliding_window[-1]:
            return

        if len(self.acc_sliding_window) >= self.sliding_window_length:
            self.acc_sliding_window = self.acc_sliding_window[1:]  # delete the olddest one
            self.timestamp_sliding_window = self.timestamp_sliding_window[1:]
            if self.last_peak_index is not None:
                self.last_peak_index = self.last_peak_index - 1
            if self.last_valley_index is not None:
                self.last_valley_index = self.last_valley_index - 1

        self.acc_sliding_window.append(acc_new)
        self.timestamp_sliding_window.append(timestamp_new)

    def smooth_fft(self):
        # # smooth the signal using fft
        if self.timestamp_sliding_window[0] == -1:
            sample_rate = self.sample_rate_default
        else:
            sample_rate = int(
                len(self.timestamp_sliding_window) / (
                        self.timestamp_sliding_window[-1] - self.timestamp_sliding_window[0]))

        acc_sw = self.acc_sliding_window - np.mean(self.acc_sliding_window)
        acc_sw = list(acc_sw)
        padding_num = 8
        acc_sliding_padding = [0] * padding_num + acc_sw + [0] * padding_num

        num_points = len(acc_sliding_padding)
        fhat = np.fft.fft(acc_sliding_padding, num_points)
        # PSD = fhat * np.conj(fhat) / num_points
        # plt.plot(range(len(fhat)), abs(fhat))
        # plt.title("fhat")
        # plt.show()
        freq = sample_rate / num_points * np.arange(num_points)
        # print(freq)
        # print("freq:", freq)
        indices = np.zeros_like(freq)
        for i in range(len(freq)):
            # if (1 <= freq[i] <=3) or (7 <= freq[i] <= 10):
            if (0.2 <= freq[i] <= 2) or (sample_rate - 2 <= freq[i] - 0.2):
                indices[i] = 1
        fhat = indices * fhat
        acc_smooth = np.fft.ifft(fhat)
        acc_smooth = np.real(acc_smooth)

        acc_smooth = acc_smooth[padding_num:num_points - padding_num]

        self.acc_sliding_window_after_smooth = acc_smooth

        return acc_smooth

    def visualize(self, peaks, valley, valid_peaks=None, figure_index=0):

        plt.figure()
        plt.subplot(211)
        plt.plot(range(len(self.acc_sliding_window)),
                 np.array(self.acc_sliding_window) - np.mean(np.array(self.acc_sliding_window)))
        plt.plot(range(len(self.acc_sliding_window)),
                 [np.mean(np.array(self.acc_sliding_window) - np.mean(np.array(self.acc_sliding_window))) for i in
                  range(len(self.acc_sliding_window))], 'r')
        plt.title("acc norm: {}".format(figure_index))

        plt.subplot(212)
        plt.plot(range(len(self.acc_sliding_window_after_smooth)), self.acc_sliding_window_after_smooth)
        plt.plot(range(len(self.acc_sliding_window_after_smooth)),
                 [np.mean(self.acc_sliding_window_after_smooth) for i in
                  range(len(self.acc_sliding_window_after_smooth))], 'r')
        plt.xlabel('Time')
        # plt.ylabel('Signal Amplitude')
        # plt.title("acc_norm filter")
        # print(f"peaks: {peaks}")
        # print(f"valley: {valley}")
        for index in peaks:
            plt.scatter(index, self.acc_sliding_window_after_smooth[index], c='y', marker='x')
        for index in valley:
            plt.scatter(index, self.acc_sliding_window_after_smooth[index], c='m', marker='x')

        for index in valid_peaks:
            plt.scatter(index, self.acc_sliding_window_after_smooth[index], c='g', marker='*')

        plt.show()

    def stastic_peaks_valley_pair(self, peaks_index, valley_index):
        """统计有效的波峰-波谷对数。"""

        valid_peaks = []
        matched_valley = []
        matched_pair = {}  # {valley_index: peak_index}

        for peak in peaks_index:
            for i in range(self.max_peak_valley_interval):  # 在波峰之后的5个数据点内寻找对应的波谷
                cur_valley_index = peak + 1 + i
                if cur_valley_index in valley_index:
                    if cur_valley_index in matched_valley:
                        invalid_peak = matched_pair[cur_valley_index]
                        valid_peaks.remove(invalid_peak)  # 如果有更近的匹配， 则删除之前较远的匹配， 保证波峰波谷的最邻近匹配
                    valid_peaks.append(peak)
                    matched_valley.append(cur_valley_index)
                    matched_pair[cur_valley_index] = peak
                    break  ## 匹配最邻近的波谷， 一旦匹配成功即结束
        last_matched_valley = max(matched_pair.keys())
        # print(f"valid peaks: {valid_peaks}")
        # print(f"step:{len(valid_peaks)}")
        return valid_peaks

    def find_valid_single(self, peaks, valley):
        """Find the single left peak or vally."""
        single_left = None
        valid_peaks, last_matched_valley = self.stastic_peaks_valley_pair(peaks, valley)
        if peaks[-1] not in valid_peaks:
            single_left = 'peak'
        elif valley[-1] != last_matched_valley:
            single_left = 'valley'
        return single_left

    def _is_valid(self, valley):
        if len(valley) > 0 and valley[-1] == len(self.acc_sliding_window) - 1:
            return True
        else:
            return False

    def step_detect_offline(self, visualize=True):

        acc_smooth = self.smooth_fft()
        peaks, _ = scipy.signal.find_peaks(acc_smooth, height=self.acc_threshold,
                                           distance=self.min_peak_peak_interval)

        valley, _ = scipy.signal.find_peaks(-1 * acc_smooth, height=self.acc_threshold,
                                            distance=self.min_peak_peak_interval)

        valid_peaks = self.stastic_peaks_valley_pair(peaks, valley)
        if visualize:
            self.visualize(peaks, valley, valid_peaks, figure_index='offline')
        return valid_peaks

    def step_detect(self, visualize=False, figure_index=0):
        if len(self.acc_sliding_window) < 2:
            return False
        step_flag = False

        acc_smooth = self.smooth_fft()
        acc_smooth = np.append(acc_smooth, 0)  # add zero in the last
        peaks, _ = scipy.signal.find_peaks(acc_smooth, height=self.acc_threshold,
                                           distance=self.min_peak_peak_interval)
        valley, _ = scipy.signal.find_peaks(-1 * acc_smooth, height=self.acc_threshold,
                                            distance=self.min_peak_peak_interval)

        # print(f"peaks: {peaks}")
        # print(f"valley:{valley}")
        if self.single_left == 'peak' and self._is_valid(valley):
            ## current value must be an valid valley
            cur_vally = valley[-1]
            if cur_vally - self.last_peak_index <= self.max_peak_valley_interval:
                step_flag = True
                self.single_left = None
            else:
                self.single_left = "vally"
                self.last_valley_index = cur_vally

        elif self.single_left == "vally" and self._is_valid(peaks):
            ## current value must be an valid peak
            cur_peak = peaks[-1]
            if cur_peak - self.last_valley_index <= self.max_peak_valley_interval:
                step_flag = True
                self.single_left = None
            else:
                self.single_left = "peak"
                self.last_peak_index = cur_peak

        elif self.single_left is None:
            if len(valley) > 0 and valley[-1] == len(self.acc_sliding_window) - 1:
                self.single_left = "vally"
                self.last_valley_index = valley[-1]
            elif len(peaks) > 0 and peaks[-1] == len(self.acc_sliding_window) - 1:
                self.single_left = "peak"
                self.last_peak_index = peaks[-1]

        # self.single_left = self.find_valid_single(peaks, valley)

        if visualize and figure_index in [90, 91, 92]:
            print("*" * 20, figure_index, self.single_left)
            self.visualize(peaks, valley, peaks, figure_index=figure_index)
        return step_flag


if __name__ == "__main__":
    acc_data_list = [[], []]  ## 加速度数据， 每个加速度包含x,y,z 三个值
    timestamp_list = []  ## 时间戳， 跟上卖弄的加速度一一对应

    ## online detect
    stepdetector = StepDetect()
    index = 0

    step_flag_list = []
    for acc, timestamp in zip(acc_data_list, timestamp_list):
        acc_norm = np.linalg.norm([acc[0], acc[1], acc[2]])
        stepdetector.update_acc_and_timestamp(acc_norm, timestamp)

        flag = stepdetector.step_detect(visualize=True, figure_index=index)

        step_flag_list.append(flag)
        index += 1

    print(step_flag_list)
    print(f"online detect: step number: {sum(step_flag_list)}")
    print("online detect pos:", np.nonzero(np.array(step_flag_list)))

    ## offline detect

    acc_norm_data = [np.linalg.norm([acc[0], acc[1], acc[2]]) for acc in acc_data_list]
    stepdetector_offline = StepDetect()
    stepdetector_offline.acc_sliding_window = acc_norm_data
    stepdetector_offline.timestamp_sliding_window = timestamp_list
    print(f"all acc norm data: {acc_norm_data}")
    print(f"all timestamp_list: {timestamp_list}")
    offline_res = stepdetector_offline.step_detect_offline()
    print("offline detect: step number:", len(offline_res))
    print("offline detect pos:", np.array(offline_res))
