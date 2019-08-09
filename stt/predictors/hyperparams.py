# This file contains some parameters that we are not going to compare...

sample_rate = 16000
preemphasis = 0.97
hop_sec = 0.01
win_sec = 0.025
hop_length = int(sample_rate * hop_sec)
win_length = int(sample_rate * win_sec)
# n_fft = 1 << (win_length - 1).bit_length()
n_fft = win_length
n_channel = (n_fft // 2) + 1
# n_fft = 1024

# others
n_mels = 80
n_mfcc = 13

# print("n_mels: {}".format(n_mels))
# print("hop_length: {}".format(hop_length))
# print("win_length: {}".format(win_length))
# print("n_fft: {}".format(n_fft))
# print("n_channel: {}".format(n_channel))
