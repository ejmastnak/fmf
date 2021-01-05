# The contents of this script come directly from the tutorial linked below. All credit to the original author.
# https://dspillustrations.com/pages/posts/misc/spectral-leakage-zero-padding-and-frequency-resolution.html

import numpy as np
from matplotlib import pyplot as plt
import pyaudio
pa = pyaudio.PyAudio()

tones_low = np.array([697, 770, 852, 941])
tones_high = np.array([1209, 1336, 1477, 1633])
characters = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "*", "#")


def get_char_tones(character):
    """
    Maps keypad characters to their tones
    :param character:
    :return:
    """
    return {'1': (697, 1209), '2': (697, 1336), '3': (697, 1477), '4': (770, 1209),
            '5': (770, 1336), '6': (770, 1477), '7': (852, 1209), '8': (852, 1336),
            '9': (852, 1477), '0': (941, 1336), '*': (941, 1209), '#': (941, 1477),
            'A': (697, 1633), 'B': (770, 1633), 'C': (852, 1633), 'D': (941, 1633)}.get(character, (941, 1336))


def play_sound_test(samples, fs):
    """
    Plays an array of samples representing a audio data sampled at frequency fs
    :return:
    """
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
    stream.write(samples)  # play
    stream.stop_stream()
    stream.close()
    pa.terminate()


def get_samples_test():
    """
    Generates a basic (superposition of) sinusoidal signal(s) formatted for use with PyAudio's write function
    :return: The audio signal and associated sample rate
    """
    fs = 44100  # sampling rate, Hz, must be integer
    duration = 1.0  # in seconds, may be float
    f1 = 440.0  # sine frequency, Hz, may be float
    f2 = 880.0  # sine frequency, Hz, may be float

    # t = np.arange(fs * duration) / fs  # generate time samples
    t = np.arange(0, duration, 1/fs)  # generate time samples

    samples = np.sin(2 * np.pi * f1 * t) + 0.1 * np.sin(2 * np.pi * f2 * t)  # generate samples
    return samples.astype(np.float32).tobytes(), fs  # convert np array to float32, then to bytes


def play_np_array(samples, fs):
    """
    Plays an array of samples representing a audio data sampled at frequency fs
    :return:
    """
    samples = samples.astype(np.float32).tobytes()  # convert np array to float32, then to bytes
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
    stream.write(samples)  # play
    stream.stop_stream()
    stream.close()
    pa.terminate()


def get_char_signal(character, duration, fs):
    """
    Returns a dual-tone audio signal corresponding to the single inputted keypad character
    :return:
    """
    f_low, f_high = get_char_tones(character)  # unpack the character's frequencies
    t = np.arange(0, duration, 1 / fs)  # time samples
    return np.cos(2 * np.pi * f_low * t) + np.cos(2 * np.pi * f_high * t)  # signal samples


def get_number_signal(sequence, char_duration, fs):
    """
    Returns a dual-tone audio signal corresponding to the list of inputed inputted keypad character.
    Represents the sound of dialing the inputted list of characters.
    :param sequence: a list of keypad characters 0-9, A-D and #, *
    :param char_duration: the duration in seconds of a single character's sound
    :param fs: sample rate [Hz]
    :return:
    """
    N = int(fs*char_duration)  # number of samples in a single character's sound
    signal = np.zeros(len(sequence) * N)  # preallocate array for entire sequence's sound data
    for i, character in enumerate(sequence):
        signal[i*N:i*N + N] = get_char_signal(character, char_duration, fs)

    # signal = np.hstack([get_char_signal(character, char_duration, fs) for character in sequence])  # one-liner, les

    return signal


def dft(signal, length=None):
    """
    Wrapper method for Fourier transform of a signal and reshifting frequency indices with fftshift
    """
    if length is None:
        length = len(signal)
    return np.fft.fftshift(np.fft.fft(signal, length))


def n(f, fs, N):
    """
    Returns the index n corresponding to the frequency f for a frequency spectrum of N points from -fs/2 to fs/2
    """
    return int((f/fs)*N + N/2)


def guess_character(signal, fs):
    """
    Input the sound signal for a single keypad character.
    Guesses the character based on the two dominant frequencies in the sound signal
    """
    N = len(signal)  # number of samples
    pad_factor = 5
    f_padded = np.arange(-fs / 2, fs / 2, fs / (pad_factor * N))  # 8 times zeropadding
    signal_dft = abs(dft(signal, pad_factor * N))  # signal's DFT

    # finds the indices of the four elements in f_padded closest to the four allowed low-frequency tones
    low_f_indices = np.array([np.argmin(abs(f_padded - f)) for f in tones_low])
    high_f_indices = np.array([np.argmin(abs(f_padded - f)) for f in tones_high])  # same, but for high-frequency tones

    low_amplitudes = signal_dft[low_f_indices]  # amplitudes of the frequencies at low_f_indices
    high_amplitudes = signal_dft[high_f_indices]  # amplitudes of the frequencies at high_f_indices

    low_f_index = np.argmax(low_amplitudes)  # choose strongest-amplitude frequency as match
    high_f_index = np.argmax(high_amplitudes)

    low_f = tones_low[low_f_index]  # guess for matching low-frequency tone
    high_f = tones_high[high_f_index]  # guess for matching high-frequency tone

    # plot_DFT(f_padded, signal_dft)

    for character in characters:  # finds matching character
        if low_f == get_char_tones(character)[0] and high_f == get_char_tones(character)[1]:
            return character

    return '0'  # return 0 by default


def guess_character_mine(signal, fs):
    """
    Input the sound signal for a single keypad character.
    Guesses the character based on the two dominant frequencies in the sound signal
    """
    N = len(signal)  # number of samples
    pad_factor = 5
    f_padded = np.arange(-fs / 2, fs / 2, fs / (pad_factor * N))  # 8 times zeropadding
    signal_dft = abs(dft(signal, pad_factor * N))  # signal's DFT

    N = N*pad_factor
    low_start_index = n(650, fs, N)
    low_end_index = n(990, fs, N)
    high_start_index = n(1140, fs, N)
    high_end_index = n(1700, fs, N)
    # print("Low Index: {}\t High: {}".format(low_start_index, low_end_index))
    # print("High Index: {}\t High: {}".format(high_start_index, high_end_index))

    low_index = np.argmax(signal_dft[low_start_index:low_end_index]) + low_start_index
    high_index = np.argmax(signal_dft[high_start_index:high_end_index]) + high_start_index

    low_f = f_padded[low_index]
    high_f = f_padded[high_index]

    # print("Low f Index: {}\t High f Index: {}".format(low_index, high_index))
    # print("Low f: {}\t High f: {}".format(low_f, high_f))

    match_low_index = np.argmin(abs(tones_low-low_f))
    match_high_index = np.argmin(abs(tones_high - high_f))

    low_f = tones_low[match_low_index]  # guess for matching low-frequency tone
    high_f = tones_high[match_high_index]  # guess for matching high-frequency tone

    # plt.plot(signal_dft, marker='.', linestyle='--')
    # plt.show()
    # plot_DFT(f_padded, signal_dft)

    for character in characters:  # finds matching character
        if low_f == get_char_tones(character)[0] and high_f == get_char_tones(character)[1]:
            return character

    return '0'  # return 0 by default


def guess_number(signal, char_duration, fs):
    samples_per_char = int(fs*char_duration)  # number of samples in a single character's sound
    tones = signal.reshape((-1, samples_per_char))  # split entire number's sound into sounds of each character
    guess = [guess_character_mine(tones[i,:], fs) for i in range(tones.shape[0])]
    return "".join(guess)


def practice():
    N = 4
    a = np.zeros(N)
    b = np.zeros(N)
    for i in range(N):
        a[i] = i*2
        b[i] = i*-2
    print(np.hstack([a, b]))


def mark_possible_tones():
    for f in tones_low:
        plt.axvline(f, color='#888888', linestyle='--')
    for f in tones_high:
        plt.axvline(f, color='#888888', linestyle='--')


def plot_DFT(f, ft):
    plt.plot(f, ft, marker='.', linestyle='--')
    plt.xlim(500, 1750)
    mark_possible_tones()
    plt.show()


def run():
    number = "9147234537"
    char_duration = 0.4
    fs = 8000
    signal = get_number_signal(number, char_duration, fs)
    # signal += 5*np.random.randn(len(signal))

    # t = time.time()

    guess = guess_number(signal, char_duration, fs)
    print("Number: {}\t Guess: {}\t Result: {}".format(number, guess, guess==number))
    play_np_array(signal, fs)

    # print(time.time() - t)


# play_sound_test(*get_samples())  # the asterisk * unpacks the output of get_samples into signal, fs
# print_tone_table()
run()
# practice()
