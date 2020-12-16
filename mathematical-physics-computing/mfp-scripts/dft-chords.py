import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

f_A4 = 440

fig_dir = "../4-dft/figures/"
chord_fig_dir = fig_dir + "chords-{}/".format(f_A4)
note_fig_dir = fig_dir + "first-note/"
data_dir = "../4-dft/data/"
save_figures = True

color_signal = "#16697a"
color_spectrum = "#ff165d"


def get_notes_plucked_chord(chord):
    """
    Returns the notes in plucked Chords
    E: E2 E3 B3 G#3
    D: D3 A3 D4 F#4
    C: C3 G3 C4 E4
    A: A2 A3 C#4 E4
    G: G2 B3 D4 G4
    :param chord:
    :return:
    """
    return {"E": ("E2", "E3", "B3", "G#3"), "D": ("D3", "A3", "D4", "F#4"), "C": ("C3", "G3", "C4", "E4"),
            "A": ("A2", "A3", "C#4", "E4"), "G": ("G2", "B3", "D4", "G4")}.get(chord, ("A2", "A3", "C#4", "E4"))


def get_notes_strummed_chord(chord):
    """
    Returns notes in strummed chords
    E: E2 B2 E3 G#3 B3 E4
    D: D3 A3 D4 F#4
    C: C3 E3 G3 C4 E4
    A: A2 E3 A3 C#4 E4 A4
    G: G2 D3 G3 B3 D4 G4
    :param chord:
    :return:
    """
    return {"E": ("E2", "B2", "E3", "G#3", "B3", "E4"), "D": ("D3", "A3", "D4", "F#4"), "C": ("C3", "E3", "G3", "C4", "E4"),
    "A": ("A2", "E3", "A3", "C#4", "E4", "A4"), "G": ("G2", "D3", "G3", "B3", "D4", "G4")}.get(chord, ("A2", "E3", "A3", "C#4", "E4", "A4"))

def get_note_number(note):
    """
    Maps C to 0, C# to 1, ..., B to 11
    Used for note to ordinal conversion
    """
    return {'C': 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F":  5,
            "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}.get(note, 0)

def get_ordinal(note_char, index):
    """
    Returns ordinal of the note-index e.g. A4 note is "A" and index is 4
    Reference ordinal is A4 at 49 (as 49th piano key)
    """
    return 49 - (9 - get_note_number(note_char)) - 12 * (4 - index)


def decompose_note(note):
    """
    Input a string, e.g. "A4" Return the note's character and index as a tuple e.g. ("A", 4)
    """
    if len(note) == 2:  # e.g. "A4"
        return note[0:1], int(note[1:2])
    elif len(note) == 3:  # e.g. "A#4"
        return note[0:2], int(note[2:3])
    else:  # should never happen
        return "A", 4

def get_note_frequency(note, f_ref):
    """
    Returns frequency of the inputted note in 12 tone equal temperment tuning system
    Uses A4 reference frequency f_ref and assumes reference ordinal 49 for A4ef:

    """
    note_char, note_index = decompose_note(note)
    ordinal = get_ordinal(note_char, note_index)
    return f_ref*((2**(1/12))**(ordinal-49))


def analyze_chord(filename, chord, notes_in_chord, plucked_or_strummed):
    """
    Used to analyze my guitary chord wav files
    """
    samples, fs = sf.read(filename)  # samples and sample rate (in Hz)
    fc = 0.5 * fs  # Nyquist frequency
    N = samples.shape[0]  # number of samples
    T = N / fs  # length of recording in seconds

    dft = np.fft.fftshift(np.fft.fft(samples[:,0]))
    # f = np.arange(-fc, fc, fs / N)  # create frequency points
    # t = np.arange(0, T, 1 / fs)  # generate time samples

    f = np.linspace(-fc, fc, N, endpoint=False)
    t = np.linspace(0, T, N)  # generate time samples

    plot_single_chord(f, dft, t, samples[:, 0], fs, chord, notes_in_chord, plucked_or_strummed)


def plot_single_chord(f, dft, t, signal, fs, chord, notes_in_chord, plucked_or_strummed):
    """
    Plots the waveform on one axis and the spectrum on the axis below
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    dft=np.abs(dft)
    N = len(dft)

    x_scale = 1.1
    y_scale = 1.3
    note_markers, max_f = get_note_markers(notes_in_chord, f_A4)
    min_f_index = int(N/2)
    max_f_index = int((max_f/fs)*N + N/2)
    max_dft = np.max(dft[min_f_index:max_f_index])
    max_f = x_scale*max_f  # show frequencies up to 1.1 * max_f

    # Plot signal
    ax = axes[0]
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Signal $h(t)$")
    ax.plot(t, signal, color=color_signal) #, label="{} major, {}".format(chord, plucked_or_strummed))

    # Plot DFT
    ax = axes[1]
    ax.set_ylabel('DFT $|H(f)|$')
    ax.set_xlabel("Frequency $f$")
    ax.set_xlim((0, max_f))
    ax.set_ylim(top=y_scale * max_dft)  # show values up to yscale * max_dft
    ax.plot(f, dft, color=color_spectrum, linestyle='--', linewidth=1.5, marker='.') #, label="{} major, {}".format(chord, plucked_or_strummed))

    for marker in note_markers:
        ax.axvline(marker[0], ymin=0, ymax=1/y_scale, color='#888888', linestyle='--', zorder=-1)
        ax.annotate(marker[1], (marker[0], 1.02*max_dft), ha="center", va="bottom")

    plt.tight_layout()
    plt.suptitle("Analysis of {} {} Major".format(plucked_or_strummed.title(), chord))
    plt.subplots_adjust(top=0.94)
    if save_figures: plt.savefig(chord_fig_dir + chord + "-" + plucked_or_strummed + "_.png", dpi=200)
    plt.show()


def plot_chords():
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    x_scale = 1.1
    y_scale = 1.4
    colors = ("#ec8689", "#c93475", "#661482", "#350b52")
    # Plot left column -- Plucked chords
    chords = ("A", "C")
    for i, chord in enumerate(chords):
        filename = data_dir + "chords-plucked-{}/".format(f_A4) + "{}.wav".format(chord)
        samples, fs = sf.read(filename)  # samples and sample rate (in Hz)
        fc = 0.5 * fs  # Nyquist frequency
        N = samples.shape[0]  # number of samples
        dft = np.abs(np.fft.fftshift(np.fft.fft(samples[:, 0])))
        f = np.linspace(-fc, fc, N, endpoint=False)

        note_markers, max_f = get_note_markers(get_notes_plucked_chord(chord), f_A4)
        min_f_index = int(N / 2)
        max_f_index = int((max_f / fs) * N + N / 2)
        max_dft = np.max(dft[min_f_index:max_f_index])
        max_f = x_scale * max_f  # show frequencies up to 1.1 * max_f

        ax = axes[i][0]
        ax.set_ylabel('DFT $|H(f)|$')
        if i == 1: ax.set_xlabel("Frequency $f$ [Hz]")
        ax.set_xlim((0, max_f))
        ax.set_ylim(top=y_scale * max_dft)  # show values up to yscale * max_dft
        ax.plot(f, dft, color=colors[i], linestyle='--', linewidth=1.5,
                marker='.', label="{} major, plucked".format(chord))

        for marker in note_markers:
            ax.axvline(marker[0], ymin=0, ymax=1/y_scale, color='#888888', linestyle='--', zorder=-1)
            ax.annotate(marker[1], (marker[0], 1.02 * max_dft), ha="center", va="bottom")
        ax.legend(loc="best", fontsize=10)

    # Plot left column -- strummed chords
    chords = ("D", "G")
    for i, chord in enumerate(chords):
        filename = data_dir + "chords-strummed-{}/".format(f_A4) + "{}.wav".format(chord)
        samples, fs = sf.read(filename)  # samples and sample rate (in Hz)
        fc = 0.5 * fs  # Nyquist frequency
        N = samples.shape[0]  # number of samples
        dft = np.abs(np.fft.fftshift(np.fft.fft(samples[:, 0])))
        f = np.linspace(-fc, fc, N, endpoint=False)

        note_markers, max_f = get_note_markers(get_notes_strummed_chord(chord), f_A4)
        min_f_index = int(N / 2)
        max_f_index = int((max_f / fs) * N + N / 2)
        max_dft = np.max(dft[min_f_index:max_f_index])
        max_f = x_scale * max_f  # show frequencies up to 1.1 * max_f

        ax = axes[i][1]
        if i == 1: ax.set_xlabel("Frequency $f$ [Hz]")
        ax.set_xlim((0, max_f))
        ax.set_ylim(top=y_scale * max_dft)  # show values up to yscale * max_dft
        ax.plot(f, dft, color=colors[i+2], linestyle='--', linewidth=1.5,
                marker='.' , label="{} major, strummed".format(chord))

        for marker in note_markers:
            ax.axvline(marker[0], ymin=0, ymax=1 / y_scale, color='#888888', linestyle='--', zorder=-1)
            ax.annotate(marker[1], (marker[0], 1.02 * max_dft), ha="center", va="bottom")
        ax.legend(loc="best", fontsize=10)

    plt.suptitle("Guitar Chord Spectra", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    if save_figures: plt.savefig(chord_fig_dir + "chords_.png", dpi=200)
    plt.show()


def plot_compare_tuning():
    fig, ax = plt.subplots(figsize=(7, 4))
    x_min = 400
    x_max = 670
    y_scale = 1.2
    color_440 = "#c93475"
    color_432= "#661482"
    marker440 = 'o'
    marker432 = 'd'
    linewidth = 2

    # Plot 440 tuning
    filename = data_dir + "chords-strummed-440/A.wav"
    samples, fs = sf.read(filename)  # samples and sample rate (in Hz)
    fc = 0.5 * fs  # Nyquist frequency
    N = samples.shape[0]  # number of samples
    dft = np.abs(np.fft.fftshift(np.fft.fft(samples[:, 0])))
    f = np.linspace(-fc, fc, N, endpoint=False)

    note_markers, _ = get_note_markers(get_notes_strummed_chord("A"), 440)
    min_f_index = int((x_min / fs) * N + N / 2)
    max_f_index = int((x_max / fs) * N + N / 2)
    max_dft_440 = np.max(dft[min_f_index:max_f_index])

    ax.plot(f, dft, color=color_440, linestyle='--', linewidth=linewidth, marker=marker440, label="A major 440 Hz")

    for marker in note_markers:
        ax.vlines(marker[0], ymin=0, ymax=max_dft_440, color='#888888', linestyle='--', zorder=-1)  # from 0 to maxdft440
        ax.annotate(marker[1], (marker[0], 1.02 * max_dft_440), ha="center", va="bottom")
    plt.subplots_adjust(hspace=0)

    # Plot 432 Tuning
    filename = data_dir + "chords-strummed-432/A.wav"
    samples, fs = sf.read(filename)  # samples and sample rate (in Hz)
    fc = 0.5 * fs  # Nyquist frequency
    N = samples.shape[0]  # number of samples
    dft = np.abs(np.fft.fftshift(np.fft.fft(samples[:, 0])))
    f = np.linspace(-fc, fc, N, endpoint=False)

    note_markers, max_f = get_note_markers(get_notes_strummed_chord("A"), 432)
    min_f_index = int((x_min / fs) * N + N / 2)
    max_f_index = int((x_max / fs) * N + N / 2)
    max_dft_432 = np.max(dft[min_f_index:max_f_index])

    down_shift = 1.2 * max_dft_432
    ax.plot(f, dft - down_shift, color=color_432, linestyle='--', linewidth=linewidth,
            marker=marker432, label="A major 432 Hz")

    for marker in note_markers:
        # ax.axvline(marker[0], ymin=0, ymax=1 / y_scale, color='#888888', linestyle='--', zorder=-1)
        ax.vlines(marker[0], ymin=-down_shift, ymax=max_dft_432-down_shift, color='#888888', linestyle='--', zorder=-1)
        ax.annotate(marker[1], (marker[0], 0.8 * (max_dft_432-down_shift)), ha="center", va="bottom")

    ax.set_xlim((x_min, x_max))  # small window around A4
    ax.set_ylim(bottom=-down_shift, top=y_scale * max_dft_440)  # show values up to yscale * max_dft
    ax.set_xlabel("Frequency $f$ [Hz]")
    ax.grid()
    # ax.set_yticklabels([])
    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    ax.legend(loc=(0.7, 0.7), fontsize=10)

    plt.title("Detecting Flat Tuning", fontsize=16)
    if save_figures: plt.savefig(fig_dir + "chord-440-432_.png", dpi=200)
    plt.show()


def get_note_markers(chord_notes, fA4_ref):
    """
    Input a list of form e.g. ("E2", "E3", "B3", "G#3")
    Pluts vertical lines on the frequency axis marking position of each note on the given axis
    """
    markers = []  # frequency, note-name tuples e.g. (440, "A4")
    max_f = 0  # highest marked frequency
    for note in chord_notes:
        note_frequency = get_note_frequency(note, fA4_ref)

        note_char, note_index = decompose_note(note)
        for i in range(1, 3):  # plot note and first few harmonics
            f_harmonic = i*note_frequency  # frequency of harmonic
            if f_harmonic > max_f:
                max_f = f_harmonic
            if not contains_frequency(markers, f_harmonic):  # sometimes harmonics will overlap with other notes; leave those out
                note = note_char + str(note_index+i-1)
                markers.append((f_harmonic, note))

    return markers, max_f


def contains_frequency(marker_list, frequency):
    for marker in marker_list:
        # print("{}\t{}".format(marker[0], frequency))
        if int(marker[0]) == int(frequency):  # round to int to avoid machine error differences
            return True
    return False


def do_chords():
    """
    Convenience method to handle the chord analysis procedure
    :return:
    """
    plucked_or_strummed = "strummed"
    for chord in ("A", "C", "D", "E", "G"):
        file = data_dir + "chords-{}-{}/".format(plucked_or_strummed, f_A4) + "{}.wav".format(chord)
        if plucked_or_strummed == "plucked":
            notes_in_chord = get_notes_plucked_chord(chord)
        elif plucked_or_strummed == "strummed":
            notes_in_chord = get_notes_plucked_chord(chord)
        else:
            continue
        analyze_chord(file, chord, notes_in_chord, plucked_or_strummed)


def run():
    # do_chords()
    # plot_chords()
    plot_compare_tuning()

# print(get_note_frequency("G3", 440))
run()