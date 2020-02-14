import glob
import os
import numpy as np
import soundfile as sf
import matplotlib
from matplotlib import pyplot as plt
from scipy import signal


def visualize(epoch_number, batch_number, dry, wet, fake):
    # reed_acoustic_011-073-075._chunk_0
    sf.write('final/Dry_epoch_{}_batch_{}.wav'.format(epoch_number, batch_number), dry, 16384, subtype='PCM_16')
    sf.write('final/Wet_epoch_{}_batch_{}.wav'.format(epoch_number, batch_number), wet, 16384, subtype='PCM_16')
    sf.write('final/Generated_epoch_{}_batch_{}.wav'.format(epoch_number, batch_number), fake, 16384, subtype='PCM_16')

    font = {'size': 7}
    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(4, 4, figsize=(25, 12))

    axs[0, 0].plot(dry, '-b', alpha=1)
    axs[0, 1].plot(wet, '-g', alpha=1)
    axs[0, 2].plot(fake, '-r', alpha=1)

    axs[0, 0].set_ylabel("Complete Waveform")

    axs[1, 0].plot(dry[:2000], '.b', alpha=0.5, ms=2)
    axs[1, 1].plot(wet[:2000], '.g', alpha=0.5, ms=2)
    axs[1, 2].plot(fake[:2000], '.r', alpha=0.5, ms=2)

    axs[1, 0].set_ylabel("0 - 2000")

    axs[2, 0].plot(dry[:300], '.b', alpha=0.4, ms=2)
    axs[2, 1].plot(wet[:300], '.g', alpha=0.4, ms=2)
    axs[2, 2].plot(fake[:300], '.r', alpha=0.4, ms=2)

    axs[2, 0].set_ylabel("0 - 300")

    axs[3, 0].plot(dry[5000:5400], '.b', alpha=0.4, ms=2)
    axs[3, 1].plot(wet[5000:5400], '.g', alpha=0.4, ms=2)
    axs[3, 2].plot(fake[5000:5400], '.r', alpha=0.4, ms=2)

    axs[3, 0].set_ylabel("5000 - 5400")

    axs[0, 3].plot(dry, '-b', alpha=0.8)
    axs[0, 3].plot(wet, '-g', alpha=0.4)
    axs[0, 3].plot(fake, '-r', alpha=0.35)

    axs[1, 3].plot(dry[:2000], '.b', alpha=0.8, ms=2)
    axs[1, 3].plot(wet[:2000], '.g', alpha=0.4, ms=2)
    axs[1, 3].plot(fake[:2000], '.r', alpha=0.35, ms=2)

    axs[2, 3].plot(dry[:300], '.b', alpha=0.8, ms=2)
    axs[2, 3].plot(wet[:300], '.g', alpha=0.4, ms=2)
    axs[2, 3].plot(fake[:300], '.r', alpha=0.35, ms=2)

    axs[3, 3].plot(dry[5000:5400], '.b', alpha=0.8, ms=2)
    axs[3, 3].plot(wet[5000:5400], '.g', alpha=0.4, ms=2)
    axs[3, 3].plot(fake[5000:5400], '.r', alpha=0.35, ms=2)

    axs[0, 0].set_title("Input")
    axs[0, 1].set_title("Target")
    axs[0, 2].set_title("Generated")
    axs[0, 3].set_title("Overlapped")

    plt.subplot_tool()
    plt.tight_layout()
    plt.savefig('spectroperiodoplots/wavs_epoch_{}_batch_{}'.format(epoch_number, batch_number))
    plt.clf()

    plt.subplot(141)
    f, t, Sxx = signal.spectrogram(np.reshape(dry, (16384)), fs=16384, nfft=2048)
    plt.pcolormesh(t, f, Sxx)
    plt.ylim(top=8000)
    plt.ylabel('Spectrogram\nFrequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('A - Spectrogram for Input signal', fontsize=7)

    plt.subplot(142)
    f, t, Sxx = signal.spectrogram(np.reshape(wet, (16384)), fs=16384, nfft=2048)
    plt.pcolormesh(t, f, Sxx)
    plt.ylim(top=8000)
    plt.ylabel('Spectrogram\nFrequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('A - Spectrogram for Target signal', fontsize=7)

    plt.subplot(143)
    f, t, Sxx = signal.spectrogram(np.reshape(fake, (16384)), fs=16384, nfft=2048)
    plt.pcolormesh(t, f, Sxx)
    plt.ylim(top=8000)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('B - Spectrogram for Generated Signal', fontsize=7)

    plt.subplot(144)
    # f, Pxx = signal.periodogram(orig, fs=sr)
    f, Pxx = signal.welch(np.reshape(dry, (16384)), 16384, 'flattop', 64, scaling='spectrum')
    plt.semilogy(f, Pxx, '-b')
    f, Pxx = signal.welch(np.reshape(wet, (16384)), 16384, 'flattop', 64, scaling='spectrum')
    plt.semilogy(f, Pxx, '-g')
    f, Pxx = signal.welch(np.reshape(fake, (16384)), 16384, 'flattop', 64, scaling='spectrum')
    plt.semilogy(f, Pxx, '-r')
    plt.subplot_tool()
    plt.tight_layout()
    plt.savefig('spectroperiodoplots/spectro_epoch_{}_batch_{}'.format(epoch_number, batch_number))
    plt.clf()
    plt.close(fig)
    plt.close()
    plt.close()
