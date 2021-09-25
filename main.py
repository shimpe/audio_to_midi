import mido
from scipy.io import wavfile
import sys
import math
import pathlib
import time
from sms_tools.software.models import harmonicModel
from sms_tools.software.models import stft
from scipy.signal import get_window
import numpy as np
from collections import namedtuple, defaultdict
from statistics import mean, median
from mapping import Mapping

from midiutil.MidiFile import MIDIFile
from mido import Message

TimelineEntry = namedtuple("TimelineEntry", "start stop note velocity")
Event = namedtuple("Event", "type time note velocity")

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier if n is not None else None


def cpsmidi(value):
    return math.log2(value * 0.0022727272727) * 12.0 + 69


def dbamp(value):
    return 10 ** (value * 0.05)


def distill_timeline(time_step, score, vel_threshold):
    freq_tracker = defaultdict(lambda:[])
    abs_time = 0
    timeline = []
    for line in score:
        for event in line:
            note = event[0]
            if note > 0:
                vel = event[1]
                stop = vel < vel_threshold
                go = not stop
                if go:
                    freq_tracker[note].append((abs_time, vel))
                elif stop:
                    if note in freq_tracker.keys():
                        timeline.append(TimelineEntry(start=freq_tracker[note][0][0],
                                                      stop=freq_tracker[note][-1][0] + time_step,
                                                      note=note,
                                                      velocity=int(round_half_up(median([el[1] for el in freq_tracker[note]])))))
                        del freq_tracker[note]

        abs_time += time_step

    return timeline


def distill_event_list(timeline):
    event_list = []
    for event in timeline:
        event_list.append(Event(type='note_on', time=event.start, note=event.note, velocity=event.velocity))
        event_list.append(Event(type='note_off', time=event.stop, note=event.note, velocity=0))
    event_list.sort(key=lambda el: el.time)
    return event_list


def remove_short_events(timeline, min_time):
    result = []
    for e in timeline:
        if e.stop - e.start > min_time:
            result.append(e)
    return result


def main():
    own_path = pathlib.Path(sys.argv[0]).parent
    fs, audio = wavfile.read(own_path.joinpath("inputs/scared.wav"))
    if audio.ndim == 1:
        mono = audio / (2**15)
    else:
        mono = audio.sum(axis=1) / audio.shape[1] / (2**15)

    #hfreq, hmag, hphase = analyse_audio_sms_tools(fs, mono, own_path)
    hfreq, hmag, hphase = analyse_audio_stft(fs, mono, own_path)

    if hfreq:
        score = []
        no_of_lines = len(hfreq)
        duration = audio.shape[0]/fs
        time_step = duration/no_of_lines
        max_amp = -1e10
        min_amp = 1e10
        for m in hmag:
            relevant_amps = [el for el in m if el != -100]
            max_amp = max([max_amp, max(relevant_amps)])
            min_amp = min([min_amp, min(relevant_amps)])
        print(f"{max_amp = }, {min_amp = }")

        for f, m in zip(hfreq, hmag):
            midinotes = [int(round_half_up(cpsmidi(freq))) if freq > 0 else 0 for freq in f ]
            midinotes_filtered = [e if 0 < e < 127 else 0 for e in midinotes]
            amps = [el if el > -100 else min_amp for el in m]
            mapped_amps = [Mapping.linlin(a, min_amp, max_amp, 0, 127) for a in amps]
            rescaled_amps = [int(round_half_up(el)) for el in mapped_amps]
            score.append([(note, amp) for note, amp in zip(midinotes_filtered, rescaled_amps) if note != 0 and amp != 0])

        timeline = distill_timeline(time_step, score, 100)
        filtered_timeline = remove_short_events(timeline, 0.03)
        event_list = distill_event_list(filtered_timeline)

        outport = mido.open_output('INTEGRA-7:INTEGRA-7 MIDI 1 28:0')

        for i in range(5):
            previous_time = 0
            for event in event_list:
                new_time = event.time
                if new_time == previous_time:
                    outport.send(Message(event.type, channel=0, note=event.note, velocity=event.velocity))
                else:
                    delta = new_time - previous_time
                    previous_time = new_time
                    time.sleep(delta)

            time.sleep(1.0)
            outport.reset()


def analyse_audio_sms_tools(fs, mono, own_path):
    analysis_window_size = 1201
    window = 'blackman'
    fft_size = 2048
    w = get_window(window, analysis_window_size)
    spectral_peak_threshold_db = -90
    max_harmonics = 60
    min_f0 = 130
    max_f0 = 1000
    error_threshold_f0_detection = 5
    harm_dev_slope = 0.1
    min_sine_dur = 0.02
    hop_size = 128
    check_model = harmonicModel.harmonicModel(x=mono, fs=fs, w=w, N=fft_size, t=spectral_peak_threshold_db,
                                              nH=max_harmonics, minf0=min_f0, maxf0=max_f0,
                                              f0et=error_threshold_f0_detection)
    wavfile.write(own_path.joinpath("outputs/check_harmonic.wav"), fs, check_model)
    hfreq, hmag, hphase = harmonicModel.harmonicModelAnal(x=mono, fs=fs, w=w, N=fft_size, H=hop_size,
                                                          t=spectral_peak_threshold_db,
                                                          nH=max_harmonics, minf0=min_f0, maxf0=max_f0,
                                                          f0et=error_threshold_f0_detection,
                                                          harmDevSlope=harm_dev_slope, minSineDur=min_sine_dur)
    return hfreq, hmag, hphase


def analyse_audio_stft(fs, mono, own_path):
    window = 'hanning'
    analysis_window_size = 1201
    fft_size = 4096
    w = get_window(window, analysis_window_size)
    hop_size = 2048
    check_model = stft.stft(x=mono, w=w, N=fft_size, H=hop_size)
    wavfile.write(own_path.joinpath("outputs/check_stft.wav"), fs, check_model)

    hmag, hphase = stft.stftAnal(x=mono, w=w, N=fft_size, H=hop_size)
    hfreq = [[ i*fs/fft_size for i in range(fft_size//2) ] for _ in range(len(hmag))]

    return hfreq, hmag, hphase


if __name__ == '__main__':
    main()
