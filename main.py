import random

import mido
from scipy.io import wavfile
import sys
import math
import pathlib
from sms_tools.software.models import stft
from scipy.signal import get_window
from collections import namedtuple, defaultdict
from statistics import mean, median
from mapping import Mapping
from jack_player import JackPlayer
from mido import Message
import time

TimelineEntry = namedtuple("TimelineEntry", "start stop note velocity")
Event = namedtuple("Event", "type time note velocity")
TestSetEntry = namedtuple("TestSetEntry",
                          "filename velocity_threshold min_duration max_note min_note min_amplitude_db transposition")

test_sets = {
    # entry 0 is the worst :)
    0: TestSetEntry(filename='inputs/hello-long-filtered.wav',  # hello, hello, hello
                    min_note=0,
                    max_note=100,
                    transposition=0,
                    min_amplitude_db=-180,
                    velocity_threshold=101,
                    min_duration=0.02),
    1: TestSetEntry(filename='inputs/scared.wav',  # i don't wanna go, i'm scared
                    min_note=30,
                    max_note=120,
                    transposition=0,
                    min_amplitude_db=-153,
                    velocity_threshold=100,
                    min_duration=0.02),
    2: TestSetEntry(filename="inputs/peace.wav",
                    # once this was a peaceful place, until one day it all went out of control
                    min_note=0,
                    max_note=127,
                    transposition=0,
                    min_amplitude_db=-160,
                    velocity_threshold=93,
                    min_duration=0.02),
    3: TestSetEntry(filename="inputs/ohmy.wav",  # oh my
                    min_note=0,
                    max_note=127,
                    transposition=0,
                    min_amplitude_db=-140,
                    velocity_threshold=100,
                    min_duration=0.02),
    4: TestSetEntry(filename="inputs/greatestshow.wav",
                    # welcome ladies and gentlemen and welcome to the greatest show on earth
                    min_note=0,
                    max_note=127,
                    transposition=-3,
                    min_amplitude_db=-190,
                    velocity_threshold=100,
                    min_duration=0.02),
    5: TestSetEntry(filename="inputs/scale.wav",
                    min_note=0,
                    max_note=127,
                    transposition=0,
                    min_amplitude_db=-120,
                    velocity_threshold=90,
                    min_duration=0.02),
    6: TestSetEntry(filename="inputs/thankyou.wav",
                    min_note=0,
                    max_note=127,
                    transposition=0,
                    min_amplitude_db=-110,
                    velocity_threshold=89,
                    min_duration=0.02),
    7: TestSetEntry(filename="inputs/rustigmaar.wav",
                    min_note=30,
                    max_note=120,
                    transposition=0,
                    min_amplitude_db=-190,
                    velocity_threshold=95,
                    min_duration=0.02)
}
test_id = 7


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier if n is not None else None


def cpsmidi(value):
    return math.log2(value * 0.0022727272727) * 12.0 + 69


def dbamp(value):
    return 10 ** (value * 0.05)


def distill_timeline(time_step, score, vel_threshold):
    freq_tracker = defaultdict(lambda: [])
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
                                                      velocity=int(
                                                          round_half_up(mean([el[1] for el in freq_tracker[note]])))))
                        del freq_tracker[note]

        abs_time += time_step

    return timeline


def distill_event_list(timeline):
    event_list = []
    for event in timeline:
        event_list.append(Event(type='note_on', time=event.start, note=event.note, velocity=event.velocity))
        event_list.append(Event(type='note_off', time=event.stop, note=event.note, velocity=0))
    event_list.sort(key=lambda el: el.time)
    print(event_list[-1])
    return event_list


def remove_short_events(timeline, min_time):
    result = []
    for e in timeline:
        if e.stop - e.start > min_time:
            result.append(e)
    return result


def print_stats(midinotes):
    # h = np.histogram(midinotes, 128, (0, 127))
    # print(h[0])
    pass


def make_odd(num):
    if num % 2 == 0:
        return num + 1
    return num


def analyse_audio_stft(fs, mono, own_path):
    window = 'hamming'
    fft_size = 8192
    analysis_window_size = make_odd(int(fft_size / 2))
    w = get_window(window, analysis_window_size)
    hop_size = 2048
    check_model = stft.stft(x=mono, w=w, N=fft_size, H=hop_size)
    wavfile.write(own_path.joinpath("outputs/check_stft.wav"), fs, check_model)

    hmag, hphase = stft.stftAnal(x=mono, w=w, N=fft_size, H=hop_size)
    spacing = fs / fft_size
    hfreq = [[i * spacing for i in range(fft_size // 2)] for _ in range(len(hmag))]
    print(f"frequency bin spacing = {spacing} Hz")
    return hfreq, hmag, hphase


def convert_freq_mag_to_event_list(duration, hfreq, hmag):
    global test_sets
    global test_id
    if len(hfreq) > 0:
        score = []
        no_of_lines = len(hfreq)
        time_step = duration / no_of_lines
        max_amp = -1e10
        min_amp = 1e10
        for m in hmag:
            relevant_amps = [el for el in m if el > test_sets[test_id].min_amplitude_db]
            max_amp = max([max_amp, max(relevant_amps)])
            min_amp = min([min_amp, min(relevant_amps)])
        print(f"{max_amp = }, {min_amp = }")

        for f, m in zip(hfreq, hmag):
            midinotes = [int(round_half_up(cpsmidi(freq))) + test_sets[test_id].transposition if freq > 0 else 0 for
                         freq in f]
            midinotes_filtered = [e if test_sets[test_id].min_note <= e <= test_sets[test_id].max_note else 0 for e in
                                  midinotes]
            print_stats(midinotes_filtered)
            amps = [el if el >= test_sets[test_id].min_amplitude_db else min_amp for el in m]
            mapped_amps = [Mapping.linlin(a, min_amp, max_amp, 0, 127) for a in amps]
            rescaled_amps = [int(round_half_up(el)) for el in mapped_amps]
            score.append(
                [(note, amp) for note, amp in zip(midinotes_filtered, rescaled_amps) if note != 0 and amp != 0])

        timeline = distill_timeline(time_step, score, test_sets[test_id].velocity_threshold)
        filtered_timeline = remove_short_events(timeline, test_sets[test_id].min_duration)
        event_list = distill_event_list(filtered_timeline)
        return event_list, filtered_timeline

    return []


def perform_event_list(event_list, use_direct_hardware_connection=True):
    if event_list:
        if use_direct_hardware_connection:
            outport = mido.open_output('INTEGRA-7:INTEGRA-7 MIDI 1 28:0')

            for i in range(100):
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
        else:
            new_event_list = event_list[:]
            # add an extra nop event to prevent last notes from keeping playing
            extra_time = event_list[-1].time + test_sets[test_id].min_duration
            new_event_list.append(Event(type='nop', time=extra_time, note=0, velocity=0))
            j = JackPlayer(new_event_list, 'ardour:MIDI 1/midi_in 1')
            j.wait_until_finished()
            j.close()

def time_to_ticks(elapsed_time, resolution, tempo,):
    return int(resolution * (1 / tempo) * 1000 * elapsed_time)


def main():
    own_path = pathlib.Path(sys.argv[0]).parent
    fs, audio = wavfile.read(own_path.joinpath(test_sets[test_id].filename))

    if audio.ndim == 1:
        mono = audio / (2 ** 15)
    else:
        mono = audio.sum(axis=1) / audio.shape[1] / (2 ** 15)

    hfreq, hmag, hphase = analyse_audio_stft(fs, mono, own_path)
    event_list, timeline = convert_freq_mag_to_event_list(audio.shape[0] / fs, hfreq, hmag)
    perform_event_list(event_list, use_direct_hardware_connection=True)


if __name__ == '__main__':
    main()
