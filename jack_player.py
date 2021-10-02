import jack
import threading

NOTEON = 0x9 << 4
NOTEOFF = 0x8 << 4
ALL_NOTES_OFF = 123
ALL_SOUND_OFF = 120
CC = 0xB << 4


class JackPlayer:
    def __init__(self, event_list, autoconnect='ardour:MIDI 1/midi_in 1'):
        self.fs = 0
        self.client = jack.Client("audio_to_midi")
        self.outport = self.client.midi_outports.register('midi_out')
        self.client.set_process_callback(self.process)
        self.client.set_samplerate_callback(self.samplerate)
        self.client.set_shutdown_callback(self.shutdown)
        self.client.activate()
        if autoconnect:
            self.client.connect('audio_to_midi:midi_out', autoconnect)

        self.offset = 0
        self.event = threading.Event()
        self.event_list = iter(event_list)
        self.msg = next(self.event_list)

    def close(self):
        self.client.midi_outports.clear()  # unregister all audio output ports
        self.client.deactivate()
        self.client.close()

    def process(self, frames):
        self.outport.clear_buffer()
        channel = 0
        previous_time = self.msg.time
        while True:
            if self.offset >= frames:
                self.offset -= frames
                return  # We'll take care of this in the next block ...
            # Note: This may raise an exception:
            if self.msg.type != 'nop':
                status = NOTEON if self.msg.type == 'note_on' and self.msg.velocity > 0 else NOTEOFF
                self.outport.write_midi_event(self.offset, [status + channel, self.msg.note, self.msg.velocity])
            try:
                self.msg = next(self.event_list)
            except StopIteration:
                self.event.set()
                raise jack.CallbackExit
            self.offset += round((self.msg.time - previous_time) * self.fs)
            previous_time = self.msg.time

    def wait_until_finished(self, timeout=None):
        return self.event.wait(timeout=timeout)

    def samplerate(self, samplerate):
        self.fs = samplerate
        print(f"{self.fs = }")

    def shutdown(self, status, reason):
        print('JACK shutdown:', reason, status)
        self.event.set()

