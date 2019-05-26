import numpy as np
from .utils import piano_roll_to_pretty_midi


class MIDIDataset:
    def __init__(self, converter, seq_len=50, song_batch_size=12, nn_batch_size=96):
        self.converter = converter
        self.seq_len = seq_len
        self.song_batch_size = song_batch_size
        self.nn_batch_size = nn_batch_size
        self._pos = 0
        self._song_batch = None
        
    def get_batch(self):
        if self._song_batch is None:
            songs_train, songs_target = self.converter.get_batch(self.song_batch_size, self.seq_len)
            shuffle_order = np.random.permutation(np.arange(songs_target.shape[0]))
            songs_train = songs_train[shuffle_order, :]
            songs_target = songs_target[shuffle_order]
            self._song_batch = (songs_train, songs_target)
            self._pos = 0
            
        songs_train, songs_target = self._song_batch
        end_pos = min(self._pos + self.nn_batch_size, songs_target.shape[0])
        train_batch, target_batch = songs_train[self._pos:end_pos, :], songs_target[self._pos:end_pos]
        if end_pos == songs_target.shape[0]:
            self._song_batch = None
        self._pos = end_pos
        return train_batch, target_batch

    def unique_notes(self):
        return self.converter.unique_notes()
    
    def sample_start(self, n=1):
        frequency = self.converter.notes_frequency()
        notes = np.zeros(shape=self.seq_len)
        for i in range(1,n+1):
            notes[-i] = np.argmax(np.random.multinomial(1, frequency))
        return notes
    
    def to_midi(self, sequence):
        pianoroll = self.converter.sequence_to_pianoroll(sequence)
        generate_to_midi = piano_roll_to_pretty_midi(pianoroll, fs=self.converter.fs)
        for note in generate_to_midi.instruments[0].notes:
            note.velocity = 100
        return generate_to_midi
