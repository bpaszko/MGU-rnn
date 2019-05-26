import os
import numpy as np
from collections import deque, Counter
from typing import Tuple, Dict, List
from tqdm import tqdm

from .utils import midi_path_to_pianoroll, pianoroll_to_time_dict


class MIDIConverter:
    def __init__(self, directory: str, frac: float=0.1, fs: int=30) -> None:
        assert 0 < frac <= 1
        self.fs = fs
        self._paths = self._locate_midi_files(directory, frac)
        self._time_dicts, self._notes_mapping, self._notes_frequency = self._convert_to_time_dicts()
        self._inverse_notes_mapping = {v: k for k, v in self._notes_mapping.items()}
        
    def get_batch(self, batch_size, seq_len):
        idx = np.random.choice(len(self._paths), size=batch_size)
        batch_train, batch_target = [], []
        for i in idx:
            time_dict = self._time_dicts[i]
            train_vals, target_vals = self._time_dict_to_seq(time_dict, seq_len)
            batch_train.append(train_vals)
            batch_target.append(target_vals)
        return np.vstack(batch_train), np.hstack(batch_target)
        
    def unique_notes(self):
        return len(self._notes_mapping)
    
    def notes_frequency(self):
        total = sum(self._notes_frequency.values())
        freqs = np.zeros(shape=(len(self._notes_mapping) + 1), dtype=np.float64)
        for note, idx in self._notes_frequency.items():
            freqs[idx] = self._notes_frequency[note] / total
        return freqs
    
    def sequence_to_pianoroll(self, sequence):
        notes = [[int(note) for note in self._inverse_notes_mapping.get(idx, '-1').split(',')] for idx in sequence]
        pianoroll = np.zeros(shape=(128, len(notes)))
        for i, note_idx in enumerate(notes):
            if note_idx != -1:
                pianoroll[note_idx, i] = 1
        return pianoroll
        
        
    def _time_dict_to_seq(self, time_dict, seq_len) -> np.ndarray:
        times = list(time_dict.keys())
        start_time, end_time = np.min(times), np.max(times)
        n_samples = end_time - start_time
        initial_values = [0]*(seq_len-1) + [time_dict[start_time]]
        train_values = np.zeros(shape=(n_samples+1, seq_len))
        target_values = np.zeros(shape=(n_samples+1))
        train_values_per_step = deque(initial_values)
        for i in range(n_samples):
            train_values[i, :] = list(train_values_per_step)
            current_target = time_dict.get(start_time + i, 0)
            target_values[i] = current_target
            train_values_per_step.popleft()
            train_values_per_step.append(current_target)
        train_values[n_samples, :] = list(train_values_per_step)
        return train_values, target_values
        
    def _convert_to_time_dicts(self) -> Tuple[Dict[int, Dict[int, str]], Dict[str, int]]:
        unique_notes = list()
        time_dicts = {}
        for i, path in tqdm(enumerate(self._paths), total=len(self._paths)):
            pianoroll = midi_path_to_pianoroll(path, fs=self.fs)
            time_dict = pianoroll_to_time_dict(pianoroll)
            time_dicts[i] = time_dict
            unique_notes += list(time_dict.values())

        notes_freq = Counter(unique_notes)
        unique_notes = set(unique_notes)
        # Replace strings with oridinal encoding
        notes_mapping = {note:(i+1) for i, note in enumerate(unique_notes)}
        for i, time_dict in time_dicts.items():
            for time, notes in time_dict.items():
                time_dict[time] = notes_mapping[notes]
        return time_dicts, notes_mapping, notes_freq
            
    def _locate_midi_files(self, base_dir: str, frac: float) -> List[str]:
        midi_files = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.midi'):
                    midi_files.append(os.path.join(root, file))
        return np.random.choice(midi_files, size=int(frac*len(midi_files)), replace=False)
