import pretty_midi
import numpy as np
from typing import Dict


def midi_path_to_pianoroll(path: str, fs: int=5) -> np.ndarray:
    pmid = pretty_midi.PrettyMIDI(path)
    piano = pmid.instruments[0]
    pianoroll = piano.get_piano_roll(fs=fs)
    return pianoroll
    

def pianoroll_to_time_dict(pianoroll: np.ndarray) -> Dict[int, str]:
    times = np.unique(pianoroll.nonzero()[1])  # czasy gdzie występuje przynajmniej jedna nuta 
    index = pianoroll.nonzero()  # indeksy wszystkich nut
    dict_keys_time = {}

    for time in times:
        index_where = (index[1] == time).nonzero()  # pozycje nut, które występują w danym czasie, w indeksie
        notes = index[0][index_where]  # odszukanie nut
        dict_keys_time[time] = ','.join(notes.astype(str))
        
    return dict_keys_time


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm
