
"""
Simple function for converting Pretty MIDI object into one-hot encoding
/ piano-roll-like to be used for machine learning.
"""
import pretty_midi
import numpy as np
import sys
import argparse

def pretty_midi_to_one_hot(pm, fs=100):
    """Compute a one hot matrix of a pretty midi object

    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.

    Returns
    -------
    one_hot : np.ndarray, shape=(128,times.shape[0])
        Piano roll of this instrument. 1 represents Note Ons,
        -1 represents Note offs, 0 represents constant/do-nothing
    """

    # Allocate a matrix of zeros - we will add in as we go
    one_hots = []

    for instrument in pm.instruments:
        one_hot = np.zeros((128, int(fs*instrument.get_end_time())+1))
        for note in instrument.notes:
            # note on
            one_hot[note.pitch, int(note.start*fs)] = 1
            print('note on',note.pitch, int(note.start*fs))
            # note off
            one_hot[note.pitch, int(note.end*fs)] = 0
            print('note off',note.pitch, int(note.end*fs))
        one_hots.append(one_hot)

    one_hot = np.zeros((128, np.max([o.shape[1] for o in one_hots])))
    for o in one_hots:
        one_hot[:, :o.shape[1]] += o

    one_hot = np.clip(one_hot,-1,1)
    return one_hot

def one_hot_to_pretty_midi(one_hot, fs=100, program=1,bpm=120):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,time)
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    bpm : int
        Beats per minute, used to decide when to re-emphasize notes left on.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    '''
    notes, frames = one_hot.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # prepend, append zeros so we can acknowledge inital and ending events
    piano_roll = np.hstack((np.zeros((notes, 1)),
                            one_hot,
                            np.zeros((notes, 1))))

    # use changes to find note on / note off events
    changes = np.nonzero(np.diff(piano_roll).T)

    # keep track of note on times and notes currently playing
    note_on_time = np.zeros(notes)
    current_notes = np.zeros(notes)

    bps = bpm / 60
    beat_interval = fs / bps
    strong_beats = beat_interval * 2 #(for 4/4 timing)

    last_beat_time = 0

    for time, note in zip(*changes):
        change = piano_roll[note, time + 1]

        if time >= last_beat_time + beat_interval:
            for note in current_notes:

        time = time / fs


        if change == 1:
            # note on
            if current_notes[note] == 0:
                # from note off
                note_on_time[note] = time
                current_notes[note] = 1
            else:
                #re-articulate (later in code)
                '''pm_note = pretty_midi.Note(
                        velocity=100, #don't care fer now
                        pitch=note,
                        start=note_on_time[note],
                        end=time)
                instrument.notes.append(pm_note)
                note_on_time[note] = time
                current_notes[note] = 1'''
        elif change == 0:
            #note off
            pm_note = pretty_midi.Note(
                    velocity=100, #don't care fer now
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
            current_notes[note] = 0
            instrument.notes.append(pm_note)
    pm.instruments.append(instrument)
    return pm

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Translate MIDI file to piano roll and back',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_midi', action='store',
                        help='Path to the input MIDI file')
    parser.add_argument('output_midi', action='store',
                        help='Path where the translated MIDI will be written')
    parser.add_argument('--fs', default=100, type=int, action='store',
                        help='Sampling rate to use between conversions')
    parser.add_argument('--program', default=1, type=int, action='store',
                        help='Program of the instrument')

    parameters = vars(parser.parse_args(sys.argv[1:]))
    pm = pretty_midi.PrettyMIDI(parameters['input_midi'])
    #print(pm.instruments[0].notes)
    oh = pretty_midi_to_one_hot(pm, fs=parameters['fs'])
    new_pm = one_hot_to_pretty_midi(oh, fs=parameters['fs'],
                                       program=parameters['program'])
    #print(new_pm.instruments[0].notes)
    new_pm.write(parameters['output_midi'])
