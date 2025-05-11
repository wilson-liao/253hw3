#!/usr/bin/env python
# coding: utf-8

# ## Homework 3: Symbolic Music Generation Using Markov Chains

# **Before starting the homework:**
# 
# Please run `pip install miditok` to install the [MiDiTok](https://github.com/Natooz/MidiTok) package, which simplifies MIDI file processing by making note and beat extraction more straightforward.
# 
# You’re also welcome to experiment with other MIDI processing libraries such as [mido](https://github.com/mido/mido), [pretty_midi](https://github.com/craffel/pretty-midi) and [miditoolkit](https://github.com/YatingMusic/miditoolkit). However, with these libraries, you’ll need to handle MIDI quantization yourself, for example, converting note-on/note-off events into beat positions and durations.

# In[ ]:


# run this command to install MiDiTok
#! pip install miditok


# In[ ]:


# import required packages
import random
from glob import glob
from collections import defaultdict

import numpy as np
from numpy.random import choice

from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile


# In[ ]:


# You can change the random seed but try to keep your results deterministic!
# If I need to make changes to the autograder it'll require rerunning your code,
# so it should ideally generate the same results each time.
random.seed(42)


# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# ### Load music dataset
# We will use a subset of the [PDMX dataset](https://zenodo.org/records/14984509).
# 
# Please find the link in the homework spec.
# 
# All pieces are monophonic music (i.e. one melody line) in 4/4 time signature.

# In[ ]:


midi_files = glob('PDMX_subset/*.mid')
print(len(midi_files))


# ### Train a tokenizer with the REMI method in MidiTok

# In[ ]:


config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=midi_files)


# ### Use the trained tokenizer to get tokens for each midi file
# In REMI representation, each note will be represented with four tokens: `Position, Pitch, Velocity, Duration`, e.g. `('Position_28', 'Pitch_74', 'Velocity_127', 'Duration_0.4.8')`; a `Bar_None` token indicates the beginning of a new bar.

# In[ ]:


# e.g.:
midi = Score(midi_files[0])
tokens = tokenizer(midi)[0].tokens
print(tokens[:10])


# 1. Write a function to extract note pitch events from a midi file; and another extract all note pitch events from the dataset and output a dictionary that maps note pitch events to the number of times they occur in the files. (e.g. {60: 120, 61: 58, …}).
# 
# `note_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of note pitch events (e.g. [60, 62, 61, ...])
# 
# `note_frequency()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to the number of times they occur, e.g {60: 120, 61: 58, …}

# In[ ]:


def note_extraction(midi_file):
    # Q1a: Your code goes here
    pitc_list = []
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    for feature in tokens:
        if "Pitch" in feature:
            pitch = feature.split("_")[1]
            pitc_list.append(int(pitch))
    return pitc_list

print(note_extraction(midi_files[0]))

# In[ ]:


def note_frequency(midi_files):
    # Q1b: Your code goes here
    pitch_dict = defaultdict(int)
    for midi_file in midi_files:
        pitch_list = note_extraction(midi_file)
        for pitch in pitch_list:
            pitch_dict[int(pitch)] += 1
    return pitch_dict

print(note_frequency(midi_files))


# 2. Write a function to normalize the above dictionary to produce probability scores (e.g. {60: 0.13, 61: 0.065, …})
# 
# `note_unigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to probabilities, e.g. {60: 0.13, 61: 0.06, …}

# In[ ]:


def note_unigram_probability(midi_files):
    note_counts = note_frequency(midi_files)
    unigramProbabilities = {}

    total_count = sum(note_counts.values())
    for pitch, count in note_counts.items():
        unigramProbabilities[pitch] = count / total_count

    return unigramProbabilities


# 3. Generate a table of pairwise probabilities containing p(next_note | previous_note) values for the dataset; 
# write a function that randomly generates the next note based on the previous note based on this distribution.
# 
# `note_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramTransitions`: key: previous_note, value: a list of next_note, e.g. {60:[62, 64, ..], 62:[60, 64, ..], ...} (i.e., this is a list of every other note that occured after note 60, every note that occured after note 62, etc.)
# 
#   - `bigramTransitionProbabilities`: key:previous_note, value: a list of probabilities for next_note in the same order of 
# `bigramTransitions`, e.g. {60:[0.3, 0.4, ..], 62:[0.2, 0.1, ..], ...} (i.e., you are converting the values above to probabilities)
# 
# `sample_next_note()`
# - **Input**: a note
# 
# - **Output**: next note sampled from pairwise probabilities

# In[ ]:


def note_bigram_probability(midi_files):
    bigramTransitions = defaultdict(list)
    bigramTransitionProbabilities = defaultdict(list)

    i = 0
    for midi_file in midi_files:
        pitch_list = note_extraction(midi_file)
        for i in range(len(pitch_list) - 1):
            prev_note = pitch_list[i]
            next_note = pitch_list[i+1]
            if prev_note not in bigramTransitions:
                bigramTransitions[prev_note] = []
                bigramTransitionProbabilities[prev_note] = []
            
            if next_note not in bigramTransitions[prev_note]:
                bigramTransitions[prev_note].append(next_note)
                bigramTransitionProbabilities[prev_note].append(1)
            else:
                next_note_ind = bigramTransitions[prev_note].index(next_note)
                bigramTransitionProbabilities[prev_note][next_note_ind] += 1
    
    for prev_note in bigramTransitionProbabilities:
        total_count = sum(bigramTransitionProbabilities[prev_note])
        for i in range(len(bigramTransitionProbabilities[prev_note])):
            bigramTransitionProbabilities[prev_note][i] = bigramTransitionProbabilities[prev_note][i] / total_count



    return bigramTransitions, bigramTransitionProbabilities


# In[ ]:


def sample_next_note(note):
    # Q3b: Your code goes here
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    probs = bigramTransitionProbabilities[note]
    return np.random.choice(list(bigramTransitions[note]), p=probs)


# 4. Write a function to calculate the perplexity of your model on a midi file.
# 
#     The perplexity of a model is defined as
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-1})))$
# 
#     where $p(w_1|w_0) = p(w_1)$, $p(w_i|w_{i-1}) (i>1)$ refers to the pairwise probability p(next_note | previous_note).
# 
# `note_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# In[ ]:


def note_bigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)

    # Q4: Your code goes here
    # Can use regular numpy.log (i.e., natural logarithm)
    pitch_list = note_extraction(midi_file)
    perplexity = np.log(unigramProbabilities[pitch_list[0]])
    for i in range(len(pitch_list) - 1):
        prev_note = pitch_list[i]
        next_note = pitch_list[i+1]
        next_note_ind = bigramTransitions[prev_note].index(next_note)
        perplexity += np.log(bigramTransitionProbabilities[prev_note][next_note_ind])
    return np.exp(-perplexity / len(pitch_list))


# 5. Implement a second-order Markov chain, i.e., one which estimates p(next_note | next_previous_note, previous_note); 
# write a function to compute the perplexity of this new model on a midi file.
# 
#     The perplexity of this model is defined as
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-2}, w_{i-1})))$
# 
#     where $p(w_1|w_{-1}, w_0) = p(w_1)$, $p(w_2|w_0, w_1) = p(w_2|w_1)$, $p(w_i|w_{i-2}, w_{i-1}) (i>2)$ refers to the probability p(next_note | next_previous_note, previous_note).
# 
# 
# `note_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramTransitions`: key - (next_previous_note, previous_note), value - a list of next_note, e.g. {(60, 62):[64, 66, ..], (60, 64):[60, 64, ..], ...}
# 
#   - `trigramTransitionProbabilities`: key: (next_previous_note, previous_note), value: a list of probabilities for next_note in the same order of `trigramTransitions`, e.g. {(60, 62):[0.2, 0.2, ..], (60, 64):[0.4, 0.1, ..], ...}
# 
# `note_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# In[ ]:


def note_trigram_probability(midi_files):
    trigramTransitions = defaultdict(list)
    trigramTransitionProbabilities = defaultdict(list)

    # Q5a: Your code goes here
    # ...
    for midi_file in midi_files:
        pitch_list = note_extraction(midi_file)
        for i in range(len(pitch_list) - 2):
            prev_note = pitch_list[i]
            next_previous_note = pitch_list[i+1]
            next_note = pitch_list[i+2]
            if (prev_note, next_previous_note) not in trigramTransitions:
                trigramTransitions[(prev_note, next_previous_note)] = []
                trigramTransitionProbabilities[(prev_note, next_previous_note)] = []
            
            if next_note not in trigramTransitions[(prev_note, next_previous_note)]:
                trigramTransitions[(prev_note, next_previous_note)].append(next_note)
                trigramTransitionProbabilities[(prev_note, next_previous_note)].append(1)
            else:
                next_note_ind = trigramTransitions[(prev_note, next_previous_note)].index(next_note)
                trigramTransitionProbabilities[(prev_note, next_previous_note)][next_note_ind] += 1
    
    for prev_note in trigramTransitionProbabilities:
        total_count = sum(trigramTransitionProbabilities[prev_note])
        for i in range(len(trigramTransitionProbabilities[prev_note])):
            trigramTransitionProbabilities[prev_note][i] = trigramTransitionProbabilities[prev_note][i] / total_count


            

    return trigramTransitions, trigramTransitionProbabilities


# In[ ]:


def note_trigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)

    pitch_list = note_extraction(midi_file)
    perplexity = np.log(unigramProbabilities[pitch_list[0]])
    index = bigramTransitions[pitch_list[0]].index(pitch_list[1])
    perplexity += np.log(bigramTransitionProbabilities[pitch_list[0]][index])
    for i in range(len(pitch_list) - 2):
        prev_note = pitch_list[i]
        next_previous_note = pitch_list[i+1]
        next_note = pitch_list[i+2]
        index = trigramTransitions[(prev_note, next_previous_note)].index(next_note)
        perplexity += np.log(trigramTransitionProbabilities[(prev_note, next_previous_note)][index])
    return np.exp(-perplexity / len(pitch_list))




# 6. Our model currently doesn’t have any knowledge of beats. Write a function that extracts beat lengths and outputs a list of [(beat position; beat length)] values.
# 
#     Recall that each note will be encoded as `Position, Pitch, Velocity, Duration` using REMI. Please keep the `Position` value for beat position, and convert `Duration` to beat length using provided lookup table `duration2length` (see below).
# 
#     For example, for a note represented by four tokens `('Position_24', 'Pitch_72', 'Velocity_127', 'Duration_0.4.8')`, the extracted (beat position; beat length) value is `(24, 4)`.
# 
#     As a result, we will obtain a list like [(0,8),(8,16),(24,4),(28,4),(0,4)...], where the next beat position is the previous beat position + the beat length. As we divide each bar into 32 positions by default, when reaching the end of a bar (i.e. 28 + 4 = 32 in the case of (28, 4)), the beat position reset to 0.

# In[ ]:


duration2length = {
    '0.2.8': 2,  # sixteenth note, 0.25 beat in 4/4 time signature
    '0.4.8': 4,  # eighth note, 0.5 beat in 4/4 time signature
    '1.0.8': 8,  # quarter note, 1 beat in 4/4 time signature
    '2.0.8': 16, # half note, 2 beats in 4/4 time signature
    '4.0.4': 32, # whole note, 4 beats in 4/4 time signature
}


# `beat_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of (beat position; beat length) values

# In[ ]:


def beat_extraction(midi_file):
    # Q6: Your code goes here
    pass


# 7. Implement a Markov chain that computes p(beat_length | previous_beat_length) based on the above function.
# 
# `beat_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatTransitions`: key: previous_beat_length, value: a list of beat_length, e.g. {4:[8, 2, ..], 8:[8, 4, ..], ...}
# 
#   - `bigramBeatTransitionProbabilities`: key - previous_beat_length, value - a list of probabilities for beat_length in the same order of `bigramBeatTransitions`, e.g. {4:[0.3, 0.2, ..], 8:[0.4, 0.4, ..], ...}

# In[ ]:


def beat_bigram_probability(midi_files):
    bigramBeatTransitions = defaultdict(list)
    bigramBeatTransitionProbabilities = defaultdict(list)

    # Q7: Your code goes here
    # ...

    return bigramBeatTransitions, bigramBeatTransitionProbabilities


# 8. Implement a function to compute p(beat length | beat position), and compute the perplexity of your models from Q7 and Q8. For both models, we only consider the probabilities of predicting the sequence of **beat lengths**.
# 
# `beat_pos_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatPosTransitions`: key - beat_position, value - a list of beat_length
# 
#   - `bigramBeatPosTransitionProbabilities`: key - beat_position, value - a list of probabilities for beat_length in the same order of `bigramBeatPosTransitions`
# 
# `beat_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: two perplexity values correspond to the models in Q7 and Q8, respectively

# In[ ]:


def beat_pos_bigram_probability(midi_files):
    bigramBeatPosTransitions = defaultdict(list)
    bigramBeatPosTransitionProbabilities = defaultdict(list)

    # Q8a: Your code goes here
    # ...

    return bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities


# In[ ]:


def beat_bigram_perplexity(midi_file):
    bigramBeatTransitions, bigramBeatTransitionProbabilities = beat_bigram_probability(midi_files)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    # Q8b: Your code goes here
    # Hint: one more probability function needs to be computed

    # perplexity for Q7
    perplexity_Q7 = None

    # perplexity for Q8
    perplexity_Q8 = None

    return perplexity_Q7, perplexity_Q8


# 9. Implement a Markov chain that computes p(beat_length | previous_beat_length, beat_position), and report its perplexity.
# 
# `beat_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramBeatTransitions`: key: (previous_beat_length, beat_position), value: a list of beat_length
# 
#   - `trigramBeatTransitionProbabilities`: key: (previous_beat_length, beat_position), value: a list of probabilities for beat_length in the same order of `trigramBeatTransitions`
# 
# `beat_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# In[ ]:


def beat_trigram_probability(midi_files):
    trigramBeatTransitions = defaultdict(list)
    trigramBeatTransitionProbabilities = defaultdict(list)

    # Q9a: Your code goes here
    # ...

    return trigramBeatTransitions, trigramBeatTransitionProbabilities


# In[ ]:


def beat_trigram_perplexity(midi_file):
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)
    # Q9b: Your code goes here


# 10. Use the model from Q5 to generate N notes, and the model from Q8 to generate beat lengths for each note. Save the generated music as a midi file (see code from workbook1) as q10.mid. Remember to reset the beat position to 0 when reaching the end of a bar.
# 
# `music_generate`
# - **Input**: target length, e.g. 500
# 
# - **Output**: a midi file q10.mid
# 
# Note: the duration of one beat in MIDIUtil is 1, while in MidiTok is 8. Divide beat length by 8 if you use methods in MIDIUtil to save midi files.

# In[ ]:


def music_generate(length):
    # sample notes
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)

    # Q10: Your code goes here ...
    sampled_notes = []

    # sample beats
    sampled_beats = []

    # save the generated music as a midi file

