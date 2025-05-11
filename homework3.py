#!/usr/bin/env python
# coding: utf-8

# ## Homework 3: Symbolic Music Generation Using Markov Chains

# **Before starting the homework:**
# 
# Please run `pip install miditok` to install the [MiDiTok](https://github.com/Natooz/MidiTok) package, which simplifies MIDI file processing by making note and beat extraction more straightforward.
# 
# You're also welcome to experiment with other MIDI processing libraries such as [mido](https://github.com/mido/mido), [pretty_midi](https://github.com/craffel/pretty-midi) and [miditoolkit](https://github.com/YatingMusic/miditoolkit). However, with these libraries, you'll need to handle MIDI quantization yourself, for example, converting note-on/note-off events into beat positions and durations.

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


# In[ ]:


def note_frequency(midi_files):
    # Q1b: Your code goes here
    pitch_dict = defaultdict(int)
    for midi_file in midi_files:
        pitch_list = note_extraction(midi_file)
        for pitch in pitch_list:
            pitch_dict[int(pitch)] += 1
    return pitch_dict



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




# 6. Our model currently doesn’t have any knowledge of beats. Write a function that extracts beat lengths and outputs 
# a list of [(beat position; beat length)] values.
# 
#     Recall that each note will be encoded as `Position, Pitch, Velocity, Duration` using REMI. 
# Please keep the `Position` value for beat position, and convert `Duration` to beat length using provided lookup table 
# `duration2length` (see below).
# 
#     For example, for a note represented by four tokens `('Position_24', 'Pitch_72', 'Velocity_127', 'Duration_0.4.8')`, 
# the extracted (beat position; beat length) value is `(24, 4)`.
# 
#     As a result, we will obtain a list like [(0,8),(8,16),(24,4),(28,4),(0,4)...], 
# where the next beat position is the previous beat position + the beat length. 
# As we divide each bar into 32 positions by default, when reaching the end of a bar (i.e. 28 + 4 = 32 in the case of (28, 4)), 
# the beat position reset to 0.

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
    beat_list = []
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    for token in tokens:
        if "Position" in token:
            position = token.split("_")[1]
        if "Duration" in token:
            duration = token.split("_")[1]
            beat_list.append((int(position), duration2length[duration]))
    return beat_list


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
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for i in range(len(beats) - 1):
            prev_beat_length = beats[i][1]
            next_beat_length = beats[i+1][1]
            
            if next_beat_length not in bigramBeatTransitions[prev_beat_length]:
                bigramBeatTransitions[prev_beat_length].append(next_beat_length)
                bigramBeatTransitionProbabilities[prev_beat_length].append(1)
            else:
                next_beat_ind = bigramBeatTransitions[prev_beat_length].index(next_beat_length)
                bigramBeatTransitionProbabilities[prev_beat_length][next_beat_ind] += 1
    
    for prev_beat_length in bigramBeatTransitionProbabilities:
        total_count = sum(bigramBeatTransitionProbabilities[prev_beat_length])
        for i in range(len(bigramBeatTransitionProbabilities[prev_beat_length])):
            bigramBeatTransitionProbabilities[prev_beat_length][i] = bigramBeatTransitionProbabilities[prev_beat_length][i] / total_count


    return bigramBeatTransitions, bigramBeatTransitionProbabilities


# 8. Implement a function to compute p(beat length | beat position), 
# and compute the perplexity of your models from Q7 and Q8. 
# For both models, we only consider the probabilities of predicting the sequence of **beat lengths**.
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
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for i in range(len(beats)):
            beat_position = beats[i][0]
            beat_length = beats[i][1]
            if beat_length not in bigramBeatPosTransitions[beat_position]:
                bigramBeatPosTransitions[beat_position].append(beat_length)
                bigramBeatPosTransitionProbabilities[beat_position].append(1)
            else:
                next_beat_ind = bigramBeatPosTransitions[beat_position].index(beat_length)
                bigramBeatPosTransitionProbabilities[beat_position][next_beat_ind] += 1

    for beat_position in bigramBeatPosTransitionProbabilities:
        total_count = sum(bigramBeatPosTransitionProbabilities[beat_position])
        for i in range(len(bigramBeatPosTransitionProbabilities[beat_position])):
            bigramBeatPosTransitionProbabilities[beat_position][i] = bigramBeatPosTransitionProbabilities[beat_position][i] / total_count

        
    return bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities


# In[ ]:

def beat_length_unigram_probability(midi_files):
    unigramProbabilities = defaultdict(int)
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for beat in beats:
            unigramProbabilities[beat[1]] += 1
    total_count = sum(unigramProbabilities.values())
    for beat_length in unigramProbabilities:
        unigramProbabilities[beat_length] = unigramProbabilities[beat_length] / total_count
    return unigramProbabilities

def beat_pos_unigram_probability(midi_files):
    unigramProbabilities = defaultdict(int)
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for beat in beats:
            unigramProbabilities[beat[0]] += 1
    total_count = sum(unigramProbabilities.values())
    for beat_position in unigramProbabilities:
        unigramProbabilities[beat_position] = unigramProbabilities[beat_position] / total_count
    return unigramProbabilities

def beat_bigram_perplexity(midi_file):
    bigramBeatTransitions, bigramBeatTransitionProbabilities = beat_bigram_probability(midi_files)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    
    length_unigramProbabilities = beat_length_unigram_probability(midi_files)
    
    # Get the beats from the midi file
    beats = beat_extraction(midi_file)
    
    # Calculate perplexity for Q7 (beat_length | previous_beat_length)
    log_prob_sum_Q7 = 0
    # First beat uses unigram probability
    log_prob_sum_Q7 += np.log(length_unigramProbabilities[beats[0][1]])
    
    # Remaining beats use bigram probabilities
    for i in range(len(beats) - 1):
        prev_beat_length = beats[i][1]
        next_beat_length = beats[i+1][1]
        
        # Handle the case where the transition hasn't been seen in training
        if next_beat_length in bigramBeatTransitions[prev_beat_length]:
            index = bigramBeatTransitions[prev_beat_length].index(next_beat_length)
            prob = bigramBeatTransitionProbabilities[prev_beat_length][index]
        else:
            # Backoff to unigram probability if transition not seen
            prob = length_unigramProbabilities.get(next_beat_length, 1e-10)
            
        log_prob_sum_Q7 += np.log(prob)
    
    perplexity_Q7 = np.exp(-log_prob_sum_Q7 / len(beats))
    
    # Calculate perplexity for Q8 (beat_length | beat_position)
    log_prob_sum_Q8 = 0
    
    for i in range(len(beats)):
        beat_position = beats[i][0]
        beat_length = beats[i][1]
        
        # Handle the case where the position-length pair hasn't been seen
        if beat_length in bigramBeatPosTransitions[beat_position]:
            index = bigramBeatPosTransitions[beat_position].index(beat_length)
            prob = bigramBeatPosTransitionProbabilities[beat_position][index]
        else:
            # Backoff to unigram probability if pair not seen
            prob = length_unigramProbabilities.get(beat_length, 1e-10)
            
        log_prob_sum_Q8 += np.log(prob)
    
    perplexity_Q8 = np.exp(-log_prob_sum_Q8 / len(beats))
    
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
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for i in range(len(beats) - 1):
            prev_beat_length = beats[i][1]
            next_beat_length = beats[i+1][1]
            next_beat_position = beats[i+1][0]
            if next_beat_length not in trigramBeatTransitions[(prev_beat_length, next_beat_position)]:
                trigramBeatTransitions[(prev_beat_length, next_beat_position)].append(next_beat_length)
                trigramBeatTransitionProbabilities[(prev_beat_length, next_beat_position)].append(1)
            else:
                next_beat_ind = trigramBeatTransitions[(prev_beat_length, next_beat_position)].index(next_beat_length)
                trigramBeatTransitionProbabilities[(prev_beat_length, next_beat_position)][next_beat_ind] += 1

    for prev_beat_length in trigramBeatTransitionProbabilities: 
        total_count = sum(trigramBeatTransitionProbabilities[prev_beat_length])
        for i in range(len(trigramBeatTransitionProbabilities[prev_beat_length])):
            trigramBeatTransitionProbabilities[prev_beat_length][i] = trigramBeatTransitionProbabilities[prev_beat_length][i] / total_count


    return trigramBeatTransitions, trigramBeatTransitionProbabilities


# In[ ]:


def beat_trigram_perplexity(midi_file):
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)
    
    length_unigramProbabilities = beat_length_unigram_probability(midi_files)
    
    beats = beat_extraction(midi_file)
    
    # Start with the first beat using position-based probability
    log_prob_sum = 0
    
    # First beat uses position-based probability
    beat_position = beats[0][0]
    beat_length = beats[0][1]
    
    if beat_length in bigramBeatPosTransitions[beat_position]:
        index = bigramBeatPosTransitions[beat_position].index(beat_length)
        prob = bigramBeatPosTransitionProbabilities[beat_position][index]
    else:
        # Backoff to unigram probability if not seen
        prob = length_unigramProbabilities.get(beat_length, 1e-10)
    
    log_prob_sum += np.log(prob)
    
    # For remaining beats, use the trigram model
    for i in range(len(beats)-1):
        prev_beat_length = beats[i][1]
        next_beat_position = beats[i+1][0]
        next_beat_length = beats[i+1][1]
        
        key = (prev_beat_length, next_beat_position)
        
        if key in trigramBeatTransitions and next_beat_length in trigramBeatTransitions[key]:
            index = trigramBeatTransitions[key].index(next_beat_length)
            prob = trigramBeatTransitionProbabilities[key][index]
        else:
            # Backoff to position-based probability if trigram not seen
            if next_beat_length in bigramBeatPosTransitions[next_beat_position]:
                index = bigramBeatPosTransitions[next_beat_position].index(next_beat_length)
                prob = bigramBeatPosTransitionProbabilities[next_beat_position][index]
            else:
                # Further backoff to unigram if needed
                prob = length_unigramProbabilities.get(next_beat_length, 1e-10)
        
        log_prob_sum += np.log(prob)
    
    perplexity = np.exp(-log_prob_sum / len(beats))
    return perplexity


# 10. Use the model from Q5 to generate N notes, and the model from Q8 to generate beat lengths for each note. 
# Save the generated music as a midi file (see code from workbook1) as q10.mid. 
# Remember to reset the beat position to 0 when reaching the end of a bar.
# 
# `music_generate`
# - **Input**: target length, e.g. 500
# 
# - **Output**: a midi file q10.mid
# 
# Note: the duration of one beat in MIDIUtil is 1, while in MidiTok is 8. 
# Divide beat length by 8 if you use methods in MIDIUtil to save midi files.

# In[ ]:


def music_generate(length):
    # Get probability models for notes
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    
    # Get probability models for beats
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    
    # Sample first note from unigram distribution
    notes = []
    pitches = list(unigramProbabilities.keys())
    probs = list(unigramProbabilities.values())
    first_note = np.random.choice(pitches, p=probs)
    notes.append(first_note)
    
    # Sample second note from bigram distribution
    if first_note in bigramTransitions:
        second_note = np.random.choice(bigramTransitions[first_note], 
                                      p=bigramTransitionProbabilities[first_note])
        notes.append(second_note)
    else:
        # Fallback to unigram if first note has no transitions
        second_note = np.random.choice(pitches, p=probs)
        notes.append(second_note)
    
    # Generate remaining notes using trigram model
    for i in range(2, length):
        prev_note_pair = (notes[i-2], notes[i-1])
        if prev_note_pair in trigramTransitions and trigramTransitions[prev_note_pair]:
            next_note = np.random.choice(trigramTransitions[prev_note_pair], 
                                        p=trigramTransitionProbabilities[prev_note_pair])
        elif notes[i-1] in bigramTransitions and bigramTransitions[notes[i-1]]:
            # Backoff to bigram if trigram not available
            next_note = np.random.choice(bigramTransitions[notes[i-1]], 
                                        p=bigramTransitionProbabilities[notes[i-1]])
        else:
            # Backoff to unigram if bigram not available
            next_note = np.random.choice(pitches, p=probs)
        notes.append(next_note)
    
    # Generate beat positions and lengths
    beat_positions = []
    beat_lengths = []
    current_position = 0
    
    for i in range(length):
        beat_positions.append(current_position)
        
        # Sample beat length based on position
        if current_position in bigramBeatPosTransitions:
            beat_length = np.random.choice(bigramBeatPosTransitions[current_position], 
                                          p=bigramBeatPosTransitionProbabilities[current_position])
        else:
            # Default to quarter note (8) if position not in training data
            beat_length = 8
        
        beat_lengths.append(beat_length)
        
        # Update position for next note, reset to 0 if we reach end of bar
        current_position = (current_position + beat_length) % 32
    
    # Create MIDI file
    midi = MIDIFile(1)
    track = 0
    time = 0
    tempo = 120
    midi.addTempo(track, time, tempo)
    
    # Add notes to MIDI file
    time = 0
    for i in range(length):
        pitch = notes[i]
        duration = beat_lengths[i] / 8  # Convert from MidiTok to MIDIUtil duration
        midi.addNote(track, 0, pitch, time, duration, 100)
        time += duration
    
    # Write MIDI file
    with open('q10.mid', 'wb') as f:
        midi.writeFile(f)

music_generate(500)

from midi2audio import FluidSynth # Import library
from IPython.display import Audio, display
fs = FluidSynth("FluidR3Mono_GM.sf3") # Initialize FluidSynth
fs.midi_to_audio('q10.mid', 'q10.wav')
display(Audio('q10.wav'))
