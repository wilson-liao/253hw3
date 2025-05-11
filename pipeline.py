#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
from glob import glob
import numpy as np
import random


# In[ ]:


import homework3


# In[ ]:


random.seed(0)


# In[ ]:


midi_files = glob('PDMX_subset/*.mid')
# Note: the autograder will only use a subset of the files.
# You might also work with a small subset to make experimentation faster


# In[ ]:


def testQ1a():
    yours = homework3.note_extraction(midi_files[0])
    print(yours)


# In[ ]:


def testQ1b():
    yours = homework3.note_frequency(midi_files)
    print(yours)


# In[ ]:


def testQ2():
    yours = homework3.note_unigram_probability(midi_files)
    print(yours)


# In[ ]:


def testQ3a():
    your_transition, your_probability = homework3.note_bigram_probability(midi_files)
    print(your_transition[74]) # Example
    print(your_probability[74])


# In[ ]:


def testQ3b():
    test_notes = [92, 35, 54] # some notes that have only one possible next note
    yours = []
    correct = []
    for note in test_notes:
        yours.append(homework3.sample_next_note(note))

    print(yours)


# In[ ]:


def testQ4():
    test_file = midi_files[0]
    yours = [homework3.note_bigram_perplexity(test_file)]
    print(yours)


# In[ ]:


def testQ5a():
    test_notes = [71,72,73]
    your_transition, your_probability = homework3.note_trigram_probability(midi_files)
    print(your_transition)
    print(your_probability)


# In[ ]:


def testQ5b():
    test_file = midi_files[0]
    yours = [homework3.note_trigram_perplexity(test_file)]
    print(yours)


# In[ ]:


def testQ6():
    test_files = midi_files[:5]
    yours = []
    for file in test_files:
        beats = homework3.beat_extraction(file)
        yours += [beat[0] for beat in beats]
        yours += [beat[1] for beat in beats]

    print(yours)


# In[ ]:


def testQ7():
    test_beats = [2,4,8]
    your_transition, your_probability = homework3.beat_bigram_probability(midi_files)
    yours = []
    correct = []
    for note in test_beats:
        index = your_transition[4].index(note)
        yours.append(your_probability[4][index])

    print(yours)


# In[ ]:


def testQ8a():
    test_beats = [2,4,8]
    your_transition, your_probability = homework3.beat_pos_bigram_probability(midi_files)
    yours = []
    for note in test_beats:
        index = your_transition[0].index(note)
        yours.append(your_probability[0][index])

    print(yours)


# In[ ]:


def testQ8b():
    test_file = midi_files[0]
    yours = list(homework3.beat_bigram_perplexity(test_file))
    print(yours)


# In[ ]:


def testQ9a():
    test_beats = [2,4,8]
    your_transition, your_probability = homework3.beat_trigram_probability(midi_files)
    yours = []
    for note in test_beats:
        index = your_transition[(4, 0)].index(note)
        yours.append(your_probability[(4, 0)][index])

    print(yours)


# In[ ]:


def testQ9b():
    test_file = midi_files[0]
    yours = [homework3.beat_trigram_perplexity(test_file)]

    print(yours)


# In[ ]:


def testQ10():
    homework3.music_generate(n)
    if not os.path.exists('q10.mid'):
        print('No q10.mid file found')
        return 0

    # requirement1: generation of n notes
    notes = homework3.note_extraction('q10.mid')
    if len(notes) == n:
        point += 0.25
    else:
        print('It looks like your solution has the wrong sequence length')

    # Various other tests about the statistics of your midi file...


# In[ ]:




