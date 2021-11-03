# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""

from collections import Counter
import numpy as np
import math

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''


    LAPLACE = 0.001
    UNKNOWN_WORD = '__UNKNOWN__'
        #setup code for counters and dictionaries
    tagCount = Counter()
    initTagCount = Counter()
    transCount = Counter()
    emissCount = {}
    uniqueWords = set()
    wordCount = Counter()
    lastTag = {}
    numWords = 0
    for sentence in train:
        for i in range(len(sentence)-1):
            word1, tag1 = sentence[i]
            word2, tag2 = sentence[i+1]
            if i == 0:
                initTagCount[tag1] += 1
            transCount[(tag1, tag2)] += 1
        for word, tag in sentence:
            if tag not in emissCount:
                emissCount[tag] = Counter()
            emissCount[tag][word] += 1

            tagCount[tag] += 1

            uniqueWords.add(word)
            wordCount[word] += 1
            lastTag[word] = tag

            numWords += 1

        #calculate number of hapax words or words that only appear once
    TOTAL_HAPAX = 0
    hapaxTagCount = Counter()
    for word in wordCount:
        if wordCount[word] == 1:
            TOTAL_HAPAX += 1
            hapaxTagCount[lastTag[word]] += 1
    uniqueWords.add(UNKNOWN_WORD)
    NUM_UNIQUE_WORDS = len(uniqueWords)
    TAGS = [tag for tag in tagCount]
    initial_probs = {tag: math.log((initTagCount[tag] + LAPLACE)/ (len(train) + 1 + LAPLACE*len(tagCount))) for tag in tagCount}
    transition_probs = {(tag1, tag2): math.log((transCount[(tag1, tag2)] + LAPLACE) / (tagCount[tag1] + LAPLACE*(1+len(tagCount)))) for tag1 in tagCount for tag2 in tagCount}

    hapaxProb = {}
    for tag in TAGS:
        hapaxProb[tag] = (hapaxTagCount[tag] + LAPLACE) / (TOTAL_HAPAX + LAPLACE*(len(TAGS) + 1))

    emissProb = {} 
    for tag in TAGS:
        emissProb[tag] = {}
        for word in uniqueWords:
            hapax_LAP = hapaxProb[tag] * LAPLACE
            emissProb[tag][word] = math.log( (emissCount[tag][word] + hapax_LAP) / (tagCount[tag] + hapax_LAP * NUM_UNIQUE_WORDS) )

    output = []
    ITER = 0

    for sentence in test:
        Vit = np.zeros((len(TAGS), len(sentence)))
        backpointer_row = np.zeros(Vit.shape, dtype=int)
        for i in range(len(TAGS)):
            if sentence[0] in emissProb[TAGS[i]]:
                Vit[i][0] = initial_probs[TAGS[i]] + emissProb[TAGS[i]][sentence[0]]
            else:
                Vit[i][0] = initial_probs[TAGS[i]] + emissProb[TAGS[i]][UNKNOWN_WORD]
        for col in range(1, Vit.shape[1]):
            curr_word = sentence[col]
            for row in range(Vit.shape[0]): 
                poss_vals = np.zeros(Vit.shape[0])
                this_emiss = emissProb[TAGS[row]][curr_word] if curr_word in emissProb[TAGS[row]] else emissProb[TAGS[row]][UNKNOWN_WORD]
                for prev_row in range(Vit.shape[0]): 
                    poss_vals[prev_row] = Vit[prev_row][col-1] + transition_probs[TAGS[prev_row], TAGS[row]] + this_emiss

                best_row = np.argmax(poss_vals)
                Vit[row][col] = poss_vals[best_row]
                backpointer_row[row, col] = best_row
        overall_best_row = np.argmax(Vit[:, -1])
        row = overall_best_row
        this_output = []
        for col in reversed(range(1, Vit.shape[1])):
            this_output.append((sentence[col], TAGS[row]))
            row = backpointer_row[row, col]
        this_output.append((sentence[0], TAGS[row]))
        this_output.reverse()
        output.append(this_output)

    return output