"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

from collections import Counter
import numpy as np
import math

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    LAPLACE = 0.001
    UNKNOWN_WORD = '__UNKNOWN__'

    tagCount = Counter()
    firstTagCount = Counter()
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
                firstTagCount[tag1] += 1
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

    TOTAL_HAPAX = 0
    hapax_tag_count = Counter()
    for word in wordCount:
        if wordCount[word] == 1:
            TOTAL_HAPAX += 1
            hapax_tag_count[lastTag[word]] += 1

    uniqueWords.add(UNKNOWN_WORD)
    NUM_UNIQUE_WORDS = len(uniqueWords)
    TAGS = [tag for tag in tagCount]
    initial_probs = {tag: math.log((firstTagCount[tag] + LAPLACE)/ (len(train) + 1 + LAPLACE*len(tagCount))) for tag in tagCount}
    transition_probs = {(tag1, tag2): math.log((transCount[(tag1, tag2)] + LAPLACE) / (tagCount[tag1] + LAPLACE*(1+len(tagCount)))) for tag1 in tagCount for tag2 in tagCount}
    hapaxProb = {}
    for tag in TAGS:
        hapaxProb[tag] = (hapax_tag_count[tag] + LAPLACE) / (TOTAL_HAPAX + LAPLACE*(len(TAGS) + 1))
    emissProb = {} 
    for tag in TAGS:
        emissProb[tag] = {}
        for word in uniqueWords:
            hapax_LAP = hapaxProb[tag] * LAPLACE
            emissProb[tag][word] = math.log( (emissCount[tag][word] + hapax_LAP) / (tagCount[tag] + hapax_LAP * NUM_UNIQUE_WORDS) )

    retList = []

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

        bestRow = np.argmax(Vit[:, -1])
        row = bestRow
        retSentence = []
        for col in reversed(range(1, Vit.shape[1])):
            retSentence.append((sentence[col], TAGS[row]))
            row = backpointer_row[row, col]
        retSentence.append((sentence[0], TAGS[row]))
        retSentence.reverse()
        retList.append(retSentence)

    return retList