"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

from collections import Counter

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
    test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
    E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    wordTagCount = {}
    tagCounter = Counter()
    retList = []

    for sentence in train:
        for word, tag in sentence:
            if word not in wordTagCount:
                wordTagCount[word] = Counter()
            wordTagCount[word][tag] += 1
            tagCounter[tag] += 1
    
    
    mostCommonTag = tagCounter.most_common(1)[0][0]

    for sentence in test:
        sentenceList = []
        for word in sentence:
            if word in wordTagCount:
                baseTag = wordTagCount[word].most_common(1)[0][0]
            else:
                baseTag = mostCommonTag
            sentenceList.append((word, baseTag))
        retList.append(sentenceList)

    return retList