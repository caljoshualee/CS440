"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

from collections import Counter
import math

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    emissLap = 1.0
    transLap = 1.0

    startTagCounter = Counter() #keeps track of total number of tags seen in the first word of a sentence
    tagCount = Counter() #keeps track of total number of tags
    transCount = {} #keeps track of the number of tag transitions 
    emissCount = {} #keeps track of number of words seen for each tag
    for sentence in train:
        startTagCounter[sentence[0][1]] += 1

        for word, tag in sentence:
            tagCount[tag] += 1
            if tag not in emissCount:
                emissCount[tag] = Counter()
            emissCount[tag][word] += 1

        tagPairs = [(sentence[i-1][1], sentence[i][1]) for i in range(1, len(sentence))]

        for tag1, tag2 in tagPairs:
            if tag1 not in transCount:
                transCount[tag1] = Counter()
            transCount[tag1][tag2] += 1

    #laplace smoothing for start prob and store in dictionary
    #startProb[tag] = probability of tag being the starting value
    startProb = {}
    numUniqueTags = len(tagCount)
    numStartTags = sum(startTagCounter.values())
    startProb['UNK'] = math.log(transLap/(numStartTags + transLap*(numUniqueTags + 1)))
    for tag in startTagCounter:
        startProb[tag] = math.log(startTagCounter[tag] + transLap/(numStartTags + transLap*(numUniqueTags + 1)))

    #laplace smoothing for transition probabliity and store in dictionary
    #transProb[tag1][tag2] = probability of tag1 transitioning to tag2
    transProb = {}
    for tag1 in transCount:
        transProb[tag1] = {}
        totalTagTrans = sum(transCount[tag1].values())
        for tag2 in transCount[tag1]:
            transProb[tag1][tag2] = math.log(transCount[tag1][tag2] + transLap/(totalTagTrans + transLap*(numUniqueTags + 1)))
        transProb[tag1]['UNK'] = math.log(transLap/(totalTagTrans + transLap*(numUniqueTags + 1)))
        
    #laplace smoothing for emission probability and store in dictionary
    #emissProb[tag][word] = probability of the word given the tag
    emissProb = {}
    for tag in emissCount:
        emissProb[tag] = {}
        numUniqueWords = len(emissCount[tag])
        numWordsWithTag = sum(emissCount[tag].values())
        for word in emissCount[tag]:
            emissProb[tag][word] = math.log(emissCount[tag][word] + emissLap/(numWordsWithTag + emissLap*(numUniqueWords + 1)))
        emissProb[tag]['UNK'] = math.log(emissLap/(numWordsWithTag + emissLap*(numUniqueWords + 1)))

    tagList = tagCount.keys()
 
    
    retList = []

    for sentence in test:
        retSentence = []
        vTrellis = {}
        bTrellis = {}

        vTrellis[0] = {}
        for tag in tagList:
            if tag != 'START':
                vTrellis[0][tag] = math.log(0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001)
            else:
                vTrellis[0][tag] = math.log(1)
            
        vTrellis[1] = {}
        initWord = sentence[1]
        for tag in tagList:
            if tag in startProb:
                if initWord in emissProb[tag]:
                    vTrellis[1][tag] = startProb[tag] + emissProb[tag][initWord]
                else:
                    vTrellis[1][tag] = startProb[tag] + emissProb[tag]['UNK']
            else:
                if initWord in emissProb[tag]:
                    vTrellis[1][tag] = startProb['UNK'] + emissProb[tag][initWord]
                else:
                    vTrellis[1][tag] = startProb['UNK'] + emissProb[tag]['UNK']

        for i in range(2, len(sentence)):
            vTrellis[i] = {}
            bTrellis[i] = {}
            currWord = sentence[i]
            for tagB in tagList:
                maxTag = ''
                maxProb = float('-inf')
                for tagA in tagList:
                    if tagA == 'END' or tagA == 'START':
                        continue
                    if tagB in transProb[tagA]:
                        if currWord in emissProb[tag]:
                            currV = vTrellis[i-1][tagA] + transProb[tagA][tagB] + emissProb[tag][currWord]
                        else:
                            currV = vTrellis[i-1][tagA] + transProb[tagA][tagB] + emissProb[tag]['UNK']
                    else:
                        if currWord in emissProb[tag]:
                            currV = vTrellis[i-1][tagA] + transProb[tagA]['UNK'] + emissProb[tag][currWord]
                        else:
                            currV = vTrellis[i-1][tagA] + transProb[tagA]['UNK'] + emissProb[tag]['UNK']
                    if currV > maxProb:
                        maxTag = tagA
                        maxProb = currV
                vTrellis[i][tagB] = maxProb
                bTrellis[i][tagB] = maxTag
        finalTag = ''
        finalMaxProb = float('-inf')
        for tag in tagList:
            if vTrellis[len(sentence)-1][tag] > finalMaxProb:
                finalMaxProb = vTrellis[len(sentence)-1][tag]
                finalTag  = tag
        retSentence.append((sentence[len(sentence)-1],finalTag))
        currTag = finalTag
        index = len(sentence)-2
        while(index >= 0):
            print(index)
            retSentence.append((sentence[index], bTrellis[index+1][currTag]))
            currTag = bTrellis[index+1][currTag]
            index -= 1
        retSentence.reverse()
        retList.append(retSentence)    

    return retList