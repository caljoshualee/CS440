# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=8.0, pos_prior=0.8,silently=False):
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)
    log_pos_prior = math.log(pos_prior)
    log_neg_prior = math.log(1 - pos_prior)

    #training set
    #count the number of words for every review for both positive and negative reviews
    pos_word_set = Counter()
    neg_word_set = Counter()
    for reviewIdx in range(len(train_set)):
        if train_labels[reviewIdx] == 1:
            for word in train_set[reviewIdx]:
                pos_word_set[word] += 1
        else:
            for word in train_set[reviewIdx]:
                neg_word_set[word] += 1
    
    #add laplace smoothing and add log probabilities to a dictionary
    pos_prob = {}
    pos_keys = pos_word_set.keys()
    pos_unique_keys = len(pos_keys)
    pos_num_words = sum(pos_word_set.values())

    neg_prob = {}
    neg_keys = neg_word_set.keys()
    neg_unique_keys = len(neg_keys)
    neg_num_words = sum(neg_word_set.values())

    for key in pos_keys:
        pos_prob[key] = math.log((pos_word_set[key]+laplace)/(pos_num_words+laplace*(pos_unique_keys+1)))
    pos_prob['UNK'] = math.log(laplace/(pos_num_words+laplace*(pos_unique_keys+1)))

    for key in neg_keys:
        neg_prob[key] = math.log((neg_word_set[key]+laplace)/(neg_num_words+laplace*(neg_unique_keys+1)))
    neg_prob['UNK'] = math.log(laplace/(neg_num_words+laplace*(pos_unique_keys+1)))
    
    #development set
    #caluculate MAP estimate and store the estimate with a larger log probability in the labels
    
    labels = []

    for doc in tqdm(dev_set,disable=silently):
        pos_est = log_pos_prior
        neg_est = log_neg_prior
        for word in doc:
            if word in pos_prob:
                pos_est += pos_prob[word]
            else:
                pos_est += pos_prob['UNK']
            if word in neg_prob:
                neg_est += neg_prob[word]
            else:
                neg_est += neg_prob['UNK']
        labels.append(1 if pos_est > neg_est else 0)

    return labels


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=9.0, bigram_laplace=0.2, bigram_lambda=0.3,pos_prior=0.8, silently=False):

    # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    log_pos_prior = math.log(pos_prior)
    log_neg_prior = math.log(1 - pos_prior)


    #training set
    #count the number of words for every review for both positive and negative reviews
    pos_word_set_bigram = Counter()
    neg_word_set_bigram = Counter()
    for reviewIdx in range(len(train_set)):
        review = train_set[reviewIdx]
        reviewTuples = [(review[i-1], review[i]) for i in range(1, len(review))]
        if train_labels[reviewIdx] == 1:
            for wordPair in reviewTuples:
                pos_word_set_bigram[wordPair] += 1
        else:
            for wordPair in reviewTuples:
                neg_word_set_bigram[wordPair] += 1
    
    #add laplace smoothing and add log probabilities to a dictionary
    pos_prob_bigram = {}
    pos_keys_bigram = pos_word_set_bigram.keys()
    pos_unique_keys_bigram = len(pos_keys_bigram)
    pos_num_words_bigram = sum(pos_word_set_bigram.values())

    neg_prob_bigram = {}
    neg_keys_bigram = neg_word_set_bigram.keys()
    neg_unique_keys_bigram = len(neg_keys_bigram)
    neg_num_words_bigram = sum(neg_word_set_bigram.values())

    for key in pos_keys_bigram:
        pos_prob_bigram[key] = math.log((pos_word_set_bigram[key]+bigram_laplace)/(pos_num_words_bigram+bigram_laplace*(pos_unique_keys_bigram+1)))
    pos_prob_bigram['UNK'] = math.log(bigram_laplace/(pos_num_words_bigram+bigram_laplace*(pos_unique_keys_bigram+1)))

    for key in neg_keys_bigram:
        neg_prob_bigram[key] = math.log((neg_word_set_bigram[key]+bigram_laplace)/(neg_num_words_bigram+bigram_laplace*(neg_unique_keys_bigram+1)))
    neg_prob_bigram['UNK'] = math.log(bigram_laplace/(neg_num_words_bigram+bigram_laplace*(pos_unique_keys_bigram+1)))

    #unigram algorithm
    #training set
    #count the number of words for every review for both positive and negative reviews
    pos_word_set = Counter()
    neg_word_set = Counter()
    for reviewIdx in range(len(train_set)):
        if train_labels[reviewIdx] == 1:
            for word in train_set[reviewIdx]:
                pos_word_set[word] += 1
        else:
            for word in train_set[reviewIdx]:
                neg_word_set[word] += 1
    
    #add laplace smoothing and add log probabilities to a dictionary
    pos_prob = {}
    pos_keys = pos_word_set.keys()
    pos_unique_keys = len(pos_keys)
    pos_num_words = sum(pos_word_set.values())

    neg_prob = {}
    neg_keys = neg_word_set.keys()
    neg_unique_keys = len(neg_keys)
    neg_num_words = sum(neg_word_set.values())

    for key in pos_keys:
        pos_prob[key] = math.log((pos_word_set[key]+unigram_laplace)/(pos_num_words+unigram_laplace*(pos_unique_keys+1)))
    pos_prob['UNK'] = math.log(unigram_laplace/(pos_num_words+unigram_laplace*(pos_unique_keys+1)))

    for key in neg_keys:
        neg_prob[key] = math.log((neg_word_set[key]+unigram_laplace)/(neg_num_words+unigram_laplace*(neg_unique_keys+1)))
    neg_prob['UNK'] = math.log(unigram_laplace/(neg_num_words+unigram_laplace*(pos_unique_keys+1)))


    #development set
    #caluculate MAP estimate and store the estimate with a larger log probability in the labels
    
    labels = []

    for doc in tqdm(dev_set,disable=silently):
        pos_est = log_pos_prior
        neg_est = log_neg_prior
        for word in doc:
            if word in pos_prob:
                pos_est += (1-bigram_lambda)*pos_prob[word]
            else:
                pos_est += (1-bigram_lambda)*pos_prob['UNK']
            if word in neg_prob:
                neg_est += (1-bigram_lambda)*neg_prob[word]
            else:
                neg_est += (1-bigram_lambda)*neg_prob['UNK']
        reviewTuples = [(doc[i-1], doc[i]) for i in range(1, len(doc))]
        for wordPair in reviewTuples:
            if wordPair in pos_prob_bigram:
                pos_est += bigram_lambda*pos_prob_bigram[wordPair]
            else:
                pos_est += bigram_lambda*neg_prob_bigram['UNK']
            if wordPair in neg_prob_bigram:
                neg_est += bigram_lambda*neg_prob_bigram[wordPair]
            else:
                neg_est += bigram_lambda*neg_prob_bigram['UNK']
            
        labels.append(1 if pos_est > neg_est else 0)

    return labels
