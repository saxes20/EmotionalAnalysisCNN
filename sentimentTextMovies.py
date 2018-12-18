#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:10:47 2018

@author: sameer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:10:17 2018

@author: sameer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 11:58:23 2018

@author: sameer
"""

##### TEXTBLOB ######
from textblob import TextBlob

#1 - positive
#0 - neutral
#-1 - negative
statement = "Today is a great day!"
wiki = TextBlob(statement)
#wiki = wiki.correct()
reliability = wiki.sentiment.subjectivity
polarity = wiki.sentiment.polarity
print(polarity)
#tokenization - wiki.words / wiki.sentences

##### Naive Bayes (from sklearn) ######
import nltk.classify.util
#from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from random import randrange
import operator
import Contractions
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt


bestWords = []

def word_features(words):
    wordList = []
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 200)
    stopWords = set(stopwords.words('english'))
    for word in words:
        if word not in stopWords and word in bestWords:
            wordList.append((word,True))
    for bigram in bigrams:
        wordList.append((bigram,True))
    return dict(wordList)

word_fd = FreqDist()
category_fd = ConditionalFreqDist()

punctuation = ['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', "-", "--", "@", "...", "[", "]"]
numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

positives = []
with open("/Users/sameer/Desktop/PJAS/positiveReviews.txt", "r", encoding='latin-1') as f:
    for l in f:
        #expand contradictions
        for k in punctuation:
            l = l.replace(k," ")
        for n in numbers:
            l = l.replace(n, " ")
        l = Contractions.expandContractions(l)
        sentenceWords = nltk.word_tokenize(l)
        if "'s" in sentenceWords:
            sentenceWords.remove("'s")
        for word in sentenceWords:
            word_fd[word.lower()] += 1
            category_fd['pos'][word.lower()] += 1
        positives.append(l)
        
negatives = []
with open("/Users/sameer/Desktop/PJAS/negativeReviews.txt", "r", encoding='latin-1') as f:
    for l in f:
        #expand contradictions
        for k in punctuation:
            l = l.replace(k," ")
        for n in numbers:
            l = l.replace(n, " ")
        l = Contractions.expandContractions(l)
        sentenceWords = nltk.word_tokenize(l)
        if "'s" in sentenceWords:
            sentenceWords.remove("'s")
        for word in sentenceWords:
            word_fd[word.lower()] += 1
            category_fd['neg'][word.lower()] += 1
        negatives.append(l)

pos_wordCnt = category_fd['pos'].N()
neg_wordCnt = category_fd['neg'].N()
total_wordCnt = pos_wordCnt + neg_wordCnt
word_scores = {}
for word, freq in word_fd.items():
    pos_score = BigramAssocMeasures.chi_sq(category_fd['pos'][word],
        (freq, pos_wordCnt), total_wordCnt)
    neg_score = BigramAssocMeasures.chi_sq(category_fd['neg'][word],
        (freq, neg_wordCnt), total_wordCnt)
    word_scores[word] = pos_score + neg_score
    
best = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)[:10000]
bestWords = set([w for w, s in best])

posfeats = []
#positives = movie_reviews.fileids('pos')
for line in positives:
    lineWiki = TextBlob(line.lower())
    words = list(lineWiki.words)
    featset = word_features(words)
    tag = 'pos'
    posfeats.append((featset, tag))

negfeats = []
#negatives = movie_reviews.fileids('neg')
for line in negatives:
    lineWiki = TextBlob(line.lower())
    words = list(lineWiki.words)
    featset = word_features(words)
    tag = 'neg'
    negfeats.append((featset, tag))

def split (feats):
    train_set = list()
    test_set = list()
    set_copy = list(feats)
    train_size = len(feats) * 3/4
    while len(train_set) < train_size:
        index = randrange(len(set_copy))
        train_set.append(set_copy.pop(index))
    test_set = list([feat for feat in set_copy])
    return train_set, test_set

posTrain, posTest = split(posfeats)
negTrain, negTest = split(negfeats)

trainFeats = negTrain + posTrain
testFeats = negTest + posTest
print('train on %d instances, test on %d instances' % (len(trainFeats), len(testFeats)))

classifier = NaiveBayesClassifier.train(trainFeats)
classifier.show_most_informative_features()
accuracy = nltk.classify.util.accuracy(classifier, testFeats)
print('accuracy: %f' % accuracy)
prediction = classifier.classify(word_features(wiki.words))
print('Testing on ' + statement)
print('Prediction: ' + str(prediction))
prob_Prediction = classifier.prob_classify(word_features(wiki.words))
print('Pos Confidence: ' + str(prob_Prediction.prob('pos')))
print('Neg Confidence: ' + str(prob_Prediction.prob('neg')))

stopwords = set(STOPWORDS)
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    
#show_wordcloud(positives)
#show_wordcloud(negatives)
   
#Calculate precision and recall?

##### Random Forest ######
