import string
from nltk.corpus import stopwords
from string import punctuation
from os import listdir
from collections import Counter

### Preprocessing Methods ###

#Open file in read-only and extract content into variable 'content'
def loadFile(filename):
    openFile = open(filename, 'r')
    content = openFile.read()
    openFile.close()
    return content

#Tokenize file
def tokenizeFile(filename):
    tokens = filename.split()                                   #remove whitespace
    tokens = [x.strip(string.punctuation) for x in tokens]           #remove punctuation
    tokens = [word for word in tokens if word.isalpha()]        #remove none alphabetic words
    stopWords = set(stopwords.words('english'))                 #remove stop words
    tokens = [word for word in tokens if not word in stopWords]
    tokens = [word for word in tokens if len(word) > 1]         #remove 1-letter tokens
    return tokens

#Convert tokens to single strings for easier encoding
def fileToLine(filename, vocab):
    content = loadFile(filename)
    tokens = tokenizeFile(content)
    tokens = [word for word in tokens if word in vocab]
    return ' '.join(tokens)

#Load all reviews and start mapping words to counter
def loadReviews(directory, vocab, is_train):
    lines = list()
    for filename in listdir(directory):
        if filename.startswith('cv9') and is_train:
            continue
        if not filename.startswith('cv9') and not is_train:
            continue
        path = directory + '/' + filename
        line = fileToLine(path, vocab)
        lines.append(line)
    return lines

#Predict reviews based on MLP network
def predictReview(review, vocab, tokenizer, model):

    #Split review into words and filter based on current vocab
    tokens = tokenizeFile(review)
    tokens = [word for word in tokens if word in vocab]
    lines = ' '.join(tokens)
    encode = tokenizer.texts_to_matrix([lines], mode='freq')

    #Predict review: 0 if positive, 1 if negative
    y = model.predict(encode, verbose=0)
    return round(y[0,0])
