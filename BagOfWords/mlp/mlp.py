import util #preprocessing
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from pandas import DataFrame
from matplotlib import pyplot

def evaluateMode(x_train, y_train, x_test, y_test):
    scores = list()
    repetition = 1 #Default 30
    nWords = x_test.shape[1]
    for i in range(repetition):
        #Define MLP network
        model = Sequential()
        model.add(Dense(50, input_shape=(nWords,), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        #Compile and Fit network to training data
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=30, verbose=2)

        #Evaluate network on testing data
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        scores.append(acc)
        print('%d accuracy: %s' % ((i+1), acc*100))
    return scores

#Encoding data
def dataPrep(train_rev, test_rev, mode):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_rev)
    x_train = tokenizer.texts_to_matrix(train_rev, mode=mode)
    x_test = tokenizer.texts_to_matrix(test_rev, mode=mode)
    return x_train, x_test

# Main
vocabFile = 'vocab.txt'
vocab = util.loadFile(vocabFile)
vocab = vocab.split()
vocab = set(vocab)

#Training set
pos_reviews = util.loadReviews('data/pos', vocab, True)
neg_reviews = util.loadReviews('data/neg', vocab, True)
train_rev = pos_reviews+neg_reviews

#Testing Set
pos_reviews = util.loadReviews('data/pos', vocab, False)
neg_reviews = util.loadReviews('data/neg', vocab, False)
test_rev = pos_reviews+neg_reviews

#Labels
y_train = array([0 for _ in range(900)] + [1 for _ in range(900)])
y_test = array([0 for _ in range(100)] + [1 for _ in range(100)])

#Obtain results for different word scoring modes
modes = ['binary', 'count', 'freq', 'tfidf']  #Binary: 1(word exists) or 0(does not exist)
                                              #Count: integer that identifies occurrence for each Word
                                              #TF-IDF: words scored on frequency, but penalized when common amongst all documents (example: stopwords)
                                              #Freq: words scored on frequency within a review
results = DataFrame()
for mode in modes:
    x_train, x_test = dataPrep(train_rev, test_rev, mode)
    print('Word Scoring Method: %s' % (mode))
    results[mode] = evaluateMode(x_train, y_train, x_test, y_test)

#Print results in table and plot format
print(results.describe())
results.boxplot()
pyplot.show()

# Sample reviews - EDIT THIS TO TEST MLP NETWORK (0 if positive, 1 if negative)
#text1 = "The first film, a frenetic and hilarious tribute to superhero films and imbued with 1950s style, was an absolute masterpiece, with great characters and hugely inventive action set pieces. This one is a worthy successor. That’s all you need to know"
#print(util.predictReview(text1, vocab, tokenizer, model))

#text2 = "I'm not usually a fan of suspenseful movies, but this one surprised me in a lot of ways. Of course, it's not without its shortcomings: the foil is pretty easy to spot early on, and the monster is fairly unimaginative, but there's a beauty in its simplicity, particularly in the family's story, that draws you in.John Krasinski does it all, providing a great performance and excellent direction of himself and his peers. Even though it's never one to subvert your expectations, the thrill of seeing it all unfold is very enjoyable. Though I didn't particularly enjoy the ending, I thought the rest of the film was strong enough to make up for it.What it does well: The quiet, honestly, is done perfectly. It's not abused with jump scares, and really feels as if it's its own character.What it could improve on:Itwill, almost certainly, leave you with some questions.Why didn't they just do this?was a common discussion topic among my group."
#print(util.predictReview(text2, vocab, tokenizer, model))
