# text_classify
Use RNN and LSTM to implement text classification on the movie review dataset.

# Dataset
Stanford University Large Movie Review Dataset（large movie review dataset） https://ai.stanford.edu/~amaas/data/sentiment/
It contains 25,000 training samples and 25,000 test samples. After downloading and decompressing, you will get the aclImdb folder. There are train and test under aclImdb, and txt files under neg and pos respectively. The txt files contain movie review texts.
(Put the aclImdb folder in the project folder)

# embedding Pre-training loading
After the model is built, the embedding layer of the word vector is randomly initialized. It is time-consuming and laborious to train a word vector representation with a certain logical relationship from scratch. Usually, a word vector matrix trained on a large scale can be used.
Here you can refer to Stanford University's GloVe（Global Vectors for Word Representation） pre-trained word vector. (https://nlp.stanford.edu/projects/glove/)
GloVe is an unsupervised learning algorithm used to obtain the vector representation of words. GloVe pre-trained word vectors can effectively capture the semantic relationship between words and are widely used in various tasks in the field of natural language processing, such as text classification, named entity recognition, and machine translation.

There are four types of Gloves, which are differentiated according to the amount of data and the same data according to the length of the vector.
Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip
Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): glove.42B.300d.zip
Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): glove.840B.300d.zip
Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download): glove.twitter.27B.zip
