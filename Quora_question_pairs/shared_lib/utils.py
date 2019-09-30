import re
import time
import itertools
import numpy as np
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import timeit

# For pretty-printing
import pandas as pd
from IPython.display import display, HTML
import jinja2

def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
    return list(itertools.chain.from_iterable(list_of_lists))

HIGHLIGHT_BUTTON_TMPL = jinja2.Template("""
<script>
colors_on = true;
function color_cells() {
  var ffunc = function(i,e) {return e.innerText {{ filter_cond }}; }
  var cells = $('table.dataframe').children('tbody')
                                  .children('tr')
                                  .children('td')
                                  .filter(ffunc);
  if (colors_on) {
    cells.css('background', 'white');
  } else {
    cells.css('background', '{{ highlight_color }}');
  }
  colors_on = !colors_on;
}
$( document ).ready(color_cells);
</script>
<form action="javascript:color_cells()">
<input type="submit" value="Toggle highlighting (val {{ filter_cond }})"></form>
""")

RESIZE_CELLS_TMPL = jinja2.Template("""
<script>
var df = $('table.dataframe');
var cells = df.children('tbody').children('tr')
                                .children('td');
cells.css("width", "{{ w }}px").css("height", "{{ h }}px");
</script>
""")

def pretty_print_matrix(M, rows=None, cols=None, dtype=float,
                        min_size=30, highlight=""):
    """Pretty-print a matrix using Pandas.

    Optionally supports a highlight button, which is a very, very experimental
    piece of messy JavaScript. It seems to work for demonstration purposes.

    Args:
      M : 2D numpy array
      rows : list of row labels
      cols : list of column labels
      dtype : data type (float or int)
      min_size : minimum cell size, in pixels
      highlight (string): if non-empty, interpreted as a predicate on cell
      values, and will render a "Toggle highlighting" button.
    """
    html = [pd.DataFrame(M, index=rows, columns=cols,
                         dtype=dtype)._repr_html_()]
    if min_size > 0:
        html.append(RESIZE_CELLS_TMPL.render(w=min_size, h=min_size))

    if highlight:
        html.append(HIGHLIGHT_BUTTON_TMPL.render(filter_cond=highlight,
                                             highlight_color="yellow"))
    display(HTML("\n".join(html)))


def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)


##
# Word processing functions
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "<unk>" # unknown token

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

##
# Data loading functions
import nltk
import vocabulary

def get_corpus(name="brown"):
    return nltk.corpus.__getattr__(name)

def build_vocab(corpus, V=10000):
    token_feed = (canonicalize_word(w) for w in corpus.words())
    vocab = vocabulary.Vocabulary(token_feed, size=V)
    return vocab

def sents_to_tokens(sents, vocab):
    """Returns an flattened list of the words in the sentences, with normal padding."""
    padded_sentences = (["<s>"] + s + ["</s>"] for s in sents)
    # This will canonicalize words, and replace anything not in vocab with <unk>
    return np.array([canonicalize_word(w, wordset=vocab.wordset)
                     for w in flatten(padded_sentences)], dtype=object)

def get_train_test_sents(corpus, split=0.8, shuffle=True):
    """Get train and test sentences.

    Args:
      corpus: nltk.corpus that supports sents() function
      split (double): fraction to use as training set
      shuffle (int or bool): seed for shuffle of input data, or False to just
      take the training data as the first xx% contiguously.

    Returns:
      train_sentences, test_sentences ( list(list(string)) ): the train and test
      splits
    """
    sentences = np.array(corpus.sents(), dtype=object)
    fmt = (len(sentences), sum(map(len, sentences)))
    print "Loaded %d sentences (%g tokens)" % fmt

    if shuffle:
        rng = np.random.RandomState(shuffle)
        rng.shuffle(sentences)  # in-place
    train_frac = 0.8
    split_idx = int(train_frac * len(sentences))
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]

    fmt = (len(train_sentences), sum(map(len, train_sentences)))
    print "Training set: %d sentences (%d tokens)" % fmt
    fmt = (len(test_sentences), sum(map(len, test_sentences)))
    print "Test set: %d sentences (%d tokens)" % fmt

    return train_sentences, test_sentences

def preprocess_sentences(sentences, vocab):
    """Preprocess sentences by canonicalizing and mapping to ids.

    Args:
      sentences ( list(list(string)) ): input sentences
      vocab: Vocabulary object, already initialized

    Returns:
      ids ( array(int) ): flattened array of sentences, including boundary <s>
      tokens.
    """
    # Add sentence boundaries, canonicalize, and handle unknowns
    words = ["<s>"] + flatten(s + ["<s>"] for s in sentences)
    words = [canonicalize_word(w, wordset=vocab.word_to_id)
             for w in words]
    return np.array(vocab.words_to_ids(words))

##
# Use this function
def load_corpus(name, split=0.8, V=10000, shuffle=0):
    """Load a named corpus and split train/test along sentences."""
    corpus = get_corpus(name)
    vocab = build_vocab(corpus, V)
    train_sentences, test_sentences = get_train_test_sents(corpus, split, shuffle)
    train_ids = preprocess_sentences(train_sentences, vocab)
    test_ids = preprocess_sentences(test_sentences, vocab)
    return vocab, train_ids, test_ids

##
# Use this function
def batch_generator(ids, batch_size, max_time):
    """Convert ids to data-matrix form."""
    # Clip to multiple of max_time for convenience
    clip_len = ((len(ids)-1) / batch_size) * batch_size
    input_w = ids[:clip_len]     # current word
    target_y = ids[1:clip_len+1]  # next word
    # Reshape so we can select columns
    input_w = input_w.reshape([batch_size,-1])
    target_y = target_y.reshape([batch_size,-1])

    # Yield batches
    for i in xrange(0, input_w.shape[1], max_time):
	yield input_w[:,i:i+max_time], target_y[:,i:i+max_time]

## -----------------------------------------------------------
## New helper function added for Open Classification Project
## -----------------------------------------------------------
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

## Clean String
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`-]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

## Taking care of the stop words and stemming
## Keep punctuation
def preprocess_stop_stem(text, punc=False, stem=False, stop=False, sent=False):
    
    # Remove punctuations
    if punc:
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub('', text)
        
    tokens = word_tokenize(text) 
    if stop:
        stop = stopwords.words('english')
        tokens =[word for word in tokens if word not in stop]
        tokens = [word.lower() for word in tokens]
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    if sent:
        tokens = ' '.join(tokens)
    return tokens

## Preprocess newsgroup data
## Cleaning, Cut or padd with given length
## Input: list of data from newsgroup library
## Output: list of data with cleaned and cut/padded strings
#Preprocessing for Doc2Vec 
def preprocess_doc(doc):
    if type(doc) is str:
        doc = clean_str((doc))
        new_doc = preprocess_stop_stem(doc, punc=False, stop=True, sent=True, stem=False)
    elif math.isnan(doc):
        new_doc = "empty string"
    elif isinstance(doc, float):
        new_doc = "empty string"
    return new_doc

def get_train_test_docs(docs, labels, split = 0.8, shuffle=True):
    """"
    Args:
      docs: a list of sample docs
      split (double): fraction to use as training set
      shuffle (int or bool): seed for shuffle of input data, or False to just
      take the training data as the first xx% contiguously.

    Returns:
      train_docs, test_docs ( list(docs in string) ): the train and test
      splits
    """
    
    docs = np.array(docs, dtype=object)
    labels = np.array(labels)
    fmt = (len(docs), sum(map(len, docs)))
    print "Loaded %d docs (%g tokens)" % fmt
    
    if shuffle:
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        docs = docs[shuffle_indices]
        labels = labels[shuffle_indices]
    train_frac = 0.8
    split_idx = int(train_frac * len(labels))
    train_docs = docs[:split_idx]
    train_labels = labels[:split_idx]
    test_docs = docs[split_idx:]
    test_labels = labels[split_idx:]

    fmt = (len(train_docs), sum(map(len, train_docs)))
    print "Training set: %d docs (%d tokens)" % fmt
    fmt = (len(test_docs), sum(map(len, test_docs)))
    print "Test set: %d docs (%d tokens)" % fmt

    return train_docs, train_labels, test_docs, test_labels

def get_train_val_docs(docs, labels, split = 0.8, shuffle=True):
    """"
    Args:
      docs: a list of sample docs
      split (double): fraction to use as training set
      shuffle (int or bool): seed for shuffle of input data, or False to just
      take the training data as the first xx% contiguously.

    Returns:
      train_docs, test_docs ( list(docs in string) ): the train and test
      splits
    """
    
    docs = np.array(docs, dtype=object)
    labels = np.array(labels)
    fmt = (len(docs), sum(map(len, docs)))
    print "Loaded %d docs (%g tokens)" % fmt
    
    if shuffle:
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        docs = docs[shuffle_indices]
        labels = labels[shuffle_indices]
    train_frac = 0.8
    split_idx = int(train_frac * len(labels))
    train_docs = docs[:split_idx]
    train_labels = labels[:split_idx]
    valid_docs = docs[split_idx:]
    valid_labels = labels[split_idx:]

    fmt = (len(train_docs), sum(map(len, train_docs)))
    print "Training set: %d docs (%d tokens)" % fmt
    fmt = (len(valid_docs), sum(map(len, valid_docs)))
    print "Validation set: %d docs (%d tokens)" % fmt

    return train_docs, train_labels, valid_docs, valid_labels
    
## Tokenize document and convert ids
## Input: docs in np.array, vocab object
## Return: converted docs np.array
def docs_to_ids(docs, vocab):
    new_doc = []
    for doc in docs:
        doc_tokens = np.array([canonicalize_word(w, wordset=vocab.wordset) for w in doc.split(' ')], dtype = object)
        doc_id = vocab.words_to_ids(doc_tokens)
        new_doc.append(doc_id)
    return np.array(new_doc, dtype = object)