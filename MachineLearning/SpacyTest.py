import spacy
import re
import codecs
import os
from gensim.models.word2vec import LineSentence
import itertools as it

nlp = spacy.load('en')

sampleDoc = "/Users/shah/Box Sync/MIMC_v2/Corpus/corpus/corpus/1.txt"


with open(sampleDoc, 'r') as f:
    body = f.read()

body = re.sub(r"\[\*\*.+\*\*\]", "", body)
body = re.sub(r"\n|\r", " ", body)
parsedDoc = nlp(body)

print(body)
print()
print("*******************************")
print()
for num, sentence in enumerate(parsedDoc.sents):
    print("Sentence {}: {}".format(num, sentence))

# print("*******************************")
#
# for num, token in enumerate(parsedDoc):
#     print("Token {}: {}".format(num, token.lemma_))

# def punc_space(token):
#     return token.is_punct or token.is_space
#
# def line_review(filename):
#     with codecs.open(filename, encoding="utf-8") as f:
#         for review in f:
#             yield review.replace('\\n', '\n')
#
# def lemmatized_sentence_corpus(filename):
#     for parsed_review in nlp.pipe(line_review(filename), batch_size=10000, n_threads=4):
#         for sent in parsed_review.sents:
#             yield u' '.join([token.lemma_ for token in sent if not punc_space(token)])
#
# unigram_sentences_filepath = os.path.join(intermediate_directory, 'unigram_sentences_all.txt')
#
# with codecs.open(unigram_sentences_filepath, 'w', encoding="utf-8") as f:
#     for sentence in lemmatized_sentence_corpus(review_txt_filepath):
#         f.write(sentence + '\n')
#
# unigram_sentences = LineSentence(unigram_sentences_filepath)
#
# for unigram_sentence in it.islice(unigram_sentences, 230, 240):
#     print(u' '.join(unigram_sentence))
# #     print(u'')