import HTMLParser, string, re, itertools, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from autocorrect import spell
from textblob import TextBlob


text = "I luv my &lt;3 iphone &amp; you're awsm apple. DisplayIsAwsome, sooo happppppy :) http://www.apple.com"

print "----------BEFORE------------"
print text,"\n"

'''
Escaping HTML characters
'''
html_parser = HTMLParser.HTMLParser()
tweet = html_parser.unescape(text)

print "---------After HTML character removal-----------"
print tweet,"\n"

'''
Removing Urls
'''
tweet = re.sub(r"http\S+", "", tweet)

print "---------After removing urls-----------"
print tweet,"\n"

'''
Decoding data to utf8
'''
tweet = tweet.decode("utf8").encode('ascii','ignore')

print "---------After Decoding data to UTF8-----------"
print tweet,"\n"

'''
Split Attached Words
'''
ans = ""
for a in re.findall('[A-Z][^A-Z]*',tweet):
   ans+=a.strip()+' '

print "---------After splitting Attached words-----------"
print ans,"\n"

'''
Removal of Punctuations
'''
lowers = ans.lower()
no_punctuation = lowers.translate(None, string.punctuation)

print "---------After removing Punctuations-----------"
print no_punctuation,"\n"

'''
Standardizing words
'''
tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(no_punctuation))

print "---------After Standardizing-----------"
print tweet,"\n"

'''
Stop word removal and Tokenization
''' 
stop_words = set(stopwords.words('english'))
stop_words.add('.')
stop_words.add(',')
 
word_tokens = word_tokenize(tweet)
 
filtered_sentence = [w for w in word_tokens if not w in stop_words]
 
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print "---------After stopword removal and Tokenization-----------"
print filtered_sentence,"\n"

'''
Spell Checker
'''
result =[]
for w in filtered_sentence:
	result.append(spell(w))

print "---------Spell checking-----------"
print result,"\n"

'''
Part of speech tagging

CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: "there is" ... think of it like "there exists")
FW foreign word
IN preposition/subordinating conjunction
JJ adjective 'big'
JJR adjective, comparative 'bigger'
JJS adjective, superlative 'biggest'
LS list marker 1)
MD modal could, will
NN noun, singular 'desk'
NNS noun plural 'desks'
NNP proper noun, singular 'Harrison'
NNPS proper noun, plural 'Americans'
PDT predeterminer 'all the kids'
POS possessive ending parent's
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO to go 'to' the store.
UH interjection errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where
'''

tweet = nltk.pos_tag(filtered_sentence)
print "---------Part of speech tagging-----------"
print tweet,"\n"

'''
Sentiment Analysis
'''
tweet = "".join(filtered_sentence)

analysis = TextBlob(tweet)
if analysis.sentiment.polarity > 0:
	print 'positive'
elif analysis.sentiment.polarity == 0:
    print 'neutral'
else:
    print 'negative'

