#------------------------------- TOKENIZATION--------------------------------------
from nltk.tokenize import sent_tokenize, word_tokenize
 
text = "Twitter was created in March 2006 by Jack Dorsey,"+ \
       "Noah Glass, Biz Stone, and Evan Williams and launched " +\
       "in July of that year. The service rapidly gained worldwide "+\
       "popularity. In 2012, more than 100 million users posted 340 "+\
       "million tweets a day, and the service handled an average of "+\
       "1.6 billion search queries per day."
 

print "------------------------------- TOKENIZATION--------------------------------------"
print "\nSENTENCE TOKENIZATION: ",sent_tokenize(text)
print "\nWORD TOKENIZATION: ",word_tokenize(text)

#-------------------------STOP WORD REMOVAL--------------------------------------
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 
stop_words = set(stopwords.words('english'))
stop_words.add('.')
stop_words.add(',')
 
word_tokens = word_tokenize(text)
 
filtered_sentence = [w for w in word_tokens if not w in stop_words]
 
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
 
print "\n-------------------------STOP WORD REMOVAL--------------------------------------"
print filtered_sentence

#------------------------------GET SYNONYMS/ANTONYMS----------------------------------
from nltk.corpus import wordnet
 
syns = wordnet.synsets("program")
 

print "\n------------------------------GET SYNONYMS/ANTONYMS----------------------------------"

print syns[0].name() 
print syns[0].lemmas()
print syns[0].lemmas()[0].name()
print syns[0].definition()
print syns[0].examples()


synonyms = []
antonyms = []
 
for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
 
print "\nSynonyms: ",set(synonyms)
print "\nAntonyms: ",set(antonyms)

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01') # n denotes noun
print "\nSimilarity between Ship and Boat",w1.wup_similarity(w2)

