#!/usr/bin/env python
# encoding: utf-8

# Written by Anindya Roy,
# CNRS-LIMSI
# roy@limsi.fr
#
# Script to implement Vietnamese tokenization.
# Version 1. 21.1.14.
#
# TO DO : Add general help, program description, format of output tokenized text.
#
#===============================================================================

import sys
if 1:
	n_args = len(sys.argv)
	if n_args < 3:
		print '\n'
		print "Vietnamese tokenizer. Version 0.1."
		print "=================================="
		print "Usage: ./vn_tokenizer.py <input file name> <output file name> <model file name>."
		print "<input file name> and <output file name> are mandatory inputs."
		print "If <model file name> is not provided, the default is './model.pkl'."
		print "The input file MUST be in UTF-8 encoding."
		print "The tokenized text will be written in the output file." 
		print "Each token will be surrounded by square brackets []." + '\n'
		print "Note: Newlines are assumed to end sentences, so no tokenized word"
		print "will continue across a newline." + '\n'
		exit()
	if n_args == 3:
		input_file_name = sys.argv[1]
		output_file_name = sys.argv[2]
		model_file_name = './model.pkl'
	if n_args == 4:
		input_file_name = sys.argv[1]
		output_file_name = sys.argv[2]
		model_file_name = sys.argv[3]

if 0:
	input_file_name = './input.txt'
	output_file_name = './output.txt'
	model_file_name = './model.pkl'

import os
if not os.path.isfile(input_file_name):
	print 'Input text file "' + input_file_name + '" does not exist. Retry with a valid file name.'
	exit(1)

if not os.path.isfile(model_file_name):
	print 'Model file "' + model_file_name + '" does not exist. Retry with a valid file name.'
	exit(1)


import sys, re, math, unicodedata, numpy as np, codecs, pickle
# import nltk : NLTK not required.


# STEP: Read input file.
# The file is stored as a list of items, each item is one line. 
# Each item (line) is itself a list of the contents of the line 
# including words, punctuation marks and special characters.

f = codecs.open(input_file_name, mode = 'r', encoding = 'utf-8', errors = 'ignore')
sents = []
for line in f:
	sents.append(line.split()) # Split line on space to get syllables + etc.
f.close()


#not_words_ = [u'!',  u'"', u'%', u'&', u"'", u'(', u')', u'*', u'+', u',', u'-', u'.',
#                      u'/', u':', u';', u'=', u'>', u'?']

punct = [u'!', u',', u'.', u':', u';', u'?']  # TO DO : Add "..." etc
quotes = [u'"', u"'"]
brackets = [u'(', u')', u'[', u']', u'{', u'}']
mathsyms = [u'%', u'*', u'+', u'-', u'/', u'=', u'>', u'<']


# STEP: Detach punctuation marks attached at the end of words.
# In general, a period (.) or a comma (,) at end of a word should 
# be detached from the word.
# Exceptions to check: initials & acronyms e.g. "D. Hằng" and dates.

if 1: 
	sents_ = []
	for sent in sents:
		sent_ = []
		for word in sent:
			# First, check if acronym or abbreviation, i.e. Z., Y.Z., X.Y.Z. etc.
			if re.search('(.\.)+\Z', word) and word.isupper(): 
				sent_.append(word) # Checked.
				continue 
			# Second, check if it is a date.
			# DD.MM.YY.
			if re.search('\A[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{2}\.\Z', word):
				sent_.append(word) # Checked.
				continue
			# DD.MM.YYYY.
			if re.search('\A[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4}\.\Z', word):
				sent_.append(word) # Checked.
				continue
			# If not, separate out punctuation mark at end of word.
			for char in punct:
				rm = re.search('\\' + char + '+\Z', word)
				if rm:
					word = re.sub('\\' + char + '+\Z', '', word) + ' ' + char
					break
			sent_.extend(word.split())	
			
		sents_.append(sent_)

# Intermediate output.
if 0:
	f = codecs.open(output_file_name, mode = 'w', encoding = 'utf-8')
	for sent in sents_:
		f.write(' '.join(sent) + '\n')
	f.close()

# Tokenization by MM+ algorithm.
if 1:
	f = open(model_file_name, 'rb')
	words_ = pickle.load(f) # Words with smoothed log probs.

	# Break word formation when encounter these characters (detached from any word).
	not_words_ = [u'!', u'"',  u'&', u"'", u'(', u')', u'*', u'+', u',', u'-', u'.',
                      u'/', u':', u';', u'=', u'>', u'?'] # u'%'
	f.close()	
	
	f = codecs.open(output_file_name, mode = 'w', encoding = 'utf-8')
	sents = [] # Tokenized sentences will be written here.

	for line in sents_:
		sent = []
		word = []

		for syl in line: # Consider each syllable in this line.

			# Check if syl is a punctuation mark or special character.
			if syl in not_words_: 
				if len(word) > 0:
					sent.append('[' + ' '.join(word) + ']') # Write current word to sentence surrounded by [].
					word = [] # Flush word.
				sent.append(syl) # Add the punct or special character (NOT as a token).
				continue
			word.append(syl)
			word1 = ' '.join(word) # Form new word by appending current syllable.

			# Check if the word exists in lexicon.
			if word1 in words_: 
				continue # Do not write anything, continue.

			# Check if the word forms the initials of a person name: "X. Y.".
			if 0: # Disabled.
				if re.search('\A(.\. )+.\.\Z', word1) and word1.isupper():
					continue 
			# Check if it is a person name in the form "X. Y. Z. Xyz Abc"
			if 0: # Disabled.
				rm = re.search('\A(.\. )+', word1)
				if rm:
					word1a = re.sub('\A' + rm.group(), '', word1) # Strip initials.
					word1a = word1a.split()
					isName = 1
					for w_ in word1a:
						if w_[0].islower(): # Initial letter.
							isName = 0
							continue
					if isName == 1:
						continue # Then do not write anything now, continue.
			# Check if it is a person name in the form "Xyz Abc Lmn"
			if 0: # Disabled.
				word1a = word1.split()
				isName = 1
				for w_ in word1a:
					if w_[0].islower():
						isName = 0
						continue
				if isName == 1:
					continue
			
			# Otherwise, check if all syllables in current word are unknown, then keep going.
			# Reason: exploit the observation that unknown foreign words are usually clumped together as 				# single words. This improves P by 0.6 %, does not alter R, and improves F-ratio by 0.3 %.
			if 1:
				all_unk = 1
				for syl_ in word:
					if syl_ in words_:
						all_unk = 0
						continue 
				if all_unk:
					continue # i.e. clump together unknown words.

			# Check if it is a single unknown syllable.
			if len(word) == 1: # Keep it -> as it may be a bounded morpheme.
				continue # This test is not required, it is covered by the above test.

			# Check if first syllable is known, second unknown.
			# (Also, the first and second together do not make a valid word.)
			if len(word) == 2:
				sent.append('[' + word[0] + ']') # Then add 1st syllable as a word to the sentence.
				word = [word[1]] # Begin new word with 2nd syllable.

			# Check 1-lookahead with overlap ambiguity resolution.
			# Compare log prob(a, b_c) vs. log prob(a_b, c) if a, b_c, a_b, c exists in lexicon.
			# and write (a, b_c) or (a_b, c) accordingly.
			if len(word) > 2:
				word2 = ' '.join(word[:-2]) # (a)
				word3 = ' '.join(word[-2:]) # (b_c)
				word4 = ' '.join(word[:-1]) # (a_b)
				word5 = word[-1] # (c)
				if word3 not in words_ or word2 not in words_:
					sent.append('[' + word4 + ']')
					word = [word[-1]]
				elif word5 in words_ and word4 in words_:
					P1 = words_[word2] + words_[word3] # P(a, b_c)
					P2 = words_[word4] + words_[word5] # P(a_b, c)
					if P1 > P2:
						sent.append('[' + word2 + ']')
						word = word[-2:]
					else:
						sent.append('[' + word4 + ']')
						word = [word[-1]]
				else:
					# syl was an unknown word.
					sent.append('[' + word4 + ']')
					word = [word[-1]]
		# Last sentence.
		if len(word) > 0:
			sent.append('[' + ' '.join(word) + ']')
		if len(sent) > 0:
			sents.append(sent)
		f.write(' '.join(sent) + '\n')

	f.close()













