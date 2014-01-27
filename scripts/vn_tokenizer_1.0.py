#!/usr/bin/env python
# encoding: utf-8

# Written by Anindya Roy,
# CNRS-LIMSI
# roy@limsi.fr
#
# Script to implement Vietnamese tokenization.
#
# Version 1.0. 26.1.14.
# -> User must choose between CRF and MM+ algorithms.
# -> To use CRF, CRFSuite and liblbfgs must be installed first.
# Source for CRFSuite and liblbfgs:
# http://www.chokkan.org/software/crfsuite/
#
#===============================================================================

import sys
if 1:
	n_args = len(sys.argv)
	if n_args < 3:
		print '\n'
		print "Vietnamese tokenizer. Version 1.0."
		print "=================================="
		print "Usage: ./vn_tokenizer_1.0.py <input file> <output file> <algorithm> <command to run crfsuite binary> <path to libcrfsuite-0.12.so>" + '\n'
		print "\t* <input file> and <output file> are mandatory inputs. Rest are optional." + '\n'
		print "\t* Choices for <algorithm> are 'mm' (Maximum Matching) and 'crf' (CRF)."
		print "\t* Default <algorithm> : CRF."
		print "\t* Default command to run crfsuite binary : 'crfsuite'"
		print "\t* Default path to libcrfsuite-0.12.so : '/usr/local/lib/'" + '\n'
		print "\t* The input file must be in UTF-8 encoding."
		print "\t* The tokenized text will be written in the output file." 
		print "\t* Each token will be surrounded by square brackets []." + '\n'
		print "\t* Newlines are assumed to end sentences, so no tokenized word will continue across a newline."
		print '\n'
		exit(1)

	if n_args == 3:
		input_file_name = sys.argv[1]
		output_file_name = sys.argv[2]
		algo = 'crf'
		model_file_name = './model.crf'
		model_file_name_pkl = './model.crf.pkl'
		CRFSUITE = 'crfsuite'
		LIBCRFPATH = "/usr/local/lib/"

	if n_args == 4:
		input_file_name = sys.argv[1]
		output_file_name = sys.argv[2]
		algo = sys.argv[3]
		if algo == 'mm':
			model_file_name = './model.pkl'
		elif algo == 'crf':
			model_file_name = './model.crf'
			model_file_name_pkl = './model.crf.pkl'
		else:
			print "Algorithm '" + algo + "' is invalid. Choose between 'mm' and 'crf' only."
			exit(1)
		CRFSUITE = 'crfsuite'
		LIBCRFPATH = "/usr/local/lib/"

	if n_args == 5:
		input_file_name = sys.argv[1]
		output_file_name = sys.argv[2]
		algo = sys.argv[3]
		if algo == 'mm':
			model_file_name = './model.pkl'
		elif algo == 'crf':
			model_file_name = './model.crf'
			model_file_name_pkl = './model.crf.pkl'
		else:
			print "Algorithm '" + algo + "' is invalid. Choose between 'mm' and 'crf' only."
			exit(1)
		CRFSUITE = sys.argv[4]
		LIBCRFPATH = "/usr/local/lib/"

	if n_args >= 6:
		input_file_name = sys.argv[1]
		output_file_name = sys.argv[2]
		algo = sys.argv[3]
		if algo == 'mm':
			model_file_name = './model.pkl'
		elif algo == 'crf':
			model_file_name = './model.crf'
			model_file_name_pkl = './model.crf.pkl'
		else:
			print "Algorithm '" + algo + "' is invalid. Choose between 'mm' and 'crf' only."
			exit(1)
		CRFSUITE = sys.argv[4]
		LIBCRFPATH = sys.argv[5]

if 0:
	input_file_name = './input1.tkn'
	output_file_name = './input1.tkn.wseg2'
	algo = 'crf'
	model_file_name = './model.crf'
	model_file_name_ = './model.crf.pkl'
	CRFSUITE = 'crfsuite'
	LIBCRFPATH = "/usr/local/lib/"

import os
if not os.path.isfile(input_file_name):
	print 'Input text file "' + input_file_name + '" does not exist. Retry with a valid file name.'
	exit(1)

if not os.path.isfile(model_file_name):
	print 'Model file "' + model_file_name + '" does not exist. Retry with a valid file name.'
	exit(1)


import sys, re, math, unicodedata, codecs, pickle
# NLTK not required.

# STEP: Read input file.
# The file is stored as a list of items, each item is one line. 
# Each item (line) is itself a list of the contents of the line 
# including words, punctuation marks and special characters.

f = codecs.open(input_file_name, mode = 'r', encoding = 'utf-8', errors = 'ignore')
sents = []
for line in f:
	sents.append(line.split()) # Split line on space to get syllables + etc.
f.close()

# Miscellaneous special characters and punctuation marks.
punct = [u'!', u',', u'.', u':', u';', u'?']  # TO DO : Add "...".
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


# STEP: Tokenization by MM+ algorithm.
if algo == 'mm':

	f = open(model_file_name, 'rb')
	words_ = pickle.load(f) # Words with smoothed log probs.
	f.close()

	# Break word formation when encounter these characters (detached from any word).
	not_words_ = [u'!', u'"',  u'&', u"'", u'(', u')', u'*', u'+', u',', u'-', u'.',
                      u'/', u':', u';', u'=', u'>', u'?'] # u'%'	
	
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

# STEP: Tokenization by CRF3 algorithm.
if algo == 'crf':
	
	import subprocess # For calling CRFSuite.

	# Write text in column format.
	f = codecs.open('./tmp.1', mode = 'w', encoding = 'utf-8')
	for line in sents_:
		for syl in line:
			f.write(syl + '\n')
		f.write('\n') # End of sentence.
	f.close()

	# Read in CRF feature attribute parameters.
	f = open(model_file_name_pkl, 'rb')
	sc = pickle.load(f) # Words with smoothed log probs.
	pname = pickle.load(f)
	puncts = pickle.load(f)
	words_ = pickle.load(f)
	first_names = pickle.load(f)
	middle_names = pickle.load(f)
	last_names = pickle.load(f)
	location_names = pickle.load(f)
	f.close()

	# Extract features and write to CRFSuite format file for tagging.
	fout = codecs.open('./tmp.1a', mode = 'w', encoding = 'utf-8')
	nSents = len(sents_)
	for sentNo in xrange(nSents):
		
		syls = sents_[sentNo]
		nSyls = len(syls)
		
		syls_ = ['BOS2', 'BOS1']
		syls_.extend(syls)
		syls_.extend(['EOS1', 'EOS2'])
		
		for sylNo in xrange(nSyls):
			sylNo_ = sylNo + 2 # Offset in syls_.
			text = 'c' # Arbitrary text to write to output file in CRFSuite format.
			attrib_no = 0 # Attribute index.
			if 1:
				for attrib in sc:
					ngram = syls_[sylNo_ + attrib[0]]
					for offset in attrib[1:]:
						ngram = ngram + '_' + syls_[sylNo_ + offset]
					
					text = text + '\t' + 'f' + str(attrib_no) + '=' + ngram
					attrib_no += 1
			
			if 1:
				for attrib in sc:
					ngram = syls_[sylNo_ + attrib[0]]
					for offset in attrib[1:]:
						ngram = ngram + '_' + syls_[sylNo_ + offset]
					if ngram in words_:
						text = text + '\t' + 'f' + str(attrib_no) + '=1'
					else:
						text = text + '\t' + 'f' + str(attrib_no) + '=0'
					attrib_no += 1

			if 1:
				# First name.
				for offset in pname:
					if syls_[sylNo_ + offset] in first_names:
						text = text + '\t' + 'f' + str(attrib_no) + '=1'
					else:
						text = text + '\t' + 'f' + str(attrib_no) + '=0'
					attrib_no += 1

				# Middle name.
				for offset in pname:
					if syls_[sylNo_ + offset] in middle_names:
						text = text + '\t' + 'f' + str(attrib_no) + '=1'
					else:
						text = text + '\t' + 'f' + str(attrib_no) + '=0'
					attrib_no += 1

				# Last name.
				for offset in pname:
					if syls_[sylNo_ + offset] in last_names:
						text = text + '\t' + 'f' + str(attrib_no) + '=1'
					else:
						text = text + '\t' + 'f' + str(attrib_no) + '=0'
					attrib_no += 1

				# Location name.
				for attrib in sc:
					ngram = syls_[sylNo_ + attrib[0]]
					for offset in attrib[1:]:
						ngram = ngram + '_' + syls_[sylNo_ + offset]
					if ngram in location_names:
						text = text + '\t' + 'f' + str(attrib_no) + '=1'
					else:
						text = text + '\t' + 'f' + str(attrib_no) + '=0'
					attrib_no += 1

			if 1:
				
				# Regular expressions:
				# Numbers, percentages and money.
				if re.search('[+-]?\d+([,.]\d+)*%*', syls_[sylNo_]):
					text = text + '\t' + 'f' + str(attrib_no) + '=1'
				else:
					text = text + '\t' + 'f' + str(attrib_no) + '=0'
				attrib_no += 1

				# Short and long dates.
				if re.search('\d+[/-:]\d+', syls_[sylNo_]) or re.search('\d+[/-:]\d+[/-:]\d+', syls_[sylNo_]):
					text = text + '\t' + 'f' + str(attrib_no) + '=1'
				else:
					text = text + '\t' + 'f' + str(attrib_no) + '=0'
				attrib_no += 1	
			
				# Initial capital.
				if syls_[sylNo_][0].isupper():
					text = text + '\t' + 'f' + str(attrib_no) + '=1'
				else:
					text = text + '\t' + 'f' + str(attrib_no) + '=0'
				attrib_no += 1

				# All capitals.
				if syls_[sylNo_].isupper():
					text = text + '\t' + 'f' + str(attrib_no) + '=1'
				else:
					text = text + '\t' + 'f' + str(attrib_no) + '=0'
				attrib_no += 1

				# Punctuation and special characters.
				if syls_[sylNo_] in puncts:
					text = text + '\t' + 'f' + str(attrib_no) + '=1'
				else:
					text = text + '\t' + 'f' + str(attrib_no) + '=0'
				attrib_no += 1

			fout.write(text + '\n')

		fout.write('\n')

    	fout.close()	

	
	# Tokenize using CRFSuite.
	cmd = "export LD_LIBRARY_PATH=" + LIBCRFPATH
	cmd = cmd + "; " + CRFSUITE + " tag -m " + model_file_name + " tmp.1a > tmp.2"
	cmd = cmd + "; paste tmp.1 tmp.2 > tmp.3"
	cmd = cmd + "; rm tmp.1 tmp.2"
	status = subprocess.call(cmd, shell = True)	

	# Read tokenized text.
	fin = codecs.open('./tmp.3', mode = 'r', encoding = 'utf-8', errors = 'ignore')
	fout = codecs.open(output_file_name, mode = 'w', encoding = 'utf-8')
	sent = []
	word = ''
	write = 0 # To write sentence.
	for line in fin:
		line = line.split()
		if len(line) == 0: # End of sentence. 
			if word != '':
				sent.append('[' + word + ']') # Write current word.
				word = '' # Flush word buffer.
			if write == 1:
				fout.write(' '.join(sent) + '\n')
				sent = [] # Flush sentence buffer.
				write = 0
			continue
		write = 1
		syl, tag = line
		if tag == 'O':
			if word != '':
				sent.append('[' + word + ']') # Write current word to sentence.
				word = '' # Flush.
			sent.append(syl) # Add the current syl (not as token).
			continue
		if tag == 'B_W': # Begin word.
			if word != '':
				sent.append('[' + word + ']') # Write current word.
			word = syl # Begin new word.
			continue
		if tag == 'I_W': # Inside word.
			word = word + ' ' + syl

		# Last sentence.
	if word != '':
		sent.append('[' + word + ']')
	if write == 1:
		fout.write(' '.join(sent) + '\n')
		sent = [] # Flush sentence buffer.
		write = 0
	fin.close()
	fout.close()
	status = subprocess.call('rm ./tmp.3 ./tmp.1a', shell = True)	

	
	












