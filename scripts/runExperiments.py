#!/usr/bin/env python
# encoding: utf-8

# Anindya Roy
# CNRS-LIMSI
# roy@limsi.fr
#
# Vietnamese tokenization experiments.
#
# + Data extraction.
# + Maximum Matching (MM) algorithm.
# + Enhanced MM algorithm.
# + 5-fold Cross-validation experiments
#
#==================================================================

import os, sys, re, math, unicodedata
import numpy as np
import codecs
import pickle

#==================================================================

# SETTINGS:
USE_LEXICON = 1
USE_LOCATIONS_FILE = 1
USE_PERSONNAMES_FILE = 1
USE_MM = 0 # Use Maximum Matching (MM) algorithm.
USE_ENHANCED_MM = 1 # Use Enhanced MM algorithm

DIR0 = '../' # Top level directory: Roy_VnTokenizer/.

# Values over cross-validation runs.
P_ = 0 # Precision.
R_ = 0 # Recall.
F_ = 0 # F-ratio.
O_ = 0 # No. of OOV words.
NSE_ = 0 # No. of sentences.
NW_ = 0 # No. of words.
NSY_ = 0 # No. of syllables.

nRuns = 5
for RUN in xrange(nRuns): # 5 cross-validation folds.

    if 1:	
	#print "STEP : Making list of train files."
	path_data = DIR0 + '/data/'
	tmp_lst = os.listdir(path_data)
	train_files_lst = []
	for item in tmp_lst:
		if re.search('\Atrain.*\.iob2', item): # Name format of train file: train*.iob2.
			train_files_lst.append(path_data + item)   # Could have used "glob" instead. Prefer not to reduce imports.

    if 1:	
	#print "STEP : Processing train files."
	words = [] # All words with counts.
	syls = [] # All syllables with counts.
	not_words = [] # Elements marked as 'O', i.e. not part of words, e.g. [,.;?0-9] etc.

	for item in train_files_lst[RUN:RUN+1]:
		f = codecs.open(item, encoding='utf-8', mode = 'r', errors = 'ignore') 
		# Vietnamese is written in utf-8. Assume file is utf-8 for now. TO DO: Check if utf-8!
		word = ''
		for line in f:
			line = line.split()
			if len(line) == 0: # End of sentence. 
				if word != '':
					words.append(word) # Write current word.
					word = '' # Flush.
				continue
			syl, tag = line
			if tag == 'O':
				not_words.append(syl)
				if word != '':
					words.append(word) # Write current word.
					word = '' # Flush.
				continue
			if tag == 'B_W': # Begin word.
				if word != '':
					words.append(word) # Write current word.
				word = syl # Begin new word.
				syls.append(syl)
				continue
			if tag == 'I_W': # Inside word.
				word = word + ' ' + syl
				syls.append(syl)
		f.close()
		# Last sentence.
		if word != '':
			words.append(word)
		words_train = words

    if 1:
	#print "STEP : Counting words in train files."
	# Unigram counts.
	words_ = {}
	syls_ = {}
	not_words_ = {}

	for word in words:
		if word in words_:
			words_[word] += 1
		else:
			words_[word] = 1

	for syl in syls:
		if syl in syls_:
			syls_[syl] += 1
		else:
			syls_[syl] = 1

	for not_word in not_words:
		if not_word in not_words_:
			not_words_[not_word] += 1
		else:
			not_words_[not_word] = 1
	del words
	del syls
	del not_words

    # Use additional material, Vietnamese dictionary, locations and person names files.
    if 1: 
	#print "STEP : Reading dictionary, locations and person names files."
	words_lex = []

    if USE_LEXICON:
	# Reading Vietnamese dictionary.
	lexfile = DIR0 + '/data/VNDic_UTF-8.txt'
	f = codecs.open(lexfile, encoding = 'utf-8', mode = 'r', errors = 'ignore')
	for line in f:
		if re.search('##', line):
			word = re.sub('##', '', line)
			word = ' '.join(word.split())
			words_lex.append(word)
		if re.search('@@', line):
			line = re.sub('@@', '', line)
			if line in ['Proverb', 'Idiom']:
				del words_lex[-1] # Remove last word added.
	f.close()

    # Reading locations file.
    if USE_LOCATIONS_FILE:
	lexfile = DIR0 + '/data/vnlocations.txt'
	f = codecs.open(lexfile, encoding = 'utf-8', mode = 'r', errors = 'ignore')
	for line in f:
		word = ' '.join(line.split())
		words_lex.append(word)
	f.close()

    # Reading person names file.
    if USE_PERSONNAMES_FILE:
	lexfile = DIR0 + '/data/vnpernames.txt'
	f = codecs.open(lexfile, encoding = 'utf-8', mode = 'r', errors = 'ignore')
	for line in f:
		word = ' '.join(line.split())
		words_lex.append(word)
	f.close()

    if 1:
	words_lex = list(set(words_lex))
	for word in words_lex:
		if word not in words_:
			words_[word] = 0.5
		

# Are there items in not_words_ which are also in words_ ? Check.
    if 0:
	amb_words = []
	for word in not_words_:
		if word in words_:
			amb_words.append(word)
				
# 106 amb_words exist. However, most occur few times as not_word (1 or 2 times mostly) compared to word.
# So, for this version of tokenizer, consider all not_words in words as words.
    if 0: 
	not_words__ = {}
	for word in not_words_:
		if word not in words_:
			not_words__[word] = not_words_[word]
	not_words_ = not_words__
	del not_words__

    if 1:
	not_words_ = [u'!',  u'"', u'%', u'&', u"'", u'(', u')', u'*', u'+', u',', u'-', u'.',
                      u'/', u':', u';', u'=', u'>', u'?'] # Just special chars.
# Observation: Using the above reduced not_words_ list improves P, R, F by about 1%.

# Convert counts to log for Algorithm MM+.
    if 1:
	#print "STEP : Smoothing and taking log of word counts."
	epsi = 0.01 # Smoothing.
	for word in words_:
		words_[word] = math.log(words_[word] + epsi)

# Run cross-validation experiment.
    if 1:
	#print "STEP : Make list of test files."
	# Make test file list.
	path_data = DIR0 + '/data/'
	tmp_lst = os.listdir(path_data)
	test_files_lst = []
	for item in tmp_lst:
		if re.search('\Atest.*\.iob2', item): # Name format of train file: test*.iob2.
			test_files_lst.append(path_data + item) 
	

    if 1:
	
	# Reference sentences. Each sentence will be a list of words.
	#============================================================
	#print "STEP : Make reference sentences."
	sents = [] 
	for item in test_files_lst[RUN:RUN+1]:
		f = codecs.open(item, encoding='utf-8', mode = 'r', errors = 'ignore') 
		sent = []
		word = ''
		for line in f:
			line = line.split()
			if len(line) == 0: # End of sentence. 
				if word != '':
					sent.append(word) # Write current word.
					word = '' # Flush word buffer.
				if len(sent) > 0:
					sents.append(sent)
					sent = [] # Flush sentence buffer.
				continue
			syl, tag = line
			if tag == 'O':
				if word != '':
					sent.append(word) # Write current word to sentence.
					word = '' # Flush.
				continue
			if tag == 'B_W': # Begin word.
				if word != '':
					sent.append(word) # Write current word.
				word = syl # Begin new word.
				continue
			if tag == 'I_W': # Inside word.
				word = word + ' ' + syl
		f.close()
		# Last sentence.
		if word != '':
			sent.append(word)
		if len(sent) > 0:
			sents.append(sent)
	sents_ref = sents

    if 1:
	# Calculate OOV words.
	nOOV = 0
	#oov = []
	for sent in sents_ref:
		for word in sent:
			if word not in words_:
				nOOV += 1
				#oov.append(word)

    if 1:
	nSents = len(sents)
	nW = 0
	nSyls = 0
	for sent in sents:
		nW += len(sent)
		nSyls += len(' '.join(sent).split())

    if 0:
	# Generate hypotheses sentences. 
	# Baseline algorithm: Maximum Matching (MM).
	#=================================================================
	#print "STEP : Tokenize input and make list of hypotheses sentences."
	sents = [] 
	for item in test_files_lst[RUN:RUN+1]:
		f = codecs.open(item, encoding='utf-8', mode = 'r', errors = 'ignore') 
		sent = []
		word = ''
		for line in f:
			line = line.split()
			if len(line) == 0: # End of sentence. 
				if word != '':
					sent.append(word) # Write current word.
					word = '' # Flush word buffer.
				if len(sent) > 0:
					sents.append(sent)
					sent = [] # Flush sentence buffer.
				continue
			syl, tag = line
			if syl in not_words_: # if tag == 'O':
				if word != '':
					sent.append(word) # Write current word to sentence.
					word = '' # Flush.
				continue
			word_ = word + ' ' + syl
			if word_ not in words_: # Compound word does not exist in lexicon.
				if word != '':
					sent.append(word) # Write current word.
				word = syl # New word starts.
			else:
				word = word_
		f.close()
		# Last sentence.
		if word != '':
			sent.append(word)
		if len(sent) > 0:
			sents.append(sent)
	sents_hyp = sents


    if 1:
	# Generate hypotheses sentences. 
	# Algorithm: MM+ i.e. MM + 1-lookahead + Unigram probs.
	#==================================================================
	# TO DO: check b_c_e, even if b_c does not exist.
	#print "STEP : Tokenize input and make list of hypotheses sentences."
	sents = [] 
	for item in test_files_lst[RUN:RUN+1]:
		f = codecs.open(item, encoding='utf-8', mode = 'r', errors = 'ignore') 
		sent = []
		word = []
		for line in f:
			line = line.split()
			if len(line) == 0: # End of sentence. 
				if len(word) > 0:
					sent.append(' '.join(word)) # Write current word.
					word = [] # Flush word buffer.
				if len(sent) > 0:
					sents.append(sent)
					sent = [] # Flush sentence buffer.
				continue
			syl, tag = line
			if syl in not_words_: # if tag == 'O':
				if len(word) > 0:
					sent.append(' '.join(word)) # Write current word to sentence.
					word = [] # Flush.
				continue
			word.append(syl)
			word1 = ' '.join(word)      # (a b c)
			# Check if the word exists in lexicon.
			if word1 in words_:
				continue # Do not write anything, continue.

			# Check if the word forms the initials of a person name: "X. Y.".
			if 0:
				if re.search('\A(.\. )+.\.\Z', word1) and word1.isupper():
					continue 
			# or if it is a person name in the form "X. Y. Z. Xyz Abc"
			if 0:
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
						continue # Do not write anything now, continue.
				
			# Otherwise,
			# check if all syllables in current word are unknown, then keep going.
			# Exploit the observation that unknown foreign words are clumped together as single words.
			# This improves P by 0.6 %, does not alter R, and improves F-ratio by 0.3 %.
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
				sent.append(word[0]) # Add 1st syllable as a word to the sentence.
				word = [word[1]] # Begin new word with 2nd syllable.

			# Check 1-lookahead with overlap ambiguity resolution.
			# Compare log prob(a, b_c) vs. log prob(a_b, c) if a, b_c, a_b, c exists in lexicon.
			if len(word) > 2:
				word2 = ' '.join(word[:-2]) # (a)
				word3 = ' '.join(word[-2:]) # (b_c)
				word4 = ' '.join(word[:-1]) # (a_b)
				word5 = word[-1] # (c)
				if word3 not in words_ or word2 not in words_:
					sent.append(word4)
					word = [word[-1]]
				elif word5 in words_ and word4 in words_:
					P1 = words_[word2] + words_[word3] # P(a, b_c)
					P2 = words_[word4] + words_[word5] # P(a_b, c)
					if P1 > P2:
						sent.append(word2)
						word = word[-2:]
					else:
						sent.append(word4)
						word = [word[-1]]
				else:
					# syl was an unknown word.
					sent.append(word4)
					word = [word[-1]]
			
		f.close()
		# Last sentence.
		if len(word) > 0:
			sent.append(' '.join(word))
		if len(sent) > 0:
			sents.append(sent)
	sents_hyp = sents


    if 1:
	# Evaluation.
	#print "STEP : Evaluation."
	#P, R, F = eval_seg(sents_ref, sents_hyp)
	n_hyp = 0
	n_corr = 0
	n_ref = 0
	nSents = len(sents_ref)
	#nSents = 6
	n_hyps = []
	n_refs = []
	n_corrs = []
	for n in xrange(nSents):
		sent1 = sents_ref[n]
		sent2 = sents_hyp[n]
		
		n_ref_ = len(sent1)
		n_hyp_ = len(sent2)
		
		# Finding optimal alignment and consequently no. of correct words in hypotheses
		# by dynamic programming. Longest Common Subsequence problem.
		l = np.zeros([n_ref_+1, n_hyp_+1])

		for row in range(1,l.shape[0]):
			for col in range(1,l.shape[1]):
				if sent1[row-1] == sent2[col-1]:
					l[row][col] = l[row-1][col-1] + 1
				else:
					l[row][col] = max([l[row][col-1], l[row-1][col]])
		n_corr_ = l[n_ref_][n_hyp_]
		
		n_hyp += n_hyp_
		n_ref += n_ref_
		n_corr += n_corr_
		n_hyps.append(n_hyp_)
		n_corrs.append(n_corr_)
		n_refs.append(n_ref_)

	prec = n_corr / n_hyp
	recall = n_corr / n_ref
	fratio = 2*prec*recall / (prec + recall)
	print  "Run " + "%.0f" % RUN + ": P = " + "%.3f" % prec + ", R = " + "%.3f" % recall + ", F = " + "%.3f" % fratio + ", OOV = " + "%.0f" % nOOV
    
    P_ += prec
    R_ += recall
    F_ += fratio
    O_ += nOOV
    NSE_ += nSents
    NW_ += nW
    NSY_ += nSyls

P_  /= nRuns
R_  /= nRuns
F_  /= nRuns
O_  /= nRuns
NSE_ /= nRuns
NW_ /= nRuns
NSY_ /= nRuns


print "===================================================="
print "Avg.   P = "+ "%.3f" % P_ + ", R = " + "%.3f" % R_ + ", F = " + "%.3f" % F_ + ", OOV = " + "%.0f" % nOOV
    
    
if 0:
	# Writing words with special chars (i.e. not_words_) in them for later analysis.
	special = []
	for word in words_:
		for c in not_words_:
			if re.search('\\' + c, word):
				special.append(word)
	f = codecs.open('./special.txt', mode = 'w', encoding = 'utf-8')
	for word in special:
		f.write(word + '\n')
	f.close()


# 19.1.14.

# 5-fold cross-validation experiments.
#===================================================

# Statistics of data in each fold and over all folds:
# Run 0: nSents = 1561, nWds = 24865, nSyls = 34405                                  
# Run 1: nSents = 1561, nWds = 26189, nSyls = 36189
# Run 2: nSents = 1561, nWds = 25474, nSyls = 35392                               
# Run 3: nSents = 1561, nWds = 25055, nSyls = 34939
# Run 4: nSents = 1562, nWds = 25910, nSyls = 35839
# =================================================
# Avg. : nSents = 1561, nWds = 25498, nSyls = 35352

# Experiment 1: Using only data in train files to create lexicon. Algo: MM.
# Run 0: P = 0.930, R = 0.931, F = 0.930, OOV = 0
# Run 1: P = 0.924, R = 0.928, F = 0.926, OOV = 0
# Run 2: P = 0.864, R = 0.895, F = 0.880, OOV = 1634
# Run 3: P = 0.927, R = 0.933, F = 0.930, OOV = 0
# Run 4: P = 0.929, R = 0.932, F = 0.930, OOV = 0
# ===================================================
# Avg.   P = 0.915, R = 0.924, F = 0.919, OOV = 0
# NWORDS_LEXICON (i.e. no. of unique words in train set) = c. 13K.
# P, R, F could improve if we have a larger lexicon e.g. Vn TreeBank has 40181 words.

# Comments:
# How many cases of overlap ambiguity are there? Overlap ambiguity resolution may not help to improve P,R,F a lot. 
# It may be better to look at resolution of numbers etc.
# How about a sentence tokenizer? [.:;]

# 20.1.14.

# Experiment 2: Use data in train files + locations and person names files to create lexicon. Algo: MM.
# + indicates improvement over previous best.
# Run 0: P = 0.932, R = 0.932, F = 0.932, OOV = 0 +
# Run 1: P = 0.928, R = 0.929, F = 0.929, OOV = 0 +
# Run 2: P = 0.867, R = 0.896, F = 0.881, OOV = 1623 +
# Run 3: P = 0.930, R = 0.934, F = 0.932, OOV = 0 +
# Run 4: P = 0.933, R = 0.933, F = 0.933, OOV = 0 +
# ====================================================
# Avg.   P = 0.918, R = 0.925, F = 0.921, OOV = 0 +
# NWORDS_LEXICON = c. 29K.

# Experiment 3: Use data in train files + dictionary, locations and person names files to create lexicon. Algo: MM.
# Run 0: P = 0.930, R = 0.925, F = 0.927, OOV = 0 -
# Run 1: P = 0.926, R = 0.922, F = 0.924, OOV = 0 -
# Run 2: P = 0.898, R = 0.909, F = 0.903, OOV = 999 +
# Run 3: P = 0.928, R = 0.928, F = 0.928, OOV = 0 -
# Run 4: P = 0.928, R = 0.924, F = 0.926, OOV = 0 -
# ====================================================
# Avg.   P = 0.922, R = 0.922, F = 0.922, OOV = 0 +
# NWORDS_LEXICON = c. 93K.

# Issue: Some words in provided lexicon may actually be phrases. But not marked as phrases (other than Proverb and Idiom - but removing them does not change result). 

# Comments:
# From experiments 1, 2 and 3, we observe that using training data, provided lexicon, locations and person names files together to create the lexicon (words_) leads to highest overall P,R,F values.
# So, we could keep this setting and test algorithm MM+ next.

# Experiment 4: Use data in train files + dictionary, locations and person names files to create lexicon. Algo: MM+.
# Run 0: P = 0.936, R = 0.929, F = 0.932, OOV = 0
# Run 1: P = 0.934, R = 0.927, F = 0.931, OOV = 0
# Run 2: P = 0.907, R = 0.912, F = 0.909, OOV = 999
# Run 3: P = 0.935, R = 0.932, F = 0.933, OOV = 0
# Run 4: P = 0.935, R = 0.929, F = 0.932, OOV = 0
# ====================================================
# Avg.   P = 0.929, R = 0.926, F = 0.927, OOV = 0

# Comments: Algo MM+ performs 0.5 % absolute better than Algo MM.

# Experiment 5: Use data in train files + locations + person names files, Algo: MM+.
# Run 0: P = 0.935, R = 0.931, F = 0.933, OOV = 0
# Run 1: P = 0.932, R = 0.928, F = 0.930, OOV = 0
# Run 2: P = 0.876, R = 0.895, F = 0.886, OOV = 1623
# Run 3: P = 0.933, R = 0.933, F = 0.933, OOV = 0
# Run 4: P = 0.935, R = 0.932, F = 0.934, OOV = 0
# ====================================================
# Avg.   P = 0.922, R = 0.924, F = 0.923, OOV = 0

# In brief:
# train + MM                                       : 0.919
# train + locations + person names + MM            : 0.921
# train + lexicon + locations + person names + MM  : 0.922
# train + lexicon + locations + person names + MM+ : 0.927


# In the LREC comparative study, it is shown that CRFs are outperformed by PVnSeg which uses MM + perl regex - "implements a series of heuristics for the detection of compound formulas such as proper nouns, common abbreviations, dates, numbers, URLs, e-mail addresses". So, we should implement this first, rather than CRF. OK but these should not improve current results as OOV low.


# Observation on OOVs:
# For runs 0, 1, 3, 4, OOV = 0. For run 2, OOV = 999 if include lexicon, and = 1623 if do not include it.
# So, the majority of errors are due to cross and overlap ambiguities, not due to OOV.
# Observing OOV words for run 2, all numeric instances are without spaces, so should not pose a problem in tokenization.
# e.g. 00:47:58, 19,99. 
# Obs.: Often consecutive foreign words form one word. May exploit this heuristic. Done.


# May mention the case of Vietnamese abbreviations and language used in SMS, emails and the internet. 
# Resource: http://vietpali.sourceforge.net/binh/VietTatChuVietTrongNgonNguChatVaTinNhan.htm
# Lexicon may be enhanced by these words.

if 0:
	# Save tokenizer model in .pkl file.
	# Tokenizer model consists of variables words_ and not_words_
	f = open('./model.pkl', 'wb')
	pickle.dump(words_,f)
	pickle.dump(not_words_,f)
	f.close()
	









 
