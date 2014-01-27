#!/usr/bin/env python
# encoding: utf-8

# Anindya Roy
# CNRS-LIMSI
# roy@limsi.fr
#
# Vietnamese tokenization experiments.
# Using CRFSuite.
#
# CRFSuite should be installed along with liblbfgs 
# Source for CRFSuite and liblbfgs: 
# http://www.chokkan.org/software/crfsuite/
#
#==================================================================

import sys, re, math, unicodedata
import nltk
import numpy as np
import codecs
import pickle
import subprocess

#==================================================================
# PATHS AND OPTIONS:
# (Edit as suitable)

# Command to run CRFSuite binary:
CRFSUITE = 'crfsuite'

# Location of libcrfsuite-0.12.so:
LIBCRFPATH = "/usr/local/lib/"

USE_SC = 1 # Use syllable conjunction features for CRF.
USE_SSC = 0 # Use sparse syllable conjunction features for CRF.
USE_DICT = 1 # Use dictionary features for CRF.
USE_ERS = 1 # Use external resource features (i.e. is it a location or person name?)
USE_MISC = 1 # Use miscellaneous features, e.g. first (or all) letters capital, punctuation marks, etc.

DIR0 = '../' # Top level directory: Roy_VnTokenizer/
DATADIR = DIR0 + '/data/' # Location of most data files.


# CRF feature function attribute offsets :
#=========================================

# (1) Offsets for syllable conjunction (SC) features: 
# 1-, 2-, 3-grams in (-2, 2) context window.
sc = [ [-2], [-1], [0], [1], [2], 
         [-2,-1], [-1, 0], [0, 1], [1, 2], 
         [-2, -1, 0], [-1, 0, 1], [0, 1, 2] ]
# -> Also used for dictionary features.

# (2) Offsets for sparse syllable conjunction (SCS) features: 
# Sparse 3-grams in (-2, 2) context window.
ssc = [ [-2, 0], [-1, 1], [0, 2] ]

# (3) Offsets for person name (MISC) features:
pname = [-2, -1, 0, 1, 2]

# Punctuation and special characters.
puncts = set([u'!',  u'"', u'%', u'&', u"'", u'(', u')', u'*', u'+', u',', u'-', u'.',
                      u'/', u':', u';', u'=', u'>', u'?'])

#==========================================================

if USE_DICT and 1:

	print "STEP : Preparing dictionary."
	lexfile = DIR0 + '/data/VNDic_UTF-8.txt'
	f = codecs.open(lexfile, encoding = 'utf-8', mode = 'r', errors = 'ignore')
	words_ = []
	for line in f:
		if re.search('##', line):
			word = re.sub('##', '', line)
			word = '_'.join(word.split()) # Note: join by '_' to match CRFSuite format.
			words_.append(word)
		if re.search('@@', line):
			line = re.sub('@@', '', line)
			if line in ['Proverb', 'Idiom']:
				del words_[-1] # Remove last item added if it is a proverb or idiom.
	f.close()
	words_ = set(words_) # IMPORTANT NOTE: set is much faster to search than list.

#===================================================================================================

if USE_ERS and 1:
	
	print "STEP : Processing external resources (location and person names)."

	first_names = []
	middle_names = []
	last_names = []
	location_names = []

	lexfile = DIR0 + '/data/vnlocations.txt'
	f = codecs.open(lexfile, encoding = 'utf-8', mode = 'r', errors = 'ignore')
	for line in f:
		word = '_'.join(line.split())
		location_names.append(word)
	f.close()
	
	lexfile = DIR0 + '/data/vnpernames.txt'
	f = codecs.open(lexfile, encoding = 'utf-8', mode = 'r', errors = 'ignore')
	for line in f:
		word = line.split()
		first_names.append(word[0])
		if len(word) > 1:
			last_names.append(word[-1])
		if len(word) > 2:
			middle_names.extend(word[1:-1])
	f.close()
	
	location_names = set(location_names)
	first_names = set(first_names)
	middle_names = set(middle_names)
	last_names = set(last_names)

#===================================================================================================

# 5 cross-validation folds.
nRuns = 5
if 1:

    print "STEP : Feature attribute extraction."
    for RUN in xrange(nRuns): 

    	print "Run# " + str(RUN)
	
	# Read in raw train file in JVnSeg format,
	# i.e. each line: "syllable \t label", label = {B_W, I_W, O}.
	# Empty line to mark end of sentences.
	if 1:
	    fin = codecs.open(DATADIR + "/train" + str(RUN+1) + ".iob2", encoding='utf-8', mode = 'r', errors = 'ignore') 
	    seqs = [] # List of sentences.
	    seq = {}
	    seq['labels'] = [] # label = {B_W, I_W, O}.
	    seq['syls'] = [] # Vietnamese syllables.

	    for line in fin:
		line_ = line.split()
		if len(line_) == 0: # End of sentence. 

			seqs.append(seq)
			seq = {}
			seq['labels'] = []
			seq['syls'] = []
			#input("Press Enter to continue...")
			continue

		seq['syls'].append(re.sub(":", "\:", line_[0])) # Must escape ':' -> specific to CRFSuite.
		seq['labels'].append(line_[1])
		#print seq

	    # Last sentence.
	    if len(seq['labels']) > 0:
		seqs.append(seq)
		seq['labels'] = []
		seq['syls'] = []
	    fin.close()

	    fout = codecs.open(DATADIR + '/crf.train.' + str(RUN+1) + '.txt', encoding = 'utf-8', mode = 'w')
	    # Attribute extraction.
	    nSents = len(seqs) # No. of sentences.
	    for sentNo in xrange(nSents):
		
		seq = seqs[sentNo]
		labels = seq['labels']
		syls = seq['syls']
		nSyls = len(syls)
		
		syls_ = ['BOS2', 'BOS1']
		syls_.extend(syls)
		syls_.extend(['EOS1', 'EOS2'])
		
		for sylNo in xrange(nSyls):
			sylNo_ = sylNo + 2 # Offset inside syls_.
			label = labels[sylNo]
			text = label # Text to write to output file in CRFSuite format.
			attrib_no = 0 # Attribute index.
			if USE_SC:
				for attrib in sc:
					ngram = syls_[sylNo_ + attrib[0]]
					for offset in attrib[1:]:
						ngram = ngram + '_' + syls_[sylNo_ + offset]
					
					text = text + '\t' + 'f' + str(attrib_no) + '=' + ngram
					attrib_no += 1
			if USE_SSC:
				for attrib in ssc:
					ngram = syls_[sylNo_ + attrib[0]]
					for offset in attrib[1:]:
						ngram = ngram + '_' + syls_[sylNo_ + offset]
					
					text = text + '\t' + 'f' + str(attrib_no) + '=' + ngram
					attrib_no += 1
			if USE_DICT:
				for attrib in sc:
					ngram = syls_[sylNo_ + attrib[0]]
					for offset in attrib[1:]:
						ngram = ngram + '_' + syls_[sylNo_ + offset]
					if ngram in words_:
						text = text + '\t' + 'f' + str(attrib_no) + '=1'
					else:
						text = text + '\t' + 'f' + str(attrib_no) + '=0'
					attrib_no += 1

			if USE_ERS:
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

			if USE_MISC:
				
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
			
	if 1:
	    # Process test file
	    fin = codecs.open(DATADIR + "/test" + str(RUN+1) + ".iob2", encoding='utf-8', mode = 'r', errors = 'ignore') 
	    seqs = [] # List of sentences.
	    seq = {}
	    seq['labels'] = [] # B_W, I_W or O.
	    seq['syls'] = [] # Vietnamese syllables.

	    for line in fin:
		line_ = line.split()
		if len(line_) == 0: # End of sentence. 
			#print seq
			seqs.append(seq)
			seq = {}
			seq['labels'] = []
			seq['syls'] = []
			#input("Press Enter to continue...")
			continue
		#print line_
		seq['syls'].append(re.sub(":", "\:", line_[0])) # IMPORTANT. Specific to CRFSuite.
		seq['labels'].append(line_[1])
		#print seq

	    # Last sentence.
	    if len(seq['labels']) > 0:
		seqs.append(seq)
		seq['labels'] = []
		seq['syls'] = []
	    fin.close()

	    fout = codecs.open(DATADIR + '/crf.test.' + str(RUN+1) + '.txt', encoding = 'utf-8', mode = 'w')
	    # Attribute extraction.
	    nSents = len(seqs) # No. of sentences.
	    for sentNo in xrange(nSents):
		
		seq = seqs[sentNo]
		labels = seq['labels']
		syls = seq['syls']
		nSyls = len(syls)
		
		syls_ = ['BOS2', 'BOS1']
		syls_.extend(syls)
		syls_.extend(['EOS1', 'EOS2'])
		
		for sylNo in xrange(nSyls):
			sylNo_ = sylNo + 2 # Offset in syls_.
			label = labels[sylNo]
			text = label # Text to write to output file in CRFSuite format.
			attrib_no = 0 # Attribute index.
			if USE_SC:
				for attrib in sc:
					ngram = syls_[sylNo_ + attrib[0]]
					for offset in attrib[1:]:
						ngram = ngram + '_' + syls_[sylNo_ + offset]
					
					text = text + '\t' + 'f' + str(attrib_no) + '=' + ngram
					attrib_no += 1
			if USE_SSC:
				for attrib in ssc:
					ngram = syls_[sylNo_ + attrib[0]]
					for offset in attrib[1:]:
						ngram = ngram + '_' + syls_[sylNo_ + offset]
					
					text = text + '\t' + 'f' + str(attrib_no) + '=' + ngram
					attrib_no += 1
			if USE_DICT:
				for attrib in sc:
					ngram = syls_[sylNo_ + attrib[0]]
					for offset in attrib[1:]:
						ngram = ngram + '_' + syls_[sylNo_ + offset]
					if ngram in words_:
						text = text + '\t' + 'f' + str(attrib_no) + '=1'
					else:
						text = text + '\t' + 'f' + str(attrib_no) + '=0'
					attrib_no += 1

			if USE_ERS:
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

			if USE_MISC:
				
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

if 1:
	print "STEP: Training CRF models."
	for RUN in xrange(nRuns): 
		print "Run# " + str(RUN)
		cmd = CRFSUITE + " learn -m " + DATADIR + "/crf." + str(RUN+1) + ".model " + DATADIR + "/crf.train." + str(RUN+1) + ".txt"
		cmd = "export LD_LIBRARY_PATH=" + LIBCRFPATH + "; " + cmd 
		status = subprocess.call(cmd, shell = True)
		

if 1: 
	print "STEP: Tagging with CRF models."
	for RUN in xrange(nRuns):
		print "Run# " + str(RUN)
		cmd = "export LD_LIBRARY_PATH=" + LIBCRFPATH
		cmd = cmd + "; " + CRFSUITE + " tag -m " + DATADIR + "/crf." + str(RUN+1) +".model " + DATADIR + "/crf.test." + str(RUN+1) + ".txt > tmp.1"
		cmd = cmd + "; " + "cut -f1 " + DATADIR + "/test" + str(RUN+1) + ".iob2" + " > tmp.2"
		cmd = cmd + "; paste tmp.2 tmp.1 > " + DATADIR + "/crf.test." + str(RUN+1) + ".hyp.iob2"
		cmd = cmd + "; rm tmp.1 tmp.2"
		status = subprocess.call(cmd, shell = True)
		
if 1:
	print "STEP: Evaluation."
	# Values over runs.
	P_ = 0 # Precision.
	R_ = 0 # Recall.
	F_ = 0 # F-ratio.
	#nRuns = 5

	#for RUN in [0, 1, 3, 4]:
	for RUN in xrange(nRuns):
		# Process reference iob2 file.
		sents = [] 
		f = codecs.open(DATADIR + "/test" + str(RUN+1) + ".iob2", encoding='utf-8', mode = 'r', errors = 'ignore') 
		sent = []
		word = ''
		write = 0 # To write sentence.
		for line in f:
			line = line.split()
			if len(line) == 0: # End of sentence. 
				if word != '':
					sent.append(word) # Write current word.
					word = '' # Flush word buffer.
				if write == 1:
					sents.append(sent)
					sent = [] # Flush sentence buffer.
					write = 0
				continue
			write = 1
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
		if write == 1:
			sents.append(sent)
			write = 0
		sents_ref = sents
	
		# Process hypothesis iob2 file (output by CRFSuite).
		sents = [] 
		f = codecs.open(DATADIR + "/crf.test." + str(RUN+1) + ".hyp.iob2", encoding='utf-8', mode = 'r', errors = 'ignore') 
		sent = []
		word = ''
		write = 0 # To write sentence.
		for line in f:
			line = line.split()
			if len(line) == 0: # End of sentence. 
				if word != '':
					sent.append(word) # Write current word.
					word = '' # Flush word buffer.
				if write == 1:
					sents.append(sent)
					sent = [] # Flush sentence buffer.
					write = 0
				continue
			write = 1
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
		if write == 1:
			sents.append(sent)
			write = 0
		sents_hyp = sents	

    		n_hyp = 0
		n_corr = 0
		n_ref = 0
		nSents = len(sents_ref)
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
		print  "Run " + "%.0f" % RUN + ": P = " + "%.3f" % prec + ", R = " + "%.3f" % recall + ", F = " + "%.3f" % fratio
    
		P_ += prec
		R_ += recall
    		F_ += fratio

	P_  /= nRuns
	R_  /= nRuns
	F_  /= nRuns


	print "===================================================="
	print "Avg.   P = "+ "%.3f" % P_ + ", R = " + "%.3f" % R_ + ", F = " + "%.3f" % F_
    

# 26.1.14.
# Performance evaluation.
# =======================
# System CRF1: Using only syllable conjunction (SC) features:
# Run 0: P = 0.903, R = 0.910, F = 0.906
# Run 1: P = 0.894, R = 0.901, F = 0.898
# Run 2: P = 0.903, R = 0.911, F = 0.907
# Run 3: P = 0.901, R = 0.912, F = 0.906
# Run 4: P = 0.906, R = 0.908, F = 0.907
# ====================================================
# Avg.   P = 0.901, R = 0.908, F = 0.905


# Comments: 
# (1) OOV rate on test data is high if use only the provided lexicon
#    - could explain lower P, R, F values for MM reported in JVnSeg article.
# (2) In contrast, OOV drops down if use train data. 
# This could explain why CRF performs better than MM in JVnSeg article - because CRF is trained on train data
# and its features (e.g. SC) are built using these words, including OOV words not present in original lexicon.
# (3) Consistent with this observation, if we use train data for MM, its performance will improve from
# F = 0.84 to F = 91. 
# (4) So, the question remains if MM and CRF were compared in a 100% unbiased way in the JVnSeg article.


# 26.1.14.
# Performance evaluation.
# =======================
# System CRF2 : Using syllable conjunction (SC) and Dictionary (DICT) features.
# Run 0: P = 0.938, R = 0.945, F = 0.941
# Run 1: P = 0.940, R = 0.947, F = 0.943
# Run 2: P = 0.934, R = 0.943, F = 0.938
# Run 3: P = 0.938, R = 0.949, F = 0.944
# Run 4: P = 0.943, R = 0.950, F = 0.946
# ====================================================
# Avg.   P = 0.939, R = 0.947, F = 0.943
# Training time: ~10 minutes on Intel Atom Acer Netbook.


# Comments:
# (1) CRF2 performs slightly better than best JVnSeg performance on same 5-fold setup reported in JVnSeg article. 
# (2) CRF2 performs 3.8% better than CRF1 (first system above).

# System CRF3: Using syllable conjunction (SC), Dictionary (DICT), External Resources (ERS) and 
# Miscellaneous (MISC) features.
# Run 0: P = 0.939, R = 0.948, F = 0.943
# Run 1: P = 0.940, R = 0.948, F = 0.944
# Run 2: P = 0.937, R = 0.945, F = 0.941
# Run 3: P = 0.938, R = 0.949, F = 0.943
# Run 4: P = 0.945, R = 0.951, F = 0.948
# ====================================================
# Avg.   P = 0.940, R = 0.948, F = 0.944
# Training time : ~20 minutes on Intel Atom Acer Netbook.

# For comparison, JVnSegmenter's best CRF performance (from viet4.pdf, page 7):
#        P = 0.938, R = 0.943, F = 0.941
# and best SVM performance (from viet4.pdf, page 7):
#        P = 0.940, R = 0.945, F = 0.942
# Note that training time for JVnSegmenter CRF is 2 hours and SVM is 4 hours (from viet4.pdf, page 8).

# Comments:
# (1) CRF3 performs 0.1% better than CRF2 on average.
# (2) CRF3 performs 0.3% better than JVnSeg CRF performance on same 5-fold experiment setup reported in JVnSeg article. 
# (3) CRF3 is 6 times faster than JVnSeg CFR. (Reason? Maybe because CRF3 uses sets rather than lists.) 
# (4) Biggest gain by adding Dict features : 90.5% (CRF1) -> 94.3% (CRF2) 
#     Not much improvement in adding other (ERS, MISC) features at the cost of doubling training time.

# Summary of my experiments (Avg. F-measure):
# ===================================
# CRF1 (SC)                     0.905         
# CRF2 (SC + DICT)              0.943
# CRF3 (SC + DICT + ERS + MISC) 0.944
# -----------------------------------
# MM+ (best configuration)      0.927
# ===================================

# Comments:
# (1) CRF2 performs 1.6% better than MM+ (best configuration) on average. 
# (2) CRF3 performs 1.7% better than MM+ (best configuration) on average.
# (3) So, CRF does perform slightly better than MM on this data, although the difference in performance (1.6-1.7%) is 
# lower than the difference of 11% shown in viet4.pdf. This is probably due to not using train data for MM in viet4.pdf.
# However, as pointed out above, an unbiased comparison is expected to use train data for MM as it is used for CRF too.


if 0:
	# Save necessary files.
	f = open('./model.crf.pkl', 'wb')
	pickle.dump(sc,f)
	pickle.dump(pname,f)
	pickle.dump(puncts,f)
	pickle.dump(words_,f)
	pickle.dump(first_names,f)
	pickle.dump(middle_names,f)
	pickle.dump(last_names,f)
	pickle.dump(location_names,f)
	f.close()




