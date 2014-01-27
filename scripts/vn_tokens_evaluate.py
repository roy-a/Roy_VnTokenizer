#!/usr/bin/env python
# encoding: utf-8

# Written by Anindya Roy,
# CNRS-LIMSI
# roy@limsi.fr
#
# Script to evaluate Vietnamese tokenization.
# Version 1. 21.1.14.
#
#
#===============================================================================

import sys
if 1:
	n_args = len(sys.argv)
	if n_args < 3:
		print '\n'
		print "Vietnamese tokenizer evaluation. Version 0.1."
		print "============================================="
		print "Usage: ./vn_tokens_evaluate.py <ref file name> <hyp file name>"
		print "<ref file name> and <hyp file name> are mandatory inputs."
		print "The hyp file must contain hypothesized tokenization."
		print "The ref file must contain reference tokenization."
		print "All tokens must be surrounded by square brackets []." + '\n'
		exit()
	if n_args >= 3:
		input_file_name = sys.argv[1]
		output_file_name = sys.argv[2]
	

if 0:
	input_file_name = './input1.tkn'
	output_file_name = './input1.tkn.wseg1'


import os
if not os.path.isfile(input_file_name):
	print 'ref file "' + input_file_name + '" does not exist. Retry with a valid file name.'
	exit(1)

if not os.path.isfile(output_file_name):
	print 'hyp file "' + output_file_name + '" does not exist. Retry with a valid file name.'
	exit(1)


import sys, re, math, unicodedata, numpy as np, codecs, pickle
# import nltk : NLTK not required.

f = codecs.open(input_file_name, mode = 'r', encoding = 'utf-8', errors = 'ignore')
sents = []
for line in f:
	sent = []
	line = line.split()
	for word in line:
		if re.search('\A\[.+\]\Z', word): # If it is a token.
			sent.append(word)
	sents.append(sent) # Split line on space to get syllables + etc.
f.close()
sents_ref = sents

f = codecs.open(output_file_name, mode = 'r', encoding = 'utf-8', errors = 'ignore')
sents = []
for line in f:
	sent = []
	line = line.split()
	for word in line:
		if re.search('\A\[.+\]\Z', word): # If it is a token.
			sent.append(word)
	sents.append(sent) # Split line on space to get syllables + etc.
f.close()
sents_hyp = sents

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
print  "P = " + "%.3f" % prec + ", R = " + "%.3f" % recall + ", F = " + "%.3f" % fratio











