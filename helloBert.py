"""
A simple example to illustrate the use of BERT for reranking documents.
"""

# Copyright (c) 2025, Carnegie Mellon University.  All Rights Reserved.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Idx import Idx


# --------- A simple wrapper for accessing BERT ------------ #

class BERT:
    """A simple wrapper API for accessing HuggingFace Transformers."""

    def __init__(self, modelPath):
        """Initialize from a pretrained model checkpoint"""
        self.tokenizer = AutoTokenizer.from_pretrained(modelPath)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            modelPath,
            num_labels=1)
        self.model.eval()	# Prepare to use the model for evaluation

        
    def display_sequence(self, tensors_dict):
        """Display the WordPiece tokens for an encoded sequence"""
        tokens = self.tokenizer.convert_ids_to_tokens(
            tensors_dict['input_ids'][0])
        print(f"\nWordPiece tokens: {tokens}")
        

    def encode_q_psg(self, q_str, psg_str, max_seq_length):
        """
        Tokenize a (query, passage) pair, convert to token ids, and
            return as tensors.
        Input: query and document strings.
        Output: a dictionary of tensors that a transformer  understands.
            input_ids: The ids for each token. 
            token_type_ids: The token type (sequence) id of each token.
            attention_mask: For each token, mask(0) or don't mask(1). Not used.
        """
        return(self.tokenizer.encode_plus(
            [q_str, psg_str],		# sequence_1, sequence_2
            add_special_tokens=True,	# Add [CLS] and [SEP] tokens?
            max_length=max_seq_length,	# Maximum sequence length
            truncation="only_second",	# If too long, truncate sequence_2
            return_tensors="pt"))	# Return PyTorch tensors


    def score_sequence(self, tensors_dict):
        """
        Score a (query, document) pair encoded as a tensor.
        Input: the tokenized sequence.
        Output: the reranking score.
        """
        with torch.no_grad():
            # Pass the tokenized sequence to the model for scoring. 
            # Extract the classification score and transform to python float.
            outputs = self.model(**tensors_dict) 
            score = outputs.logits.data.item()
            return(score)


# ------------------ Configuration ------------------------- #

bert_max_sequence_length = 512				# Max WordPiece tokens
bert_modelPath = "INPUT_DIR/ms-marco-MiniLM-L-12-v2"	# Stored BERT model
indexPath = "INPUT_DIR/index-cw09"			# Stored CW09 index


# ------------------ Script body --------------------------- #

Idx.open(indexPath)

# Initialize BERT from a pretrained model checkpoint
bert = BERT(bert_modelPath)

# Match a query to two documents.
query = "quit smoking"
print(f'QUERY:\t{query}\n')

# doc1
doc = Idx.getAttribute("title-string", 304969)
print(f'--- DOC 304969: ---\n{doc}')

encoded_sequence = bert.encode_q_psg(query, doc, bert_max_sequence_length)
bert.display_sequence(encoded_sequence)		# Just FYI, if you are curious

score = bert.score_sequence(encoded_sequence)
print(f'\n(q, d) score:\t{score}\n' )

# doc2
doc = Idx.getAttribute("body-string", 288258)	
print(f'\n--- DOC 288258: ---\n{doc}')

encoded_sequence = bert.encode_q_psg(query, doc, bert_max_sequence_length)

score = bert.score_sequence(encoded_sequence)
print(f'\n(q, d) score:\t{score}\n' )

