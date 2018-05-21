"""Defines a simple Convolutional sequence-to-sequence model."""

import opennmt as onmt

def model():
    return onmt.models.FConvModel(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512)
    )
