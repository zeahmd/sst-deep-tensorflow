from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from loguru import logger
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


class SSTTokenizer:
    def __init__(self, corpus):
        logger.info("Preparing SST Tokenizer")
        self.vocab_size, self.max_sen_len = self.__get_vocab_size(corpus)
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.tokenizer.fit_on_texts(corpus)

    def get_word_index(self):
        return self.tokenizer.word_index

    def texts_to_sequences(self, sentences):
        return sequence.pad_sequences(
            self.tokenizer.texts_to_sequences(sentences),
            maxlen=self.max_sen_len,
            padding="post",
        )

    def __get_vocab_size(self, corpus):
        words_set = set()
        max_sentence_len = 0
        for sentence in corpus:
            sentence_tokens = sentence.split()
            if len(sentence_tokens) > max_sentence_len:
                max_sentence_len = len(sentence_tokens)
            for token in sentence_tokens:
                words_set.add(token)

        return len(words_set), max_sentence_len
