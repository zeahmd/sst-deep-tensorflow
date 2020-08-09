import pytreebank
from sst.tokenizer import SSTTokenizer
import os
from loguru import logger
from tensorflow.keras.utils import to_categorical
from sst.preprocessing import preprocess_sst
from sst.utils import get_binary_label


class SSTContainer:

    def __init__(self, root=False, binary=False):
        self.root = root
        self.binary = binary
        logger.info(f"Loading SST Dataset with config root: {root}, binary: {binary}")
        sst = pytreebank.load_sst()

        self.train_set = self.__preprocess_dataset(split='train', dataset=sst)
        self.dev_set = self.__preprocess_dataset(split='dev', dataset=sst)
        self.test_set = self.__preprocess_dataset(split='test', dataset=sst)

        self.tokenizer = SSTTokenizer(corpus=self.train_set[0]+self.dev_set[0]+self.test_set[0])
        del sst


    def __preprocess_dataset(self, split, dataset):
        logger.info(f"Preprocessing {split} set")
        split_dataset = dataset[split]
        text, sentiment = list(), list()
        if self.root:
            if self.binary:
                for tree in split_dataset:
                    if tree.to_labeled_lines()[0][0] != 2:
                        text.append(
                            preprocess_sst(tree.to_labeled_lines()[0][1])
                        )
                        sentiment.append(get_binary_label(tree.to_labeled_lines()[0][0]))
            else:
                for tree in split_dataset:
                    text.append(
                        preprocess_sst(tree.to_labeled_lines()[0][1])
                    )
                    sentiment.append(
                        tree.to_labeled_lines()[0][0]
                    )
        else:
            if self.binary:
                for tree in split_dataset:
                    for subtree in tree.to_labeled_lines():
                        if subtree[0] != 0:
                            text.append(
                                preprocess_sst(subtree[1])
                            )
                            sentiment.append(
                                get_binary_label(subtree[0])
                            )

            else:
                for tree in split_dataset:
                    for subtree in tree.to_labeled_lines():
                        text.append(
                            preprocess_sst(subtree[1])
                        )
                        sentiment.append(
                            subtree[0]
                        )

        return (text, sentiment)

    def sst_tokenizer_word_index(self):
        return self.tokenizer.get_word_index()

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def max_sen_len(self):
        return self.tokenizer.max_sen_len


    def data(self, split="train"):
        if self.binary:
            num_classes = 2
        else:
            num_classes = 5

        logger.info(f"Making token sequences of {split} set")
        if split is "train":
            return (
                self.tokenizer.texts_to_sequences(self.train_set[0]),
                to_categorical(self.train_set[1], num_classes= num_classes)
            )

        elif split is 'dev':
            return (
                self.tokenizer.texts_to_sequences(self.dev_set[0]),
                to_categorical(self.dev_set[1], num_classes=num_classes)
            )
        elif split is 'test':
            return (
                self.tokenizer.texts_to_sequences(self.test_set[0]),
                to_categorical(self.test_set[1], num_classes=num_classes)
            )
        else:
            logger.warning("Invalid split!")
            logger.info("Terminating!")
            os._exit(1)



