import numpy as np
import os
import pomegranate as pom
import xml.etree.ElementTree as ET

def alphabet_from(corpus) -> set:
    '''Creates the alphabet of a given dataset
    '''

    alphabet = set()
    for sentence in corpus.data:
        alphabet.update(sentence)
    return alphabet

class BNC(pom.io.BaseGenerator):
    '''Unpacks the .xml of a section of BNC in the same directory as a
    dataloader of sequences of tags associated with each word, removing
    exceptions
    '''

    train_test = 0.8
    batch_size = 200
    num_batches = 1
    rng = np.random.default_rng()

    def __init__(self, path: str, tag = None, exceptions: list=None):
        if exceptions == None:
            exceptions = []
        files = []

        for file in os.listdir(path=path):
            parser = ET.parse(path + '/' + file)
            root = parser.getroot()
            file_data = np.array([
                # pom requires that the sequences be np.arrays
                np.array([
                    w.get(tag) if tag != None else w.text
                    for w in s.iter(tag='w') # gets every word
                    if w.get(tag) not in exceptions
                ])
                for s in root.iter(tag='s') # in every sentence
            ], dtype=object)
            # removes empty entries
            sentence_is_nonempty = [bool(s.size) for s in file_data]
            files.append(file_data[sentence_is_nonempty])
        
        self.data = np.concatenate(files)
    
    def __len__(self):
        return self.num_batches

    @property
    def shape(self):
        return self.batch_size * self.num_batches, 1

    @property
    def ndim(self):
        return 2
    
    @property
    def train_data(self):
        num_train = int(len(self.data) * self.train_test)
        return self.data[:num_train]
    
    @property
    def test_data(self):
        num_train = int(len(self.data) * self.train_test)
        return self.data[num_train:]
    
    def batches(self):
        weight = np.ones((self.batch_size,))
        for _ in range(self.num_batches):
            yield self.rng.choice(self.train_data, self.batch_size), weight


def main():
    corpus = BNC('bncJJ')
    print(corpus.batches())

if __name__ == '__main__':
    main()