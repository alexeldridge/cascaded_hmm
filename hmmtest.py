from corpora import BNC
import json
import numpy as np
import os
import pomegranate as pom

rng = np.random.default_rng()

def my_from_json(file_text):
    d = json.loads(file_text)
    model = pom.HiddenMarkovModel(str(d['name']))

    states = [pom.State.from_dict(j) for j in d['states']]
    for i, j in d['distribution ties']:
        # Tie appropriate states together
        states[i].tie(states[j])

    # Add all the states to the model
    model.add_states(states)

    # Indicate appropriate start and end states
    model.start = states[d['start_index']]
    model.end = states[d['end_index']]

    # Add all the edges to the model
    for start, end, probability, pseudocount, group in d['edges']:
        model.add_transition(states[start], states[end], probability,
            pseudocount, group)

    # Bake the model
    model.bake(verbose=True, merge="None")
    return model

def get_corpus_by_name():
    output = {}
    for name in ['bncAK', 'bncJJ']:
        output[name] = (BNC(name, tag='pos'), BNC(name))
    return output

def get_model_by_corpus_name():
    # setting up a dict of corpora and corresponding models
    models = {'bncAK': [], 'bncJJ': []}

    for file_name in os.listdir(path='models'):

        for corpus_name in models:
            if corpus_name in file_name:
                with open('models\{}'.format(file_name), mode='r') as file:
                    try:
                        model = my_from_json(file.read())
                        models[corpus_name].append(model)
                    except RuntimeError:
                        print('messed up on {}'.format(file_name))
    
    return models

def fake_corpus(corpus):
    alph_counts = {}
    for seq in corpus.test_data:
        for word in seq:
            alph_counts.setdefault(word, 0)
            alph_counts[word] += 1

    total = sum(alph_counts.values())
    alph = np.array(list(alph_counts.keys()))
    alph_freqs = [val / total for val in alph_counts.values()]

    return np.array([
        rng.choice(alph, size=len(sen), p=alph_freqs)
        for sen in corpus.test_data
    ], dtype=object)

def average_prob(sens, model):
    return sum([
        model.probability(sen)
        for sen in sens
    ]) / len(sens)

# compare the probability the model gives to corpus data to fake similar data
def test_1():
    corpora_by_name = get_corpus_by_name()
    models = get_model_by_corpus_name()

    for name in ['bncAK', 'bncJJ']:
        corpus = corpora_by_name[name][0]
        fake_test_data = fake_corpus(corpus)

        for model in models[name]:
            print(model.name)
            corpus_avg = average_prob(corpus.test_data, model)
            fake_avg = average_prob(fake_test_data, model)
            print("Corpus: ", corpus_avg)
            print("Fake: ", fake_avg)
            print("Ratio: ", corpus_avg / fake_avg)

def name_to_phrases(state_name, length):
    phrases = tuple(state_name.split(',')[1:-1])
    phrases += ('',) * (length - len(phrases))
    return phrases

# print sequences aligned with phrase labels
def test_2():
    corpora_by_name = get_corpus_by_name()
    models = get_model_by_corpus_name()
    model = models['bncAK'][0]
    print(model.name)
    poses, words = corpora_by_name['bncAK']

    for pos_seq, word_seq in zip(poses.test_data[:100], words.test_data[:100]):
        state_ids = model.predict(pos_seq, algorithm='viterbi')
        state_names = [
            name_to_phrases(model.states[id].name, 2)
            for id in state_ids
            if id < model.silent_start]
        for i in range(2):
            print(np.array([phrase[i] for phrase in state_names]))
        print(pos_seq)
        print(word_seq)
        print()

if __name__ == '__main__':
    # test_1()
    test_2()