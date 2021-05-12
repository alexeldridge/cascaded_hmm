from cascadedhmm import cascadedHMM, new_phrase
from corpora import alphabet_from, BNC
import pomegranate as pom

def main():
    corpus_paths = ['bncAK', 'bncJJ']
    tags = ['pos']
    nums_states = [100, 200]
    chmm_architectures = [
        ((4,4,4,4), (3,3,2,0)), # control
        ((4,4,4,4), (4,3,1,0)), # many big phrases
        ((4,4,4,4), (2,3,4,0)), # many small phrases
        ((8,4,4,2), (3,3,2,0)), # complicated big phrases
        ((2,4,4,8), (3,3,2,0)), # complicated small phrases
        ((6,6,6), (4,4,0)), # less layers
        ((3,3,3,3,3), (2,2,2,2,0)) # more layers
    ]

    for path in corpus_paths:
        for tag in tags:
            corpus = BNC(path, tag=tag)
            alphabet = alphabet_from(corpus)

            for num_states in nums_states:
                print('HMM -- Corpus: {}, Tag: {}, {} States'.format(path, tag, num_states))
                hmm = new_phrase(
                    num_states,
                    [],
                    alphabet,
                    'hmm_{}_{}_{}states'.format(path, tag, num_states)
                )
                hmm.fit(
                    corpus,
                    stop_threshold = 0.001,
                    max_iterations = 100,
                    min_iterations = 100,
                    batches_per_epoch = 1,
                    n_jobs = 4,
                    verbose = True
                )
                hmm.bake()

                file_name = 'models/hmm_{}_{}_{}states.json'.format(
                    path, tag, num_states
                )
                with open(file_name, mode='w') as file:
                    file.write(hmm.to_json())
            
            for arch in chmm_architectures:
                print('CHMM -- Corpus: {}, Tag: {}, Architecture: {}'.format(path, tag, arch))
                hmm = cascadedHMM(
                    *arch,
                    alphabet,
                    'chmm_{}_{}_{}'.format(path, tag, arch)
                )
                hmm.fit(
                    corpus,
                    stop_threshold = 0.001,
                    max_iterations = 500,
                    min_iterations = 500,
                    batches_per_epoch = 1,
                    n_jobs = 4,
                    verbose = True
                )

                file_name = 'models/chmm_{}_{}_{}.json'.format(
                    path, tag, arch
                )
                with open(file_name, mode='w') as file:
                    file.write(hmm.to_json())
                


if __name__ == '__main__':
    main()