import math
import numpy as np
import pomegranate as pom

def default_dist(alphabet) -> pom.DiscreteDistribution:
    '''Generates a new uniform distribution on the given alphabet
    '''
    return pom.DiscreteDistribution({pos : 1/len(alphabet) for pos in alphabet})

def add_tied_copy(location, phrase, prefix: str):
    '''Adds a copy of the phrase HMM into the location HMM with all states
    using the same distribution and all edges tied to the same group. States'
    names are prefixed with the given prefix. Returns the start and end state
    of the copied model
    '''

    # creates a new state with same distribution and stores pairs in a dict
    new_state_map = {}
    for state in phrase.states:
        state_copy = pom.State(state.distribution, name = prefix + ',' + state.name)
        new_state_map[state] = state_copy
    
    location.add_states(*new_state_map.values())
    
    # adds edges with the same probability and group
    for s1 in phrase.states:
        for s2 in phrase.states:
            if phrase.graph.has_edge(s1, s2):
                data = phrase.graph.get_edge_data(s1, s2)
                location.add_transition(
                    new_state_map[s1],
                    new_state_map[s2],
                    math.exp(data['probability']), # prob stored as log prob
                    group = data['group']
                )
    return new_state_map[phrase.start], new_state_map[phrase.end]

def new_phrase(num_states, lower_phrases, alphabet, name):
    '''Creates a new HMM that links together some number of lower phrases and
    and a number of new states over a given alphabet, all prefixed with name
    '''

    start = pom.State(None, name = name + ',<')
    end = pom.State(None, name = name + ',>')
    model = pom.HiddenMarkovModel(start=start, end=end, name=name)
    
    states = [
        pom.State(default_dist(alphabet), name=name+',s'+str(i))
        for i in range(num_states)
    ]
    model.add_states(states)
    
    # adds tied copies of each lower phrase, stores their starts and ends
    phrase_starts, phrase_ends = [], []
    for phrase in lower_phrases:
        phrase_start, phrase_end = add_tied_copy(model, phrase, name)
        phrase_starts.append(phrase_start)
        phrase_ends.append(phrase_end)
    
    for s1 in states + phrase_ends + [start]:
        for s2 in states + phrase_starts + [end]:
            if s1 == start and s2 == end:
                # if there is an edge directly from start to end, and this is
                # connected to another phrase like it, then there is a cycle of
                # silent states, which pom doesn't like
                continue
            model.add_transition(s1, s2, 1, group = s1.name + ' & ' + s2.name)
    
    model.bake()
    return model

def cascadedHMM(num_states_per_layer, num_phrases_per_sublayer, alphabet, name):
    '''Creates a large HMM with a by recursively making new phrases composed of
    lower phrases. At each stage some number of new states is added and some
    number of new phrases are composed of lower phrases and these new states.
    The 
    '''

    lower_phrases = [
        cascadedHMM(num_states_per_layer[1:], num_phrases_per_sublayer[1:], alphabet, str(i))
        for i in range(num_phrases_per_sublayer[0])
    ]
    model = new_phrase(num_states_per_layer[0], lower_phrases, alphabet, name)
    return model

def main():
    chmm_architectures = [
        ((4,4,4,4), (3,3,2,0)),
        ((4,4,4,4), (4,3,1,0)),
        ((4,4,4,4), (2,3,4,0)),
        ((8,4,4,2), (3,3,2,0)),
        ((2,4,4,8), (3,3,2,0)),
        ((6,6,6), (4,4,0)),
        ((3,3,3,3,3), (2,2,2,2,0))
    ]
    for arch in chmm_architectures:
        chmm = cascadedHMM(*arch, 'abc', 's')
        print(len(chmm.states), chmm.edge_count())

if __name__ == '__main__':
    main()