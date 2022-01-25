import numpy as np

# FAI / fairness strategy (https://www.researchgate.net/publication/227132202_Group_Recommender_Systems_Combining_Individual_Models)
# Individual objectives take turns one by another and at each iteration, item maximizing the current objective is
class fai_strategy:
    def __init__(self, first_party=None, initialize_first_party_random=False):
        self.first_party = first_party
        self.initialize_first_party_random = initialize_first_party_random
        self._initialize()

    def _initialize_next_party(self, parties):
        if not self.first_party:
            if self.initialize_first_party_random:
                return np.random.choice(parties, 1)[0]
            return parties[0]
        return self.first_party

    # Expected to be called iteratively
    def __call__(self, candidate_groups, votes, partial_list, num_mandates):
        parties = list(candidate_groups.keys())
        if not self.last_party_on_turn:
            # Initialize the party selected for the current turn
            next_party = self._initialize_next_party(parties)
        else:
            idx = parties.index(self.last_party_on_turn)
            next_party = parties[(idx + 1) % len(parties)]
        
        self.last_party_on_turn = next_party
        best_candidate = max(candidate_groups[next_party], key=lambda x: candidate_groups[next_party][x]) # Get the one with highest supports
        best_candidate_support = candidate_groups[next_party][best_candidate] # TODO REMOVE
        del candidate_groups[next_party][best_candidate]
        
        if len(partial_list) + 1 == num_mandates:
            self._reset()
        
        return best_candidate, { next_party: best_candidate_support } # TODO Remove

    def __repr__(self):
        return f"fai_strategy(first_party={self.first_party}, initialize_first_party_random={self.initialize_first_party_random}"

    def __str__(self):
        return self.__class__.__name__

    def _reset(self):
        self._initialize()

    def _initialize(self):
        self.last_party_on_turn = None