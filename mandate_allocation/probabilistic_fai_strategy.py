import numpy as np

# Similar to FAI strategy, but instead of rotating between parties, we sample next party on random, weighted by the weights of the parties
class probabilistic_fai_strategy:
    def __init__(self, first_party=None, initialize_first_party_random=False):
        self.first_party = first_party
        self.initialize_first_party_random = initialize_first_party_random
        self._initialize()

    def _initialize_next_party(self, parties, votes):
        if not self.first_party:
            if self.initialize_first_party_random:
                return np.random.choice(list(votes.keys()), p=list(votes.values()), size=1)[0]
            return parties[0]
        return self.first_party

    # Expected to be called iteratively
    def __call__(self, candidate_groups, votes, partial_list, num_mandates, *args):
        votes = {p: 0.0 if v == 0.0 else v / sum(votes.values()) for p, v in votes.items()}

        parties = list(candidate_groups.keys())
        if not self.last_party_on_turn:
            # Initialize the party selected for the current turn
            next_party = self._initialize_next_party(parties, votes)
        else:
            next_party = np.random.choice(list(votes.keys()), p=list(votes.values()), size=1)[0]
        
        self.last_party_on_turn = next_party
        best_candidate = max(candidate_groups[next_party], key=lambda x: candidate_groups[next_party][x]) # Get the one with highest supports
        best_candidate_support = candidate_groups[next_party][best_candidate] # TODO REMOVE
        del candidate_groups[next_party][best_candidate]
        
        if len(partial_list) + 1 == num_mandates:
            self._reset()
        
        return best_candidate, { next_party: best_candidate_support } # TODO Remove

    def __repr__(self):
        return f"probabilistic_fai_strategy(first_party={self.first_party}, initialize_first_party_random={self.initialize_first_party_random}"

    def __str__(self):
        return self.__class__.__name__

    def _reset(self):
        self._initialize()

    def _initialize(self):
        self.last_party_on_turn = None