import numpy as np

class exactly_proportional_fai_strategy:
    def __init__(self, initialize_first_party_random=False):
        self.initialize_first_party_random = initialize_first_party_random
        self._initialize()

    def _initialize_next_party(self, parties, party_weights):
        if self.initialize_first_party_random:
            next_party = np.random.choice(parties, p=party_weights)
            return next_party, parties.index(next_party)
        
        next_party_idx = np.argmax(party_weights)
        return parties[next_party_idx], next_party_idx # Select party with largest weight

    def _get_next_party(self, parties, mandates_per_party):
        idx = parties.index(self.last_party_on_turn)
        next_idx = (idx + 1) % len(parties)
        while self.per_party_allocated_mandates[next_idx] >= mandates_per_party[next_idx]:
            next_idx = (next_idx + 1) % len(parties)
        return parties[next_idx], next_idx

    # Expected to be called iteratively
    def __call__(self, candidate_groups, votes, partial_list, num_mandates):
        vote_values = np.fromiter(votes.values(), dtype=np.float64)
        assert np.isclose(np.sum(vote_values), 1.0), f"Votes {votes} must be normalized" # So they correspond to objective weights directly
        
        if self.per_party_allocated_mandates is None:
            self.per_party_allocated_mandates = [0 for _ in votes] # Initialize
            self.last_call_votes = votes
        
        assert self.last_call_votes == votes, "Votes cannot change in-between calls" # TODO implement

        mandates_per_party = [int(num_mandates * weight) for weight in vote_values]
        mandate_remainder = num_mandates - sum(mandates_per_party)
        
        for _ in range(mandate_remainder): # We should distribute the remaining mandates between the parties
            winning_party = np.random.choice(np.arange(len(mandates_per_party)), p=vote_values) # Always sample the winning party based on its weight
            mandates_per_party[winning_party] += 1
        
        assert np.sum(mandates_per_party) == num_mandates, f"Number of mandates per party {mandates_per_party} must sum to the total number of mandates {num_mandates}"
            
        parties = list(candidate_groups.keys())
        if not self.last_party_on_turn:
            # Initialize the party selected for the current turn
            next_party, next_party_idx = self._initialize_next_party(parties, vote_values)
        else:
            next_party, next_party_idx = self._get_next_party(parties, mandates_per_party)

        
        self.last_party_on_turn = next_party
        
        best_candidate = max(candidate_groups[next_party], key=lambda x: candidate_groups[next_party][x]) # Get the one with highest supports
        best_candidate_support = candidate_groups[next_party][best_candidate] # TODO REMOVE
        del candidate_groups[next_party][best_candidate]
        self.per_party_allocated_mandates[next_party_idx] += 1
        
        if len(partial_list) + 1 == num_mandates:
            self._reset()

        return best_candidate, (next_party, best_candidate_support)

    def __repr__(self):
        return f"fai_strategy(first_party={self.first_party}, initialize_first_party_random={self.initialize_first_party_random}"

    def __str__(self):
        return self.__class__.__name__

    def _reset(self):
        self._initialize()

    def _initialize(self):
        self.last_party_on_turn = None
        self.per_party_allocated_mandates = None
        self.last_call_votes = None