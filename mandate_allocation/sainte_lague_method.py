from collections import defaultdict

class sainte_lague_method:

    def __init__(self):
        self._initialize()

    # Expected to be called iteratively
    def __call__(self, candidate_groups, votes, partial_list, num_mandates, *args):
        if self.last_call_votes is None:
            self.last_call_votes = votes
        assert self.last_call_votes == votes, "Votes cannot change in-between calls" # TODO implement

        max_quotient = 0.0
        max_quotient_party = None
        max_quotient_party_candidates = None
        
        for party_name, party_candidates in candidate_groups.items():
            quotient = votes[party_name] / (2 * self.seats_per_party[party_name] + 1)
            if quotient > max_quotient:
                max_quotient = quotient
                max_quotient_party = party_name
                max_quotient_party_candidates = party_candidates

        best_candidate = max(max_quotient_party_candidates, key=lambda x: max_quotient_party_candidates[x]) # Get the one with highest supports
        
        self.seats_per_party[max_quotient_party] += 1

        supports = dict()
        for party_name, candidates in candidate_groups.items():
            if best_candidate in candidates:
                supports[party_name] = candidates[best_candidate]

        del max_quotient_party_candidates[best_candidate] # TODO remove from all candidate groups

        if len(partial_list) + 1 == num_mandates:
            self._reset()
        
        return best_candidate, supports

    # TODO implement "using"/"with" pattern
    def _reset(self):
        #self._initialize()
        pass

    def _initialize(self):
        self.seats_per_party = defaultdict(int)
        self.last_call_votes = None

    def __repr__(self):
        return "sainte_lague_method()"
    
    def __str__(self):
        return self.__class__.__name__