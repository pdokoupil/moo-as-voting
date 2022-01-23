from collections import defaultdict
import numpy as np

# New Version of exactly proportional fuzzy dhondt.
# The difference from the previous version is that now we allow
# items with negative support/score
class exactly_proportional_fuzzy_dhondt_2:

    def __init__(self):
        self._initialize()

    def _initialize(self):
        self.tot = 0.0
        self.s_r = defaultdict(float)
        self.last_call_votes = None

    # Expected to be called iteratively
    def __call__(self, candidate_groups, votes, partial_list, num_mandates, extremes_per_party):
        if self.last_call_votes is None:
            self.last_call_votes = votes
        assert self.last_call_votes == votes, "Votes cannot change in-between calls" # TODO implement

        # Normalize votes
        votes = {p: 0.0 if v == 0.0 else v / sum(votes.values()) for p, v in votes.items()}

        # Total relevance of the items
        tot_items = defaultdict(lambda: self.tot)

        def support_towards_party(item, party):
            if item in candidate_groups[party]:
                return candidate_groups[party][item]    
            #return min(candidate_groups[party])
            return extremes_per_party[party]["min"]

        def total_support_towards_party(party):
            return sum(support_towards_party(item, party) for item in partial_list)

        max_gain = 0.0
        max_gain_item = None

        for party_name, candidate_group in candidate_groups.items():
            for candidate, support in candidate_group.items():
                tot_items[candidate] += support #* votes[party_name]
                
        gain_items = defaultdict(float) # Initialize to negative infinity
        for party_name, candidate_group in candidate_groups.items():
            if votes[party_name] == 0:
                continue
            for candidate in tot_items.keys(): # We need to iterate over all items, note that candidates for parties could differ
                limit = tot_items[candidate] * votes[party_name]

                if limit <= self.s_r[party_name]:
                    e_r = 0
                else:
                    e_r = limit - self.s_r[party_name]

                # sup = max(
                #     support_towards_party(candidate, party_name),
                #     support_towards_party(candidate, party_name) + (self.s_r[party_name] - tot_items[candidate] * votes[party_name])
                # )
                sup = support_towards_party(candidate, party_name)
                gain_items[candidate] += min(sup, e_r)

        max_gain_item, max_gain =  max(gain_items.items(), key=lambda x: x[1])

        self.tot = 0.0
        for p, s in self.s_r.items():
            self.s_r[p] = s + support_towards_party(max_gain_item, p)
            self.tot += self.s_r[p]

        for _, candidates in candidate_groups.items():
            if max_gain_item in candidates:
                del candidates[max_gain_item]
    
        if len(partial_list) + 1 == num_mandates:
            self._reset()

        assert max_gain_item is not None, "Valid item must be returned"
        return max_gain_item, ("rating_based_relevance", 123.0)

    # TODO implement "using"/"with" pattern
    def _reset(self):
        self._initialize()

    def __name__(self):
        return self.get_name()

    def get_name(self):
        return self.__class__.__name__

    def __repr__(self):
        return  "exactly_proportional_fuzzy_dhondt()"
    
    def __str__(self):
        return self.get_name()