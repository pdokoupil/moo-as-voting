from collections import defaultdict
import numpy as np

class exactly_proportional_fuzzy_dhondt:

    def __init__(self):
        self._initialize()

    def _initialize(self):
        self.tot = 0.0
        self.s_r = defaultdict(float)
        self.last_call_votes = None

    # Expected to be called iteratively
    def __call__(self, candidate_groups, votes, partial_list, num_mandates):
        if self.last_call_votes is None:
            self.last_call_votes = votes
        assert self.last_call_votes == votes, "Votes cannot change in-between calls" # TODO implement

        parties = candidate_groups.keys()
        # Normalize votes
        votes = {p: 0.0 if v == 0.0 else v / sum(votes.values()) for p, v in votes.items()}

        
        # Map supports from [-1, 1] to [0, 1]
        def map_support(support):
            return (support + 1.0) / 2.0

        # Normalize supports (sum of squares per party should be 1)
        per_party_support_sum = {p: sum(map(lambda support: map_support(support) ** 2, candidate_groups[p].values())) for p in parties}
        
        for party, candidate_group in candidate_groups.items():
            for item, support in candidate_group.items():
                if per_party_support_sum[party] > 0.0:
                    candidate_group[item] = map_support(support) / np.sqrt(per_party_support_sum[party])

        tot_items = defaultdict(lambda: self.tot)

        def support_towards_party(item, party):
            if item in candidate_groups[party]:
                return map_support(candidate_groups[party][item])    
            return 0.0

        def total_support_towards_party(party):
            return sum(support_towards_party(item, party) for item in partial_list)

        max_gain = 0.0
        max_gain_item = None

        for _, candidate_group in candidate_groups.items():
            for candidate, support in candidate_group.items():
                tot_items[candidate] += map_support(support)
                
        gain_items = defaultdict(float)
        for party_name, candidate_group in candidate_groups.items():
            for candidate, _ in candidate_group.items():
                #e_r = {p: max(0.0, tot_items[candidate] * votes[p] - self.s_r[p]) for p in parties}
                e_r = max(0.0, tot_items[candidate] * votes[party_name] - self.s_r[party_name])
                gain_items[candidate] += min(support_towards_party(candidate, party_name), e_r)

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