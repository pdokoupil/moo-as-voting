from collections import defaultdict
import numpy as np

class weighted_average_strategy:
    # Expected to be called iteratively
    def __call__(self, candidate_groups, votes, partial_list, num_mandates, extremes_per_party):
        votes = {p: 0.0 if v == 0.0 else v / sum(votes.values()) for p, v in votes.items()}

        def support_towards_party(item, party):
            if item in candidate_groups[party]:
                return candidate_groups[party][item]    
            #return min(candidate_groups[party])
            return extremes_per_party[party]["min"]

        item_gains = defaultdict(float)
        for party_name, supports in candidate_groups.items():
            for item, support in supports.items():
                item_gains[item] += support * votes[party_name]

        best_candidate = max(item_gains.keys(), key=lambda x: item_gains[x]) # Get the one with highest supports
        
        party_supports = {
            party_name: support_towards_party(best_candidate, party_name) for party_name in candidate_groups.keys()
        }

        for party_name, candidates in candidate_groups.items():
            if best_candidate in candidates:
                del candidate_groups[party_name][best_candidate]

        return best_candidate, party_supports # TODO Remove

    def __repr__(self):
        return f"weighted_average(first_party={self.first_party}, initialize_first_party_random={self.initialize_first_party_random}"

    def __str__(self):
        return self.__class__.__name__