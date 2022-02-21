import numpy as np
class random_mandate_allocation:
    # Expected to be called iteratively
    def __call__(self, candidate_groups, votes, partial_list, num_mandates, *args):
        objective_names = list(candidate_groups.keys())
        # For the given position, we sample randomly the canditate groups
        objective_name = np.random.choice(objective_names, 1)[0]
        #candidate_index = np.random.randint(0, len(candidate_groups[objective_name]))
        #candidate, _ = candidate_groups[objective_name][candidate_index]
        candidate = np.random.choice(list(candidate_groups[objective_name].keys()), 1)[0]
        return candidate

    def __repr__(self):
        return "random_mandate_allocation()"

    def __str__(self):
        return self.__class__.__name__