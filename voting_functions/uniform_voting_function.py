class uniform_voting_function:
    def __init__(self, user, objectives, num_votes_per_party):
        self.user = user
        self.objectives = objectives
        self.num_votes_per_party = num_votes_per_party

    def __call__(self, ctx):
        return { obj.get_name(): self.num_votes_per_party for obj in self.objectives }