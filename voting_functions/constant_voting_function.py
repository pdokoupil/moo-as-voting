class constant_voting_function:
    def __init__(self, user, objectives, votes):
        self.user = user
        self.objectives = objectives
        self.votes = votes
        assert len(objectives) == len(votes), "there must be a vote for each of the parties/objectives"

    def __call__(self, ctx):
        return { obj.get_name(): vote for obj, vote in zip(self.objectives, self.votes) }