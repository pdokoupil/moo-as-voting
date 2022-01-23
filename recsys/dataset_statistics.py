class dataset_statistics:
    # users: list of user ids
    # items: list of item ids
    # feedback: dictionary mapping each user to dictionary with elements item: rating
    # sparsity
    # number_interactions
    # users_viewed_item: maps each item to set of users who viewed it
    # items_unobserved: items that were not observed (rated) by any user
    # items_seen_by_user: maps each user to a set of items seen (rated) by that user
    def __init__(
        self,
        users,
        items,
        feedback,
        sparsity,
        number_interactions,
        users_viewed_item,
        items_unobserved,
        items_seen_by_user,    
    ):
        self.users = users
        self.items = items
        self.feedback = feedback
        self.sparsity = sparsity
        self.number_interactions = number_interactions
        self.users_viewed_item = users_viewed_item
        self.items_unobserved = items_unobserved
        self.items_seen_by_user = items_seen_by_user
