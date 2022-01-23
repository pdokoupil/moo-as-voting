from collections import defaultdict
from typing import DefaultDict
from recsys.recommendation_list import recommendation_list

# This context contains current recommendation list and all the past recommendations for the given user
class user_historical_rankings_context:
    def __init__(self, user, current_list, past_lists):
        self.user = user
        self.current_list = current_list
        self.past_lists = past_lists

    # Callback that should be called whenever a new item was recommended to the given user
    def on_item_recommended(self, item):
        assert self.current_list.append_item(item), "Should never happen" # TODO

    def on_list_recommended(self, list):
        self.past_lists.append(list)
        self.current_list = recommendation_list(self.current_list.k, [])

# This context contains current recommendation list and all the past recommendations for EVERY user
class global_historical_rankings_context:
    def __init__(self, k):
        self.per_user_rankings_context = dict()
        self.num_users = 0
        self.users = set()
        self.relevant_items = defaultdict(lambda: set()) #for each user, we have a set of its relevant items
        self.objective_names = []
        self.recsys_statistics = None
        self.k = k
        self.dataset_statistics = None

    def on_item_recommended(self, item, user):
        if user not in self.per_user_rankings_context:
            ctx = user_historical_rankings_context(user, recommendation_list(self.k), [])
            ctx.on_item_recommended(item)
            self.per_user_rankings_context[user] = ctx
        else:
            self.per_user_rankings_context[user].on_item_recommended(item)

    def on_list_recommended(self, list, user):
        assert user in self.per_user_rankings_context, "We are finishing list, so on_item_recommended must have been called previously" # TODO
        self.per_user_rankings_context[user].on_list_recommended(list)

    def on_new_user(self, user):
        pass

    def on_received_feedback(self, user, feedback):
        pass

    def on_recommendation_list_full(self, user):
        assert user in self.per_user_rankings_context
        self.per_user_rankings_context[user]

    def get_current_recommendation_list_of_user(self, user):
        if user not in self.per_user_rankings_context:
            return None
        return self.per_user_rankings_context[user].current_list

    def get_current_recommendation_lists(self):
        return dict(map(lambda x: (x[0], x[1].current_list), self.per_user_rankings_context.items()))

    def add_objective_name(self, objective_name):
        self.objective_names.append(objective_name)

    def set_objective_names(self, objective_names):
        self.objective_names = objective_names

    def get_objective_names(self):
        return self.objective_names

    def set_recsys_statistics(self, recsys_statistics):
        self.recsys_statistics = recsys_statistics

    def get_recsys_statistics(self):
        return self.recsys_statistics

    def set_dataset_statistics(self, statistics):
        self.dataset_statistics = statistics
        self.num_users = len(statistics.users) # TODO update at different place
    
    def get_dataset_statistics(self):
        return self.dataset_statistics