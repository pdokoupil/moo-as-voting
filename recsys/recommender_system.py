from collections import defaultdict
from email.policy import default
import itertools
import random

import numpy as np

import time

from scipy.spatial.distance import squareform, pdist
from recsys.dataset_statistics import dataset_statistics

from recsys.recommendation_list import recommendation_list
from recsys.recommender_statistics import recommender_statistics
from contexts.historical_rankings_context import global_historical_rankings_context, user_historical_rankings_context

from multiprocessing import Pool

# TODO https://stackoverflow.com/questions/2912231/is-there-a-clever-way-to-pass-the-key-to-defaultdicts-default-factory
class defaultdict_with_key(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

class recommender_system:
    def __init__(self, voting_function_factory, supports_function_factories, filter_function, mandate_allocator, k, enable_normalization):
        self.voting_function_factory = voting_function_factory
        self.supports_function_factories = supports_function_factories
        self.filter_function = filter_function
        self.data_statistics = None
        self.context = global_historical_rankings_context(k)
        self.context.set_objective_names([factory(None).get_name() for factory in supports_function_factories])
        self.candidate_groups = defaultdict(lambda: dict()) # Mapping user to mapping of party name -> candidates
        
        self.mandate_allocator = mandate_allocator
        self.k = k
        
        self.voting_functions = defaultdict_with_key(lambda user: self.voting_function_factory(user)) # Each user has its own instance
        self.supports_functions = defaultdict_with_key(lambda user: [factory(user) for factory in self.supports_function_factories])
        self.enable_normalization = enable_normalization

    # Adds new objective to the system
    def add_new_objective(self, objective_supports):
        pass

    def _known_user(self, user):
        return user in self.context.recsys_statistics.user_to_user_id

    # Returns recommendation list for the given user
    def get_recommendations(self, user):
        if not self._known_user(user):
            self._on_new_user(user)
            
        votes = self.voting_functions[user](self.context)
        mandates = []
        per_user_support = dict() # TODO REMOVE maps step for i=1,..,k to support for this given user
        for i in range(self.k):
            # Before getting first candidate in order to solve situation where feedback was received (thus candidates update is needed) prior to getting first recommendation
            #self.candidate_groups[user] = self.measure_call(self._update_candidate_groups, user, self.data_statistics.items.difference(mandates))
            self.candidate_groups[user], extremes_per_party = self._update_candidate_groups(user, self._get_user_unseen_items(user, self.data_statistics).difference(mandates))
            #item = self.measure_call(self.mandate_allocator, self.candidate_groups[user], votes, mandates, self.k)
            item, item_support = self.mandate_allocator(self.candidate_groups[user], votes, mandates, self.k, extremes_per_party)
            per_user_support[i] = item_support # TODO REMOVE
            mandates.append(item)
            self.context.on_item_recommended(item, user)
        
        result = recommendation_list(self.k, mandates)
        self.context.on_list_recommended(result, user)
        return result, per_user_support

    def train(self, data_statistics):
        self.data_statistics = data_statistics
        self.context.set_dataset_statistics(data_statistics)
        #self.voting_functions = { user : self.voting_function_factory(user) for user in data_statistics.items }
        #self.supports_functions = { user: [factory(user) for factory in self.supports_function_factories] for user in data_statistics.users }
        
        statistics = self.measure_call(self._build_recsys_statistics)
        self.candidate_groups = self.measure_call(self._initialize_candidate_groups, data_statistics)
        
        return statistics

    def _get_user_unseen_items(self, user, data_statistics):
        if user in data_statistics.items_seen_by_user:
            return data_statistics.items.difference(data_statistics.items_seen_by_user[user])
        return data_statistics.items

    def _initialize_candidate_groups(self, data_statistics):
        # [0] to skip extremes per party
        return { user: self._update_candidate_groups(user, self._get_user_unseen_items(user, data_statistics))[0] for user in data_statistics.users }

    # TODO move to some "utils" file
    def measure_call(self, func, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        if hasattr(func, "get_name"):
            name = func.get_name()
        else:
            name = func.__name__
        print(f"Calling function: {name} took: {time.perf_counter() - start_time} seconds")
        return result

    def _build_recsys_statistics(self):
        item_to_item_id = self.measure_call(self._build_item_mapping)
        user_to_user_id = self.measure_call(self._build_user_mapping)
        rating_matrix = self.measure_call(self._build_rating_matrix, user_to_user_id, item_to_item_id)
        similarity_matrix = self.measure_call(self._build_similarity_matrix, np.transpose(rating_matrix))
        statistics = self.measure_call(recommender_statistics, rating_matrix, similarity_matrix, user_to_user_id, item_to_item_id)
        self.context.set_recsys_statistics(statistics)
        return statistics

    def _build_rating_matrix(self, user_to_user_id, item_to_item_id):
        num_users = len(self.data_statistics.users)
        num_items = len(self.data_statistics.items)

        rating_matrix = np.zeros((num_users, num_items), dtype=np.float32)
        
        for user, ratings in self.data_statistics.feedback.items():
            user_id = user_to_user_id[user]
            for item, rating in ratings.items():
                item_id = item_to_item_id[item]
                rating_matrix[user_id, item_id] = rating

        return rating_matrix

    # Based on: https://github.com/caserec/CaseRecommender/blob/master/caserec/recommenders/item_recommendation/base_item_recommendation.py
    def _build_similarity_matrix(self, features):
        similarity_matrix = np.float32(squareform(pdist(features, "cosine")))

        # Get rid of NANs
        similarity_matrix[np.isnan(similarity_matrix)] = 1.0
        
        # Transform distances to similarities
        similarity_matrix = 1.0 - similarity_matrix

        return similarity_matrix

    def _build_item_mapping(self):
        mapping = dict()
        i = 0
        for item in self.data_statistics.items:
            mapping[item] = i
            i += 1
        return mapping

    def _build_user_mapping(self):
        mapping = dict()
        i = 0
        for user in self.data_statistics.users:
            mapping[user] = i
            i += 1
        return mapping

    def predict_batched(self, users):
        ranking = []
        start_time = time.perf_counter()
        per_user_supports = defaultdict(lambda: defaultdict(lambda: list())) # TODO REMOVE
        for i, user in enumerate(users):
            if i % 100 == 0:
                print(f"Predicting for: {user}, took: {time.perf_counter() - start_time}")
                start_time = time.perf_counter()
            top_k_list, per_user_support = self.get_recommendations(user) # TODO remove per user support
            for j, party_supports in per_user_support.items(): # TODO REMOVE
                for party, support in party_supports.items():
                    per_user_supports[party][j].append(support) # TODO REMOVE

            user_id = self.context.recsys_statistics.user_to_user_id[user]
            ranking.extend([(user, item, self.context.recsys_statistics.rating_matrix[user_id, self.context.recsys_statistics.item_to_item_id[item]]) for item in top_k_list.items])

        return ranking, per_user_supports # TODO REMOVE per user support

    # Feedback should have related users&items in it
    # feedback be anything -- new user appearing, new item appearech, user interaction, etc.
    def on_receive_feedback(self, feedback):
        #TODO: invoke _on_new_item and _on_new_user accordingly
        self._update_context_with_feedback(feedback)
        pass

    # Should be invokend whenever new item appears in the system
    def _on_new_item(self):
        pass

    # Should be invoked whenever new user appears in the system
    def _on_new_user(self, user):
        assert max(self.context.recsys_statistics.user_to_user_id.values()) + 1 == len(self.context.recsys_statistics.user_to_user_id), "aa"
        self.context.recsys_statistics.user_to_user_id[user] = len(self.context.recsys_statistics.user_to_user_id)
        # Estimate the new user by average of movie ratings
        user_rating = np.zeros_like(self.context.recsys_statistics.rating_matrix[0]) #np.mean(self.context.recsys_statistics.rating_matrix, axis=0)
        self.context.recsys_statistics.rating_matrix = np.vstack([self.context.recsys_statistics.rating_matrix, user_rating])

    def _map_range(self, x, original_range, new_range):
        #assert original_range[0] <= original_range[1], f"Invalid original range: {original_range}"
        #assert new_range[0] <= new_range[1], f"Invalid new range: {new_range}"
        #if np.isclose(original_range[0], original_range[1]):
        if original_range[0] == original_range[1]:
            return (new_range[1] + new_range[0]) / 2.0
        #assert x >= original_range[0] and x <= original_range[1], f"Value x={x} must fit into the original range: {original_range}"
        result = new_range[0] + ((new_range[1] - new_range[0])/(original_range[1] - original_range[0])) * (x - original_range[0])
        #assert result >= new_range[0] and result <= new_range[1], f"New value {result} must fit into the new range: {new_range}"
        return result

    # Normalize candidates from [min, max] to [-1, 1]
    # Candidates are a list of pairs (item, support)
    def _normalize_candidates(self, candidates):
        #min_support = min(candidates, key=lambda x: x[1])[1]
        #max_support = max(candidates, key=lambda x: x[1])[1]
        min_support = np.inf
        max_support = np.NINF
        for _, support in candidates:
            if support < min_support:
                min_support = support
            if support > max_support:
                max_support = support
        return [(item, self._map_range(support, [min_support, max_support], [0.0, 1.0])) for item, support in candidates]
        
    # Important assumption is that these items do not contain items already seen by the user or already recommended to the user
    def _update_candidate_groups(self, user, items):
        candidate_groups = dict()
        extremes_per_party = defaultdict(dict)
        for support_func in self.supports_functions[user]:
            # Each support function will correspond to a single group
            # Each support function should have a name associated with it
            item_support = [(item, support_func(item, self.context)) for item in items] # Get support value for all the items
            if self.enable_normalization:
                item_support = self._normalize_candidates(item_support) # TODO turn on/off normalization, possibly merge with filter function to improve performance?
            extremes_per_party[support_func.get_name()]["max"] = max(item_support, key=lambda x: x[1])[1]
            extremes_per_party[support_func.get_name()]["min"] = min(item_support, key=lambda x: x[1])[1]
            candidate_groups[support_func.get_name()] = self.filter_function(item_support) # TODO compare performance with the one below
            # candidate_groups[support_func.get_name()] = self.filter_function({ item: support_func(item, self.context) for item in items}.items()) 
        return candidate_groups, extremes_per_party

    def _update_context_with_feedback(self, feedback):
        pass

    