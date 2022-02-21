from collections import defaultdict
import itertools
import pickle
import random

import numpy as np

import time
import os

from scipy.spatial.distance import squareform, pdist
from objective_functions.diversity.intra_list_diversity import intra_list_diversity
from recsys.dataset_statistics import dataset_statistics

from recsys.recommendation_list import recommendation_list
from recsys.recommender_statistics import recommender_statistics
from contexts.historical_rankings_context import global_historical_rankings_context, user_historical_rankings_context

from multiprocessing import Pool

import matplotlib.pyplot as plt

from support_functions.normalizing_marginal_gain_support_function import normalizing_marginal_gain_support_function

# TODO https://stackoverflow.com/questions/2912231/is-there-a-clever-way-to-pass-the-key-to-defaultdicts-default-factory
class defaultdict_with_key(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

class recommender_system:
    def __init__(self, voting_function_factory, supports_function_factories, filter_function, mandate_allocator, k, support_normalization_factory, shift, cache_dir, baseline_name):
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
        self.support_normalization = None if support_normalization_factory is None \
                                          else {
                                              obj_name: support_normalization_factory(
                                                  shift,
                                                  os.path.join(cache_dir, f"{support_normalization_factory.__name__}_{obj_name}_{baseline_name}.pckl"),
                                                ) for obj_name in self.context.get_objective_names()
                                            }

        self.recsys_statistics_cache_path = os.path.join(cache_dir, "recsys_statistics.pckl")

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
        extremes_per_party = dict()
        for i in range(self.k):
            # Before getting first candidate in order to solve situation where feedback was received (thus candidates update is needed) prior to getting first recommendation
            #self.candidate_groups[user] = self.measure_call(self._update_candidate_groups, user, self.data_statistics.items.difference(mandates))
            
            self.candidate_groups[user], extremes_per_party = self._update_candidate_groups(user, self._get_user_unseen_items(user, self.data_statistics).difference(mandates))
            #self.candidate_groups[user], extremes_per_party = self._update_candidate_groups_fast(user, self._get_user_unseen_items(user, self.data_statistics), mandates, extremes_per_party)
            
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
        
        #self.candidate_groups = self.measure_call(self._initialize_candidate_groups, data_statistics) # TODO verify not needed

        if self.support_normalization:
            print("Training support (objective) normalization")
            self.measure_call(self._train_normalization, data_statistics)

        return statistics

    def _get_user_unseen_items(self, user, data_statistics):
        if user in data_statistics.items_seen_by_user:
            return data_statistics.items.difference(data_statistics.items_seen_by_user[user])
        return data_statistics.items

    def _train_normalization(self, data_statistics):
        item_combinations = list(itertools.combinations(data_statistics.items, 1))
        item_pair_combinations = list(itertools.combinations(data_statistics.items, 2))
        sampled_users = random.sample(data_statistics.users, len(data_statistics.users))
        
        obj_names = self.context.get_objective_names()
        
        objectives = {}
        
        for obj_idx, obj_name in enumerate(obj_names):
            objectives[obj_name] = (obj_idx, self.supports_functions[sampled_users[0]][obj_idx].objective)

        for obj_name, (obj_idx, obj) in objectives.items():
            print(f"Training {obj_name}")
            start_time = time.perf_counter()
            if hasattr(obj, "user"): # depends on user
                self.support_normalization[obj_name].train(
                    sampled_users,
                    item_combinations,
                    lambda user, obj_idx=obj_idx: self.supports_functions[user][obj_idx].objective,
                    self.context
                )
            elif type(obj) is intra_list_diversity:
                print("Diversity")
                self.support_normalization[obj_name].train({}, item_pair_combinations, obj, self.context)
            else:
                print("Other than diversity")
                self.support_normalization[obj_name].train({}, item_combinations, obj, self.context)
            print(f"Obj: {obj_name} took: {time.perf_counter() - start_time}")

        self._set_normalization(data_statistics.users)

    def _set_normalization(self, users):
        # Inject the normalization into the support function
        start_time = time.perf_counter()
        for user in users:
            for support_normalization, support_function in zip(self.support_normalization.values(), self.supports_functions[user]):
                support_function.set_normalization(support_normalization)

        print(f"Setting normalization took: {time.perf_counter() - start_time}")

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
        statistics = None
        if os.path.exists(self.recsys_statistics_cache_path):
            with open(self.recsys_statistics_cache_path, 'rb') as f:
                print(f"Loading recsys statistics from: {self.recsys_statistics_cache_path}")
                statistics = pickle.load(f)
        else:
            item_to_item_id = self.measure_call(self._build_item_mapping)
            user_to_user_id = self.measure_call(self._build_user_mapping)
            rating_matrix = self.measure_call(self._build_rating_matrix, user_to_user_id, item_to_item_id)
            similarity_matrix = self.measure_call(self._build_similarity_matrix, np.transpose(rating_matrix))
            statistics = self.measure_call(recommender_statistics, rating_matrix, similarity_matrix, user_to_user_id, item_to_item_id)

            with open(self.recsys_statistics_cache_path, 'wb') as f:
                print(f"Saving recsys statistics to: {self.recsys_statistics_cache_path}")
                pickle.dump(statistics, f)
        
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
        self._set_normalization(users) # Set normalization for previously unknown users

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

    def _normalize_candidates(self, candidates, support_normalization):
        #candidate_items, candidate_supports = zip(*candidates)
        #support_normalization.train(candidate_supports)
        #return list(zip(candidate_items, support_normalization.predict_batched(candidate_supports)))
        return candidates

    # Important assumption is that these items do not contain items already seen by the user or already recommended to the user
    def _update_candidate_groups(self, user, items):
        candidate_groups = dict()
        extremes_per_party = defaultdict(dict)
        for support_func in self.supports_functions[user]:
            sup_name = support_func.get_name()
            # Each support function will correspond to a single group
            # Each support function should have a name associated with it
            item_support = [(item, support_func(item, self.context)) for item in items] # Get support value for all the items
            extremes_per_party[sup_name]["max"] = max(item_support, key=lambda x: x[1])[1]
            extremes_per_party[sup_name]["min"] = min(item_support, key=lambda x: x[1])[1]
            candidate_groups[sup_name] = self.filter_function(item_support) # TODO compare performance with the one below
            # candidate_groups[support_func.get_name()] = self.filter_function({ item: support_func(item, self.context) for item in items}.items()) 
        return candidate_groups, extremes_per_party

    # TODO: this decreases generality of the algorithm
    def _update_candidate_groups_fast(self, user, items, mandates, extremes_per_party):
        candidate_groups = dict()
        if user in self.candidate_groups:
            for support_func in self.supports_functions[user]:
                sup_name = support_func.get_name()
                # if len(mandates) > 0 and mandates[-1] in self.candidate_groups[user][sup_name]:
                #     del self.candidate_groups[user][sup_name][mandates[-1]]
                if sup_name == "intra_list_diversity":
                    item_support = [(item, support_func(item, self.context)) for item in items.difference(mandates)] # Get support value for all the items
                    extremes_per_party[sup_name]["max"] = max(item_support, key=lambda x: x[1])[1]
                    extremes_per_party[sup_name]["min"] = min(item_support, key=lambda x: x[1])[1]
                    candidate_groups[sup_name] = self.filter_function(item_support)
                else:
                    candidate_groups[sup_name] = self.candidate_groups[user][sup_name]
            return candidate_groups, extremes_per_party
        else:
            return self._update_candidate_groups(user, items.difference(mandates))

    def _update_context_with_feedback(self, feedback):
        pass

    