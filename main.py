import argparse
from collections import defaultdict
import os
from time import perf_counter

from caserec.recommenders.item_recommendation.itemknn import ItemKNN
from caserec.utils.cross_validation import CrossValidation
from objective_functions.diversity.intra_list_diversity import intra_list_diversity
from recsys.recommendation_list import recommendation_list

import itertools

from scipy.spatial.distance import squareform, pdist

from recsys.recommender_system import recommender_system
from recsys.dataset_statistics import dataset_statistics
from recsys.recommender_statistics import recommender_statistics
from objective_functions.relevance.average_precision import average_precision
from objective_functions.relevance.mean_average_precision import mean_average_precision
from objective_functions.relevance.rating_based_relevance import rating_based_relevance
from objective_functions.relevance.precision import precision
from objective_functions.diversity.expected_intra_list_diversity import expected_intra_list_diversity
from objective_functions.novelty.expected_popularity_complement import expected_popularity_complement
from objective_functions.novelty.popularity_complement import popularity_complement

from support_functions.normalization.cdf import SHIFT, cdf
from support_functions.normalization.min_max_scaling import min_max_scaling
from support_functions.normalization.standardization import standardization
from support_functions.normalization.robust_scaling import robust_scaling

from filter_functions.top_k_filter_function import top_k_filter_function
from support_functions.marginal_gain_support_function import marginal_gain_support_function
from support_functions.normalizing_marginal_gain_support_function import normalizing_marginal_gain_support_function
from support_functions.relative_gain_support_function import relative_gain_support_function
from voting_functions.constant_voting_function import constant_voting_function
from voting_functions.uniform_voting_function import uniform_voting_function
from mandate_allocation.sainte_lague_method import sainte_lague_method
from mandate_allocation.fai_strategy import fai_strategy
from mandate_allocation.exactly_proportional_fuzzy_dhondt import exactly_proportional_fuzzy_dhondt
from mandate_allocation.exactly_proportional_fuzzy_dhondt_2 import exactly_proportional_fuzzy_dhondt_2
from mandate_allocation.exactly_proportional_fai_strategy import exactly_proportional_fai_strategy
from mandate_allocation.random_mandate_allocation import random_mandate_allocation

import random

import math
import time
import numpy as np
import matplotlib.pyplot as plt

import copy

from caserec.recommenders.rating_prediction.itemknn import ItemKNN as RatingItemKNN
from caserec.recommenders.rating_prediction.userknn import UserKNN as RatingUserKNN
from caserec.utils.process_data import ReadFile, WriteFile
from caserec.evaluation.rating_prediction import RatingPredictionEvaluation
from caserec.recommenders.rating_prediction.svd import SVD

def calculate_diversity(top_k, recsys_statistics):
    d = 0.0
    for i in range(len(top_k)):
        i_index = recsys_statistics.item_to_item_id[top_k[i][1]]
        for j in range(len(top_k)):
            if i != j:
                j_index = recsys_statistics.item_to_item_id[top_k[j][1]]
                d += (1.0 - recsys_statistics.similarity_matrix[i_index, j_index])
    return d / (len(top_k) * (len(top_k) - 1))

def calculate_novelty(top_k, recsys_statistics):
    n = 0.0
    num_users = recsys_statistics.rating_matrix.shape[0]
    c = 0.0
    for idx, i in enumerate(top_k):
        item = i[1]
        u_i = np.count_nonzero(recsys_statistics.rating_matrix[:, recsys_statistics.item_to_item_id[item]])
        popularity = u_i / num_users
        c += np.power(0.85, idx)
        n += np.power(0.85, idx) * (1.0 - popularity)
    return (1.0 / c) * n

def trim_total_ranking(ranking, total_ranking_size):
    return ranking[:total_ranking_size]

# Calculates diversity of the given recommender
def evaluate_diversity(args, ranking, recsys_statistics):
    assert len(ranking) % args.ranking_size == 0

    total_diversity = 0.0
    n = 0

    for i in range(0, len(ranking), args.ranking_size):
        top_k_per_user = ranking[i:i+args.ranking_size]
        d = calculate_diversity(top_k_per_user, recsys_statistics)
        assert d <= 1.0 and d >= 0.0
        total_diversity += d
        n += 1

    return total_diversity / n

# Calculates novelty of the given recommender
def evaluate_novelty(args, ranking, recsys_statistics):
    assert len(ranking) % args.ranking_size == 0

    total_novelty = 0.0
    n = 0

    for i in range(0, len(ranking), args.ranking_size):
        top_k_per_user = ranking[i:i+args.ranking_size]
        novelty = calculate_novelty(top_k_per_user, recsys_statistics)
        assert novelty <= 1.0 and novelty >= 0.0
        total_novelty += novelty
        n += 1

    return total_novelty / n

def _precision_at_k(single_ranking, test_dataset_statistics, k):
    user = single_ranking[0][0]
    k = min(k, len(single_ranking))
    p = 0.0
    for _, item, _ in single_ranking[:k]:
        #if user in test_dataset_statistics.feedback and item in test_dataset_statistics.feedback[user] and test_dataset_statistics.feedback[user][item] >= 3.0:
        if _is_relevant(item, user, test_dataset_statistics):
            p += 1

    assert p / k <= 1.0
    return p / k
    
# https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
# https://machinelearninginterview.com/topics/machine-learning/mapatk_evaluation_metric_for_ranking/
def _average_precision_at_k(single_ranking, test_dataset_statistics, k):
    k = min(k, len(single_ranking))
    user = single_ranking[0][0]
    p = 0.0
    n = 0.0
    for i in range(1, k + 1):
        rel = float(_is_relevant(single_ranking[i - 1][1], user, test_dataset_statistics))
        n += rel
        p += _precision_at_k(single_ranking, test_dataset_statistics, i) * rel
        
    #if n == 0.0:
        #return 0.0
    #return p / n
    m = _num_relevant_items(user, test_dataset_statistics)
    assert n <= m
    if m == 0:
        return 0
    ap = p / min(m, k)
    assert ap <= 1.0
    return p / min(m, k)

def _mean_average_precision(args, ranking, test_dataset_statistics, k):
    p = 0.0
    users = 0
    for i in range(0, len(ranking), args.ranking_size):
        single_ranking = ranking[i:i+args.ranking_size]
        users += 1
        p += _average_precision_at_k(single_ranking, test_dataset_statistics, k)
    assert p / users <= 1.0
    return p / users

def _is_relevant(item, user, test_dataset_statistics):
    if user in test_dataset_statistics.feedback and item in test_dataset_statistics.feedback[user]: # and test_dataset_statistics.feedback[user][item] >= 5.0:
        return True
    return False
    #return user in test_dataset_statistics.items_seen_by_user and item in test_dataset_statistics.items_seen_by_user[user]

def _num_relevant_items(user, test_dataset_statistics):
    num_relevant = 0
    if user in test_dataset_statistics.items_seen_by_user:
        for item in test_dataset_statistics.items_seen_by_user[user]:
            if _is_relevant(item, user, test_dataset_statistics):
                num_relevant += 1
    return num_relevant

def evaluate_map(args, ranking, test_dataset_statistics):
    return _mean_average_precision(args, ranking, test_dataset_statistics, args.ranking_size)


# Calculates diversity, all default metrics and other needed metrics
def custom_evaluate(args, ranking, recsys_statistics, test_dataset_statistics, normalized_ranking=None):
    diversity = evaluate_diversity(args, ranking, recsys_statistics)
    novelty = evaluate_novelty(args, ranking, recsys_statistics)
    map_value = evaluate_map(args, ranking, test_dataset_statistics)
    print(f"DIVERSITY: {diversity}")
    print(f"NOVELTY: {novelty}")
    print(f"MAP@10: {map_value}")
    print(f"Precision@10: {sum([_precision_at_k(ranking[i:i+args.ranking_size], test_dataset_statistics, args.ranking_size) for i in range(0, len(ranking), args.ranking_size)]) / (len(ranking) / args.ranking_size)}")
    results = {
        "diversity": diversity,
        "novelty": novelty,
        "map": map_value,
        "per-user-diversity": [calculate_diversity(ranking[i:i+args.ranking_size], recsys_statistics) for i in range(0, len(ranking), args.ranking_size)],
        "per-user-novelty": [calculate_novelty(ranking[i:i+args.ranking_size], recsys_statistics) for i in range(0, len(ranking), args.ranking_size)]
    }
    if normalized_ranking:
        mean_estimated_rating = np.mean([r[2] for r in normalized_ranking]) # Average over all users
        per_user_mean_estimated_rating = [np.mean(list(map(lambda x: x[2], normalized_ranking[i:i+args.ranking_size]))) for i in range(0, len(normalized_ranking), args.ranking_size)]
        print(f"MEAN ESTIMATED RATING: {mean_estimated_rating}")
        results["mer"] = mean_estimated_rating
        results["per-user-mer"] = per_user_mean_estimated_rating
    
    return results
    # for metric, value in recommender.evaluation_results.items():
    #     print(f"{metric} = {value}")

def custom_evaluate_voting(args, ranking, recsys_statistics,
    test_dataset_statistics, normalized_ranking,
    voting_recommender, rating_matrix, similarity_matrix):
    ctx = voting_recommender.context
    results = custom_evaluate(args, ranking, recsys_statistics, test_dataset_statistics, normalized_ranking)
    
    div = intra_list_diversity(similarity_matrix)
    nov = popularity_complement()
    
    total_novelty = 0.0
    total_diversity = 0.0
    n = 0

    per_user_diversity = []
    per_user_novelty = []

    for i in range(0, len(ranking), args.ranking_size):
        top_k_per_user = recommendation_list(args.ranking_size, [i for u, i, r in ranking[i:i+args.ranking_size]])

        diversity = div(top_k_per_user, ctx)
        novelty = nov(top_k_per_user, ctx)

        per_user_diversity.append(diversity)
        per_user_novelty.append(novelty)

        total_diversity += diversity
        total_novelty += novelty
        n += 1
    
    total_diversity = total_diversity / n
    total_novelty = total_novelty / n

    print(f"DIVERSITY2: {total_diversity}")
    print(f"NOVELTY2: {total_novelty}")
    results["diversity"] = total_diversity
    results["novelty"] = total_novelty
    results["per-user-diversity"] = per_user_diversity
    results["per-user-novelty"] = per_user_novelty


    print("-------------------")
    normalizations = voting_recommender.support_normalization

    mer_norm = normalizations[rating_based_relevance.__name__]
    div_norm = normalizations[intra_list_diversity.__name__]
    nov_norm = normalizations[popularity_complement.__name__]

    normalized_mer = 0.0
    normalized_diversity = 0.0
    normalized_novelty = 0.0
    
    normalized_per_user_mer = []
    normalized_per_user_diversity = []
    normalized_per_user_novelty = []

    # Calculate normalized MER per user
    n = 0
    for i in range(0, len(ranking), args.ranking_size):
        top_k_per_user = recommendation_list(ctx.k, [x[1] for x in ranking[i:i+args.ranking_size]])
        user, _, _ = ranking[i]

        rel = rating_based_relevance(user, rating_matrix)
        normalized_per_user_mer.append(mer_norm.predict(np.mean([rel(recommendation_list(ctx.k, [i]), ctx) for i in top_k_per_user.items]), user))
        #normalized_per_user_mer.append(np.mean([mer_norm.predict(rel(recommendation_list(ctx.k, [i]), ctx)) for i in top_k_per_user.items]))
        normalized_mer += normalized_per_user_mer[-1]

        cmbs = list(itertools.combinations(top_k_per_user.items, 2))
        normalized_per_user_diversity.append(div_norm.predict(np.mean([div(recommendation_list(ctx.k, [i, j]), ctx) for i, j in cmbs]), user))
        #normalized_per_user_diversity.append(np.mean([div_norm.predict(div(recommendation_list(ctx.k, [i, j]), ctx)) for i, j in cmbs]))
        normalized_diversity += normalized_per_user_diversity[-1]


        normalized_per_user_novelty.append(nov_norm.predict(np.mean([nov(recommendation_list(ctx.k, [i]), ctx) for i in top_k_per_user.items]), user))
        #normalized_per_user_novelty.append(np.mean([nov_norm.predict(nov(recommendation_list(ctx.k, [i]), ctx)) for i in top_k_per_user.items]))
        normalized_novelty += normalized_per_user_novelty[-1]

        n += 1

    normalized_mer /= n
    normalized_diversity /= n
    normalized_novelty /= n

    print(f"Normalized MER: {normalized_mer}")
    print(f"Normalized DIVERSITY2: {normalized_diversity}")
    print(f"Normalized NOVELTY2: {normalized_novelty}")
    
    results["normalized-mer"] = normalized_mer
    results["normalized-diversity"] = normalized_diversity
    results["normalized-novelty"] = normalized_novelty
    
    #results["normalized-per-user-mer"] = [np.mean(list(map(lambda x: mer_norm.predict(x[2]), normalized_ranking[i:i+args.ranking_size]))) for i in range(0, len(normalized_ranking), args.ranking_size)]
    #results["normalized-per-user-diversity"] = [div_norm.predict(x) for x in results["per-user-diversity"]]
    #results["normalized-per-user-novelty"] = [nov_norm.predict(x) for x in results["per-user-novelty"]]
    results["normalized-per-user-mer"] = normalized_per_user_mer
    results["normalized-per-user-diversity"] = normalized_per_user_diversity
    results["normalized-per-user-novelty"] = normalized_per_user_novelty

    # Plot histogram of normalized, per-user, sum-to-1 objectives
    normalized_sum_to_one_per_user_mer = []
    normalized_sum_to_one_per_user_diversity = []
    normalized_sum_to_one_per_user_novelty = []
    for mer, div, nov in zip(normalized_per_user_mer, normalized_per_user_diversity, normalized_per_user_novelty):
        s = np.abs(mer) + np.abs(div) + np.abs(nov)
        normalized_sum_to_one_per_user_mer.append(mer / s)
        normalized_sum_to_one_per_user_diversity.append(div / s)
        normalized_sum_to_one_per_user_novelty.append(nov / s)

    plt.hist(normalized_sum_to_one_per_user_mer)
    plt.title(f"Normalized, sum-to-one, per-user MER")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path_prefix, f"mer_hist_{args.experiment_name}.png"))
    plt.close()

    plt.hist(normalized_sum_to_one_per_user_diversity)
    plt.title(f"Normalized, sum-to-one, per-user Diversity")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path_prefix, f"div_hist_{args.experiment_name}.png"))
    plt.close()

    plt.hist(normalized_sum_to_one_per_user_novelty)
    plt.title(f"Normalized, sum-to-one, per-user Novelty")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path_prefix, f"nov_hist_{args.experiment_name}.png"))
    plt.close()


    # Print sum-to-1 results
    s = normalized_mer + normalized_diversity + normalized_novelty
    print(f"Sum-To-1 Normalized MER: {normalized_mer / s}")
    print(f"Sum-To-1 Normalized DIVERSITY2: {normalized_diversity / s}")
    print(f"Sum-To-1 Normalized NOVELTY2: {normalized_novelty / s}")

    return results

def get_voting_recommender(objective_factories, args):

    voting_function_factory = \
        lambda user: constant_voting_function(
            user, 
            [obj_factory(user) for obj_factory in objective_factories],
            args.objective_weights
        )
        # lambda user: uniform_voting_function(
        #     user, 
        #     [obj_factory(user) for obj_factory in objective_factories],
        #     10 * len(objective_factories)
        # )
        
    # https://stackoverflow.com/questions/32595586/in-python-why-do-lambdas-in-list-comprehensions-overwrite-themselves-in-retrosp
    supports_function_factories = [lambda user, obj_factory=obj_factory: args.support_function(obj_factory(user), user) for obj_factory in objective_factories]
    filter_function = top_k_filter_function(2000)
    mandate_allocator = args.mandate_allocation() #fai_strategy() # sainte_lague_method() # random_mandate_allocation()

    recommender = recommender_system(
        voting_function_factory,
        supports_function_factories,
        filter_function,
        mandate_allocator,
        args.ranking_size,
        args.support_normalization
    )
    return recommender

def get_separator():
    return "".join(['='] * 30)

def dataset_to_statistics(dataset):
    return dataset_statistics(
        set(dataset["users"]),
        set(dataset["items"]),
        dataset["feedback"],
        dataset["sparsity"],
        dataset["number_interactions"],
        dataset["users_viewed_item"],
        dataset["items_unobserved"],
        dataset["items_seen_by_user"]
    )

def merge_statistics(train_statistics, test_statistics):
    train_statistics = copy.deepcopy(train_statistics)
    test_statistics = copy.deepcopy(test_statistics)
    users = train_statistics.users.union(test_statistics.users)
    items = train_statistics.items.union(test_statistics.items)
    feedback = train_statistics.feedback
    num_interactions = train_statistics.number_interactions
    for user, ratings in test_statistics.feedback.items():
        if user not in feedback:
            feedback[user] = dict()
        for movie, rating in ratings.items():
            feedback[user][movie] = rating
            num_interactions += 1
    sparsity = num_interactions / (len(items) * len(users))
    items_seen_by_user = train_statistics.items_seen_by_user
    for user, items in test_statistics.items_seen_by_user.items():
        if user not in items_seen_by_user:
            items_seen_by_user[user] = set()
        items_seen_by_user[user] = items_seen_by_user[user].union(items)
    return dataset_statistics(
        users,
        items,
        feedback,
        sparsity,
        num_interactions,
        None,
        None,
        items_seen_by_user
    )

def tmp_evaluate(baseline, dataset):
    prec = 0.0
    n = 0
    for user in dataset.users:
        ranking = baseline.predict_scores(user, user - 1)
        prec += _precision_at_k(ranking, dataset, -1)
        n += 1
    print(f"Precision@{5} = {prec / n}")

# def evaluate_precision(ranking, dataset):
#     prec = 0.0
#     n = 0
#     for i in range(0, len(ranking), args.ranking_size):
#         rank = ranking[i:i+args.ranking_size]
#         prec += _precision_at_k(rank, dataset, args.ranking_size)
#         n += 1
#     print(f"Prec@{args.ranking_size} = {prec / n}")

def lightfm_data_to_statistics(data_dict):
    users = set()
    items = set()
    num_interactions = 0
    feedback = dict()
    items_seen_by_user = dict()
    for (user, item), rating in data_dict.items():
        users.add(user)
        items.add(item)
        num_interactions += 1
        if user not in items_seen_by_user:
            items_seen_by_user[user] = set()
        items_seen_by_user[user].add(item)
        
        if user not in feedback:
            feedback[user] = dict()
        feedback[user][item] = rating

    return dataset_statistics(
        users,
        items,
        feedback,
        num_interactions / (len(users) * len(items)),
        num_interactions,
        None,
        None,
        items_seen_by_user
    )


def lightfm_comparison():
    from lightfm import LightFM
    from lightfm.datasets import fetch_movielens
    from lightfm.evaluation import precision_at_k
    
    def save_lightfm_data(data, file_path):
        with open(file_path, "w+") as f:
            for (user, item), rating in data.todok().items():
                print(f"{user}\t{item}\t{rating}", file=f)
    
    data = fetch_movielens()
    model = LightFM(loss="warp")
    model.fit(data["train"], epochs=5, num_threads=4)
    print(f"Lightfm precision@5: {precision_at_k(model, data['test'], k=5).mean()}")

    save_lightfm_data(data["train"], "lightfm_train_new.dat")
    save_lightfm_data(data["test"], "lightfm_test_new.dat")


    return lightfm_data_to_statistics(data["train"].todok()), lightfm_data_to_statistics(data["test"].todok())

def validate_statistics(data_statistics, name):
    print(f"Validating statistics {name}")
    for user, ratings in data_statistics.feedback.items():
        for item, rating in ratings.items():
            assert item in data_statistics.items, f"error validation check item {item}"

def validate_dataset(data_statistics):
    print("Validating dataset")
    for user, ratings in data_statistics["feedback"].items():
        for item, rating in ratings.items():
            assert item in data_statistics["items"], f"error validation check item {item}"

def norm(x, old_min, old_max, new_min, new_max):
    scale = (new_max - new_min) / (old_max - old_min)
    return x * scale + (new_min - old_min * scale)

def normalize_recommendation_ranking(ranking, min_rating, max_rating):
    # Zeros will map to zeros
    # the rest will be mapped to [min_rating, max_rating]
    
    
    old_min, old_max = min(ranking, key=lambda x: x[2])[2], max(ranking, key=lambda x: x[2])[2]

    normalized = []
    for u, i, r in ranking:
        if r > 0.0:
            r = norm(r, old_min, old_max, min_rating, max_rating)
        assert r == 0.0 or (r >= min_rating and r <= max_rating), f"rating {r} is not normalized to [{min_rating}, {max_rating}]"
        normalized.append((u, i, r))

    return normalized

# Extends rating matrix based on ranking
def project_ranking_into_rating_matrix(ranking, recsys_statistics):
    rating_matrix_copy = np.zeros_like(recsys_statistics.rating_matrix) #recsys_statistics.rating_matrix.copy()
    for u, i, r in ranking:
        rating_matrix_copy[recsys_statistics.user_to_user_id[u], recsys_statistics.item_to_item_id[i]] = r
    return rating_matrix_copy

# Extends rating matrix based on similarities
def extend_rating_matrix(recsys_statistics):
    rating_matrix_copy = np.zeros_like(recsys_statistics.rating_matrix) #recsys_statistics.rating_matrix.copy() # Otherwise we tend to recommend known items
    
    n_items = rating_matrix_copy.shape[1]
    k_neighbors = int(np.sqrt(n_items))

    assert np.all(recsys_statistics.similarity_matrix >= 0.0) and np.all(recsys_statistics.similarity_matrix <= 1.0)

    # Go over all users
    for user, user_id in recsys_statistics.user_to_user_id.items():
        u_list = np.flatnonzero(recsys_statistics.rating_matrix[user_id] == 0)
        seen_items_id = np.flatnonzero(recsys_statistics.rating_matrix[user_id])
        seen_items_ratings = np.take(recsys_statistics.rating_matrix[user_id], seen_items_id)
        
        # For each user take all unseen items
        for item_id in u_list:
            # Get similarities between item being predicted and all other user's seen items
            seen_items_similarities = np.take(recsys_statistics.similarity_matrix[item_id], seen_items_id)
            most_similar_indices = np.argsort(-seen_items_similarities)

            most_similar_similarities = np.take(seen_items_similarities, most_similar_indices)
            most_similar_ratings = np.take(seen_items_ratings, most_similar_indices)

            # Sum the similarities for top k items
            similarities_weighted_sum = np.sum(most_similar_similarities[:k_neighbors] * most_similar_ratings[:k_neighbors]) # * 

            # Predict the rating based on using top-k similar items
            rating_matrix_copy[user_id, item_id] = similarities_weighted_sum / k_neighbors
            
    original_ratings_nonzero = np.flatnonzero(recsys_statistics.rating_matrix)
    rating_matrix_copy = norm(rating_matrix_copy, rating_matrix_copy.min(), rating_matrix_copy.max(), 1, 5)
    np.put(rating_matrix_copy, original_ratings_nonzero, np.take(recsys_statistics.rating_matrix, original_ratings_nonzero))
    return rating_matrix_copy

def get_baseline(args):
    print(f"########### 3. Baseline with {args.train_fold_path} train fold and {args.test_fold_path} test fold and CUSTOM evaluation ###########")
    baseline = ItemKNN(args.train_fold_path)
    baseline.compute(verbose=False)
    test_set = ReadFile(args.test_fold_path).read()
    train_set_statistics, test_set_statistics = dataset_to_statistics(baseline.train_set), dataset_to_statistics(test_set)
    print("Custom evaluate on normalized ranking from item recommendation ItemKNN")
    normalized_ranking = normalize_recommendation_ranking(baseline.ranking, 1.0, 5.0)
    recsys_statistics = recommender_statistics(baseline.matrix, baseline.si_matrix, baseline.user_to_user_id, baseline.item_to_item_id)
    data_statistics = test_set_statistics #merge_statistics(train_set_statistics, test_set_statistics)
    custom_evaluate(
        args,
        baseline.ranking, #trim_total_ranking(normalized_ranking, 50 * args.ranking_size),
        recsys_statistics,
        data_statistics,
        normalized_ranking
    )
    extended_rating_matrix = extend_rating_matrix(recsys_statistics) #project_ranking_into_rating_matrix(normalized_ranking, recsys_statistics)
    # print("########### 4. Baseline with merged train+test (Case recommender folds) statistics for evaluation ###########")
    # tmp_evaluate(baseline, merge_statistics(train_set_statistics, test_set_statistics))
    # evaluate_precision(baseline.ranking, merge_statistics(train_set_statistics, test_set_statistics))
    
    # print("########### 5. Baseline (trained on Case recommender folds) with merged train+test (Lightfm folds) statistics for evaluation ###########")
    # tmp_evaluate(baseline, merge_statistics(train, test))
    # evaluate_precision(baseline.ranking, merge_statistics(train, test))

    # print("########### 5+. Baseline (trained on lightfm folds) with merged train+test (Lightfm folds) statistics for evaluation ###########")
    # evaluate_precision(lightfm_baseline.ranking, merge_statistics(train, test))
    
    # def normalize_ranking(ranking):
    #     max_acc_rating = max(ranking, key=lambda x: x[2])[2]
    #     min_acc_rating = min(ranking, key=lambda x: x[2])[2]
    #     def rescale(x, initial_range, new_range):
    #         if initial_range[1] - initial_range[0] == 0:
    #             if x != new_range[0] and x != new_range[1]:
    #                 return new_range[0]
    #             else:
    #                 return x
    #         normalized = (x - initial_range[0]) / (initial_range[1] - initial_range[0])
    #         return (normalized * (new_range[1] - new_range[0])) + new_range[0]
    #     assert min_acc_rating >= 0.0
    #     return list(map(lambda x: (x[0], x[1], rescale(x[2], (min_acc_rating, max_acc_rating), (0.0, 5.0))), ranking))

    # def run_rating_item_knn(recommender, train_statistics, test_statistics, recsys_statistics):
    
    #     per_user_predictions = defaultdict(lambda: [])
    #     for u, i, r in recommender.predictions:
    #         per_user_predictions[u].append((i, r))

    #     ranking = []
    #     for u, predictions in per_user_predictions.items():
    #         sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    #         sorted_predictions = sorted_predictions[:10]
    #         assert len(sorted_predictions) == 10
    #         ranking.extend(list(map(lambda x: (u, x[0], x[1]), sorted_predictions)))

    #     merged_statistics = merge_statistics(train_set_statistics, test_set_statistics)

    #     print("Custom evaluate on non-normalized ranking:")
    #     normalized_ranking = normalize_ranking(ranking)
    #     custom_evaluate(
    #         ranking,
    #         recsys_statistics,
    #         merged_statistics
    #     )

    #     print("Custom evaluate on normalized ranking")
    #     custom_evaluate(
    #         normalized_ranking,
    #         recsys_statistics,
    #         merged_statistics
    #     )

    #     evaluate_precision(normalized_ranking, merge_statistics(train_statistics, test_statistics))

    # Takes rating predictions and fills them into (partially-filled) rating matrix
    



    # print("########### 6. RatingItemKNN with Case recommender folds ###########")
    # baseline = RatingItemKNN(train_fold_path, as_similar_first=True)
    # baseline.compute()
    # baseline_recsys_statistics = recommender_statistics(baseline.matrix, baseline.si_matrix, baseline.user_to_user_id, baseline.item_to_item_id)
    # run_rating_item_knn(baseline, train_set_statistics, test_set_statistics, baseline_recsys_statistics)
    
    # print("########### 7. RatingItemKNN with Lightfm folds ###########")
    # lightfm_baseline = RatingItemKNN("lightfm_train.dat")
    # lightfm_baseline.compute()
    # run_rating_item_knn(lightfm_baseline, train, test)

    # def calculate_user_overlap(train_statistics, test_statistics):
    #     return {
    #         "train_users": len(train_statistics.users),
    #         "test_users": len(test_statistics.users),
    #         "total_users": len(train_statistics.users.union(test_statistics.users)),
    #         "users_overlap": len(train_statistics.users.intersection(test_statistics.users))
    #     }

    # print(f"Lightfm folds user overlap: {calculate_user_overlap(train, test)}")
    # print(f"Case recommender folds user overlap: {calculate_user_overlap(train_set_statistics, test_set_statistics)}")

    # def fill_rating_matrix(rating_predictions, rating_matrix, recsys_statistics):
    #     filled_rating_matrix = rating_matrix.copy()
    #     for u, i, r in rating_predictions:
    #         filled_rating_matrix[recsys_statistics.user_to_user_id[u], recsys_statistics.item_to_item_id[i]] = r
    #     return filled_rating_matrix

    # Return some statistics for the selected basel

    extended_similarity_matrix = np.float32(squareform(pdist(baseline.matrix.T, "cosine")))
    #extended_similarity_matrix = np.float32(squareform(pdist(extended_rating_matrix.T, "cosine")))
    extended_similarity_matrix[np.isnan(extended_similarity_matrix)] = 1.0
    extended_similarity_matrix = 1.0 - extended_similarity_matrix

    return train_set_statistics, test_set_statistics, extended_rating_matrix, extended_similarity_matrix
    
def voting_recommendation(args):
    
    print(get_separator())
    print(get_separator())
    print("Voting case")
    print(get_separator())
    print(get_separator())
    train, test, filled_rating_matrix, filled_similarity_matrix = get_baseline(args)
    # TODO do rating prediction and update the matrix below (for the unseen values inside result of rating prediction)
    #for u, i, r in recommender.predictions:
    #    rating_matrix[recommender.user_to_user_id[u], recommender.item_to_item_id[i]] = r
    

    # Normalize from [1, 5] to [0, 1]
    filled_rating_matrix = (filled_rating_matrix - 1.0) / 4.0
    objective_factories = [
        lambda user: rating_based_relevance(user, filled_rating_matrix),
        lambda _: intra_list_diversity(filled_similarity_matrix),
        lambda _: popularity_complement() #expected_popularity_complement()
    ]
    voting = get_voting_recommender(objective_factories, args)
    print("Starting training of voting recommender")
    recsys_statistics = voting.train(train) # Trains the recommender
    print("Predicting with voting recommender")

    def take_users(users, n):
        if n < 0:
            return users
        return set(list(users)[:n])

    ranking, per_user_supports = voting.predict_batched(take_users(test.users, -1)) #voting.predict_batched(list(test.users)[:50]) # Generates ranking for all the users in the test dataset
    # Add ratings to the voting (as estimated by the base recommender) because the ratings in ranking come from a rating matrix which contained only known interactions + those estimated FOR UKNOWN users (i.e. mostly zeros everywhere)
    extended_ranking = []
    for u, i, r in ranking:
        if u in train.items_seen_by_user:
            assert i not in train.items_seen_by_user[u], "We should predict only unseen items"
        u_id = recsys_statistics.user_to_user_id[u]
        i_id = recsys_statistics.item_to_item_id[i]
        if u_id < filled_rating_matrix.shape[0] and i_id < filled_rating_matrix.shape[1]:
            extended_ranking.append((u, i, filled_rating_matrix[u_id, i_id]))
        else:
            extended_ranking.append((u, i, r))

    ranking = extended_ranking
    
    print("Starting evaluation of voting recommender")
    normalized_ranking = ranking #normalize_recommendation_ranking(ranking, 1.0, 5.0)
    data_statistics = test #merge_statistics(train, test)
    #results = custom_evaluate(args, ranking, recsys_statistics, data_statistics, normalized_ranking)
    results = custom_evaluate_voting(args, ranking, recsys_statistics, data_statistics, normalized_ranking, voting, filled_rating_matrix, filled_similarity_matrix)

    averaged_supports = defaultdict(lambda: dict()) # TODO REMOVE
    for party, values in per_user_supports.items():
        for step, supports in values.items():
            averaged_supports[party][step] = np.mean(supports)
        
        plt.scatter(averaged_supports[party].keys(), averaged_supports[party].values())
        plt.xticks(list(range(args.ranking_size)))
        plt.title(f"Avg. support {party}")
        for x, y in averaged_supports[party].items():
            plt.annotate(round(y, 2), (x, y))
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path_prefix, f"avg_{party}_{args.experiment_name}.png"))
        plt.close()

        plt.boxplot([values[i] for i in range(args.ranking_size)])
        plt.xticks([i + 1 for i in range(args.ranking_size)], list(range(args.ranking_size)))
        plt.title(f"Boxplot support {party}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path_prefix, f"box_{party}_{args.experiment_name}.png"))
        plt.close()

        
    print("Done")
    return results

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/Users/pdokoupil/Downloads/ml-100k/u.data", help="Path to the dataset file")
    parser.add_argument("--fold_dest", type=str, default="C:/Users/PD/Downloads/ml-100k-folds/newlightfmfolds", help="Path to the directory where folds could be stored")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--separator", type=str, default="\t")
    parser.add_argument("--objective_weights", type=str, default="0.5,0.25,0.25", help="Weights of the individual objectives, in format 'x, y, z'")
    parser.add_argument(
        "--mandate_allocation", type=str, default="exactly_proportional_fuzzy_dhondt_2",
        help="allowed values are {exactly_proportional_fuzzy_dhondt, exactly_proportional_fuzzy_dhondt_2, fai_strategy, random_mandate_allocation, sainte_lague_method, exactly_proportional_fai_strategy}"
    )
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--ranking_size", type=int, default=10)
    parser.add_argument("--output_path_prefix", type=str, default=".") # TODO CHANGE TO /MNT/...
    parser.add_argument("--support_normalization", type=str, default="cdf", help="which normalization to use, allowed values are {None, standardization, cdf, min_max_scaling}")
    parser.add_argument("--support_function", type=str, default="normalizing_marginal_gain_support_function")
    args = parser.parse_args()
    args.objective_weights = list(map(float, args.objective_weights.split(",")))

    if not args.experiment_name:
        args.experiment_name = f"N={args.support_normalization},MA={args.mandate_allocation},W={args.objective_weights},SF={args.support_function}" # Default experiment name

    args.mandate_allocation = globals()[args.mandate_allocation] # Get factory/constructor for mandate allocation algorithm
    args.support_function = globals()[args.support_function]
    args.support_normalization = globals()[args.support_normalization] if args.support_normalization else None

    if args.support_normalization and args.support_function is not normalizing_marginal_gain_support_function:
        assert False, f"using support normalization: {args.support_normalization} but not normalizing support function: {args.support_function}"

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.train_fold_path = f"{args.fold_dest}/0/train.dat"
    args.test_fold_path = f"{args.fold_dest}/0/test.dat"
    print(f"Fold paths: {args.train_fold_path}, {args.test_fold_path}")
    
    print(f"Starting experiment: {args.experiment_name}, with arguments:")
    for arg_name in dir(args):
        if arg_name[0] != '_':
            print(f"\t{arg_name}={getattr(args, arg_name)}")
    

    # CrossValidation(input_file=args.dataset_path, recommender=ItemKNN(), dir_folds=args.fold_dest, header=1, k_folds=5).compute()
    start_time = time.perf_counter()
    voting_recommendation(args)
    print(f"Whole voting took: {time.perf_counter() - start_time}")

if __name__ == "__main__":
    main()