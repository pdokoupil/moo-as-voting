from main import *
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import os

from collections import defaultdict

from scipy.special import rel_entr

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path_prefix", type=str, default="/mnt/0")
parser.add_argument("--output_path_prefix", type=str, default="/mnt/1/outputs")
parser.add_argument("--enable_normalization", type=bool, default=False)
cmd_args = parser.parse_args()

SEED = 42
RANKING_SIZE=10
CASEREC_TRAIN_FOLD_PATH = f"{cmd_args.dataset_path_prefix}/ml-100k-folds/folds/0/train.dat"
CASEREC_TEST_FOLD_PATH = f"{cmd_args.dataset_path_prefix}/ml-100k-folds/folds/0/test.dat"

LIGHTFM_TRAIN_FOLD_PATH = f"{cmd_args.dataset_path_prefix}/ml-100k-folds/newlightfmfolds/0/train.dat"
LIGHTFM_TEST_FOLD_PATH = f"{cmd_args.dataset_path_prefix}/ml-100k-folds/newlightfmfolds/0/test.dat"

method_to_method_name = {
    "exactly_proportional_fuzzy_dhondt": "EP-FuzzDA",
    "exactly_proportional_fuzzy_dhondt_2": "EP-FuzzDA2",
    "sainte_lague_method": "Sainte Lague",
    "fai_strategy": "FAI",
    "exactly_proportional_fai_strategy": "EP-FAI",
}

class dummy_args:
    def __init__(self, **kwargs):
        for arg_name, arg_value in kwargs.items():
            setattr(self, arg_name, arg_value)

def get_arguments():
    use_case_recommender_folds = [
        # True,
        False,
        ]
    objective_weights = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25], [0.25, 0.25, 0.5],
        [1.0/3.0, 1.0/3.0, 1.0/3.0],
        [0.5, 0.3, 0.2], [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2], [0.2, 0.5, 0.3],
        [0.2, 0.3, 0.5], [0.3, 0.2, 0.5]
    ]
    mandate_allocation = [
        #exactly_proportional_fuzzy_dhondt,
        exactly_proportional_fuzzy_dhondt_2,
        #fai_strategy,
        #sainte_lague_method,
        #exactly_proportional_fai_strategy
    ]

    for use_folds, weights, method in itertools.product(use_case_recommender_folds, objective_weights, mandate_allocation):
        if use_folds:
            train_fold_path = CASEREC_TRAIN_FOLD_PATH
            test_fold_path = CASEREC_TEST_FOLD_PATH
            folds = "C"
        else:
            train_fold_path = LIGHTFM_TRAIN_FOLD_PATH
            test_fold_path = LIGHTFM_TEST_FOLD_PATH
            folds = "L"

        name = f"{folds};{weights};{method.__name__}".replace(" ", "")
        yield dummy_args(seed=SEED, train_fold_path=train_fold_path, test_fold_path=test_fold_path,
            objective_weights=weights, mandate_allocation=method,
            ranking_size=RANKING_SIZE, experiment_name=name, support_function=normalizing_marginal_gain_support_function,
            support_normalization=cdf
        )

def generate_latex_tables(all_results, plot_save_path_prefix):
    latex_tables = ""

    groupped_results = defaultdict(lambda: defaultdict(dict))
    for experiment_name, results in all_results.items():
        parsed_experiment_name = experiment_name.replace(" ", "").split(";")
        folds = parsed_experiment_name[0]
        weights = tuple(map(float, parsed_experiment_name[1][1:-1].split(",")))
        mandate_allocation_method = parsed_experiment_name[2]
        groupped_results[weights][mandate_allocation_method][folds] = results

    # For each method and folds we also store reference MER (with weights [1, 0, 0])
    weights = [
        ("mer", (1.0, 0.0, 0.0)),
        ("diversity", (0.0, 1.0, 0.0)),
        ("novelty", (0.0, 0.0, 1.0)),
        ("normalized-mer", (1.0, 0.0, 0.0)),
        ("normalized-diversity", (0.0, 1.0, 0.0)),
        ("normalized-novelty", (0.0, 0.0, 1.0)),
    ]
    per_method_maxima = defaultdict(lambda: dict())
    across_method_maxima = defaultdict(float)
    for obj_name, w in weights:
        for method_name, fold_results in groupped_results[w].items():
            for fold_name, results in fold_results.items():
                per_method_maxima[method_name][obj_name] = results[obj_name]
                across_method_maxima[obj_name] = max(across_method_maxima[obj_name], results[obj_name])

    globally_max_objective_values = np.array([
        across_method_maxima["mer"],
        across_method_maxima["diversity"],
        across_method_maxima["novelty"]
    ])

    globally_max_objective_values_normalized = np.array([
        across_method_maxima["normalized-mer"],
        across_method_maxima["normalized-diversity"],
        across_method_maxima["normalized-novelty"]
    ])

    print(f"Globally objective maxima: {globally_max_objective_values}")
    print(f"Globally objective maxima (normalized objectives): {globally_max_objective_values_normalized}")

    def kl_divergence(objective_weights, objectives):
        return sum(rel_entr(objective_weights, objectives))

    # Calculates percentage change in MER
    def mer_change(mer, reference_mer):
        return ((mer / reference_mer) - 1.0) * 100.0

    def normalize_objective_values(max_objective_values, objective_values):
        assert len(max_objective_values) == len(objective_values)
        
        objective_values_normed = objective_values / max_objective_values # Normalize all components to [0, 1]
        objective_values_normed[objective_values_normed < 0] = 0 # TODO remove if needed. Was added to solve issues with negative objectives (those were appearing after standardization)
        assert np.all(objective_values_normed) >= 0.0 and np.all(objective_values_normed) <= 1.0
        print(f"Per-element normalized obj values: [{objective_values}] -> {objective_values_normed}")

        objective_values_normed = objective_values_normed + 1e-6 # To prevent division by zero etc.
        normalized_objective_values = objective_values_normed / np.sum(objective_values_normed)
        print(f"Fully normalized obj values: {normalized_objective_values}")
        objective_values_sum = np.sum(normalized_objective_values)
        assert np.isclose(objective_values_sum, 1.0), f"{normalized_objective_values}, {objective_values_normed}, {objective_values_sum}"
        
        return normalized_objective_values
    
    def calculate_per_user_kl_divergence(weights, rest_results):
        results = dict()
        results_normalized = dict()

        for method_name in rest_results.keys():
            per_user_mer = rest_results[method_name]["L"]["per-user-mer"]
            per_user_diversity = rest_results[method_name]["L"]["per-user-diversity"]
            per_user_novelty = rest_results[method_name]["L"]["per-user-novelty"]
            per_user_mer_normalized = rest_results[method_name]["L"]["normalized-per-user-mer"]
            per_user_diversity_normalized = rest_results[method_name]["L"]["normalized-per-user-diversity"]
            per_user_novelty_normalized = rest_results[method_name]["L"]["normalized-per-user-novelty"]
            assert len(per_user_mer) == len(per_user_diversity) == len(per_user_novelty) == len(per_user_mer_normalized) == len(per_user_diversity_normalized) == len(per_user_novelty_normalized), "All per-user results must have a same length"
            
            assert np.all(np.isfinite(np.array(per_user_mer))), f"MER must by finite: {per_user_mer}"
            assert np.all(np.isfinite(np.array(per_user_diversity))), f"Diversity must by finite: {per_user_diversity}"
            assert np.all(np.isfinite(np.array(per_user_novelty))), f"Novelty must by finite: {per_user_novelty}"
            assert np.all(np.isfinite(np.array(per_user_mer_normalized))), f"MER must by finite: {per_user_mer_normalized}"
            assert np.all(np.isfinite(np.array(per_user_diversity_normalized))), f"Diversity must by finite: {per_user_diversity_normalized}"
            assert np.all(np.isfinite(np.array(per_user_novelty_normalized))), f"Novelty must by finite: {per_user_novelty_normalized}"
            
            
            kl_divergences = []
            for mer, div, nov in zip(per_user_mer, per_user_diversity, per_user_novelty):
                normalized_objective_values = normalize_objective_values(globally_max_objective_values, np.array([mer, div, nov]))
                
                assert np.all(np.isfinite(normalized_objective_values)), f"Normalized objective values must be finite: {normalized_objective_values}, {globally_max_objective_values}, {mer}, {div}, {nov}"
                divergence = kl_divergence(weights, normalized_objective_values)
                print(f"KL-Divergence: {divergence}\n\n")
                assert np.isfinite(divergence), f"KL-Divergence must be finite: {divergence}, {normalized_objective_values}, {globally_max_objective_values}, {mer}, {div}, {nov}"
                kl_divergences.append(divergence)
            results[method_name] = kl_divergences

            kl_divergences_normalized = [] #Over normalized objective values
            for mer, div, nov in zip(per_user_mer_normalized, per_user_diversity_normalized, per_user_novelty_normalized):
                normalized_objective_values = normalize_objective_values(globally_max_objective_values_normalized, np.array([mer, div, nov]))
                
                assert np.all(np.isfinite(normalized_objective_values)), f"Normalized objective values must be finite: {normalized_objective_values}, {globally_max_objective_values_normalized}, {mer}, {div}, {nov}"
                divergence = kl_divergence(weights, normalized_objective_values)
                print(f"KL-Divergence: {divergence}\n\n")
                assert np.isfinite(divergence), f"KL-Divergence must be finite: {divergence}, {normalized_objective_values}, {globally_max_objective_values_normalized}, {mer}, {div}, {nov}"
                kl_divergences_normalized.append(divergence)
            results_normalized[method_name] = kl_divergences_normalized

        return results, results_normalized

    for weights, rest_results in groupped_results.items():
        if len(latex_tables) > 0:
            latex_tables += "\n\n"
        
        per_user_kl_divergence, per_user_kl_divergence_normalized = calculate_per_user_kl_divergence(weights, rest_results)
        mean_kl_divergence = {method_name: np.mean(results) for method_name, results in per_user_kl_divergence.items()}
        mean_kl_divergence_normalized = {method_name: np.mean(results) for method_name, results in per_user_kl_divergence_normalized.items()}
        
        # Create and save plot
        data = []
        labels = []
        indices = np.arange(1, len(per_user_kl_divergence) + 1)
        for method_name, results in per_user_kl_divergence.items():
            labels.append(method_name)
            data.append(results)
        plt.boxplot(data)
        
        # Map the method names to closer, more readable names
        for method_name, new_method_name in method_to_method_name.items():
            if method_name in labels:
                labels[labels.index(method_name)] = new_method_name

        plt.xticks(indices, labels, rotation=90)
        plt.title(f"KL-D for weights: {str(weights).replace(' ','')}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_path_prefix, "kl_d_" + str(weights).replace(" ","") + ".png"))
        plt.close()

        # Create and save plot (Normalized)
        data = []
        labels = []
        indices = np.arange(1, len(per_user_kl_divergence_normalized) + 1)
        for method_name, results in per_user_kl_divergence_normalized.items():
            labels.append(method_name)
            data.append(results)
        plt.boxplot(data)
        
        # Map the method names to closer, more readable names
        for method_name, new_method_name in method_to_method_name.items():
            if method_name in labels:
                labels[labels.index(method_name)] = new_method_name

        plt.xticks(indices, labels, rotation=90)
        plt.title(f"KL-D normalized for weights: {str(weights).replace(' ','')}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_path_prefix, "kl_d_n_" + str(weights).replace(" ","") + ".png"))
        plt.close()

        # END PLOTTING

        method_latex_tables = ""
        for method_name, fold_results in rest_results.items():
            for fold_name, results in fold_results.items():
                print(results)
                method_latex_tables += \
"""  \\textbf{%s} &
        %.3f (%.1f\\%%) | %.3f (%.1f\\%%) & %.3f | %.3f & %.3f | %.3f & %0.3f & %.3f | %.3f \\\\
  \\hline"""    % (
                    method_to_method_name[method_name],
                    
                    results["mer"],
                    mer_change(results["mer"], per_method_maxima[method_name]["mer"]),
                    
                    results["normalized-mer"],
                    mer_change(results["normalized-mer"], per_method_maxima[method_name]["normalized-mer"]),

                    results["diversity"],
                    results["normalized-diversity"],

                    results["novelty"],
                    results["normalized-novelty"],

                    results["map"],

                    mean_kl_divergence[method_name],
                    mean_kl_divergence_normalized[method_name]
                )
                

        latex_tables += """%% [%.3f, %.3f, %.3f]
\\begin{center}
\\begin{tabular}{ | c | c | c | c | c | c | } 
  \\hline
  \\textit{Method} & \\textbf{MER} & \\textbf{Diversity} & \\textbf{Novelty} & \\textbf{MAP} & \\textbf{KL} \\\\ 
  \\hline
%s
 \\multicolumn{6}{| c |}{\\textbf{Weights} [%.3f, %.3f, %.3f]}\\\\
\\end{tabular}
\\end{center}""" % (
    weights[0], weights[1], weights[2],
    method_latex_tables,
    weights[0], weights[1], weights[2]
)

    return latex_tables


if __name__ == "__main__":
    print("Starting the work")
    random.seed(SEED)
    np.random.seed(SEED)

    all_results = dict() # Maps: (folds, weights, algorithm) i.e. experiment_name to results

    print("Getting experiments")
    arguments = list(get_arguments())
    num_experiments = len(arguments)
    for i, args in enumerate(arguments):
        setattr(args, "output_path_prefix", cmd_args.output_path_prefix)
        setattr(args, "enable_normalization", cmd_args.enable_normalization)
        print(f"Starting experiment: {args.experiment_name} {i+1}/{num_experiments}, with arguments:")
        for arg_name in dir(args):
            if arg_name[0] != '_':
                print(f"\t{arg_name}={getattr(args, arg_name)}")
        start_time = time.perf_counter()
        results = voting_recommendation(args)
        all_results[args.experiment_name] = results
        print(f"Whole voting took: {time.perf_counter() - start_time}")
        print("\n\n")


    latex_tables = generate_latex_tables(all_results, cmd_args.output_path_prefix)

    print("Resulting latex tables:\n\n")
    print(latex_tables)