from main import *
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import os
import pathlib

from collections import defaultdict

from scipy.special import rel_entr

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_label", type=str)
parser.add_argument("--dataset_path_prefix", type=str, default="/mnt/0")
parser.add_argument("--output_path_prefix", type=str, default="/mnt/1/outputs")
parser.add_argument("--algorithms", type=str) # format fai,probabilistic_fai
parser.add_argument("--normalizations", type=str) # format cdf,standardization
parser.add_argument("--weights", type=str) # format a,b,c;d,e,f;g,h,i
parser.add_argument("--shift", type=float, default=0.0)
parser.add_argument("--use_mf_baseline", action="store_true", default=False)
parser.add_argument("--use_cb_diversity", action="store_true", default=False, help="Use content based diversity")
parser.add_argument("--metadata_path", type=str)
parser.add_argument("--use_discounting", action="store_true", default=False)
cmd_args = parser.parse_args()

# Update some parameters
cmd_args.output_path_prefix = os.path.join(cmd_args.output_path_prefix, cmd_args.experiment_label)
cmd_args.algorithms = [globals()[alg_name] for alg_name in cmd_args.algorithms.split(",")]
cmd_args.normalizations = [globals()[normalization_name] for normalization_name in cmd_args.normalizations.split(",")]
cmd_args.weights = [[float(w) for w in weight_vector.split(",")] for weight_vector in cmd_args.weights.split(";")]

print(f"Output path prefix: {cmd_args.output_path_prefix}")
pathlib.Path(cmd_args.output_path_prefix).mkdir(parents=True, exist_ok=True)


SEED = 42
RANKING_SIZE=10

# CASEREC_TRAIN_FOLD_PATH = f"{cmd_args.dataset_path_prefix}/ml-100k-folds/folds/0/train.dat"
# CASEREC_TEST_FOLD_PATH = f"{cmd_args.dataset_path_prefix}/ml-100k-folds/folds/0/test.dat"

# # LIGHTFM_TRAIN_FOLD_PATH = f"{cmd_args.dataset_path_prefix}/ml-100k-folds/newlightfmfolds/0/train.dat"
# # LIGHTFM_TEST_FOLD_PATH = f"{cmd_args.dataset_path_prefix}/ml-100k-folds/newlightfmfolds/0/test.dat"

# LIGHTFM_TRAIN_FOLD_PATH = f"{cmd_args.dataset_path_prefix}/ml-1m-folds/rndlightfmfolds/0/train.dat"
# LIGHTFM_TEST_FOLD_PATH = f"{cmd_args.dataset_path_prefix}/ml-1m-folds/rndlightfmfolds/0/test.dat"

# CACHE_DIR = f"{cmd_args.dataset_path_prefix}/ml-1m-folds/rndlightfmfolds/0/"

TRAIN_FOLD_PATH = os.path.join(cmd_args.dataset_path_prefix, "train.dat")
TEST_FOLD_PATH = os.path.join(cmd_args.dataset_path_prefix, "test.dat")
CACHE_DIR = cmd_args.dataset_path_prefix

method_to_method_name = {
    "exactly_proportional_fuzzy_dhondt": "EP-FuzzDA",
    "exactly_proportional_fuzzy_dhondt_2": "EP-FuzzDA2",
    "sainte_lague_method": "Sainte Lague",
    "fai_strategy": "FAI",
    "exactly_proportional_fai_strategy": "EP-FAI",
    "probabilistic_fai_strategy": "Prob-FAI",
    "weighted_average_strategy": "WA"
}

class dummy_args:
    def __init__(self, **kwargs):
        for arg_name, arg_value in kwargs.items():
            setattr(self, arg_name, arg_value)

def get_arguments(cmd_args):
    # objective_weights = [
    #     [1, 0, 0], [0, 1, 0], [0, 0, 1],
    #     [0.5, 0.25, 0.25],
    #     [0.25, 0.5, 0.25], [0.25, 0.25, 0.5],
    #     [1.0/3.0, 1.0/3.0, 1.0/3.0],
    #     [0.5, 0.3, 0.2], [0.5, 0.2, 0.3],
    #     [0.3, 0.5, 0.2], [0.2, 0.5, 0.3],
    #     [0.2, 0.3, 0.5], [0.3, 0.2, 0.5]
    # ]
    # mandate_allocation = [
    #     #exactly_proportional_fuzzy_dhondt,
    #     exactly_proportional_fuzzy_dhondt_2,
    #     #fai_strategy,
    #     #sainte_lague_method,
    #     #exactly_proportional_fai_strategy
    # ]
    objective_weights = cmd_args.weights
    mandate_allocation = cmd_args.algorithms
    normalizations = cmd_args.normalizations

    for weights, method, normalization in itertools.product(objective_weights, mandate_allocation, normalizations):
        if "ml-100k" in TRAIN_FOLD_PATH:
            folds = "ml-100k"
        elif "ml-1m" in TRAIN_FOLD_PATH:
            folds = "ml-1m"
        elif "lastfm" in TRAIN_FOLD_PATH:
            folds = "lastfm"
        else:
            folds = "unknown"

        baseline_name = 'mf' if cmd_args.use_mf_baseline else 'knn'
        div_name = 'cb' if cmd_args.use_cb_diversity else 'col'
        disc_name = 'disc' if cmd_args.use_discounting else 'no_disc'
        name = f"{folds};{weights};{method.__name__};{normalization.__name__};{baseline_name};{div_name};{disc_name}".replace(" ", "")
        yield dummy_args(seed=SEED, train_fold_path=TRAIN_FOLD_PATH, test_fold_path=TEST_FOLD_PATH,
            objective_weights=weights, mandate_allocation=method,
            ranking_size=RANKING_SIZE, experiment_name=name, support_function=normalizing_marginal_gain_support_function,
            support_normalization=normalization, metadata_path=cmd_args.metadata_path, cache_dir=CACHE_DIR, shift=cmd_args.shift,
            use_mf_baseline=cmd_args.use_mf_baseline, use_cb_diversity=cmd_args.use_cb_diversity, use_discounting=cmd_args.use_discounting
        )

def generate_latex_tables(all_results, plot_save_path_prefix):
    latex_tables = ""

    groupped_results = defaultdict(lambda: defaultdict(dict))
    for experiment_name, results in all_results.items():
        parsed_experiment_name = experiment_name.replace(" ", "").split(";")
        folds = parsed_experiment_name[0]
        weights = tuple(map(float, parsed_experiment_name[1][1:-1].split(",")))
        mandate_allocation_method = parsed_experiment_name[2]
        normalization_name = parsed_experiment_name[3]
        groupped_results[weights][mandate_allocation_method][normalization_name] = results


    def kl_divergence(objective_weights, objectives):
        return sum(rel_entr(objective_weights, objectives))

    # # Calculates percentage change in MER
    # def mer_change(mer, reference_mer):
    #     return ((mer / reference_mer) - 1.0) * 100.0
    
    def calculate_per_user_kl_divergence(weights, rest_results):
        results_normalized = dict()

        for method_name in rest_results.keys():
            for normalization_name in rest_results[method_name].keys():
                per_user_mer_normalized = rest_results[method_name][normalization_name]["normalized-per-user-mer"]
                per_user_diversity_normalized = rest_results[method_name][normalization_name]["normalized-per-user-diversity"]
                per_user_novelty_normalized = rest_results[method_name][normalization_name]["normalized-per-user-novelty"]
                assert len(per_user_mer_normalized) == len(per_user_diversity_normalized) == len(per_user_novelty_normalized), "All per-user results must have a same length"
                
                assert np.all(np.isfinite(np.array(per_user_mer_normalized))), f"MER must by finite: {per_user_mer_normalized}"
                assert np.all(np.isfinite(np.array(per_user_diversity_normalized))), f"Diversity must by finite: {per_user_diversity_normalized}"
                assert np.all(np.isfinite(np.array(per_user_novelty_normalized))), f"Novelty must by finite: {per_user_novelty_normalized}"
                
                print("Calculating KL-Divergence over normalized values")
                kl_divergences_normalized = [] #Over normalized objective values
                c = 0
                for mer, div, nov in zip(per_user_mer_normalized, per_user_diversity_normalized, per_user_novelty_normalized):
                    #assert mer <= 1.0 and mer >= 0.0 and div <= 1.0 and div >= 0.0 and nov <= 1.0 and nov >= 0.0, f"{mer},{div},{nov}"
                    objs = np.array([mer, div, nov])
                    if np.any(objs <= 0):
                        print(f"Warning: objs={objs} contains something non-positive")
                        objs[objs <= 0] = 1e-6
                        print(f"Replaced with epsilon: {objs}")
                        c += 1
                    normalized_objective_values = objs / objs.sum()

                    assert np.all(np.isfinite(normalized_objective_values)), f"Normalized objective values must be finite: {normalized_objective_values}, {mer}, {div}, {nov}"
                    divergence = kl_divergence(weights, normalized_objective_values)
                    print(f"KL-Divergence: {divergence}\n\n")
                    assert np.isfinite(divergence), f"KL-Divergence must be finite: {divergence}, {normalized_objective_values}, {mer}, {div}, {nov}"
                    kl_divergences_normalized.append(divergence)
                
                if method_name not in results_normalized:
                    results_normalized[method_name] = dict()
                results_normalized[method_name][normalization_name] = kl_divergences_normalized

                print(f"Warning: negative values found for {c} users, out of {len(per_user_mer_normalized)} ({(c / len(per_user_mer_normalized)) * 100} %)")

        return results_normalized

    def calculate_per_user_errors(weights, rest_results):
        weights = np.array(weights)
        mean_absolute_errors = dict()
        mean_errors = dict()
        for method_name in rest_results.keys():
            for normalization_name in rest_results[method_name].keys():
                per_user_mer_normalized = rest_results[method_name][normalization_name]["normalized-per-user-mer"]
                per_user_diversity_normalized = rest_results[method_name][normalization_name]["normalized-per-user-diversity"]
                per_user_novelty_normalized = rest_results[method_name][normalization_name]["normalized-per-user-novelty"]
                
                errors = []
                absolute_errors = []
                for mer, div, nov in zip(per_user_mer_normalized, per_user_diversity_normalized, per_user_novelty_normalized):
                    # Normalize to 1 sum
                    objs = np.array([mer, div, nov])
                    objs[objs <= 0] = 1e-6
                    objs = objs / objs.sum()

                    absolute_errors.append(np.abs(objs - weights).mean())
                    errors.append(objs - weights)

                if method_name not in mean_absolute_errors:
                    mean_absolute_errors[method_name] = dict()
                    mean_errors[method_name] = dict()

                mean_absolute_errors[method_name][normalization_name] = absolute_errors
                mean_errors[method_name][normalization_name] = errors

        return mean_absolute_errors, mean_errors


    for weights, rest_results in groupped_results.items():
        if len(latex_tables) > 0:
            latex_tables += "\n\n"
        
        per_user_kl_divergence_normalized = calculate_per_user_kl_divergence(weights, rest_results)
        mean_kl_divergence_normalized = dict()
        for method_name, results in per_user_kl_divergence_normalized.items():
            mean_kl_divergence_normalized[method_name] = {normalization_name: np.mean(rest_results) for normalization_name, rest_results in results.items()}

        per_user_mean_absolute_error, per_user_error = calculate_per_user_errors(weights, rest_results)
        mean_absolute_error = dict()
        mean_error = dict()
        for method_name, results in per_user_mean_absolute_error.items():
            mean_absolute_error[method_name] = {normalization_name: np.mean(rest_results) for normalization_name, rest_results in results.items()}
        for method_name, results in per_user_error.items():
            mean_error[method_name] = {normalization_name: np.mean(rest_results, axis=0) for normalization_name, rest_results in results.items()}
        
        # Create and save plot (Normalized)
        normalization_names = []
        for _, res in groupped_results.items():
            for _, r in res.items():
                normalization_names = list(r.keys())
                break
        
        for normalization_name in normalization_names:
            data = []
            labels = []
            indices = np.arange(1, len(per_user_kl_divergence_normalized) + 1)
            for method_name, results in per_user_kl_divergence_normalized.items():
                labels.append(method_name)
                data.append(results[normalization_name])
            plt.boxplot(data)
            
            # Map the method names to closer, more readable names
            for method_name, new_method_name in method_to_method_name.items():
                if method_name in labels:
                    labels[labels.index(method_name)] = new_method_name

            plt.xticks(indices, labels, rotation=90)
            plt.title(f"KL-D {normalization_name} for weights: {str(weights).replace(' ','')}")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_save_path_prefix, "kl_d_n_" + str(weights).replace(" ","") + "_" + normalization_name + ".png"))
            plt.close()

        # END PLOTTING

        
        method_latex_tables = ""
        for method_name, fold_results in rest_results.items():
            for normalization_name, results in fold_results.items():
                print(results)
                method_latex_tables += \
"""  \\textbf{%s} &
        %.3f \\# %.3f & %.3f \\# %.3f & %.3f \\# %.3f & %0.3f & %.3f & %.3f (%s) \\\\
  \\hline"""    % (
                    method_to_method_name[method_name],
                    
                    results["mer"],
                    
                    results["normalized-mer"],
                    
                    results["diversity"],
                    results["normalized-diversity"],

                    results["novelty"],
                    results["normalized-novelty"],

                    results["map"],

                    mean_kl_divergence_normalized[method_name][normalization_name],

                    mean_absolute_error[method_name][normalization_name],
                    [round(x, 3) for x in mean_error[method_name][normalization_name]]
                )
                

        latex_tables += """%% [%.3f, %.3f, %.3f]
\\begin{center}
\\begin{tabular}{ | c | c | c | c | c | c | c | } 
  \\hline
  \\textit{Method} & \\textbf{MER} & \\textbf{Diversity} & \\textbf{Novelty} & \\textbf{MAP} & \\textbf{KL} & \\textbf{E} \\\\ 
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
    arguments = list(get_arguments(cmd_args))
    num_experiments = len(arguments)
    for i, args in enumerate(arguments):
        setattr(args, "output_path_prefix", cmd_args.output_path_prefix)
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