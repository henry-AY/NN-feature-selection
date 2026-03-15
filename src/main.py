import sys
import math
import argparse
import glob
import os
import csv
import re


"""
this function forward_selection() is the main driver for the forward selection process. It works, by for every possible feature, 
finding the best possible set of features. It does this by calculating the 1-NN score accuracy, and a greedy choice property.

Sources: https://stackoverflow.com/questions/64475526/how-do-i-generate-a-list-of-all-possible-combinations-from-a-single-element-in-a
To find all the possible combinations
"""
def forward_selection(data, features, csv_filepath):
    selected_features = []
    best_features = []
    history = []
    highest_global_accuracy = 0

    with open(csv_filepath, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Features", "NumberFeatures" "Accuracy"])

        for i in range(1, features + 1):
            accuracy = 0    
            best_choice = None

            for j in range(1, features + 1):
                if j in selected_features:
                    continue # skip if the feature is already inside selected features

                test_feature_selection = selected_features + [j]
                curr_accuracy = nearest_neighbor(data, test_feature_selection)
                print(f"Using feature(s): {test_feature_selection} accuracy is {curr_accuracy}%")

                if curr_accuracy > accuracy:
                    if DEBUG:
                        print(f"Current accuracy is greater than known best accuracy with same feature list:\n {curr_accuracy} > {accuracy} improving feature is: {j}")
                    accuracy = curr_accuracy
                    best_choice = j
            
            if best_choice is not None:
                selected_features.append(best_choice)
            else:
                print("ERROR occured when appending best feature choice in forward_selection(...)")

            print(f"\nFeature set: {selected_features} performed the best with an accuracy of {accuracy}%\n")

            writer.writerow([i, selected_features.copy(), len(selected_features), accuracy]) # write to CSV so we can make graphs easily later.

            if accuracy > highest_global_accuracy:
                if DEBUG:
                    print(f"New global accuracy found: {accuracy}")
                highest_global_accuracy = accuracy
                best_features = selected_features.copy()

            history.append(accuracy)

            if len(history) >= (features // 2): # Consider exciting early, as long as the there exists negative growth from the known peak.
                if DEBUG:
                    print(f"Considering exiting early...")
                if is_negative_growth(history, highest_global_accuracy):
                    print(f"General downwards trend found, exiting early...")
                    return highest_global_accuracy, best_features
            
    return highest_global_accuracy, best_features


"""
this function backward_elimination is the main driver for the backward selection/elimination process. It works, by starting off with every single feature
and then removing the worse performing set of features. Similarly, this has a greedy choice property of the highest accuracy.
"""
def backward_elimination(data, features, csv_filepath):
    history = []
    highest_global_accuracy = 0

    selected_features = list(range(1, features + 1))
    best_features = selected_features.copy()

    with open(csv_filepath, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Features", "NumberFeatures" "Accuracy"])

        for i in range(1, features + 1):
            accuracy = 0 
            worst_choice = None

            for j in selected_features: # no need to skip anything here
                test_feature_selection = selected_features.copy()
                test_feature_selection.remove(j)

                if not test_feature_selection:
                    return highest_global_accuracy, best_features

                if DEBUG:
                    print(f"Removing feature: {j}")

                curr_accuracy = nearest_neighbor(data, test_feature_selection)
                print(f"Using feature(s): {test_feature_selection} accuracy is {curr_accuracy}%")

                if curr_accuracy > accuracy:
                    if DEBUG:
                        print(f"Current accuracy is greater than known best accuracy with same feature list:\n {curr_accuracy} > {accuracy} feature removed is: {j}")
                    accuracy = curr_accuracy
                    worst_choice = j
            
            if worst_choice is not None:
                selected_features.remove(worst_choice)
            else:
                print("ERROR occured when removing the worse feature choice in backward_elimination(...)")

            print(f"\nFeature set: {selected_features} performed the best with an accuracy of {accuracy}%\n")

            writer.writerow([i, selected_features.copy(), len(selected_features), accuracy]) # write to CSV so we can make graphs easily later.

            if accuracy > highest_global_accuracy:
                if DEBUG:
                    print(f"New global accuracy found: {accuracy}")
                highest_global_accuracy = accuracy
                best_features = selected_features.copy()

            history.append(accuracy)

            if len(history) >= (features // 2): # Consider exciting early, as long as the there exists negative growth from the known peak.
                if DEBUG:
                    print(f"Considering exiting early...")
                if is_negative_growth(history, highest_global_accuracy):
                    print(f"General downwards trend found, exiting early...")
                    return highest_global_accuracy, best_features
                
    return highest_global_accuracy, best_features

"""
This nearest_neighbor(...) function is a 1-NN. It finds the nearest neighbor for every data point, 
classifies it, and returns the overall model accuracy.
"""
def nearest_neighbor(data, features):
    overall_preductions = 0
    correct_predictions = 0

    for i in data:
        shortest_distance = sys.maxsize
        neighbor = 0

        for j in data:
            if i != j:
                distance = compute_distance(i, j, features)

                if distance < shortest_distance:
                    shortest_distance = distance
                    neighbor = j

        prediction = neighbor[0]

        if prediction == i[0]:
            correct_predictions += 1

        overall_preductions += 1

    return correct_predictions / overall_preductions

# https://www.khanacademy.org/math/geometry/hs-geo-analytic-geometry/hs-geo-distance-and-midpoints/a/distance-formula
def compute_distance(i, j, features):
    distance_sum = 0
    for f in features:
        distance_sum += (i[f] - j[f]) ** 2

    return math.sqrt(distance_sum)


def is_negative_growth(history, accuracy, passing_iterations = 35):
    if not history:
        return False # Default to false just in case
    
    if len(history) < passing_iterations:
        return False
    
    for val in history[-passing_iterations: ]:
        if val >= accuracy:
            return False
        
    return True

"""
is_negative_growth(...) function determines if there exists negative growth, and there exists a function history.
This is to prevent false positives and early exists. This function allows us to exit early, as generally, the more features added
the lower the accuracy is. If there exists a negative trend for 5 or more instances of history, we are highly confident 
we found the highest accuracy.
"""
# def is_negative_growth(history, known_peak):
#     if not history:
#         return False # Default to false just in case
    
#     known_peak_index = history.index(known_peak)

#     prev = history[known_peak_index]

#     # this exists to prevent false positives, there must be a minimum of 5 pieces of data after the known_peak_index
#     if len(history) - known_peak_index < 5:
#         return False

#     for i in range(known_peak_index + 1, len(history)):
#         curr = history[i]

#         if curr >= prev:
#             return False
#         prev = curr

#     return True

def run(header, data, function, csv):
    print(f"Running {header}\n")

    accuracy, features = function(data, len(data[0]) - 1, f"{csv}_{header}.csv")

    output = f"{header} found the following best combination:\nFeatures: {features} with an accuracy of: {accuracy}\n"
    print(output)

    return output


"""
Main function
"""
print("Welcome to Henry's Feature Selection Algorithm")

parser = argparse.ArgumentParser(description="A script that does something with flags.")
parser.add_argument("-s", "--simple", action="store_true", help="Use the Sanity check dataset")
parser.add_argument("-d", "--debug", action="store_true", help="Verbose debugging")

args = parser.parse_args()

DEBUG = args.debug

if args.simple:
    pattern = "**/SanityCheck_DataSet__1.txt"
else:
    data_uncleaned = input("Type the name of the file to test: ").strip()
    pattern = "**/" + data_uncleaned

files_found = glob.glob(os.path.join("./data", pattern), recursive=True)

if not files_found:
    raise FileNotFoundError("File not found in NN-feature-selection directory or an subdirectories.")

if len(files_found) > 1: # warning just in case
    print(f"Multiple matches found, using the first one")

file = files_found[0] # using the first one just in case there are multiple files

filename = os.path.basename(file)
csv_file = os.path.splitext(filename)[0]

data = []
try:
    with open(file, mode='r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            vals = line.split() # split on spaces

            numeric_vals = [float(val) for val in vals] # convert to floats so we can use .math, and because they're quite small/large this prevents overflow
            data.append(numeric_vals)
except Exception:
    print(f"Error reading from .txt file: {pattern}")

# Make sure that the file has at least 1 line

print("Succesfully copied .txt data")

forward_results = run("forward_selection", data, forward_selection, csv_file)
backward_results = run("backwards_elimination", data, backward_elimination, csv_file)

print("\n\n----- Final Results -----")
print(forward_results)
print(backward_results)
