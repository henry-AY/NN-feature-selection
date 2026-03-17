import sys
import math
import argparse
import glob
import os
import csv


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

    with open(csv_filepath, "w", newline = "") as f: # this allows us to write to the CSV, so that we can easily graph the data later
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Features", "NumberFeatures", "Accuracy"])

        for i in range(1, features + 1): # iterate over the outer features (allows us to compare every combination of i in j)
            accuracy = 0    
            best_choice = None

            for j in range(1, features + 1):
                if j in selected_features:
                    continue # skip if the feature is already inside selected features
                
                test_feature_selection = selected_features + [j] # append to the test feature set, and calculate the resulting accuracy
                curr_accuracy = nearest_neighbor(data, test_feature_selection)
                print(f"Using feature(s): {test_feature_selection} accuracy is {curr_accuracy}%")

                # If the current accuracy beats the best known accuracy, then we know the test feature we just added is valuable
                if curr_accuracy > accuracy:
                    if DEBUG:
                        print(f"Current accuracy is greater than known best accuracy with same feature list:\n {curr_accuracy} > {accuracy} improving feature is: {j}")
                    accuracy = curr_accuracy
                    best_choice = j
            
            if best_choice is not None: # safety check
                selected_features.append(best_choice)
            else:
                print("ERROR occured when appending best feature choice in forward_selection(...)")

            print(f"\nFeature set: {selected_features} performed the best with an accuracy of {accuracy}%\n")

            writer.writerow([i, selected_features.copy(), len(selected_features), accuracy]) # write to CSV so we can make graphs easily later.

            # similar to internal accuracy condition, here we check compared to the global accuracy
            if accuracy > highest_global_accuracy:
                if DEBUG:
                    print(f"New global accuracy found: {accuracy}")
                highest_global_accuracy = accuracy
                best_features = selected_features.copy()

            history.append(accuracy)

            if len(history) >= (features // 2): # Consider exciting early, as long as the there exists negative growth from the known peak.
                if DEBUG:
                    print(f"Considering exiting early...")
                if is_negative_growth(history, highest_global_accuracy): # we only exit early as long as there's a general negative growth trend beyond the known peak.
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

    # because backward elim needs the entire feature list, we start by creating a list of all features.
    selected_features = list(range(1, features + 1))
    best_features = selected_features.copy()

    with open(csv_filepath, "w", newline = "") as f: # similar to the comment above in forward selection, this allows us to easily download and work with the results.
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Features", "NumberFeatures", "Accuracy"])

        for i in range(1, features + 1):
            accuracy = 0 
            worst_choice = None

            for j in selected_features: # no need to skip anything here
                test_feature_selection = selected_features.copy() # copying here is much smarter, because it prevents modifications in use errors, and also gives us a unique 
                test_feature_selection.remove(j)                  # set to experiment with.

                if not test_feature_selection:
                    return highest_global_accuracy, best_features

                if DEBUG:
                    print(f"Removing feature: {j}")

                curr_accuracy = nearest_neighbor(data, test_feature_selection) # run nearest neighbor on the current testing feature set, and grab the resulting accuracy
                print(f"Using feature(s): {test_feature_selection} accuracy is {curr_accuracy}%")

                if curr_accuracy > accuracy: # similar to above, we compare the accuracy to make sure we're keeping track of the best features and accuracy.
                    if DEBUG:
                        print(f"Current accuracy is greater than known best accuracy with same feature list:\n {curr_accuracy} > {accuracy} feature removed is: {j}")
                    accuracy = curr_accuracy
                    worst_choice = j
            
            if worst_choice is not None: # safety check to not index on an empty array
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

            # Realistically, the backward selection will rarely exit early, it might consider exiting early, but not actually exit early because of it's reverse search pattern.
            # However, I've left it here just in case.
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
        shortest_distance = sys.maxsize # Since nothing will be larger than sys.max, we know for a fact the first iteration will pick up the proper distance. 
        neighbor = 0

        # iterate over every data point, and find the closest neighbor to j, and classify.
        for j in data:
            if i != j:
                distance = compute_distance(i, j, features)

                if distance < shortest_distance:
                    shortest_distance = distance
                    neighbor = j

        prediction = neighbor[0] # we classify off the neighbors label

        if prediction == i[0]: # compare to the actual label, and increment if correct.
            correct_predictions += 1

        overall_preductions += 1

    return correct_predictions / overall_preductions

# https://www.khanacademy.org/math/geometry/hs-geo-analytic-geometry/hs-geo-distance-and-midpoints/a/distance-formula
def compute_distance(i, j, features):
    distance_sum = 0
    for f in features:
        distance_sum += (i[f] - j[f]) ** 2

    return math.sqrt(distance_sum)

"""
is_negative_growth(...) function determines if there exists negative growth, and there exists a function history.
This is to prevent false positives and early exists. This function allows us to exit early, as generally, the more features added
the lower the accuracy is. If there exists a negative trend for 5 or more instances of history, we are highly confident 
we found the highest accuracy.
"""
def is_negative_growth(history, accuracy, passing_iterations = 10):
    if not history:
        return False # Default to false just in case
    
    # a small safety check to verify that there are at least 10 iterations of history. This prevents this being called on extremely small datasets.
    if len(history) < passing_iterations:
        return False
    
    for val in history[-passing_iterations: ]:
        if val >= accuracy:
            return False
        
    return True

"""
A simple runner function to make the process more modular.
"""
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
csv_file = os.path.splitext(filename)[0] # just a nice way to dynamically write the data out with the file name

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

print("Succesfully copied .txt data")

forward_results = run("forward_selection", data, forward_selection, csv_file)
backward_results = run("backwards_elimination", data, backward_elimination, csv_file)

print("\n\n----- Final Results -----") # print all of our results at the end, to make sure the user doesn't need to scroll up.
print(forward_results)
print(backward_results)
