import sys
import math
import argparse
import glob
import os
import csv


def main():
    print("Welcome to Henry's Feature Selection Algorithm")

    parser = argparse.ArgumentParser(description="A script that does something with flags.")
    parser.add_argument("-s", "--simple", action="store_true", help="Use the Sanity check dataset")

    args = parser.parse_args()

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

    print("Succesfully copied CSV data")
    
    print("Running Forward Selection: ")

    forward_selection(data, len(data[0]) - 1)







def forward_selection(data, features):
    selected_features = []

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
                accuracy = curr_accuracy
                best_choice = j
        
        if best_choice is not None:
            selected_features.append(best_choice)
        else:
            print("ERROR appending best feature choice in forward_selection()")

        print(f"Feature set: {selected_features} performed the best with an accuracy of {accuracy}%")


def backward_elimination(data, features):
    pass

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

        # Outside of the loop, we then predict based on the closest score saying it's a 1 or a 2. Then we see if we scored correct or not. 
        # We use this to calculate the overall_score. 
    return correct_predictions / overall_preductions

# https://www.khanacademy.org/math/geometry/hs-geo-analytic-geometry/hs-geo-distance-and-midpoints/a/distance-formula
def compute_distance(i, j, features):
    distance_sum = 0
    for f in features:
        distance_sum += (i[f] - j[f]) ** 2

    return math.sqrt(distance_sum)

if __name__ == "__main__":
    main()