import sys
import math


def main():
    pass







def forward_selection(data, features):
    pass


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