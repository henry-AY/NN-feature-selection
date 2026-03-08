import sys
import math


def main():
    pass







def forward_selection(data, features):
    pass


def backward_elimination(data, features):
    pass

def nearest_neighbor(data, features):
    overall_score = 0

    for i in data:
        shortest_distance = sys.maxsize

        for j in data:
            if i != j:
                # Somehow we only pick the selected features here
                pass



                # Calculate euclid distance from i to j, and if its lower than best known score save it.

        # Outside of the loop, we then predict based on the closest score saying it's a 1 or a 2. Then we see if we scored correct or not. 
        # We use this to calculate the overall_score. 
    return

# https://www.khanacademy.org/math/geometry/hs-geo-analytic-geometry/hs-geo-distance-and-midpoints/a/distance-formula
def compute_distance(i, j, features):
    sum = 0;
    for f in len(range(features)):
        sum += (i[f] - j[f]) ** 2

    return math.sqrt(sum)
