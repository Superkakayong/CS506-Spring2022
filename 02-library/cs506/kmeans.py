from collections import defaultdict
import random
import csv

from numpy import sqrt, inf


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    points = zip(*points)
    avg = lambda points, i: sum(points[i]) / float(len(points[i]))

    return [avg(points, 0), avg(points, 1)]
    # raise NotImplementedError()


def update_centers(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, dataset):
        new_means[assignment].append(point)

    for points in new_means.values():
        centers.append(point_avg(points))

    return centers
    # raise NotImplementedError()


def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(x, y):
    """
    Returns the Euclidean distance between a and b
    """
    return distance_squared(x, y) ** (1 / 2)
    # raise NotImplementedError()


def distance_squared(a, b):
    res = 0
    for i in range(len(a)) :
        res += (a[i] - b[i]) * (a[i] - b[i])

    return res
    # raise NotImplementedError()


def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    centers = []
    dimension = len(dataset[0])
    data = zip(*dataset)

    for _ in range(k):
        rand_point = []
        for i in range(dimension):
            min_val = min(data[i])
            max_val = max(data[i])
            rand_point.append(random.uniform(min_val, max_val))

        centers.append(rand_point)

    return centers
    # raise NotImplementedError()


def cost_function(clustering):
    res = 0
    for center in clustering:
        d_sum = 0
        center_point = point_avg(clustering[center])
        for point in clustering[d_sum]:
            d_sum += distance(center_point, point)
        res += d_sum
    return res
    # raise NotImplementedError()


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """
    centers = []
    dimension = len(dataset[0])
    data = zip(*dataset)

    for _ in range(k):
        rand_point = []
        for i in range(dimension):
            min_val = min(data[i])
            max_val = max(data[i])
            rand_point.append(random.uniform(min_val, max_val))

        centers.append(rand_point)

    return centers
    # raise NotImplementedError()


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset) + 1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset) + 1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)
