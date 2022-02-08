from scipy import spatial


def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return res ** (1 / 2)


def manhattan_dist(x, y):
    return sum(abs(x_val - y_val) for x_val, y_val in zip(x, y))
    # raise NotImplementedError()


def jaccard_dist(x, y):
    intersection = len(list(set(x).intersection(y)))
    union = len(list(x)) + len(list(y)) - intersection
    return float(intersection) / union
    # raise NotImplementedError()


def cosine_sim(x, y):
    result = 1 - spatial.distance.cosine(x, y)
    return result
    # raise NotImplementedError()

# Feel free to add more
