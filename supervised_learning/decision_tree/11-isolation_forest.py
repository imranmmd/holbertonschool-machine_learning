#!/usr/bin/env python3
"""Module that defines an isolation random forest."""


import numpy as np

Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest:
    """Class that represents an isolation random forest."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize an Isolation_Random_Forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """Return the mean depth predicted by the forest."""
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Train the isolation random forest."""
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []

        for i in range(n_trees):
            tree = Isolation_Random_Tree(
                max_depth=self.max_depth,
                seed=self.seed + i
            )
            tree.fit(explanatory)
            self.numpy_preds.append(tree.predict)
            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))

        if verbose == 1:
            print(
                """  Training finished.
    - Mean depth                     : {}
    - Mean number of nodes           : {}
    - Mean number of leaves          : {}""".format(
                    np.array(depths).mean(),
                    np.array(nodes).mean(),
                    np.array(leaves).mean()
                )
            )

    def suspects(self, explanatory, n_suspects):
        """Return rows with the smallest mean depths and their depths."""
        depths = self.predict(explanatory)
        indices = np.argsort(depths)[:n_suspects]
        return explanatory[indices], depths[indices]
