#!/usr/bin/env python3
def max_depth_below(self):
        """
        Returns the maximum depth below the current node
        (including itself and all descendants)
        """
        # If both children are None (safety case)
        if self.left_child is None and self.right_child is None:
            return self.depth

        left_depth = self.depth
        right_depth = self.depth

        if self.left_child is not None:
            left_depth = self.left_child.max_depth_below()

        if self.right_child is not None:
            right_depth = self.right_child.max_depth_below()

        return max(left_depth, right_depth)
