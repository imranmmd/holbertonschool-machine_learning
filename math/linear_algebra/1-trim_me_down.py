#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
the_middle.append(matrix[0][len(matrix[0])//2-1:len(matrix[0])//2+1])
the_middle.append(matrix[1][len(matrix[1])//2-1:len(matrix[1])//2+1])
the_middle.append(matrix[2][len(matrix[2])//2-1:len(matrix[2])//2+1])
print("The middle columns of the matrix are: {}".format(the_middle))
