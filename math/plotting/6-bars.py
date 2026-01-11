#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))


    people = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']  # apples, bananas, oranges, peaches

    # Bottom positions for stacking
    bottom = np.zeros(3)

    for i, color in enumerate(colors):
        plt.bar(
            people,
            fruit[i],
            bottom=bottom,
            color=color,
            width=0.5,
            label=['Apples', 'Bananas', 'Oranges', 'Peaches'][i]
        )
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()
