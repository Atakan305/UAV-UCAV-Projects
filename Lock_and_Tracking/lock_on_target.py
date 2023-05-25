import random
import time
import math

a = (33.65771288706495, 92.21570806743144, 78.32326161621613)


def distance(x, y, z, s, d, f):
    c = ((x - s) ** 2) + ((y - d) ** 2) + ((z - f) ** 2)
    return math.sqrt(c)


def nearest_plane(a, b, c, d, e, f, g, h, k, l):
    liste = [a, b, c, d, e, f, g, h, k, l]
    return min(liste)


while True:
    time.sleep(1)
    counter = 0
    liste = [None] * 10
    liste1 = [None] * 10
    number = 10
    n = 0
    while counter < 10:
        e = random.uniform(30, 40)
        b = random.uniform(90, 99)
        i = random.uniform(50, 100)

        liste[counter] = [e, b, i]
        # print(liste[counter])
        counter = counter + 1

    for g in range(number):
        liste1[g] = [distance(liste[g][n], liste[g][n + 1], liste[g][n + 2], a[0], a[1], a[2])]
        # print(liste1[g])

    print(
        nearest_plane(liste1[0], liste1[1], liste1[2], liste1[3], liste1[4], liste1[5], liste1[6], liste1[7], liste1[8],
                      liste1[9]))
    time.sleep(10)
