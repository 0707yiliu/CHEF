import random

import numpy as np

q1 = np.array([-0.1830127, -0.1830127, -0.6830127, 0.6830127])
q2 = np.array([0.6830127, -0.6830127, 0.1830127, 0.1830127])
print(np.linalg.norm(q1), np.linalg.norm(q2))

q1 = q1 / np.linalg.norm(q1)
q2 = q2 / np.linalg.norm(q2)


dis = (abs(sum(q1*q2)))

print(dis)