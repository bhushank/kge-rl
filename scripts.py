import numpy as np
import util

scores = np.asarray([[2,3,1],[5,1,2]])
ranks = util.ranks(scores)
print(ranks)