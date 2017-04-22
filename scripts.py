import numpy as np
import util
from torch.nn import SoftMarginLoss


logistic = SoftMarginLoss()
score = util.to_var(-100*np.ones(1000),volatile=True)
targets = util.to_var(np.ones(1000),volatile=True)
loss = logistic(score,targets)
print(loss.data.numpy())