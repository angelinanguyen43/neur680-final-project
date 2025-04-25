# quick 5-line notebook
import numpy as np
y_true = np.load('/Users/ajshul/Projects/final_proj/final_proj/results/sub-CSI1_resnet50_layer1_true.npy')[:, 12345]  # voxel index
y_pred = np.load('/Users/ajshul/Projects/final_proj/final_proj/results/sub-CSI1_resnet50_layer1_pred.npy')[:, 12345]

print(np.corrcoef(y_true, y_pred)[0,1])        # should be >.2 for a good voxel
print(((y_true-y_pred)**2).mean())             # small MSE if good
