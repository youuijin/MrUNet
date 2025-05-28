import torch
import time

a = torch.zeros((256, 256, 256)).cuda()
print(a.shape)

time.sleep(5)
print('finish')