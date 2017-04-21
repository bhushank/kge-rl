import torch
x = torch.randn(2,3,5)
y = torch.randn(2,3,1)
r = torch.randn(2,3,1)
r = r.expand_as(x)
y = y.expand_as(x)
z = torch.sum(torch.mul(torch.mul(x,y),r),1)
#z = x+r.expand_as(x)-y.expand_as(x)
print(z)