from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/test_minimal')
writer.add_scalar('test/metric', 1.23, 0)
print('Wrote test/metric=1.23 to runs/test_minimal')
writer.close()
