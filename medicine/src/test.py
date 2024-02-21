import torch
import numpy as np

gt = np.random.randint(0,3, size=[5,5]) #先生成一个15*15的label，值在5以内，意思是5类分割任务
gt = torch.LongTensor(gt)

def get_one_hot(label,N):
  size = list(label.size())
  label = label.view(-1)              # reshape 为向量 25
  #ones = torch.sparse.torch.eye(N)
  ones = torch.eye(N)                 #[3,3]
  ones = ones.index_select(0, label)  # 用上面的办法转为换one hot [25,3]
  size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
  ones = ones.view(*size)
  return torch.permute(ones,[2,0,1])

if __name__ == '__main__':
  print(gt)
  gt_one_hot = get_one_hot(gt, 3) #[3,5,5]
  print(gt_one_hot)
  print(gt_one_hot.shape)
  result = gt_one_hot.argmax(0)
  print(result)
  print(gt_one_hot.argmax(0) == gt)  # 判断one hot 转换方式是否正确，全是1就是正确的

# tensor([[2, 2, 1, 1, 0],
#         [2, 1, 0, 2, 2],
#         [0, 2, 2, 0, 0],
#         [2, 0, 2, 1, 0],
#         [0, 1, 2, 0, 2]])
# tensor([[[0., 0., 0., 0., 1.],
#          [0., 0., 1., 0., 0.],
#          [1., 0., 0., 1., 1.],
#          [0., 1., 0., 0., 1.],
#          [1., 0., 0., 1., 0.]],
#
#         [[0., 0., 1., 1., 0.],
#          [0., 1., 0., 0., 0.],
#          [0., 0., 0., 0., 0.],
#          [0., 0., 0., 1., 0.],
#          [0., 1., 0., 0., 0.]],
#
#         [[1., 1., 0., 0., 0.],
#          [1., 0., 0., 1., 1.],
#          [0., 1., 1., 0., 0.],
#          [1., 0., 1., 0., 0.],
#          [0., 0., 1., 0., 1.]]])
# torch.Size([3, 5, 5])
# tensor([[2, 2, 1, 1, 0],
#         [2, 1, 0, 2, 2],
#         [0, 2, 2, 0, 0],
#         [2, 0, 2, 1, 0],
#         [0, 1, 2, 0, 2]])
# tensor([[True, True, True, True, True],
#         [True, True, True, True, True],
#         [True, True, True, True, True],
#         [True, True, True, True, True],
#         [True, True, True, True, True]])
