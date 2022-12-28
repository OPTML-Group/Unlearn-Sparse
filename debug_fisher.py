import torch
import numpy as np
# path_fm = "unlearn_results/debug/fisher_new_{}_{}/fisher_newcheckpoint.pth.tar"
# nums = [100, 450, 2250, 4500]
# alphas = [2e-8, 22e-9, 24e-9, 25e-9, 26e-9, 28e-9, 3e-8]
# for num in nums:
#     print(f"{num}:")
#     for alpha in alphas:
#         path = path_fm.format(num, alpha)
#         ret = torch.load(path)
#         acc = ret['evaluation_result']['accuracy']
#         a, b, c = acc['test'], acc['forget'], acc['retain']
#         print(int(alpha * 1e9), c, b, a)

# path_fm = "unlearn_results/debug_32/fisher_new_{}_{}_{}/fisher_newcheckpoint.pth.tar"
# nums = [100, 450, 2250, 4500]
# alphas = [2e-8, 22e-9, 24e-9, 25e-9, 26e-9, 28e-9, 3e-8]
# for num in nums:
#     print(f"{num}:")
#     for alpha in alphas:
#         q1, q2, q3 = [], [], []
#         for seed in [1,2,3,4,5]:
#             path = path_fm.format(num, alpha, seed)
#             ret = torch.load(path)
#             acc = ret['evaluation_result']['accuracy']
#             a, b, c = acc['test'], acc['forget'], acc['retain']
#             q1.append(c)
#             q2.append(b)
#             q3.append(a)
#             # print(int(alpha * 1e9), c, b, a)
#         print(int(alpha * 1e9), np.mean(q1), np.mean(q2), np.mean(q3))

path_fm = "unlearn_results/debug_w/wfisher_{}_{}_{}/wfishercheckpoint.pth.tar"
nums = [100, 450, 2250, 4500]
alphas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 0]
seeds = [1,2,3]
for num in nums:
    print(f"{num}:")
    for alpha in alphas:
        q1, q2, q3 = [], [], []
        for seed in seeds:
            path = path_fm.format(num, alpha, seed)
            ret = torch.load(path)
            acc = ret['evaluation_result']['accuracy']
            q1.append(acc['retain'])
            q2.append(acc['forget'])
            q3.append(acc['test'])
            # print(int(alpha * 1e9), c, b, a)
        print(alpha, np.mean(q1), np.mean(q2), np.mean(q3))