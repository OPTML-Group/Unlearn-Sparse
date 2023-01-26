import os
import shutil
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

rerun = True
sparsities = "0.0 0.5 0.75 0.9 0.95 0.99 0.995".split(' ')
ks = [1, 3, 5]
epochs = list(range(0, 181, 20))
lanczos_dim = 10
eigens = np.zeros([len(sparsities), len(epochs), lanczos_dim])


def plot_eigens(eigens, remove_previous=False):
    dir = "fig_eigen"
    if remove_previous:
        shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir, exist_ok=True)

    topks = [np.mean(eigens[:, :, :k], axis=-1) for k in ks]
    for sp_id, sparsity in enumerate(sparsities):
        plt.clf()
        for k_id, k in enumerate(ks):
            plt.plot(epochs, topks[k_id][sp_id], label=f"top{k}")
        plt.legend()
        plt.savefig(os.path.join(dir, f"sparsity{sparsity}.png"))

    for k_id, k in enumerate(ks):
        plt.clf()
        for sp_id, sparsity in enumerate(sparsities):
            plt.plot(epochs, topks[k_id][sp_id], label=f"{sparsity}")
        plt.legend()
        plt.savefig(os.path.join(dir, f"top{k}.png"))


def load_eigens():
    for sp_id, sparsity in enumerate(sparsities):
        for ep_id, epoch in enumerate(epochs):
            path = f"eigen_results/single/{sparsity}_epoch{epoch}/eigen_{lanczos_dim}.pkl"
            with open(path, 'rb') as fin:
                vec, weight = pkl.load(fin)
            eigens[sp_id, ep_id] = np.abs(np.linalg.eigvals(weight))

            # from IPython import embed
            # embed()

    np.sort(eigens, axis=-1)
    return eigens


def main():
    save_path = "eigen_results/eigen_values.pkl"

    if not os.path.exists(save_path) or rerun:
        eigens = load_eigens()
        with open(save_path, 'wb') as fout:
            pkl.dump(eigens, fout)
    else:
        with open(save_path, 'rb') as fin:
            eigens = pkl.load(fin)

    plot_eigens(eigens, False)


if __name__ == "__main__":
    main()
