import numpy as np

from pymoo.problems import get_problem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions


def run():

    # =========================
    # problem
    # =========================
    problem = get_problem("dtlz1")
    # =========================
    # MOEA/D
    # =========================
    ref_dirs = get_reference_directions(
        "das-dennis",
        3,              # ⚠️ DTLZ1 默认至少建议 3 objectives
        n_partitions=12
    )

    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", 200),
        seed=0,
        verbose=True
    )

    Y = res.F

    # =========================
    # preference weight
    # =========================
    w = np.random.dirichlet(np.ones(Y.shape[1]))
    print("Preference weight w:", w)

    # =========================
    # min-max normalization
    # =========================
    y_min = Y.min(axis=0)
    y_max = Y.max(axis=0)
    Y_norm = (Y - y_min) / (y_max - y_min + 1e-12)

    # =========================
    # preference score
    # =========================
    pref_scores = (Y_norm * w).sum(axis=1)

    best_idx = np.argmin(pref_scores)

    best_score = pref_scores[best_idx]
    best_Y = Y[best_idx]

    # =========================
    # output
    # =========================
    print("=================================")
    print("Preference scores (top 5):")

    topk = np.argsort(pref_scores)[:5]
    for i in topk:
        print(f"idx={i}, score={pref_scores[i]:.6f}, Y={Y[i]}")

    print("=================================")
    print("Best preference solution:")
    print("best idx:", best_idx)
    print("best score:", best_score)
    print("best Y:", best_Y)
    # =========================
    # preference weight w
    # =========================
    # w = np.random.dirichlet(np.ones(Y.shape[1]))  # 或者你手动指定
    w = [0.1,0.2,0.7]
    print("Preference weight w:", w)

    # =========================
    # select best solution for w
    # =========================
    weighted_scores = (Y_norm * w).sum(axis=1)
    best_idx = np.argmin(weighted_scores)

    best_Y = Y[best_idx]
    best_Y_norm = Y_norm[best_idx]

    print("=================================")
    print("MOEA/D (pure pymoo)")
    print("Best index:", best_idx)
    print("Best objective (original):", best_Y)
    print("Best objective (normalized):", best_Y_norm)

    # =========================
    # metrics
    # =========================
    hv_proxy = np.mean(1 - np.linalg.norm(Y_norm, axis=1))
    spread = np.mean(np.std(Y_norm, axis=0))

    dist = np.linalg.norm(Y_norm[:, None] - Y_norm[None, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    spacing = np.mean(np.min(dist, axis=1))

    print("HV proxy:", hv_proxy)
    print("Spread:", spread)
    print("Spacing:", spacing)


if __name__ == "__main__":
    run()