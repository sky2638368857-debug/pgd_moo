import os
import sys
import wandb
import torch
import numpy as np
import pandas as pd
import datetime
import json
from copy import deepcopy

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(BASE_PATH)

import off_moo_bench as ob
from utils import (
    set_seed,
    get_quantile_solutions,
    spread,
    spacing,
    pairwise_euclidean_distances,
)
from off_moo_baselines.diffusion_guidance.ddpm_guidance import (
    train,
    Diffusion,
    train_preference,
    train_preference_1,
    train_preference_2,
)
from off_moo_baselines.diffusion_guidance.modules import (
    Preference_model,
    Preference_model_1,
    Preference_model_3,
    Model_unconditional,
    save_model,
    load_model,
)
from off_moo_baselines.data import tkwargs, get_dataloader, get_dataloader_1,get_preference_rankings
from off_moo_bench.task_set import *
from off_moo_bench.evaluation.metrics import hv

def evaluate_preference(model, dataloader, diffusion, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_1, x_2, y in dataloader:
            x_1 = x_1.to(device)
            x_2 = x_2.to(device)
            y = y.to(device)

            # 和训练保持一致
            x_1 = x_1 * 2 - 1
            x_2 = x_2 * 2 - 1

            t = diffusion.sample_timesteps(x_1.shape[0]).to(device)
            x_1, _ = diffusion.noise_images(x_1, t)
            x_2, _ = diffusion.noise_images(x_2, t)

            pred = model(x_1, x_2, t)

            # 取预测类别
            pred_label = torch.argmax(pred, dim=1)

            correct += (pred_label == y.squeeze().long()).sum().item()
            total += y.shape[0]

    acc = correct / total
    print(f"Preference Model Accuracy: {acc:.4f}")

    return acc

def run(config: dict):
    if config["task"] in ALLTASKSDICT.keys():
        config["task"] = ALLTASKSDICT[config["task"]]
    results_dir = os.path.join(
        config["results_dir"],
        f"{config['model']}-{config['train_mode']}-{config['task']}",
    )
    config["results_dir"] = results_dir
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f"-ts-{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"
    run_name = f"{config['model']}-{config['train_mode']}-seed{config['seed']}-{config['task']}"

    logging_dir = os.path.join(config["results_dir"], run_name + ts_name)
    os.makedirs(logging_dir, exist_ok=True)

    with open(os.path.join(logging_dir, "params.json"), "w") as f:
        json.dump(config, f, indent=4)

    set_seed(config["seed"])

    task = ob.make(config["task"])

    X = task.x.copy()
    y = task.y.copy()
    if config["subsample"]:
        X_pref, y_pref = task.get_N_non_dominated_solutions(
            N=int(X.shape[0] * config["subsample_ratio"]), return_x=True, return_y=True
        )

    X_test = task.x_test.copy()
    y_test = task.y_test.copy()

    if config["to_logits"]:
        assert task.is_discrete
        task.map_to_logits()
        X = task.to_logits(X)
        X_test = task.to_logits(X_test)
    if config["normalize_xs"]:
        task.map_normalize_x()
        X = task.normalize_x(X)
        X_test = task.normalize_x(X_test)
    if config["normalize_ys"]:
        task.map_normalize_y()
        y = task.normalize_y(y)
        y_test = task.normalize_y(y_test)

    if config["to_logits"]:
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
        X_test = X_test.reshape(-1, n_dim * n_classes)
    else:
        data_size, n_dim = tuple(X.shape)
    n_obj = y.shape[1]
    hypervolumes = []
    for i in range(y.shape[0]):
        # hypervolumes.append(hv(task.normalize_y(task.nadir_point), y[i], config['task']))
        hypervolumes.append(1.0)
    ind_pareto_rank = None
    if config["use_diversity_metric"]:
        ind_pareto_rank = get_preference_rankings(
            y_pref if config["subsample"] else y,
            task.normalize_y(task.nadir_point),
            config["task"],
            config["diversity_metric"],
        )
    hypervolumes = np.array(hypervolumes)
    model_save_dir = config["model_save_dir"]
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(
        model_save_dir,
        f"{config['model']}-{config['train_mode']}-{config['task']}-{config['seed']}-0.pt",
    )

    versions = "V_3"

    if versions == "V_0":
        preference_save_path = model_save_path.replace("-0.pt", "-preference.pt")

    elif versions == "V_1":
        preference_save_path = model_save_path.replace("-0.pt", "-preference_1.pt")

    elif versions == "V_2":
        preference_save_path = model_save_path.replace("-0.pt", "-preference.pt")
        preference_save_path_2 = model_save_path.replace("-0.pt", "-preference_2.pt")
    
    elif versions == "V_3":
        preference_save_path = model_save_path.replace("-0.pt", "-preference_3.pt")

    if versions == "V_0" or versions == "V_3":
        (train_loader_pref, val_loader_pref, _, train_loader, _, _) = get_dataloader(
            X,
            y,
            X_test,
            y_test,
            X_pref=X_pref if config["subsample"] else None,
            y_pref=y_pref if config["subsample"] else None,
            val_ratio=0.9,
            batch_size=config["batch_size"],
            preference_loader=True,
            hypervolumes=hypervolumes,
            three_dim_out=config["three_dim_out"],
            use_diversity_metric=config["use_diversity_metric"],
            pareto_rankings=ind_pareto_rank,
            diversity_score_threshold=config["diversity_score_threshold"],
        )
    elif versions == "V_1":
        (train_loader_pref, val_loader_pref, _, train_loader, _, _) = get_dataloader_1(
            X,
            y,
            X_test,
            y_test,
            X_pref=X_pref if config["subsample"] else None,
            y_pref=y_pref if config["subsample"] else None,
            val_ratio=0.9,
            batch_size=config["batch_size"],
            preference_loader=True,
            hypervolumes=hypervolumes,
            three_dim_out=config["three_dim_out"],
            use_diversity_metric=config["use_diversity_metric"],
            pareto_rankings=ind_pareto_rank,
            diversity_score_threshold=config["diversity_score_threshold"],
        )
    elif versions == "V_2":
        (train_loader_pref, val_loader_pref, _, train_loader, _, _) = get_dataloader(
            X,
            y,
            X_test,
            y_test,
            X_pref=X_pref if config["subsample"] else None,
            y_pref=y_pref if config["subsample"] else None,
            val_ratio=0.9,
            batch_size=config["batch_size"],
            preference_loader=True,
            hypervolumes=hypervolumes,
            three_dim_out=config["three_dim_out"],
            use_diversity_metric=config["use_diversity_metric"],
            pareto_rankings=ind_pareto_rank,
            diversity_score_threshold=config["diversity_score_threshold"],
        )

        (train_loader_pref_p2, val_loader_pref_p2, _, train_loader, _, _) = get_dataloader_1(
            X,
            y,
            X_test,
            y_test,
            X_pref=X_pref if config["subsample"] else None,
            y_pref=y_pref if config["subsample"] else None,
            val_ratio=0.9,
            batch_size=config["batch_size"],
            preference_loader=True,
            hypervolumes=hypervolumes,
            three_dim_out=config["three_dim_out"],
            use_diversity_metric=config["use_diversity_metric"],
            pareto_rankings=ind_pareto_rank,
            diversity_score_threshold=config["diversity_score_threshold"],
        )

    if os.path.exists(model_save_path):
        model_uncond = Model_unconditional(dim=n_dim)
        load_model(model_uncond, model_save_path, device=tkwargs["device"])
        diffusion = Diffusion(img_size=n_dim, device=tkwargs["device"])
    else:
        print("===============Training unconditional model..." + model_save_path)
        model_uncond, diffusion = train(train_loader)
        save_model(
            model=model_uncond, save_path=model_save_path, device=model_uncond.device
        )
        
    if versions == "V_0":
        if os.path.exists(preference_save_path):
            preference_model = Preference_model(
                input_dim=train_loader.dataset[0][0].shape[-1],
                device=tkwargs["device"],
                three_dim_out=config["three_dim_out"],
            ).to(tkwargs["device"])
            load_model(preference_model, preference_save_path, device=tkwargs["device"])
        else:
            preference_model = train_preference(
                dataloader=train_loader_pref,
                diffusion=diffusion,
                val_loader=val_loader_pref,
                config=config,
                model_save_path=preference_save_path,
                three_dim_out=config["three_dim_out"],
            )
    elif versions == "V_1":
        if os.path.exists(preference_save_path):
            preference_model = Preference_model_1(
                input_dim=train_loader.dataset[0][0].shape[-1],
                device=tkwargs["device"],
                three_dim_out=config["three_dim_out"],
                w_dim = n_obj,
            ).to(tkwargs["device"])
            load_model(preference_model, preference_save_path, device=tkwargs["device"])
        else:
            preference_model = train_preference_1(
                dataloader=train_loader_pref,
                diffusion=diffusion,
                val_loader=val_loader_pref,
                config=config,
                model_save_path=preference_save_path,
                three_dim_out=config["three_dim_out"],
                w_dim = n_obj,
            )
    elif versions == "V_2":

        if os.path.exists(preference_save_path):
            preference_model = Preference_model(
                input_dim=train_loader.dataset[0][0].shape[-1],
                device=tkwargs["device"],
                three_dim_out=config["three_dim_out"],
            ).to(tkwargs["device"])
            load_model(preference_model, preference_save_path, device=tkwargs["device"])
        else:
            preference_model = train_preference(
                dataloader=train_loader_pref,
                diffusion=diffusion,
                val_loader=val_loader_pref,
                config=config,
                model_save_path=preference_save_path,
                three_dim_out=config["three_dim_out"],
            )

        if os.path.exists(preference_save_path_2):
            Preference_model_2 = Preference_model_1(
                input_dim=train_loader.dataset[0][0].shape[-1],
                device=tkwargs["device"],
                three_dim_out=config["three_dim_out"],
                w_dim = n_obj,
            ).to(tkwargs["device"])
            load_model(Preference_model_2, preference_save_path_2, device=tkwargs["device"])
        else:
            Preference_model_2 = train_preference_2(
                dataloader=train_loader_pref_p2,
                diffusion=diffusion,
                val_loader=val_loader_pref_p2,
                config=config,
                model_save_path=preference_save_path_2,
                three_dim_out=config["three_dim_out"],
                w_dim = n_obj,
            )
    elif versions == "V_3":
        if os.path.exists(preference_save_path):
            preference_model = Preference_model_3(
                input_dim=train_loader.dataset[0][0].shape[-1],
                device=tkwargs["device"],
                three_dim_out=config["three_dim_out"],
            ).to(tkwargs["device"])
            load_model(preference_model, preference_save_path, device=tkwargs["device"])
        else:
            preference_model = train_preference(
                dataloader=train_loader_pref,
                diffusion=diffusion,
                val_loader=val_loader_pref,
                config=config,
                model_save_path=preference_save_path,
                three_dim_out=config["three_dim_out"],
            )
    
    acc = evaluate_preference(preference_model, val_loader_pref, diffusion, tkwargs["device"])
    # ===== 保存到统一文件 =====
    result_file = "./preference_results.csv"

    result_dict = {
        "task": config["task"],
        "model": config["model"],
        "train_mode": config["train_mode"],
        "seed": config["seed"],
        "accuracy": acc,
        "timestamp": ts_name,
    }

    df = pd.DataFrame([result_dict])

    # 如果文件存在 → 追加
    if os.path.exists(result_file):
        df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        df.to_csv(result_file, index=False)
if __name__ == "__main__":
    from utils import process_args

    config = process_args(return_dict=True)

    save_dir = "./"
    results_dir = os.path.join(save_dir, "results")
    model_save_dir = os.path.join(save_dir, "model")

    config["results_dir"] = results_dir
    config["model_save_dir"] = model_save_dir
    run(config)
