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
from off_moo_baselines.data import tkwargs, get_dataloader, get_dataloader_1, get_preference_rankings
from off_moo_bench.task_set import *
from off_moo_bench.evaluation.metrics import hv


def run(config: dict):

    # versions : V_0(original_edition)、 V_1(单引导、添加权重排序规则)、 V_2(双引导,增加偏好权重引导模块)、 V_3(原版+交互)
    versions = config.get("version", "V_0")
    print(f"Running version: {versions}")

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

    logging_dir = os.path.join(config["results_dir"], run_name + ts_name + versions)
    os.makedirs(logging_dir, exist_ok=True)

    if config["use_wandb"]:
        if "wandb_api" in config.keys():
            wandb.login(key=config["wandb_api"])

        wandb.init(
            project="Offline-MOO",
            name=run_name + ts_name,
            config=config,
            group=f"{config['model']}-{config['train_mode']}",
            job_type=config["run_type"],
            mode="online",
            dir=os.path.join(config["results_dir"], ".."),
        )

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

    if os.path.exists(model_save_path) and None:
        model_uncond = Model_unconditional(dim=n_dim)
        load_model(model_uncond, model_save_path, device=tkwargs["device"])
        diffusion = Diffusion(img_size=n_dim, device=tkwargs["device"])
    else:
        model_uncond, diffusion = train(train_loader,model_save_path=model_save_path)
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

        if os.path.exists(preference_save_path) and None:
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

        if os.path.exists(preference_save_path_2) and None:
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
            
    X_d_best, d_best = task.get_N_non_dominated_solutions(
        N=256, return_x=True, return_y=True
    )
    try:
        res_y_pf_ideal = task.problem.get_pareto_front()
    except NotImplementedError:
        res_y_pf_ideal = None
    X_d_best = torch.tensor(X_d_best[-1]).unsqueeze(0).repeat(256, 1)

    # =========================
    # Dirichlet multi-w evaluation (FULL VERSION)
    # =========================

    def sample_dirichlet_w(n_obj, alpha=1.0):
        return np.random.dirichlet(alpha * np.ones(n_obj))

    def compute_weighted_scores(y, w):
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy()
        return (y * w).sum(axis=1)

    num_w_samples = config.get("num_w_samples", 1)

    print("===============================================")
    print(f"Running FULL multi-w evaluation, num_w_samples = {num_w_samples}")

    hv_list, hv20_list = [], []
    spread_list, spread20_list = [], []
    spacing_list, spacing20_list = [], []
    ped_list, ped20_list = [], []
    dataset_pref_scores = []
    all_max_weighted, all_mean_weighted = [], []


    if versions == "V_0" or versions == "V_3":

        samples = diffusion.sample_with_preference(
            model_uncond, 256, preference_model,
            torch.tensor(X_d_best),
            cfg_scale=10.0, return_latents=False, ddim=False
        )

        samples_20 = diffusion.sample_with_preference(
            model_uncond, 256, preference_model,
            torch.tensor(X_d_best),
            cfg_scale=20.0, return_latents=False, ddim=False
        )

        if config["normalize_xs"]:
            task.map_denormalize_x()
            samples = task.denormalize_x(samples.cpu().numpy())
            samples_20 = task.denormalize_x(samples_20.cpu().numpy())
        else:
            samples = samples.cpu().numpy()
            samples_20 = samples_20.cpu().numpy()

        res_y = task.predict(samples)
        res_y_20 = task.predict(samples_20)

        res_y_norm = task.normalize_y(res_y)
        res_y_20_norm = task.normalize_y(res_y_20)

        # ===== 所有非偏好指标只算一次 =====
        hv_list.append(hv(task.normalize_y(task.nadir_point), res_y_norm, config["task"]))
        hv20_list.append(hv(task.normalize_y(task.nadir_point), res_y_20_norm, config["task"]))

        spread_list.append(spread(res_y_norm))
        spread20_list.append(spread(res_y_20_norm))

        spacing_list.append(spacing(res_y_norm))
        spacing20_list.append(spacing(res_y_20_norm))

        ped_list.append(pairwise_euclidean_distances(res_y_norm))
        ped20_list.append(pairwise_euclidean_distances(res_y_20_norm))


        for _ in range(num_w_samples):
            w = sample_dirichlet_w(n_obj)
            print(f"[Sample {_}] w = {w}")
            dataset_scores = compute_weighted_scores(y, w)
            dataset_pref_scores.append(dataset_scores.max())
            weighted_scores = compute_weighted_scores(res_y_norm, w)
            all_max_weighted.append(weighted_scores.max())
            all_mean_weighted.append(weighted_scores.mean())
    else:
        for i in range(num_w_samples):
            w = sample_dirichlet_w(n_obj)
            w_tensor = torch.tensor(w, dtype=torch.float32).to(tkwargs["device"])

            print(f"[Sample {i}] w = {w_tensor}")

            if versions == "V_1":
                samples = diffusion.sample_with_preference_1(
                    model_uncond, 256, preference_model,
                    torch.tensor(X_d_best),
                    cfg_scale=10.0, return_latents=False, ddim=False, w=w_tensor
                )

                samples_20 = diffusion.sample_with_preference_1(
                    model_uncond, 256, preference_model,
                    torch.tensor(X_d_best),
                    cfg_scale=20.0, return_latents=False, ddim=False, w=w_tensor
                )

            elif versions == "V_2":
                samples = diffusion.sample_with_preference_2(
                    model_uncond, 256, preference_model, Preference_model_2,
                    torch.tensor(X_d_best),
                    cfg_scale=1, cfg_scale_p=2,return_latents=False, ddim=False, w=w_tensor
                )

                samples_20 = diffusion.sample_with_preference_2(
                    model_uncond, 256, preference_model, Preference_model_2,
                    torch.tensor(X_d_best),
                    cfg_scale=2, cfg_scale_p=4,return_latents=False, ddim=False, w=w_tensor
                )
                
                # samples = diffusion.sample_with_preference_3(
                #     model_uncond, 256, n_obj,preference_model, Preference_model_2,
                #     torch.tensor(X_d_best),
                #     cfg_scale=10.0, cfg_scale_p=10000,return_latents=False, ddim=False
                # )

                # samples_20 = diffusion.sample_with_preference_3(
                #     model_uncond, 256, n_obj, preference_model, Preference_model_2,
                #     torch.tensor(X_d_best),
                #     cfg_scale=20.0, cfg_scale_p=20000,return_latents=False, ddim=False
                # )

            if config["normalize_xs"]:
                task.map_denormalize_x()
                samples = task.denormalize_x(samples.cpu().numpy())
                samples_20 = task.denormalize_x(samples_20.cpu().numpy())
            else:
                samples = samples.cpu().numpy()
                samples_20 = samples_20.cpu().numpy()

            res_y = task.predict(samples)
            res_y_20 = task.predict(samples_20)

            res_y_norm = task.normalize_y(res_y)
            res_y_20_norm = task.normalize_y(res_y_20)

            # ===== 所有指标都 multi-w =====
            hv_list.append(hv(task.normalize_y(task.nadir_point), res_y_norm, config["task"]))
            hv20_list.append(hv(task.normalize_y(task.nadir_point), res_y_20_norm, config["task"]))

            spread_list.append(spread(res_y_norm))
            spread20_list.append(spread(res_y_20_norm))

            spacing_list.append(spacing(res_y_norm))
            spacing20_list.append(spacing(res_y_20_norm))

            ped_list.append(pairwise_euclidean_distances(res_y_norm))
            ped20_list.append(pairwise_euclidean_distances(res_y_20_norm))

            weighted_scores = compute_weighted_scores(res_y_norm, w)

            all_max_weighted.append(weighted_scores.max())
            all_mean_weighted.append(weighted_scores.mean())
            
            
            dataset_scores = compute_weighted_scores(y, w)
            dataset_pref_scores.append(dataset_scores.max())

    # =========================
    # FINAL aggregation
    # =========================
    final_hv = np.mean(hv_list)
    final_hv20 = np.mean(hv20_list)

    final_spread = np.mean(spread_list)
    final_spread20 = np.mean(spread20_list)

    final_spacing = np.mean(spacing_list)
    final_spacing20 = np.mean(spacing20_list)

    final_ped = np.mean(ped_list)
    final_ped20 = np.mean(ped20_list)

    final_max = np.mean(all_max_weighted)
    final_mean = np.mean(all_mean_weighted)

    dataset_pref_score_mean = np.mean(dataset_pref_scores)

    result_str = f"""
    === Final Averaged Metrics ===
    task: {config['task']}
    HV: {final_hv} | HV20: {final_hv20}
    Spread: {final_spread}
    Spacing: {final_spacing}
    PED: {final_ped}
    Pref max: {final_max}
    Pref mean: {final_mean}
    dataset_pref_score_mean: {dataset_pref_score_mean}
    """

    print(result_str)

    with open("./result_temp_v0.txt", "a") as f:
        f.write(result_str)

    def count_invalid_samples(samples):
        samples = np.array(samples)  # 转成 numpy
        lower = 0.0
        upper = 1.0
        
        invalid_mask = (samples < lower) | (samples > upper)
        invalid_count = np.sum(invalid_mask)
        invalid_rows = np.sum(np.any(invalid_mask, axis=1))
        
        return invalid_count, invalid_rows
    
    # invalid_count, invalid_rows = count_invalid_samples(samples)

    # print(f"元素级别不合法数量: {invalid_count}")
    # print(f"行级别至少有一个元素不合法的样本数量: {invalid_rows}")
    # # =========================
    # 写入结果
    # =========================
    hv_results = {
        "hypervolume_10/avg": final_hv,
        "hypervolume_20/avg": final_hv20,
        "Spread_10/avg": final_spread,
        "Spread_20/avg": final_spread20,
        "Spacing_10/avg": final_spacing,
        "Spacing_20/avg": final_spacing20,
        "ped_10/avg": final_ped,
        "ped_20/avg": final_ped20,
        "weighted_10/max_avg": final_max,
        "weighted_10/mean_avg": final_mean,
        "num_w_samples": num_w_samples,
        "dataset_pref_score_mean:": dataset_pref_score_mean,
    }
    
    df = pd.DataFrame([hv_results])

    filename = os.path.join(logging_dir, "hv_results.csv")

    df.to_csv(filename, index=False)

if __name__ == "__main__":
    from utils import process_args

    config = process_args(return_dict=True)

    save_dir = "./"
    results_dir = os.path.join(save_dir, "results")
    model_save_dir = os.path.join(save_dir, "model")

    config["results_dir"] = results_dir
    config["model_save_dir"] = model_save_dir
    run(config)
