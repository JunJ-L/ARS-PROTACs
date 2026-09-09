import os
import random
import logging
import collections

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import test


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_physchem_stats(model, dataloader):
    """Estimate component-wise physicochemical means and standard deviations and store them in model.physchem buffers."""
    fp_dim = model.physchem.fp_dim
    chunk_size = model.physchem.chunk_size
    num_components = model.physchem.num_components

    all_physchem = []
    for batch in dataloader:
        desc = batch.get("mol_descriptors")
        if desc is None:
            break
        B = desc.size(0)
        chunks = desc.view(B, num_components, chunk_size)
        physchem = chunks[:, :, fp_dim:]  # [B, 3, 6]
        all_physchem.append(physchem)

    if len(all_physchem) == 0:
        return

    all_physchem = torch.cat(all_physchem, dim=0)  # [N, 3, 6]
    mean = all_physchem.mean(dim=0)  # [3, 6]
    std = all_physchem.std(dim=0)    # [3, 6]

    model.physchem.physchem_mean.copy_(mean)
    model.physchem.physchem_std.copy_(std)
    print(f"  [init_physchem_stats] per-component mean:")
    for i, name in enumerate(["Warhead", "Linker", "E3_lig"]):
        print(f"    {name}: {mean[i].tolist()}")
    print(f"  [init_physchem_stats] per-component std:")
    for i, name in enumerate(["Warhead", "Linker", "E3_lig"]):
        print(f"    {name}: {std[i].tolist()}")


def valids(model, test_loader, device, desc="Validating"):
    """
    Returns (avg_loss, y_true, y_score).
    Threshold-dependent metrics are computed externally from the raw prediction scores.
    """
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        y_true, y_score = [], []
        total_loss, total_samples = 0.0, 0

        for data_sample in tqdm(test_loader, desc=desc, leave=False):
            y = data_sample['label'].to(device)
            batch_size = y.size(0)

            model_args = [
                data_sample['target_embed'].to(device),
                data_sample['target_tokens'].to(device),
                data_sample['warhead_graph'].to(device),
                data_sample['linker_graph'].to(device),
                data_sample['e3_ligand_graph'].to(device),
                data_sample['ligase_embed'].to(device),
                data_sample['ligase_tokens'].to(device)
            ]
            model_kwargs = {
                'mol_descriptors': data_sample['mol_descriptors'].to(device)
            }

            outputs = model(*model_args, **model_kwargs)

            loss = criterion(outputs, y)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.nn.functional.softmax(outputs, 1)
            y_score.extend(probs[:, 1].cpu().tolist())
            y_true.extend(y.cpu().tolist())

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        model.train()
        return avg_loss, y_true, y_score


def _cpu_state_dict(model):
    """Deep-copy an fp32 state_dict to CPU without consuming global RNG state."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train(model, lr=0.0001, epoch=100, train_loader=None, valid_loader=None,
          test_loader=None, device=None, writer=None, LOSS_NAME=None,
          accumulation_steps=8, patience=20, min_lr=1e-5,
          model_dir=None, test_name="test", random_seed=42):
    """
    Use the epoch with the highest validation AUROC as the anchor for six checkpoints:
      c1=best-9, c2=best-5, c3=best, c4=best+5, c5=best+14, and c6=final.
    Ignore out-of-range offsets, save all available checkpoints after training, and evaluate
    them on the independent test set. Freeze alpha_t and alpha_l for the first 100 epochs,
    then unfreeze them automatically. Both rpi.alpha_* and top-level alpha_* names are supported.
    """

    # Estimate physicochemical normalization statistics from the training set, then reset RNG.
    init_physchem_stats(model, train_loader)
    set_seed(random_seed)

    model = model.to(device)

    parameter_names = {name for name, _ in model.named_parameters()}
    if {'rpi.alpha_t', 'rpi.alpha_l'} <= parameter_names:
        freeze_params_until = {'rpi.alpha_t': 100, 'rpi.alpha_l': 100}
    elif {'alpha_t', 'alpha_l'} <= parameter_names:
        freeze_params_until = {'alpha_t': 100, 'alpha_l': 100}
    else:
        raise RuntimeError(
            "Cooperativity parameters not found: expected either "
            "rpi.alpha_t/rpi.alpha_l or alpha_t/alpha_l"
        )

    best_val_auroc = float('-inf')
    best_epoch = 0
    epochs_no_improve = 0
    early_stop = False

    weight = torch.Tensor([0.8158508, 1.29151292]).to(device)  # Class weights (positive/negative ≈ 1.583)
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5,
                                  min_lr=min_lr)

    val_loss, val_true, val_score = valids(model, valid_loader, device, desc="Initial Val")
    val_auroc = roc_auc_score(val_true, val_score)
    val_aupr = average_precision_score(val_true, val_score)
    logging.info(f"Initial Validation - Loss: {val_loss:.6f}, AUROC: {val_auroc:.6f}, AUPR: {val_aupr:.6f}")
    print(f"Initial Validation - Loss: {val_loss:.6f}, AUROC: {val_auroc:.6f}, AUPR: {val_aupr:.6f}")

    WINDOW = 10  # Minimum lookback window required to retain best-9.
    weight_window = collections.OrderedDict()  # {epoch_num: cpu_state_dict}
    saved_ckpts = {}       # {'c1'..'c6': (epoch_num, cpu_state_dict)}
    pending_offsets = {}   # {future_epoch_num: 'c4'/'c5'}
    last_epoch_num = 0
    last_sd = None
    total_oom = 0  # Total number of OOM events during training.

    for epo in range(epoch):
        if early_stop:
            print(f'Early stopping triggered at epoch {epo}')
            break

        if freeze_params_until:
            for pname, unfreeze_epoch in freeze_params_until.items():
                for name, param in model.named_parameters():
                    if name == pname:
                        if epo + 1 >= unfreeze_epoch and not param.requires_grad:
                            param.requires_grad_(True)
                            print(f'  [Unfreeze] {pname} at epoch {epo+1}')
                        elif epo + 1 < unfreeze_epoch and param.requires_grad:
                            param.requires_grad_(False)

        model.train()
        running_loss, total_num = 0.0, 0
        oom_count = 0
        opt.zero_grad()

        for i, data_sample in enumerate(train_loader):
            try:
                model_args = [
                    data_sample['target_embed'].to(device),
                    data_sample['target_tokens'].to(device),
                    data_sample['warhead_graph'].to(device),
                    data_sample['linker_graph'].to(device),
                    data_sample['e3_ligand_graph'].to(device),
                    data_sample['ligase_embed'].to(device),
                    data_sample['ligase_tokens'].to(device)
                ]
                model_kwargs = {
                    'mol_descriptors': data_sample['mol_descriptors'].to(device)
                }

                outputs = model(*model_args, **model_kwargs)

                y = data_sample['label'].to(device)
                current_batch_size = y.size(0)

                loss = criterion(outputs, y)
                loss = loss / accumulation_steps
                loss.backward()

                running_loss += loss.item() * accumulation_steps * current_batch_size
                total_num += current_batch_size

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    opt.zero_grad()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    oom_count += 1
                    torch.cuda.empty_cache()
                    opt.zero_grad()
                else:
                    raise e

        if (i + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()

        avg_loss = running_loss / total_num if total_num > 0 else 0

        val_loss, val_true, val_score = valids(model, valid_loader, device, desc="Validating")
        val_auroc = roc_auc_score(val_true, val_score)
        val_aupr = average_precision_score(val_true, val_score)
        scheduler.step(val_auroc)
        cur_lr = opt.param_groups[0]['lr']

        logging.info(f"Epoch {epo+1:>3d}/{epoch}  |  Train Loss: {avg_loss:<10.6f}  |  Val Loss: {val_loss:<10.6f}  |  "
                     f"AUROC: {val_auroc:<10.6f}  |  AUPR: {val_aupr:<10.6f}  |  LR: {cur_lr:<9.2e}  |  OOM: {oom_count:<3d}")

        print(f"Epoch {epo+1:>3d}/{epoch}  |  Train-loss: {avg_loss:<10.6f}  |  Val-loss: {val_loss:<10.6f}  |  "
              f"Val-AUROC: {val_auroc:<10.6f}  |  LR: {cur_lr:<9.2e}  |  OOM: {oom_count:<3d}")
        if oom_count > 0:
            print(f"  [WARNING] {oom_count} batch(es) skipped this epoch due to CUDA out of memory.")
        total_oom += oom_count

        if writer:
            writer.add_scalar(f"{LOSS_NAME}/train_loss", avg_loss, epo)
            writer.add_scalar(f"{LOSS_NAME}/val_loss", val_loss, epo)
            writer.add_scalar(f"{LOSS_NAME}/val_AUROC", val_auroc, epo)
            writer.add_scalar(f"{LOSS_NAME}/val_AUPR", val_aupr, epo)
            writer.add_scalar(f"{LOSS_NAME}/learning_rate", cur_lr, epo)

        epoch_num = epo + 1
        cur_sd = _cpu_state_dict(model)
        last_epoch_num = epoch_num
        last_sd = cur_sd

        # Insert before trimming so a new optimum retains the full [best-9, best] window.
        weight_window[epoch_num] = cur_sd
        while len(weight_window) > WINDOW:
            weight_window.popitem(last=False)

        # Cache future offsets (c4/c5) when their registered epochs are reached.
        if epoch_num in pending_offsets:
            saved_ckpts[pending_offsets.pop(epoch_num)] = (epoch_num, cur_sd)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epo
            epochs_no_improve = 0
            saved_ckpts = {}
            pending_offsets = {}
            saved_ckpts['c3'] = (epoch_num, cur_sd)
            if (epoch_num - 5) in weight_window:
                saved_ckpts['c2'] = (epoch_num - 5, weight_window[epoch_num - 5])
            if (epoch_num - 9) in weight_window:
                saved_ckpts['c1'] = (epoch_num - 9, weight_window[epoch_num - 9])
            pending_offsets[epoch_num + 5] = 'c4'
            pending_offsets[epoch_num + 14] = 'c5'
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True

    if last_sd is not None:
        saved_ckpts['c6'] = (last_epoch_num, last_sd)  # c6 always stores the final epoch.

    if model_dir is not None and saved_ckpts:
        os.makedirs(model_dir, exist_ok=True)
        for tag in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']:
            if tag in saved_ckpts:
                en, sd = saved_ckpts[tag]
                torch.save(sd, os.path.join(model_dir, f"{tag}-ep{en}.pt"))  # Keep duplicate epochs under distinct tags.

        if test_loader is not None:
            test.evaluate_checkpoints(model, model_dir, test_loader, test_name, device)

    if total_oom > 0:
        logging.info(f"[WARNING] This run triggered CUDA out-of-memory (OOM) and skipped {total_oom} "
                     f"batch(es) in total. Please switch to a device with larger VRAM or reduce the "
                     f"batch size; the current results are UNRELIABLE.")

    print(f'Training complete. Best Val AUROC: {best_val_auroc:.6f} at epoch {best_epoch+1}')

    return model, best_val_auroc
