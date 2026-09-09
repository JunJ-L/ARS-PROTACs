import os
import importlib
import random
import logging
import datetime
from pathlib import Path

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from torch.utils.data import DataLoader

from dataset import PROTACDataset, collater
from train import train
from test import THRESHOLDS, DEFAULT_THRESHOLD

BATCH_SIZE = 2
EPOCH = 200
PATIENCE = 30
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

MODEL_MODULE = "model"

# Log timezone offset in hours. The default is UTC+8 (Beijing time); many servers use UTC.
# Change this value if log timestamps differ from local time (use 0 for UTC or -5 for US Eastern Time).
LOG_TZ_OFFSET_HOURS = 8

# All four splits share the same training and evaluation workflow. Reporting thresholds are
# centralized in test.THRESHOLDS, and the best checkpoint is selected by AUROC.
SPLIT_CONFIGS = [
    {"name": "random",      "prefix": "Random_",   "test_csv": "data/Random_test.csv",      "test_name": "Random_test"},
    {"name": "cold-drug",   "prefix": "Cold_",     "test_csv": "data/Cold_test_drug.csv",   "test_name": "Cold_test_drug"},
    {"name": "cold-target", "prefix": "Cold_",     "test_csv": "data/Cold_test_target.csv", "test_name": "Cold_test_target"},
    {"name": "temporal",    "prefix": "Temporal_", "test_csv": None,                        "test_name": "Temporal"},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _log_now():
    """Return the current time using LOG_TZ_OFFSET_HOURS to keep server logs in the intended timezone."""
    tz = datetime.timezone(datetime.timedelta(hours=LOG_TZ_OFFSET_HOURS))
    return datetime.datetime.now(tz)


def get_next_demo_dir():
    """Create and return the next run directory under log/ (log/demo_001, log/demo_002, ...)."""
    log_root = Path('log')
    log_root.mkdir(exist_ok=True)
    max_idx = 0
    for p in log_root.glob('demo_*'):
        suffix = p.name[len('demo_'):]
        if p.is_dir() and suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    demo_dir = log_root / f"demo_{max_idx + 1:03d}"
    demo_dir.mkdir()
    return demo_dir


def _setup_logging(log_path):
    """Direct the root logger to the split-specific Markdown file and return its FileHandler."""
    logger = logging.getLogger()
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return fh


def _finalize_log_header(log_path, start_time, end_time):
    """Insert the end time and duration after the first log line when training completes successfully."""
    duration = end_time - start_time
    total_seconds = int(duration.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    insert = [
        f"结束时间: {end_time:%Y-%m-%d %H:%M:%S}\n",
        f"持续时间: {h:d}:{m:02d}:{s:02d}\n",
    ]
    lines[1:1] = insert
    with open(log_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def run_single_model(config, log_dir):
    name = config['name']
    prefix = config['prefix']
    test_name = config['test_name']

    log_path = f"{log_dir}/{name}.md"
    start_time = _log_now()
    fh = _setup_logging(log_path)
    logging.info(f"运行时间: {start_time:%Y-%m-%d %H:%M:%S}")

    success = False
    best_val_auroc = None
    end_time = None

    try:
        # Import the model module before setting the random seed to preserve reproducible initialization order.
        mod = importlib.import_module(MODEL_MODULE)
        ESMWrapper = mod.ESMWrapper
        GraphTransformer = mod.GraphTransformer
        Model = mod.Model
        esm_model_name = getattr(mod, 'ESM_MODEL_NAME', 'esm2_t6_8M_UR50D')
        esm_repr_layer = getattr(mod, 'ESM_REPR_LAYER', 6)

        set_seed(RANDOM_SEED)

        train_csv = f'data/{prefix}train.csv'
        val_csv = f'data/{prefix}val.csv'
        test_csv = config['test_csv'] if config['test_csv'] is not None else val_csv

        train_mol2_dir = 'data/mol2_files/'
        val_mol2_dir = 'data/mol2_files/'
        test_mol2_dir = 'data/mol2_files/'

        # Protein ESM representations are inferred on demand by ESMEmbedder in dataset.__getitem__;
        # no cached embedding directory is read.
        print(f"\n{'='*80}")
        print(f"  RUNNING: {name} (prefix: {prefix})")
        print(f"{'='*80}")

        desc_mode = "all"  # Three-component physicochemical descriptors matching the [3, 6] buffers
        train_dataset = PROTACDataset(train_mol2_dir, train_csv, use_descriptors=True, desc_mode=desc_mode,
                                      esm_model_name=esm_model_name, esm_repr_layer=esm_repr_layer)
        val_dataset = PROTACDataset(val_mol2_dir, val_csv, use_descriptors=True, desc_mode=desc_mode,
                                    esm_model_name=esm_model_name, esm_repr_layer=esm_repr_layer)
        print(f"  Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")

        num_workers = 0
        trainloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collater, num_workers=num_workers, pin_memory=False, drop_last=True
        )
        valloader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE,
            collate_fn=collater, num_workers=num_workers, pin_memory=False
        )

        ligase_model = ESMWrapper()
        target_model = ESMWrapper()
        ESM_DIM = 128
        esm_protein_dim = getattr(mod, 'ESM_PROTEIN_DIM', 320)

        target_ligand_model = GraphTransformer(num_embeddings=10, dim=ESM_DIM)
        ligase_ligand_model = GraphTransformer(num_embeddings=10, dim=ESM_DIM)
        linker_model = GraphTransformer(num_embeddings=10, dim=ESM_DIM)

        model = Model(
            ligase_ligand_model, ligase_model,
            target_ligand_model, target_model,
            linker_model, dim=esm_protein_dim
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,} | Trainable: {trainable_params:,}")

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {device}")
        model = model.to(device)

        demo_name = os.path.basename(str(log_dir))
        if SummaryWriter is not None:
            writer = SummaryWriter(f'runs/{demo_name}/{name}')
            print(f"  TensorBoard logging to: runs/{demo_name}/{name}")
        else:
            writer = None
            print(f"  TensorBoard unavailable (tensorboard not installed), skipping")

        # Preserve RNG state because constructing the test dataset must not alter training randomness.
        if test_csv == val_csv:
            testloader = valloader
            test_size = len(val_dataset)
            print(f"  test_csv == val_csv, reusing valloader")
        else:
            rng_state = random.getstate()
            np_rng_state = np.random.get_state()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

            test_dataset = PROTACDataset(test_mol2_dir, test_csv, use_descriptors=True, desc_mode=desc_mode,
                                         esm_model_name=esm_model_name, esm_repr_layer=esm_repr_layer)
            test_size = len(test_dataset)
            print(f"  Test: {len(test_dataset)}")
            testloader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE,
                collate_fn=collater, num_workers=num_workers, pin_memory=False
            )

            random.setstate(rng_state)
            np.random.set_state(np_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

        sep = "=" * 60
        logging.info(sep)
        logging.info(f"  Split      : {name}   [prefix = {prefix}]")
        logging.info(f"  Model      : {MODEL_MODULE}")
        logging.info(f"  Train CSV  : {train_csv}   (N={len(train_dataset)})")
        logging.info(f"  Val   CSV  : {val_csv}   (N={len(val_dataset)})")
        logging.info(f"  Test  CSV  : {test_csv}   (N={test_size})")
        logging.info(f"  Threshold  : {THRESHOLDS.get(test_name, DEFAULT_THRESHOLD)}")
        logging.info(f"  Config     : BATCH={BATCH_SIZE}  |  EPOCH={EPOCH}  |  LR={LEARNING_RATE:.1e}  |  SEED={RANDOM_SEED}  |  accum=8  |  patience={PATIENCE}")
        logging.info(f"  Device     : {device}")
        logging.info(f"  Params     : total={total_params:,}  |  trainable={trainable_params:,}")
        logging.info(sep)

        model_dir = f"{log_dir}/{name}-model"
        print(f"  STARTING TRAINING: {name}\n")

        model, best_val_auroc = train(
            model,
            train_loader=trainloader,
            valid_loader=valloader,
            test_loader=testloader,
            device=device,
            writer=writer,
            LOSS_NAME=name,
            epoch=EPOCH,
            lr=LEARNING_RATE,
            accumulation_steps=8,
            patience=PATIENCE,
            model_dir=model_dir,
            test_name=test_name,
            random_seed=RANDOM_SEED
        )

        if writer is not None:
            writer.close()
        end_time = _log_now()
        success = True
        print(f"\n  DONE: {name}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n[ERROR] {name} failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
        torch.cuda.empty_cache()

    # Add the end time and duration only after a successful training run.
    if success and end_time is not None:
        _finalize_log_header(log_path, start_time, end_time)

    return best_val_auroc


def main():
    demo_dir = get_next_demo_dir()
    print(f"\n本次运行日志目录: {demo_dir}")

    for config in SPLIT_CONFIGS:
        try:
            run_single_model(config, log_dir=demo_dir)
        except Exception as e:
            print(f"\n[ERROR] {config['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue


if __name__ == "__main__":
    main()
