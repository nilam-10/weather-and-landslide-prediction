"""Generate and cache test predictions for spatial verification across all models and seeds.

Loads frozen checkpoints without retraining and evaluates strictly on the
independent primary test set (July 12-14, 2023, 129 sequences).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import h5py
import numpy as np
import torch

from src.models.convlstm import MultiTaskWeatherConvLSTM
from src.models.experimental.simvp import SimVPWeather
from src.models.experimental.unet_convlstm import UNetConvLSTMWeather
from scripts.run_rigorous_model_improvement_program import build_advanced_sequences, DEVICE

OUT_DIR = Path("outputs/reports/model_research/step2_spatial_verification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PREPARING TEST PREDICTION COHORT FOR SPATIAL VERIFICATION")
print(f"Device: {DEVICE}")
print("=" * 80)

# 1. Load Primary Test Data (July 12-14, 2023, 129 sequences)
test_primary_dates = ["20230712", "20230713", "20230714"]
te_x, te_y_b, te_y_c, te_m, te_t = build_advanced_sequences(test_primary_dates, t_in=4)

gt_bin = np.stack(te_y_b, axis=0)       # (129, 5, 25, 40)
gt_cont = np.stack(te_y_c, axis=0)      # (129, 5, 25, 40)
mask_stack = np.stack(te_m, axis=0)     # (129, 5, 25, 40)
ref_times = np.array(te_t)

x_tensor = torch.from_numpy(np.stack(te_x, axis=0)).to(DEVICE)
print(f"Loaded {len(x_tensor)} test sequences of shape {x_tensor.shape}.")

# 2. Load existing baseline predictions from PHASE3_TEST_PREDICTIONS_2023.npz
npz_existing = np.load("outputs/experiments/PHASE3_TEST_PREDICTIONS/PHASE3_TEST_PREDICTIONS_2023.npz")
persist_qpf = npz_existing["persistence_continuous"]
dense_flow_qpf = npz_existing["dense_flow_continuous"]

predictions_dict = {
    "ground_truth_continuous": gt_cont,
    "ground_truth_binary": gt_bin,
    "observation_mask": mask_stack,
    "reference_times": ref_times,
    "persistence_qpf": persist_qpf,
    "dense_flow_qpf": dense_flow_qpf,
}

# 3. Evaluate ConvLSTM Seeds (Seed 42, 123, 999)
conv_seeds = [42, 123, 999]
for s in conv_seeds:
    ckpt_path = Path(f"outputs/models/phase3/convlstm_seed{s}.pt")
    if ckpt_path.exists():
        print(f"Running inference for ConvLSTM (Seed {s})...")
        m = MultiTaskWeatherConvLSTM(in_channels=3, hidden_channels=(32, 32), num_leads=5).to(DEVICE)
        m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        m.eval()
        with torch.no_grad():
            c_logits, qpf = m(x_tensor)
            predictions_dict[f"convlstm_seed{s}_probs"] = torch.sigmoid(c_logits).cpu().numpy()
            predictions_dict[f"convlstm_seed{s}_qpf"] = qpf.cpu().numpy()

# 4. Evaluate SimVP Optimal
simvp_ckpt = Path("outputs/experiments/model_research/checkpoints/simvp_optimal_best.pt")
if simvp_ckpt.exists():
    print("Running inference for SimVP (Optimal)...")
    m_simvp = SimVPWeather(in_channels=3, hidden_dim=32, t_in=4, num_leads=5).to(DEVICE)
    m_simvp.load_state_dict(torch.load(simvp_ckpt, map_location=DEVICE))
    m_simvp.eval()
    with torch.no_grad():
        c_logits, qpf = m_simvp(x_tensor)
        predictions_dict["simvp_probs"] = torch.sigmoid(c_logits).cpu().numpy()
        predictions_dict["simvp_qpf"] = qpf.cpu().numpy()

# 5. Evaluate U-Net-ConvLSTM
unet_ckpt = Path("outputs/experiments/model_research/checkpoints/unet_convlstm_best.pt")
if unet_ckpt.exists():
    print("Running inference for U-Net-ConvLSTM...")
    m_unet = UNetConvLSTMWeather(in_channels=3, base_channels=24, num_leads=5).to(DEVICE)
    m_unet.load_state_dict(torch.load(unet_ckpt, map_location=DEVICE))
    m_unet.eval()
    with torch.no_grad():
        c_logits, qpf = m_unet(x_tensor)
        predictions_dict["unet_convlstm_probs"] = torch.sigmoid(c_logits).cpu().numpy()
        predictions_dict["unet_convlstm_qpf"] = qpf.cpu().numpy()

# Save complete cohort
cohort_path = OUT_DIR / "test_predictions_cohort.npz"
np.savez_compressed(cohort_path, **predictions_dict)
print(f"\nSaved all test prediction arrays to: {cohort_path} ({cohort_path.stat().st_size:,} bytes)")
