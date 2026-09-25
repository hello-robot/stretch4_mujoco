#!/usr/bin/env bash
# Set up MolmoBot RBY1 (Rainbow Robotics RB-Y1) simulation eval in third_party/MolmoBot.
#
# Idempotent -- safe to re-run. Does four things:
#   1. uv sync --extra eval          (installs the pinned molmospaces + eval deps)
#   2. re-installs torch for your GPU if the pinned wheel does not cover its arch
#   3. downloads the RBY1 checkpoint from HuggingFace
#   4. locates a cached RBY1 JSON benchmark and prints ready-to-run commands
#
# Usage:
#   examples/machine_learning/molmobot/setup_rby1_sim_eval.sh
#   MOLMOBOT_SKIP_CHECKPOINT=1 examples/machine_learning/molmobot/setup_rby1_sim_eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MOLMOBOT_DIR="$REPO_ROOT/third_party/MolmoBot/MolmoBot"
VENV_PY="$MOLMOBOT_DIR/.venv/bin/python"
CKPT_REPO="${MOLMOBOT_CKPT_REPO:-allenai/MolmoBot-RBY1Multitask}"
CKPT_DIR="$MOLMOBOT_DIR/ckpts/molmobot/$(basename "$CKPT_REPO")"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33mwarning: %s\033[0m\n' "$1"; }

if [[ ! -d "$MOLMOBOT_DIR" ]]; then
  echo "error: $MOLMOBOT_DIR not found -- clone https://github.com/allenai/MolmoBot into third_party/" >&2
  exit 1
fi
cd "$MOLMOBOT_DIR"

# ---------------------------------------------------------------- 1. deps ----
step "Syncing eval dependencies (uv sync --extra eval)"
uv sync --extra eval

# ------------------------------------------------------------- 2. torch ------
# The pinned uv.lock resolves a CUDA 12.6 torch wheel. Blackwell cards (RTX 50xx,
# sm_120) are not in that wheel's arch list: torch.cuda.is_available() still
# returns True, and the first kernel launch dies with "no kernel image is
# available for execution on the device". Detect that and swap in a cu128 build.
#
# NOTE: a later `uv sync` reverts this, because +cu128 is not what uv.lock pins.
# Re-run this script (or the uv pip install below) if you sync again.
step "Checking the torch build against your GPU"
TORCH_STATUS="$("$VENV_PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    print("NO_GPU"); raise SystemExit
major, minor = torch.cuda.get_device_capability(0)
need = f"sm_{major}{minor}"
print("MISMATCH" if need not in torch.cuda.get_arch_list() else "OK", need, torch.__version__)
PY
)"
echo "  $TORCH_STATUS"
case "$TORCH_STATUS" in
  MISMATCH*)
    step "Re-installing torch with CUDA 12.8 support for this GPU"
    uv pip install --python "$VENV_PY" --reinstall \
      torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    "$VENV_PY" - <<'PY'
import torch
x = torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")
torch.cuda.synchronize()
print(f"  verified: {torch.__version__} runs on {torch.cuda.get_device_name(0)}")
PY
    ;;
  NO_GPU*) warn "no CUDA device visible -- inference will be unusably slow on CPU" ;;
esac

# -------------------------------------------------------- 3. checkpoint ------
if [[ -n "${MOLMOBOT_SKIP_CHECKPOINT:-}" ]]; then
  step "Skipping checkpoint download (MOLMOBOT_SKIP_CHECKPOINT set)"
elif [[ -f "$CKPT_DIR/config.yaml" || -f "$CKPT_DIR/config.json" ]]; then
  step "Checkpoint already present at $CKPT_DIR"
else
  step "Downloading $CKPT_REPO (several GB)"
  "$MOLMOBOT_DIR/.venv/bin/hf" download "$CKPT_REPO" --local-dir "$CKPT_DIR"
fi

# --------------------------------------------------------- 4. benchmark ------
# --benchmark_path wants a directory holding benchmark.json + benchmark_metadata.json.
# MolmoSpaces caches these under ~/.cache/molmo-spaces-resources (or $MLSPACES_CACHE_DIR).
step "Locating a cached RBY1 benchmark"
BENCH="$("$VENV_PY" - <<'PY'
from pathlib import Path
from molmo_spaces.molmo_spaces_constants import DATA_CACHE_DIR

roots = sorted((Path(DATA_CACHE_DIR) / "benchmarks").glob("*/*"), reverse=True)
wanted = ("door_opening_benchmark", "RBY1PickAndPlaceDataGenConfig", "RBY1PickDataGenConfig")
for root in roots:
    for name in wanted:
        for hit in sorted(root.glob(f"**/{name}*")):
            if (hit / "benchmark.json").is_file():
                print(hit); raise SystemExit
    for hit in sorted(root.glob("**/rby1*/**/benchmark.json")):
        print(hit.parent); raise SystemExit
PY
)"

if [[ -z "$BENCH" ]]; then
  warn "no RBY1 benchmark found in the MolmoSpaces cache."
  echo "  MolmoSpaces downloads benchmark archives on demand; to fetch them all now:"
  echo "    cd $MOLMOBOT_DIR && .venv/bin/python -c \\"
  echo "      \"from molmo_spaces.molmo_spaces_constants import get_resource_manager;\\"
  echo "       get_resource_manager().install_all_for_data_type('benchmarks')\""
  BENCH="<benchmark_dir>"
else
  echo "  found: $BENCH"
fi

# ------------------------------------------------------------- summary -------
step "Ready"
cat <<EOF
Scripted eval over the whole benchmark (scores each episode's own instruction):

  $VENV_PY $REPO_ROOT/examples/machine_learning/molmobot/run_eval_rby1.py \\
    --checkpoint_path $CKPT_DIR \\
    --benchmark_path $BENCH \\
    --eval_config_cls olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorPlusOpenEvalConfig \\
    --task_horizon 400

Interactive -- you type the instructions:

  $VENV_PY $REPO_ROOT/examples/machine_learning/molmobot/rby1_interactive.py \\
    --checkpoint $CKPT_DIR \\
    --benchmark $BENCH

NOTE: MolmoBot's own launch_scripts/run_eval.py cannot run the RBY1 eval configs
as shipped -- the RBY1 policy chain reads three attributes only the Franka config
and policy define, so it raises before the first step. Both scripts above patch
that at runtime and leave the checkout untouched; see patch_released_rby1_policy
in rby1_interactive.py for what is wrong and why the fixes are what they are.
EOF
