# WyckoffDiff-Adapter
![](assets/graphical_abstract.png)

## Installation

These commands take few minutes
```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python --no-build-isolation --link-mode=symlink -e .
```

## Prepare dataset

Before training WyckoffDiff-Adapter from scratch, we have to unpack and preprocess the dataset files.

You can run the following command for `mp_20`:

```bash
# Download file from LFS
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets
```

You can also download fine-tuned model from Git LFS via

```
git lfs pull -I outputs/checkpoints/ --exclude="" 
```


run:
```bash
csv-to-wyckoff-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20_wyckoff --cache-folder datasets/cache
```

## Training

To train WyckoffDiff-Adapter, use the following command:

```bash
wyckoffdiff-adapter-train data_module.batch_size.train=32 trainer.devices=1 +trainer.logger.name=wyckoffdiff_train
```

## Finetuning

run the following command to finetune your model.

```bash
wyckoffdiff-adapter-finetune data_module.batch_size.train=32 trainer.devices=1 adapter.full_finetuning=false lightning_module.optimizer_partial.lr=1e-4
```

## Generate


Run the following commands:

(e.g. Li-O system with conditioning energy above hull and space group)

```bash
export MODEL_PATH=/path/to/your/model/
SPACE_GROUP = 225
export RESULTS_PATH="results/Li-O/${SPACE_GROUP}"

wyckoffdiff-adapter-generate \
    --output_path "$RESULTS_PATH" \
    --model_path "$MODEL_PATH" \
    --num_samples 64 \
    --adapter_condition_on chemical_system,energy_above_hull \
    --properties_to_condition_on='{chemical_system:Li-O,energy_above_hull:0.0}' \
    --guidance_scale=2.0 \
    --target_spg=${SPACE_GROUP} 
```

## Postprocess for WyckoffDiff-Adapter

run the following command:

```bash
export MODEL_PATH=/path/to/your/model/
SPACE_GROUP=225
export RESULTS_PATH="results/Li-O/${SPACE_GROUP}"

export CIF_PATH="${RESULTS_PATH}/cifs"
export CSV_FILE="${RESULTS_PATH}/post_processed/generated_samples_N=64_spg=${SPACE_GROUP}_seed=42_protostructures.csv"

postprocess \
    --load "$RESULTS_PATH" \
    --save_protostructures \
    --device cpu \
    --use_processed_data


SKIP_LARGE=${SKIP_LARGE:-1}     # 1: valid skip, 0: invalid skip(default: 1)
MAX_ATOMS=${MAX_ATOMS:-100}    # threshold of max atom in unitcell

EXTRA_FLAGS=()
if [ "$SKIP_LARGE" = "1" ]; then
EXTRA_FLAGS+=(--skip-large --max-atoms "$MAX_ATOMS")
fi

uv run python -m wyckoffdiff_adapter.scripts.gen_cif \
    --csv "$CSV_FILE" \
    --out-dir "$CIF_PATH" \
    "${EXTRA_FLAGS[@]}"

```

## Geometry Optimization using MatterSim

run the following command:

```bash
export MODEL_PATH=/path/to/your/model/
SPACE_GROUP=225
export RESULTS_PATH=results/Li-O/${SPACE_GROUP}/
export SAVE_PATH=/path/to/save/your/output/

mattergen-evaluate \
    --structures_path="$RESULTS_PATH/cifs" \
    --relax=True \
    --structure_matcher='disordered' \
    --save_as="$RESULTS_PATH/metrics_${SPACE_GROUP}.json" \
    --structures_output_path="${SAVE_PATH}/relaxed_structures_${SPACE_GROUP}.extxyz"
```

## Citation
Takanori Ishii, Kaoru Hisama, and Kohei Shinohara, "Symmetry-aware Conditional Generation of Crystal Structures Using Diffusion Models", ArXiv 2601.08115

The code has dependency on the MatterGen (https://github.com/microsoft/mattergen) and is based on its architecture.
The WyckoffGNN architecture is based on the original implementation of WyckoffDiff (https://github.com/httk/wyckoffdiff).

Some parts of the modules in this repository are adapted from the MatterGen. 

## License
This code is primarily licensed with the MIT License available in the file [`LICENSE`](LICENSE). The parts under [`wyckoffdiff_adapter/common/wyckoffgnn/d3pm`](wyckoffdiff_adapter/common/wyckoffgnn/d3pm) are based on the official public D3PM implementation <https://github.com/google-research/google-research/tree/master/d3pm> and therefore licensed separately under the Apache 2.0 license available at [`wyckoffdiff_adapter/common/wyckoffgnn/d3pm/LICENSE.txt`](wyckoff_generation/models/d3pm/LICENSE.txt).
