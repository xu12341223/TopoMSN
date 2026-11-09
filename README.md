# TopoMSN

A unified implementation for four EEG datasets, selected with `--dataset`.
The code runs independently of the experiment scripts in the parent directory.
Data, checkpoints, manuscripts, and historical results are not bundled.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | Dataset channel orders, classes, segment lengths, contexts, and default hyperparameters |
| `data.py` | Data validation, normalization, causal contexts, subject splits, and training-set connectivity priors |
| `model.py` | Position augmentation, four-source connectivity fusion, sparse graphs, Chebyshev filtering, regional readout, cross-band encoding, and gated temporal fusion |
| `train.py` | Five-fold training, validation macro F1 checkpoint selection, and split/checkpoint export |
| `test.py` | Fixed-checkpoint evaluation, prediction export, and six evaluation metrics |
| `utils.py` | Random seeds, evaluation metrics, data loaders, and serialization helpers |

## Data Format

Use the project's prepared EEG arrays. The loader does not read raw EEG recordings
or regenerate subject identifiers. Raw-recording filtering, resampling, and
segmentation must be completed before using this directory.
Each dataset directory must contain:

```text
DATASET/
  Feature/feature_1.npy       # [segments, time, channels], in the original segment order
  Feature/feature_2.npy
  Label/label.npy             # Each row: [class_id, subject_id]
  preprocess_config.json     # Optional: channel_names and sfreq
```

| `--dataset` | Dataset | Sampling Rate (Hz) | Time Points | Channels | Context Segments | Class IDs |
| --- | --- | --- | --- | --- | --- | --- |
| `ds004584` | OpenNeuro-PD | 256 | 512 | 60 | 2 | HC=0, PD=1 |
| `adftd` | ADFTD | 256 | 256 | 19 | 4 | HC=0, FTD=1, AD=2 |
| `apava` | APAVA | 256 | 256 | 16 | 4 | HC=0, AD=1 |
| `stroke_mca` | MCA | 256 | 512 | 58 | 2 | Control=0, Stroke=1 |

Channel order must match `config.py` or the dataset metadata. Contexts contain the
current segment and preceding stored segments from the same subject. If recordings
contain separate trials, preserve the original trial-boundary handling during data
preparation: this interface stores segment order but does not encode trial boundaries.
Stored segment adjacency does not necessarily imply continuous recording time.
The first `context_segments - 1` segments of each subject are excluded as classification
targets. The current segment and context are independently z-score normalized
over time for each channel.

## Usage

Run the following commands from this directory:

```bash
python -m pip install -r requirements.txt
python data.py --dataset ds004584 --data-root /path/to/DS004584
python train.py --dataset ds004584 --data-root /path/to/DS004584 --output /path/to/new_run
python test.py --data-root /path/to/DS004584 --run /path/to/new_run
```

For another dataset, change `--dataset` and `--data-root`. Testing restores the dataset
configuration from the checkpoint rather than accepting a separate dataset or model
configuration. Use `--device cpu` or `--device cuda:0` to select a device.
Training runs all five folds by default. Use `--fold 1` to train only one fold;
separate single-fold training jobs require separate output directories.

To reuse an existing experimental split, pass
`--split-manifest /path/to/splits.json`. The manifest uses the following structure:

```json
{"dataset":"ds004584","folds":[
  {"fold_index":1,"train":[1,2],"validation":[3,4],"test":[5,6]}
]}
```

This example illustrates the fields only. A valid manifest must contain all five
folds and all subjects, with every class represented in each partition.
Without a manifest, the code creates subject-stratified five-fold splits with seed 41
and reserves 20% of each development partition for validation. Use the original split
manifest when reproducing an existing experiment.

## Training and Evaluation

- Training reads only training and validation features. Population connectivity
  priors are estimated exclusively from training subjects.
- A checkpoint is saved only when validation macro F1 strictly improves. Test
  performance is not used for checkpoint selection.
- Existing run directories and exported test results are not overwritten.
- Checkpoints store the configuration, label mapping, subject split, channel order,
  model buffers, and a source-code digest.
- Testing exports `predictions.npz` and `metrics.json`. Once all five folds have
  test outputs, it also writes `summary.json`.
- The six segment-level metrics are accuracy, macro precision, macro recall,
  macro F1, AUROC, and AUPRC. Binary AUROC/AUPRC use class 1 as the positive class;
  multiclass AUROC/AUPRC use one-vs-rest macro averaging.
- Fold summaries report the mean and sample standard deviation (`ddof=1`).
  Segments are not treated as independent subjects.
