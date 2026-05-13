"""Wrapper around piper.train that:

1. Allowlists trusted pickled types (``pathlib.PosixPath`` etc.) so the
   PyTorch >= 2.6 strict-safe ``torch.load`` defaults don't reject the
   rhasspy Piper base checkpoints.

2. Strips legacy / unsupported keys from the checkpoint's
   ``hyper_parameters`` dict on load. The rhasspy checkpoints were saved
   with an older Piper API that bundled Lightning trainer args and dataset
   args into the same flat hparams dict (``batch_size``, ``max_epochs``,
   ``sample_bytes``, ``dataset_dir``, ``tpu_cores``, etc.). The current
   ``piper1-gpl`` only accepts a smaller, model-only set on the VITS class,
   so Lightning's CLI errors with
   ``Subcommand 'fit' does not accept option 'model.sample_bytes'`` on the
   first unknown key. We filter the dict down to the keys the current model
   class understands.

3. Injects a frequent-save ModelCheckpoint callback into Lightning's
   Trainer at init time. Default behaviour is "save once per epoch, keep
   only the latest" — useless for the experiment-2 plan where we want a
   step-by-step quality ladder (1k, 2k, ... 10k checkpoints) to A/B test
   on the phone. Configured via env vars TTS_CKPT_EVERY_N_STEPS and
   TTS_CKPT_KEEP_ALL.

The list of allowed keys is the union of arguments shown by
``python3 -m piper.train fit --help`` under ``--model.*`` (sans the
``model.`` prefix, since hyper_parameters stores them flat).

After the patches we just forward to ``piper.train.__main__:main`` so all
CLI args and behaviour stay identical.
"""
import os
import pathlib

import torch
import torch.serialization

# 1. Allow the path types embedded in the checkpoint.
torch.serialization.add_safe_globals(
    [
        pathlib.PosixPath,
        pathlib.WindowsPath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
    ]
)

# 2. Filter hyper_parameters to keys the current Piper VITS class accepts.
_ALLOWED_MODEL_HPARAMS = frozenset(
    {
        # Audio / mel front-end
        "sample_rate",
        "filter_length",
        "hop_length",
        "win_length",
        "mel_channels",
        "mel_fmin",
        "mel_fmax",
        "segment_size",
        # Speakers / phoneme embedding
        "num_speakers",
        "gin_channels",
        # Vocoder (HiFi-GAN-style decoder)
        "resblock",
        "resblock_kernel_sizes",
        "resblock_dilation_sizes",
        "upsample_rates",
        "upsample_initial_channel",
        "upsample_kernel_sizes",
        # Text encoder / flow / posterior
        "inter_channels",
        "hidden_channels",
        "filter_channels",
        "n_heads",
        "n_layers",
        "kernel_size",
        "p_dropout",
        "n_layers_q",
        "use_spectral_norm",
        "use_sdp",
        # Training schedule
        "learning_rate",
        "learning_rate_d",
        "betas",
        "betas_d",
        "eps",
        "lr_decay",
        "lr_decay_d",
        "init_lr_ratio",
        "warmup_epochs",
        # Loss weights
        "c_mel",
        "c_kl",
        # Misc
        "grad_clip",
        "vocoder_warmstart_ckpt",
        "dataset",
    }
)

_original_torch_load = torch.load


def _filtered_torch_load(*args, **kwargs):
    result = _original_torch_load(*args, **kwargs)
    if isinstance(result, dict) and "hyper_parameters" in result and isinstance(
        result["hyper_parameters"], dict
    ):
        result["hyper_parameters"] = {
            k: v
            for k, v in result["hyper_parameters"].items()
            if k in _ALLOWED_MODEL_HPARAMS
        }
    return result


torch.load = _filtered_torch_load  # type: ignore[assignment]


# 3. Inject a ModelCheckpoint that saves every N training steps and (optionally)
#    keeps every snapshot, so we can A/B test a quality ladder of checkpoints.
import lightning.pytorch as pl  # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402

_CKPT_EVERY_N_STEPS = int(os.environ.get("TTS_CKPT_EVERY_N_STEPS", "1000"))
_CKPT_KEEP_ALL = os.environ.get("TTS_CKPT_KEEP_ALL", "1") == "1"

_original_trainer_init = pl.Trainer.__init__


def _patched_trainer_init(self, *args, **kwargs):
    extra_cb = ModelCheckpoint(
        every_n_train_steps=_CKPT_EVERY_N_STEPS,
        save_top_k=(-1 if _CKPT_KEEP_ALL else 1),
        save_last=True,
        filename="step_{step}",
        auto_insert_metric_name=False,
    )
    cbs = kwargs.get("callbacks") or []
    if not isinstance(cbs, list):
        cbs = [cbs]
    cbs = list(cbs) + [extra_cb]
    kwargs["callbacks"] = cbs
    _original_trainer_init(self, *args, **kwargs)


pl.Trainer.__init__ = _patched_trainer_init  # type: ignore[assignment]


from piper.train.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
