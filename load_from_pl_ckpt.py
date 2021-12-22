import torch
from transformers import AutoModelForSequenceClassification

from nn_pruning.patch_coordinator import (
    SparseTrainingArguments,
    ModelPatchingCoordinator,
)


def rename_key(key):
    return key.replace('model_debias.model.', '')


def load_patched_from_pl_ckpt(
    pl_ckpt,
    config,
    device,
):
    # Recover state dict and hparams
    payload = torch.load(pl_ckpt)

    state_dict = payload['state_dict']
    state_dict = {rename_key(key): val for key, val in state_dict.items() if 'model_debias' in key}

    model_name = payload['hyper_parameters']['model_name']
    global_step = payload['global_step']
    sparse_args = payload['hyper_parameters']['sparse_train_args']

    # Load vanilla model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    sparse_args = SparseTrainingArguments(**sparse_args)

    # Patch vanilla model
    model_patcher = ModelPatchingCoordinator(
        sparse_args=sparse_args,
        device=device,
        cache_dir='tmp/',
        model_name_or_path=model_name,
        logit_names='logits',
        teacher_constructor=None,
    )
    model_patcher.patch_model(model)

    # Load state dict
    model.bert.load_state_dict(state_dict)

    # Schedule the threshold only once - it won't change
    model_patcher.schedule_threshold(
        step=global_step,  # This sets threshold to the final onee
        total_step=global_step,
        training=True,
    )

    return model
