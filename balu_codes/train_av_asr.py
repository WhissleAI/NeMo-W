import os
import sys
sys.path.insert(0, os.path.abspath('/home/bld56/gsoc/nemo/NeMo-opensource/'))
import nemo.core as nemo_core
from nemo.core import adapter_mixins
from nemo.utils import exp_manager
import nemo.collections.asr as nemo_asr
import nemo
import json
from omegaconf import OmegaConf, open_dict
import torch
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.text import WordErrorRate
import warnings
import argparse

# Function to load and configure the model
def load_and_configure_model(config_file_path):
    conf = OmegaConf.load(config_file_path)
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(conf, overrides)
    OmegaConf.set_struct(updated_conf, True)
    model = nemo_asr.models.AV_EncDecCTCModelBPE(updated_conf)

    model.setup_training_data(model.cfg.train_ds)
    return model, conf

# Function to freeze and unfreeze model parameters based on adapters
def manage_model_adapters(model, conf):
    # Freeze the entire model
    model.freeze()
    
    # Determine which modules to train based on configuration
    if model.cfg.use_video_modality:
        modules_to_train = [
            model.a_linear, model.v_linear, model.av_encoder, model.av_enocder_layer, 
            model.a_modal_embs, model.v_modal_embs, model.decoder, model.a_pos_enc, model.v_pos_enc
        ]
    elif not model.cfg.use_video_modality and model.cfg.use_pretrained_dec:
        modules_to_train = [model.a_model.decoder]
    else:  # not model.cfg.use_video_modality and not model.cfg.use_pretrained_dec
        modules_to_train = [model.decoder]
    
    # Set the selected modules to training mode and enable gradients
    for module in modules_to_train:
        module.train()
        for param in module.parameters():
            param.requires_grad = True

    # Handle adapter configurations if needed
    if conf.adapters.linear_adapter.keep:
        model.a_model.freeze()
        model.a_model.set_enabled_adapters(enabled=False)
        model.a_model.set_enabled_adapters(name=conf.adapters.linear_adapter.name, enabled=True)
        model.a_model.unfreeze_enabled_adapters()
    else:
        model.a_model.unfreeze()

# Function to set up the trainer
def setup_trainer():
    torch.set_float32_matmul_precision('high')
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        devices=-1, accelerator=accelerator, strategy="ddp_find_unused_parameters_true",
        max_epochs=100,
        enable_checkpointing=False, logger=False,
        log_every_n_steps=5, check_val_every_n_epoch=1,
    )
    return trainer

# Function to set up experiment manager
def setup_exp_manager(trainer, model):
    os.environ.pop('NEMO_EXPM_VERSION', None)

    exp_config = exp_manager.ExpManagerConfig(
        exp_dir=model.cfg.exp_dir,
        name=f'{model.cfg.wandb.run_name}',
        checkpoint_callback_params=exp_manager.CallbackParams(
            monitor="val_u_wer",
            mode="min",
            always_save_nemo=True,
            save_best_model=True,
        ),
        create_wandb_logger=model.cfg.wandb.create_wandb_logger,
        wandb_logger_kwargs=OmegaConf.create({"project": f"{model.cfg.wandb.project}", "name": f"{model.cfg.wandb.run_name}_{model.cfg.train_ds.override_snr_ratio}", "log_model": model.cfg.wandb.log_model}),
    )

    exp_config = OmegaConf.structured(exp_config)
    logdir = exp_manager.exp_manager(trainer, exp_config)
    if model.cfg.wandb.create_wandb_logger:
        trainer.loggers[1].log_hyperparams(OmegaConf.to_container(model.cfg)) # wandb logger
        # log the manifest file to wandb server
        trainer.loggers[1].experiment.log_artifact(f"{model.cfg.train_ds.manifest_filepath}")
        trainer.loggers[1].experiment.log_artifact(f"{model.cfg.validation_ds.manifest_filepath}")
        
    return logdir

# Main function to execute the workflow
def main(config_file_path, args):
    model, conf = load_and_configure_model(config_file_path)
    if args.resume_pretrained and False:
        print(f"{args.resume_pretrained} is set")
        # ckpt_path = f"/tmp/bld56_dataset_v1/saved_models/pre_av_ndec_uman_ntok--val_u_wer=0.0809-epoch=11.ckpt"
        ckpt_path = f"/tmp/bld56_dataset_v1/tmp/av_ndec_lman_ntokpre+/2024-09-07_06-19-52/checkpoints/av_ndec_lman_ntokpre+--val_u_wer=0.4031-epoch=10-last.ckpt"
        # ckpt_path = f"/tmp/bld56_dataset_v1/saved_models/av_ndec_lman_ntokpre_snr0.5/checkpoints/av_ndec_lman_ntokpre+--val_u_wer=0.3356-epoch=6.ckpt"
        # ckpt_path = f"/tmp/bld56_dataset_v1/tmp/av_ndec_lman_ntokpre+/2024-09-06_14-22-50/checkpoints/av_ndec_lman_ntokpre+--val_u_wer=0.2183-epoch=2.ckpt"
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(model)
        model.cfg.wandb.run_name += 'pre+'
    manage_model_adapters(model, conf)
    
    trainer = setup_trainer()
    model.set_trainer(trainer)
    logdir = setup_exp_manager(trainer, model)
    #trainer.fit(model)
    model.eval()
    trainer.validate(model)

def float_or_str(value):
    try:
        # Try to parse it as an integer
        return float(value)
    except ValueError:
        # If it fails, try to parse it as a float
        try:
            return str(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Value '{value}' is neither int nor float")

if __name__ == "__main__":
    # add config number args
    parser = argparse.ArgumentParser(description='Train AV ASR model')
    parser.add_argument('--config', type=int, default=5, help='Config number to use for training')
    parser.add_argument('--snr', type=float_or_str, default=0.7, help='SNR ratio to use for training')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use for training')
    parser.add_argument('--resume_pretrained', type=bool, default=False, help='Resume training from pretrained model')
    args = parser.parse_args()
    config_file_path = f"/home/bld56/gsoc/nemo/NeMo-opensource/balu_codes/final_configs_icassp_24/c{args.config}.yaml"
    # load yaml file
    with open(config_file_path) as file:
        config = OmegaConf.load(file)
    config['train_ds']['override_snr_ratio'] = args.snr
    config['validation_ds']['override_snr_ratio'] = args.snr
    config['test_ds']['override_snr_ratio'] = args.snr
    with open(config_file_path, 'w') as file:
        OmegaConf.save(config, file)
    warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed. Trying audioread instead.")
    warnings.filterwarnings("ignore", category=FutureWarning, message="librosa.core.audio.__audioread_load\n\tDeprecated as of librosa version 0.10.0.\n\tIt will be removed in librosa version 1.0.")
    main(config_file_path, args)
