# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import json
import os
import tempfile
from math import ceil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.av_to_text import _AVTextDataset
# from nemo.collections.asr.data.audio_to_text_dali import AudioToCharDALIDataset, DALIOutputs
# from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.av_wer import AV_WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRModuleMixin, ASRTranscriptionMixin, InterCTCMixin, TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import GenericTranscriptionType, TranscriptionReturnType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.parts.utils.asr_batching import get_semi_sorted_batch_sampler
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
# from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.parts.preprocessing.parsers import make_parser
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType, ImageFeatureValue
from nemo.utils import logging

#ADAPTERS
from nemo.core import adapter_mixins
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module import MultiHeadAttentionAdapterConfig
from nemo.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module import RelPositionMultiHeadAttentionAdapterConfig
__all__ = ['AV_EncDecCTCModel']


class AV_EncDecCTCModel(ASRModel, ExportableEncDecModel, ASRModuleMixin, InterCTCMixin, ASRTranscriptionMixin):
    """Base class for encoder decoder CTC-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        super().__init__(cfg=cfg, trainer=trainer)
        if "BPE:" in cfg.a_model_name:
            a_model_cfg = EncDecCTCModelBPE.from_pretrained(cfg.a_model_name[4:], return_config=True)
            a_model_cfg = self.update_model_config_to_support_adapter(a_model_cfg) # for adapters
            self.a_model = EncDecCTCModelBPE.from_pretrained(cfg.a_model_name[4:], override_config_path=a_model_cfg)
        else:
            a_model_cfg = EncDecCTCModel.from_pretrained(cfg.a_model_name, return_config=True)
            a_model_cfg = self.update_model_config_to_support_adapter(a_model_cfg)
            self.a_model = EncDecCTCModel.from_pretrained(cfg.a_model_name, override_config_path=a_model_cfg)
        
        self.labelled_manifest = cfg.labelled_manifest
        
        
        if cfg.adapters.linear_adapter.keep:
            linear_adapter_cfg = LinearAdapterConfig(
                in_features=self.a_model.encoder.d_model,
                dim = cfg.adapters.linear_adapter.dim,
                activation=cfg.adapters.linear_adapter.activation,
                norm_position=cfg.adapters.linear_adapter.norm_position,
                dropout=cfg.adapters.linear_adapter.dropout,
            )
            linear_adapter_name = cfg.adapters.linear_adapter.name
            self.a_model.add_adapter(name=linear_adapter_name, cfg=linear_adapter_cfg)
        with open_dict(self._cfg):
            if "feat_in" not in self._cfg.decoder or (
                not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self._cfg.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.decoder.num_classes < 1 and self.cfg.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.decoder.num_classes, len(self.cfg.decoder.vocabulary)
                    )
                )
                cfg.decoder["num_classes"] = len(self.cfg.decoder.vocabulary)
        assert not (self.cfg.use_pretrained_dec and self.cfg.use_video_modality), "Pretrained decoder is not supported for video modality"

        # initialize a transformer encoder and decoder
        if cfg.use_video_modality:
            self.a_linear = torch.nn.Linear(in_features = self.a_model.encoder._feat_out, out_features = self.cfg.av_encoder.d_model)
            self.v_linear = torch.nn.Linear(in_features = self.cfg.v_model.feat_dim, out_features = self.cfg.av_encoder.d_model)
            self.av_enocder_layer = torch.nn.TransformerEncoderLayer(d_model = self.cfg.av_encoder.d_model, nhead = self.cfg.av_encoder.nhead, dropout = self.cfg.av_encoder.dropout, batch_first=True)
            self.av_encoder = torch.nn.TransformerEncoder(self.av_enocder_layer, num_layers = self.cfg.av_encoder.num_layers)
        
            # Modality embeddings
            self.a_modal_embs = torch.nn.Embedding(1, self.cfg.av_encoder.d_model)
            self.v_modal_embs = torch.nn.Embedding(1, self.cfg.av_encoder.d_model)
        
            # Trainable positional encodings
            self.a_pos_enc = torch.nn.Embedding(10000, self.cfg.av_encoder.d_model)
            self.v_pos_enc = torch.nn.Embedding(10000, self.cfg.av_encoder.d_model)

        
        self.decoder = EncDecCTCModel.from_config_dict(self._cfg.decoder)
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )
        
        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        self.decoding = CTCDecoding(self.cfg.decoding, vocabulary=OmegaConf.to_container(self.decoder.vocabulary))

        # Setup metric with decoding strategy
        self.wer = AV_WER(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
            labelled_manifest=self.labelled_manifest
        )

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        # setting up interCTC loss (from InterCTCMixin)
        self.setup_interctc(decoder_name='decoder', loss_name='loss', wer_name='wer')

    def update_model_config_to_support_adapter(self, model_cfg):
        with open_dict(model_cfg):
            adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
            if adapter_metadata is not None:
                model_cfg.encoder._target_ = adapter_metadata.adapter_class_path
        
        print("Updated encoder _target_ model :", model_cfg.encoder._target_)
        return model_cfg

    def transcribe(
        self,
        audio: Union[str, List[str], torch.Tensor, np.ndarray],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """
        If modify this function, please remember update transcribe_partial_audio() in
        nemo/collections/asr/parts/utils/trancribe_utils.py

        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray audio array. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
                **Note**: All other arguments in the function will be ignored if override_config is passed.
                You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.

        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        return super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            override_config=override_config,
        )

    def change_vocabulary(self, new_vocabulary: List[str], decoding_cfg: Optional[DictConfig] = None):
        """
        Changes vocabulary used during CTC decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        If new_vocabulary == self.decoder.vocabulary then nothing will be changed.

        Args:

            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
            this is target alphabet.

        Returns: None

        """
        if self.decoder.vocabulary == new_vocabulary:
            logging.warning(f"Old {self.decoder.vocabulary} and new {new_vocabulary} match. Not changing anything.")
        else:
            if new_vocabulary is None or len(new_vocabulary) == 0:
                raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config['vocabulary'] = new_vocabulary
            new_decoder_config['num_classes'] = len(new_vocabulary)

            del self.decoder
            self.decoder = EncDecCTCModel.from_config_dict(new_decoder_config)
            del self.loss
            self.loss = CTCLoss(
                num_classes=self.decoder.num_classes_with_blank - 1,
                zero_infinity=True,
                reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            )

            if decoding_cfg is None:
                # Assume same decoding config as before
                decoding_cfg = self.cfg.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(CTCDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

            self.decoding = CTCDecoding(
                decoding_cfg=decoding_cfg, vocabulary=OmegaConf.to_container(self.decoder.vocabulary)
            )

            self.wer = AV_WER(
                decoding=self.decoding,
                use_cer=self._cfg.get('use_cer', False),
                dist_sync_on_step=True,
                log_prediction=self._cfg.get("log_prediction", False),
                labelled_manifest=self.labelled_manifest
            )

            # Update config
            with open_dict(self.cfg.decoder):
                self._cfg.decoder = new_decoder_config

            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            ds_keys = ['train_ds', 'validation_ds', 'test_ds']
            for key in ds_keys:
                if key in self.cfg:
                    with open_dict(self.cfg[key]):
                        self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

            logging.info(f"Changed decoder to output to {self.decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig):
        """
        Changes decoding strategy used during CTC decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = CTCDecoding(
            decoding_cfg=decoding_cfg, vocabulary=OmegaConf.to_container(self.decoder.vocabulary)
        )

        self.wer = AV_WER(
            decoding=self.decoding,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
            labelled_manifest=self.labelled_manifest
        )

        self.decoder.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')

        dataset = audio_to_text_dataset.get_av_to_text_char_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            preprocessor_cfg=self._cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None
        
        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        batch_sampler = None
        if config.get('use_semi_sorted_batching', False): # This is in usable format even for our 
            if not isinstance(dataset, _AVTextDataset):
                raise RuntimeError(
                    "Semi Sorted Batch sampler can be used with AudioToCharDataset or AudioToBPEDataset "
                    f"but found dataset of type {type(dataset)}"
                )
            # set batch_size and batch_sampler to None to disable automatic batching
            batch_sampler = get_semi_sorted_batch_sampler(self, dataset, config)
            config['batch_size'] = None
            config['drop_last'] = False
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            sampler=batch_sampler,
            batch_sampler=None,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.a_model.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.a_model.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "audio_input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "audio_input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "video_input_signal": NeuralType(('B', 'T', 'D'), ImageFeatureValue(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
        }

    @typecheck()
    def forward(
        self, audio_input_signal=None, audio_input_signal_length=None, video_input_signal= None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model.

        Args:
            audio_input: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            audio_input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = audio_input_signal is not None and audio_input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``audio_input`` and ``audio_input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.a_model.preprocessor(
                input_signal=audio_input_signal, length=audio_input_signal_length,
            )

        if self.a_model.spec_augmentation is not None and self.training:
            processed_signal = self.a_model.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_output = self.a_model.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoder_output[0]
        encoded_len = encoder_output[1]
        if self.cfg.use_video_modality and not self.cfg.use_pretrained_dec:
            # B,C,T -> B,T,C
            encoded = encoded.permute(0, 2, 1)
            a_encoded = self.a_linear(encoded)
            v_encoded = self.v_linear(video_input_signal)
        
            # Add modality embeddings
            B, T, C = a_encoded.size()
            B, F, D = v_encoded.size()
            assert C == D, "The audio and video features must have the same dimensionality"
            
            # Expand modality embeddings to match the dimensions of a_encoded and v_encoded
            a_modal_emb_expanded = self.a_modal_embs.weight.expand(B, T, -1)  # Shape: (B, T, feat_in)
            v_modal_emb_expanded = self.v_modal_embs.weight.expand(B, F, -1)  # Shape: (B, F, feat_in)
            
            a_encoded = a_encoded + a_modal_emb_expanded
            v_encoded = v_encoded + v_modal_emb_expanded
            
            # Add positional encodings
            a_pos_enc = self.a_pos_enc(torch.arange(T, device=a_encoded.device)).unsqueeze(0).expand(B, -1, -1)
            v_pos_enc = self.v_pos_enc(torch.arange(F, device=v_encoded.device)).unsqueeze(0).expand(B, -1, -1)
            
            a_encoded = a_encoded + a_pos_enc
            v_encoded = v_encoded + v_pos_enc
            
            # Concat and pass them through the transformer encoder
            av_encoded = torch.cat((a_encoded, v_encoded), dim=1)
            av_encoded = self.av_encoder(av_encoded)
            
            # remove the v_encoded tokens
            av_encoded = av_encoded[:, :T, :]
            
            # B,T,C -> B,C,T
            av_encoded = av_encoded.permute(0, 2, 1)
            
            # remove 
            log_probs = self.decoder(encoder_output=av_encoded)
            greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        elif (not self.cfg.use_video_modality) and (not self.cfg.use_pretrained_dec):
            log_probs = self.decoder(encoder_output=encoded)
            greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        elif (not self.cfg.use_video_modality) and self.cfg.use_pretrained_dec:
            log_probs = self.a_model.decoder(encoder_output=encoded)
            greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        elif self.cfg.use_video_modality and self.cfg.use_pretrained_dec:
            raise ValueError("Pretrained decoder is not supported for video modality")
        
        return (
            log_probs,
            encoded_len,
            greedy_predictions,
        )

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, video_input_signal, transcript, transcript_len = batch
        # if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
        #     log_probs, encoded_len, predictions = self.forward(
        #         processed_signal=signal, processed_signal_length=signal_len
        #     )
        # else:
        log_probs, encoded_len, predictions = self.forward(audio_input_signal=signal, audio_input_signal_length=signal_len, video_input_signal=video_input_signal)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

        # Add auxiliary losses, if registered
        loss_value = self.add_auxiliary_losses(loss_value)
        # only computing WER when requested in the logs (same as done for final-layer WER below)
        loss_value, tensorboard_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=((batch_nb + 1) % log_every_n_steps == 0)
        )

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        tensorboard_logs.update(
            {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
        )

        if (batch_nb + 1) % log_every_n_steps == 0:
            self.wer.update(
                predictions=log_probs,
                targets=transcript,
                targets_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            # wer, _, _ = self.wer.compute()
            labelled_wer, unlabelled_wer, acc, scores_unlabelled, words_unlabelled = self.wer.compute()
            self.wer.reset()
            # tensorboard_logs.update({'training_batch_l_wer': labelled_wer,
            #                          'training_batch_u_wer': unlabelled_wer,
            #                          'training_batch_l_acc': acc,
            #                          })
            if labelled_wer is not None:
                tensorboard_logs.update({'train_l_wer': labelled_wer}) 
                self.log('train_l_wer', labelled_wer, on_step=True, on_epoch=False)
            if unlabelled_wer is not None:
                tensorboard_logs.update({'train_u_wer': unlabelled_wer})
                self.log('train_u_wer', unlabelled_wer, on_step=True, on_epoch=False)
            if acc is not None:
                tensorboard_logs.update({'train_acc': acc})
                self.log('train_acc', acc, on_step=True, on_epoch=False)

        return {'loss': loss_value, 'log': tensorboard_logs}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, video_input_signal, transcript, transcript_len, sample_id = batch
        # if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
        #     log_probs, encoded_len, predictions = self.forward(
        #         processed_signal=signal, processed_signal_length=signal_len
        #     )
        # else:
        log_probs, encoded_len, predictions = self.forward(audio_input_signal=signal, audio_input_signal_length=signal_len, video_input_signal=video_input_signal)

        transcribed_texts, _ = self.wer.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs, decoder_lengths=encoded_len, return_hypotheses=False,
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, transcribed_texts))

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, video_input_signal, transcript, transcript_len = batch
        # if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
        #     log_probs, encoded_len, predictions = self.forward(
        #         processed_signal=signal, processed_signal_length=signal_len
        #     )
        # else:
        log_probs, encoded_len, predictions = self.forward(audio_input_signal=signal, audio_input_signal_length=signal_len, video_input_signal=video_input_signal)

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        loss_value, metrics = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=True, log_wer_num_denom=True, log_prefix="val_",
        )

        self.wer.update(
            predictions=log_probs, targets=transcript, targets_lengths=transcript_len, predictions_lengths=encoded_len,
        )
        # wer, wer_num, wer_denom = self.wer.compute()
        labelled_wer, unlabelled_wer, acc, scores_unlabelled, words_unlabelled = self.wer.compute()
        self.wer.reset()
        metrics.update({'val_loss': loss_value, 'val_labelled_wer': labelled_wer, 'val_unlabelled_wer': unlabelled_wer, 'val_acc': acc, 'val_wer_num': scores_unlabelled, 'val_wer_denom': words_unlabelled})
        
        # self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))
        if labelled_wer is not None:
            self.log('val_l_wer', labelled_wer, on_epoch=True, sync_dist=True)
        if unlabelled_wer is not None:
            self.log('val_u_wer', unlabelled_wer, on_epoch=True, sync_dist=True)
        if acc is not None:
            self.log('val_acc', acc, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss_value, sync_dist=True)
        

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        metrics = super().multi_validation_epoch_end(outputs, dataloader_idx)
        self.finalize_interctc_metrics(metrics, outputs, prefix="val_")
        return metrics

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        metrics = super().multi_test_epoch_end(outputs, dataloader_idx)
        self.finalize_interctc_metrics(metrics, outputs, prefix="test_")
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    """ Transcription related methods """

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        super()._transcribe_on_begin(audio, trcfg)

        # Freeze the encoder and decoure_exder modules
        self.encoder.freeze()
        self.decoder.freeze()

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        super()._transcribe_on_end(trcfg)

        # Unfreeze the encoder and decoder modules
        self.encoder.unfreeze()
        self.decoder.unfreeze()

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        logits, logits_len, greedy_predictions = self.forward(audio_input=batch[0], audio_input_signal_length=batch[1], video_input_signal=batch[2])
        output = dict(logits=logits, logits_len=logits_len)
        del greedy_predictions
        return output

    def _transcribe_output_processing(self, outputs, trcfg: TranscribeConfig) -> GenericTranscriptionType:
        logits = outputs.pop('logits')
        logits_len = outputs.pop('logits_len')

        current_hypotheses, all_hyp = self.decoding.ctc_decoder_predictions_tensor(
            logits, decoder_lengths=logits_len, return_hypotheses=trcfg.return_hypotheses,
        )
        if trcfg.return_hypotheses:
            if logits.is_cuda:
                # See comment in
                # ctc_greedy_decoding.py::GreedyCTCInfer::forward() to
                # understand this idiom.
                logits_cpu = torch.empty(logits.shape, dtype=logits.dtype, device=torch.device("cpu"), pin_memory=True)
                logits_cpu.copy_(logits, non_blocking=True)
            else:
                logits_cpu = logits
            logits_len = logits_len.cpu()
            # dump log probs per file
            for idx in range(logits_cpu.shape[0]):
                current_hypotheses[idx].y_sequence = logits_cpu[idx][: logits_len[idx]]
                if current_hypotheses[idx].alignments is None:
                    current_hypotheses[idx].alignments = current_hypotheses[idx].y_sequence
            del logits_cpu

        # cleanup memory
        del logits, logits_len

        hypotheses = []
        if all_hyp is None:
            hypotheses += current_hypotheses
        else:
            hypotheses += all_hyp

        return hypotheses

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.a_model.preprocessor._sample_rate,
            'labels': OmegaConf.to_container(self.decoder.vocabulary),
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'channel_selector': config.get('channel_selector', None),
        }
        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        return results

    @property
    def wer(self):
        return self._wer

    @wer.setter
    def wer(self, wer):
        self._wer = wer
