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
import io
import json
import math
import multiprocessing
import os
from collections.abc import Iterable as IterableABC
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import braceexpand
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import ChainDataset
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import available_formats as valid_sf_formats
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.data_utils import (
    DataStoreObject,
    datastore_object_get,
    datastore_path_to_webdataset_url,
    is_datastore_cache_shared,
    is_datastore_path,
    is_tarred_path,
)
from nemo.utils.distributed import webdataset_split_by_workers
from nemo.utils.get_rank import is_global_rank_zero

import numpy as np

# FOR NOISE LOADING
from pydub import AudioSegment

__all__ = [
    'AVToCharDataset',
    'AVToBPEDataset',
]

VALID_FILE_FORMATS = ';'.join(
    ['wav', 'mp3', 'flac', 'opus'] + [fmt.lower() for fmt in valid_sf_formats.keys()])


def _speech_collate_fn(batch, pad_id, get_vid_feats):
    """collate batch of audio sig, audio len, video sig, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], Optional[LongTensor],
               LongTensor, LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    packed_batch = list(zip(*batch))
    if get_vid_feats:
        if len(packed_batch) == 6:
            _, audio_lengths, _, _, tokens_lengths, sample_ids = packed_batch
        elif len(packed_batch) == 5:
            sample_ids = None
            _, audio_lengths, _, _, tokens_lengths = packed_batch
        else:
            raise ValueError(f"Expects 5 or 6 tensors in the batch!")
    else:
        if len(packed_batch) == 4:
            sample_ids = None
            _, audio_lengths, _, tokens_lengths = packed_batch
        elif len(packed_batch) == 5:
            _, audio_lengths, _, tokens_lengths, sample_ids = packed_batch
        else:
            raise ValueError(f"Expects 4 or 5 tensors in the batch!")
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens, video_feat_signal = [], [], []
    for b in batch:
        if len(b) == 6 and get_vid_feats:
            sig, sig_len, video_feat, tokens_i, tokens_i_len, _ = b
        elif len(b) == 5 and get_vid_feats:
            sig, sig_len, video_feat, tokens_i, tokens_i_len = b
        elif len(b) == 5 and not get_vid_feats:
            sig, sig_len, tokens_i, tokens_i_len, _ = b
        elif len(b) == 4 and not get_vid_feats:
            sig, sig_len, tokens_i, tokens_i_len = b
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        if get_vid_feats:
            video_feat_signal.append(video_feat)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    if get_vid_feats:
        video_feat_signal = torch.stack(video_feat_signal)
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)
    base_output = [audio_signal, audio_lengths, tokens, tokens_lengths]

    if get_vid_feats:
        base_output.insert(2, video_feat_signal)

    if sample_ids is not None:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        base_output.append(sample_ids)

    return tuple(base_output)

class ASR_AV_ManifestProcessor:
    """
    Class that processes a manifest json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A", "video_featpath": "/path/to/video_feat.npy"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        bos_id: Id of beginning of sequence symbol to append if not None.
        eos_id: Id of end of sequence symbol to append if not None.
        pad_id: Id of pad symbol. Defaults to 0.
    """

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        index_by_file_id: bool = False,
    ):
        self.parser = parser

        self.collection = collections.ASR_AV_AudioText(
            manifests_files=manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
        )

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    def process_text_by_id(self, index: int) -> Tuple[List[int], int]:
        sample = self.collection[index]
        return self.process_text_by_sample(sample)

    def process_text_by_file_id(self, file_id: str) -> Tuple[List[int], int]:
        manifest_idx = self.collection.mapping[file_id][0]
        sample = self.collection[manifest_idx]
        return self.process_text_by_sample(sample)

    def process_text_by_sample(self, sample: collections.ASR_AV_AudioText.OUTPUT_TYPE) -> Tuple[List[int], int]:
        t, tl = sample.text_tokens, len(sample.text_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl


def cache_datastore_manifests(
    manifest_filepaths: Union[str, List[str]],
    cache_audio: bool = False,
    shared_cache: Optional[bool] = None,
    num_workers: Optional[int] = None,
    max_num_workers: int = 20,
):
    """Cache manifests and audio from an object store.
    It is assumed that remote manifests are using relative paths.

    Args:
        manifest_filepaths: list of paths to manifest files (list of strings or a string with `,` as separator)
        cache_audio: If True, audio from manifest will also be cached
        shared_cache: Optional, True if cache is shared across all nodes
        num_workers: Optional, number of workers to be used for download
        max_num_workers: max number of workers to be used for download, used when setting num_workers automatically
    """
    if isinstance(manifest_filepaths, str):
        manifest_filepaths = manifest_filepaths.split(',')

    num_datastore_manifests = sum(
        [is_datastore_path(f) for f in manifest_filepaths])

    if num_datastore_manifests > 0:
        # Local utility function
        def cache_data(manifest_filepaths, cache_audio, num_workers, max_num_workers):
            """Cache manifests and audio data from object store.
            """
            # Determine the number of workers to use
            if num_workers is None:
                num_workers = os.cpu_count() - 1
            num_workers = min(num_workers, max_num_workers)

            # Process each manifest file
            for manifest_file in manifest_filepaths:
                # If manifest is on a data store, then cache it.
                # Otherwise, nothing to do.
                if is_datastore_path(manifest_file):
                    logging.info('Cache manifest file: %s', manifest_file)
                    cached_manifest_file = DataStoreObject(manifest_file).get()
                    logging.info('Cached at: %s', str(cached_manifest_file))

                    if cache_audio:
                        # Each audio file from manifest will be cached.
                        logging.info(
                            'Cache audio from manifest file: %s', manifest_file)
                        # Assumes that manifest is using relative paths
                        manifest_dir = os.path.dirname(manifest_file)
                        # Prepare all store objects
                        audio_objects = []
                        with open(cached_manifest_file, 'r') as f:
                            for line in f:
                                item = json.loads(line)
                                store_path = os.path.join(
                                    manifest_dir, item['audio_filepath'])
                                audio_objects.append(
                                    DataStoreObject(store_path=store_path))

                        if num_workers is not None and num_workers > 1:
                            logging.debug(
                                'Using multiprocessing with num_workers: %d.', num_workers)
                            with multiprocessing.Pool(processes=num_workers) as p:
                                result = list(
                                    tqdm(p.imap(datastore_object_get, audio_objects), total=len(
                                        audio_objects))
                                )
                        else:
                            logging.debug('Using a single process.')
                            result = []
                            for audio_object in tqdm(audio_objects):
                                result.append(audio_object.get() is not None)

                        if not all(result):
                            raise RuntimeError(
                                'Some files not downloaded successfully')
                        logging.info('Caching complete')

                else:
                    # Nothing to do here
                    logging.debug(
                        'Manifest is not on a data store: %s', manifest_file)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            logging.debug(
                'Distributed environment is available and initialized.')

            # Handle distributed environment
            if shared_cache is None:
                shared_cache = is_datastore_cache_shared()

            if shared_cache:
                logging.debug(
                    'Cache is shared among nodes, cache data on global rank zero.')
                is_rank_zero = is_global_rank_zero()
            else:
                logging.debug(
                    'Cache is not shared among nodes, cache data on local rank zero.')
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                is_rank_zero = local_rank == 0

            if is_rank_zero:
                logging.info('Cache data from %s rank 0',
                             'global' if shared_cache else 'local')
                cache_data(
                    manifest_filepaths=manifest_filepaths,
                    cache_audio=cache_audio,
                    num_workers=num_workers,
                    max_num_workers=max_num_workers,
                )
            logging.debug('Reached barrier')
            torch.distributed.barrier()

        elif is_global_rank_zero():
            # Handle non-distributed environment, e.g., if running on a single GPU
            logging.warning(
                'Torch distributed is not initialized and caching may be prone to data race conditions. '
                'Now caching data from global rank 0. If there are other ranks and they pass this '
                'before rank 0, errors might result.'
            )
            cache_data(
                manifest_filepaths=manifest_filepaths,
                cache_audio=cache_audio,
                num_workers=num_workers,
                max_num_workers=max_num_workers,
            )
        else:
            raise RuntimeError(
                'Torch distributed is not initialized and caching on nodes other than global rank zero is disabled '
                'to avoid race condition between different ranks. To ensure distributed environment is '
                'initialized, please update data config to use `defer_setup = True`.'
            )


class _AVTextDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded
            audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        max_utts: Limit number of utterances
        trim: whether or not to trim silence. Defaults to False
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        pad_id: Id of pad symbol. Defaults to 0
        return_sample_id (bool): whether to return the sample_id as a part of each sample
        channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
        video_frame_rate (int): Frame rate of video, used to calculate duration of video
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': [NeuralType(('B', 'T'), AudioSignal())],
            'a_sig_length': [NeuralType(tuple('B'), LengthsType())],
            'video_input_signal': [NeuralType(('B', 'T', 'D'), ChannelType(), optional=True)],
            'transcripts': [NeuralType(('B', 'T'), LabelsType())],
            'transcript_length': [NeuralType(tuple('B'), LengthsType())],
            'sample_id': [NeuralType(tuple('B'), LengthsType(), optional=True)],
        }

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        video_frame_rate: int = 5,
        get_vid_feats: bool = True,
        get_zero_vid_feats: bool = False,
        override_snr_ratio: Optional[float] = None,
    ):
        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(",")

        # If necessary, cache manifests and audio from object store
        # TODO: @Balu, include cache_video
        cache_datastore_manifests(
            manifest_filepaths=manifest_filepath, cache_audio=True)

        self.manifest_processor = ASR_AV_ManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
        )
        self.featurizer = WaveformFeaturizer(
            sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.return_sample_id = return_sample_id
        self.channel_selector = channel_selector
        self.video_frame_rate = video_frame_rate
        self.get_vid_feats = get_vid_feats
        self.get_zero_vid_feats = get_zero_vid_feats
        self.override_snr_ratio = override_snr_ratio
        self.uniform_snr_list = None
        # choose a list of snr ratios from 0.2 to 0.6 with step 0.1 for __len__ samples
        if self.override_snr_ratio == "rand":
            # set seed as 42
            np.random.seed(42)
            self.uniform_snr_list = np.random.uniform(0.3, 0.6, self.__len__())

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __getitem__(self, index):
        if isinstance(index, IterableABC):
            return [self._process_sample(_index) for _index in index]
        else:
            return self._process_sample(index)

    def calculate_rms(self, audio):
        """Calculate the RMS (root mean square) level of an audio signal."""
        return torch.sqrt(torch.mean(audio ** 2))

    def adjust_volume(self, audio, target_rms):
        """Adjust the audio's volume to a target RMS level."""
        current_rms = self.calculate_rms(audio)
        return audio * (target_rms / (current_rms + 1e-9))  # Avoid division by zero

    def _mix_audios(self, noisy_audio_feats, clean_audio_feats, index, snr, target_sr=16000):
        if self.override_snr_ratio is not None:
            if self.override_snr_ratio == "rand":
                snr = self.uniform_snr_list[index]
            else:
                snr = self.override_snr_ratio
        rms1 = self.calculate_rms(clean_audio_feats)
        rms2 = self.calculate_rms(noisy_audio_feats)
        mean_rms = (rms1 + rms2) / 2
        
        noisy_audio_feats = self.adjust_volume(noisy_audio_feats, mean_rms)
        clean_audio_feats = self.adjust_volume(clean_audio_feats, mean_rms)
        
        # assert len(clean_audio_feats) >= 10*target_sr, f"Audio length is too short: {len(clean_audio_feats)}"
        
        if len(noisy_audio_feats) < len(clean_audio_feats):
            noisy_audio_feats = torch.nn.functional.pad(noisy_audio_feats, (0, len(clean_audio_feats) - len(noisy_audio_feats)))

        # min_len = min(10*target_sr, len(clean_audio_feats))
        min_len = min(len(clean_audio_feats), len(noisy_audio_feats))
        noisy_audio_feats = noisy_audio_feats[:min_len]
        clean_audio_feats = clean_audio_feats[:min_len]
        
        mixed_audio = snr * clean_audio_feats + (1 - snr) * noisy_audio_feats
        
        return mixed_audio      

    
    def _process_sample(self, index):
        sample = self.manifest_processor.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        clean_audio_features = self.featurizer.process(
            sample.audio_file,
            offset=offset,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr,
            channel_selector=self.channel_selector,
        )
        if self.override_snr_ratio != float(0): 
            audio = AudioSegment.from_file(sample.video_file, format="mp4")
            samples_pydub = np.array(audio.get_array_of_samples(), dtype=np.float32)
            noise_features = torch.tensor(samples_pydub, dtype=torch.float32)
            noise_features = noise_features / (2**(8 * audio.sample_width) / 2)
            mixed_features = self._mix_audios(noise_features, clean_audio_features, index, snr = sample.snr)
        else:
            mixed_features = clean_audio_features
        f, fl = mixed_features, torch.tensor(mixed_features.shape[0]).long()

        # TODO: @Balu, saving audio temporarily
        # save_audio_path = f"/tmp/bld56_dataset_v1/audioset/temp_sample_check/{index}.wav"
        # import torchaudio
        # torchaudio.save(save_audio_path, f.unsqueeze(0), 16000)
        
        if self.get_vid_feats:
            if not self.get_zero_vid_feats:
                # check if file exists
                assert os.path.exists(
                    sample.video_featfile), f"Video feature file {sample.video_featfile} does not exist"
                vf = np.load(sample.video_featfile)
                # uniformly sample self.video_frame_rate frames from video at shape 0.
                assert vf.shape[0] == self.video_frame_rate*sample.duration, f"Video feature file {sample.video_featfile} has {vf.shape[0]} frame_feats, expected {self.video_frame_rate}"
                vf = torch.from_numpy(vf)
                # make it torch float
                vf = vf.float()
            else:
                vf = torch.zeros(
                    int(self.video_frame_rate * 10), 768).float()
        
        t, tl = self.manifest_processor.process_text_by_sample(sample=sample)

        output = [f, fl, torch.tensor(t).long(), torch.tensor(tl).long()]

        if self.get_vid_feats:
            output.insert(2, vf)

        if self.return_sample_id:
            output.append(index)

        output = tuple(output)

        return output

    def __len__(self):
        # return 100
        return len(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=self.manifest_processor.pad_id, get_vid_feats=self.get_vid_feats)


class AVToCharDataset(_AVTextDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels: String containing all the possible characters to map to
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        return_sample_id (bool): whether to return the sample_id as a part of each sample
        channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
        video_frame_rate (int): Frame rate of video, used to calculate duration of video
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'hidden_states': NeuralType(('B', 'T', 'D'), ImageFeatureValue(), optional=True),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        labels: Union[str, List[str]],
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        parser: Union[str, Callable] = 'en',
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        video_frame_rate: int = 3,
        get_vid_feats: bool = True,
        get_zero_vid_feats: bool = False,
        override_snr_ratio: Optional[float] = None,
    ):
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=parser,
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
            channel_selector=channel_selector,
            video_frame_rate=video_frame_rate,
            get_vid_feats=get_vid_feats,
            get_zero_vid_feats=get_zero_vid_feats,
            override_snr_ratio=override_snr_ratio,
        )


class AVToBPEDataset(_AVTextDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    In practice, the dataset and manifest used for character encoding and byte pair encoding
    are exactly the same. The only difference lies in how the dataset tokenizes the text in
    the manifest.

    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        tokenizer: A subclass of the Tokenizer wrapper found in the common collection,
            nemo.collections.common.tokenizers.TokenizerSpec. ASR Models support a subset of
            all available tokenizers.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        trim: Whether to trim silence segments
        use_start_end_token: Boolean which dictates whether to add [BOS] and [EOS]
            tokens to beginning and ending of speech respectively.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
        channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
        video_frame_rate (int): Frame rate of video, used to calculate duration of video
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        if self.get_vid_feats:
            return {
                'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
                'a_sig_length': NeuralType(tuple('B'), LengthsType()),
                'hidden_states': NeuralType(('B', 'T', 'D'), ImageFeatureValue(), optional=True),
                'transcripts': NeuralType(('B', 'T'), LabelsType()),
                'transcript_length': NeuralType(tuple('B'), LengthsType()),
                'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        else:
            return {
                'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
                'a_sig_length': NeuralType(tuple('B'), LengthsType()),
                'transcripts': NeuralType(('B', 'T'), LabelsType()),
                'transcript_length': NeuralType(tuple('B'), LengthsType()),
                'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
            }


    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        use_start_end_token: bool = True,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        video_frame_rate: int = 3,
        get_vid_feats: bool = True,
        get_zero_vid_feats: bool = False,
        override_snr_ratio: Optional[float] = None,
    ):
        if use_start_end_token and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                if isinstance(args[0], List) and self.is_aggregate:
                    t = []
                    for span in args[0]:
                        t.extend(self._tokenizer.text_to_ids(
                            span['str'], span['lang']))
                    return t

                t = self._tokenizer.text_to_ids(*args)
                return t

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            trim=trim,
            return_sample_id=return_sample_id,
            channel_selector=channel_selector,
            video_frame_rate=video_frame_rate,
            get_vid_feats=get_vid_feats,
            get_zero_vid_feats=get_zero_vid_feats,
            override_snr_ratio=override_snr_ratio,
        )
