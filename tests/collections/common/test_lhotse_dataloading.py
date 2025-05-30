# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from collections import Counter
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple

import lhotse
import numpy as np
import pytest
import torch
from lhotse import CutSet, MonoCut, NumpyFilesWriter, Recording, compute_num_samples
from lhotse.audio import AudioLoadingError
from lhotse.cut import Cut, MixedCut, PaddingCut
from lhotse.dataset import RoundRobinSampler, ZipSampler
from lhotse.shar import JsonlShardWriter
from lhotse.testing.dummies import dummy_recording
from lhotse.testing.random import deterministic_rng
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.text_adapters import SourceTargetTextExample, TextExample
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model


@pytest.fixture(scope="session")
def cutset_path(tmp_path_factory) -> Path:
    """10 utterances of length 1s as a Lhotse CutSet."""
    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest

    cuts = DummyManifest(CutSet, begin_id=0, end_id=10, with_data=True)
    for c in cuts:
        c.features = None
        c.custom = None
        c.supervisions[0].custom = None

    tmp_path = tmp_path_factory.mktemp("data")
    p = tmp_path / "cuts.jsonl.gz"
    pa = tmp_path / "audio"
    cuts.save_audios(pa).to_file(p)
    return p


@pytest.fixture(scope="session")
def cutset_shar_path(cutset_path: Path) -> Path:
    """10 utterances of length 1s as a Lhotse Shar (tarred) CutSet."""
    from lhotse import CutSet

    cuts = CutSet.from_file(cutset_path)
    p = cutset_path.parent / "shar"
    p.mkdir(exist_ok=True)
    cuts.to_shar(p, fields={"recording": "wav"}, shard_size=5)
    return p


@pytest.fixture(scope="session")
def cutset_shar_path_other(cutset_path: Path) -> Path:
    """10 utterances of length 1s as a Lhotse Shar (tarred) CutSet, but with different IDs."""
    from lhotse import CutSet

    cuts = CutSet.from_file(cutset_path).modify_ids(lambda id: f"other-{id}")
    p = cutset_path.parent / "shar-other"
    p.mkdir(exist_ok=True)
    cuts.to_shar(p, fields={"recording": "wav"}, shard_size=5)
    return p


@pytest.fixture(scope="session")
def nemo_manifest_path(cutset_path: Path):
    """10 utterances of length 1s as a NeMo manifest."""
    from lhotse import CutSet
    from lhotse.serialization import save_to_jsonl

    nemo = []
    for c in CutSet.from_file(cutset_path):
        nemo.append(
            {
                "audio_filepath": c.recording.sources[0].source,
                "text": "irrelevant",
                "text-other": "not relevant",
                "duration": c.duration,
                "my-custom-field": "irrelevant",
                "lang": "en",
                "custom-lang": "pl",
            }
        )
    p = cutset_path.parent / "nemo_manifest.json"
    save_to_jsonl(nemo, p)
    return p


@pytest.fixture(scope="session")
def nemo_manifest_with_skipme_path(nemo_manifest_path: Path) -> Path:
    """Create a nemo manifest with last 2 utterances out of 10 with `_skipme` key enabled"""
    from lhotse.serialization import load_jsonl, save_to_jsonl

    all_items = list(load_jsonl(nemo_manifest_path))

    for item in all_items[-2:]:
        item['_skipme'] = True

    p = nemo_manifest_path.parent / "nemo_manifest_with_skipme.json"
    save_to_jsonl(all_items, p)
    return p


@pytest.fixture(scope="session")
def mc_cutset_path(tmp_path_factory) -> Path:
    """10 two-channel utterances of length 1s as a Lhotse CutSet."""
    from lhotse import CutSet, MultiCut
    from lhotse.testing.dummies import DummyManifest

    num_examples = 10  # number of examples
    num_channels = 2  # number of channels per example

    # create a dummy manifest with single-channel examples
    sc_cuts = DummyManifest(CutSet, begin_id=0, end_id=num_examples * num_channels, with_data=True)
    mc_cuts = []

    for n in range(num_examples):
        # sources for individual channels
        mc_sources = []
        for channel in range(num_channels):
            source = sc_cuts[n * num_channels + channel].recording.sources[0]
            source.channels = [channel]
            mc_sources.append(source)

        # merge recordings
        rec = Recording(
            sources=mc_sources,
            id=f'mc-dummy-recording-{n:02d}',
            num_samples=sc_cuts[0].num_samples,
            duration=sc_cuts[0].duration,
            sampling_rate=sc_cuts[0].sampling_rate,
        )

        # multi-channel cut
        cut = MultiCut(
            recording=rec, id=f'mc-dummy-cut-{n:02d}', start=0, duration=1.0, channel=list(range(num_channels))
        )
        mc_cuts.append(cut)

    mc_cuts = CutSet.from_cuts(mc_cuts)

    tmp_path = tmp_path_factory.mktemp("data")
    p = tmp_path / "mc_cuts.jsonl.gz"
    pa = tmp_path / "mc_audio"
    mc_cuts.save_audios(pa).to_file(p)
    return p


@pytest.fixture(scope="session")
def nemo_tarred_manifest_path(nemo_manifest_path: Path) -> Tuple[str, str]:
    """10 utterances of length 1s as a NeMo tarred manifest."""
    from lhotse.serialization import SequentialJsonlWriter, load_jsonl
    from lhotse.shar.writers import TarWriter

    root = nemo_manifest_path.parent / "nemo_tar"
    root.mkdir(exist_ok=True)

    with (
        TarWriter(f"{root}/audios_%01d.tar", shard_size=5) as tar_writer,
        SequentialJsonlWriter(root / "tarred_audio_filepaths.jsonl") as mft_writer,
    ):
        for idx, d in enumerate(load_jsonl(nemo_manifest_path)):
            p = d["audio_filepath"]
            name = Path(p).name
            with open(p, "rb") as f:
                tar_writer.write(name, BytesIO(f.read()))
            mft_writer.write({**d, "audio_filepath": name, "shard_id": int(idx > 4)})
    return mft_writer.path, f"{root}/audios__OP_0..1_CL_.tar"


@pytest.fixture(scope="session")
def nemo_tarred_manifest_with_skipme_path(nemo_tarred_manifest_path: Path) -> Tuple[str, str]:
    """Create a nemo tarred manifest with last 2 utterances out of 10 with `_skipme` key enabled."""
    from lhotse.serialization import load_jsonl, save_to_jsonl

    json_p, tar_p = nemo_tarred_manifest_path

    all_items = list(load_jsonl(json_p))

    for item in all_items[-2:]:
        item['_skipme'] = True

    p = json_p.parent / "tarred_audio_filepaths_with_skipme.jsonl"
    save_to_jsonl(all_items, p)

    return p, tar_p


@pytest.fixture(scope="session")
def nemo_tarred_manifest_path_multi(nemo_tarred_manifest_path: tuple[str, str]) -> Tuple[str, str]:
    """10 utterances of length 1s as a NeMo tarred manifest. Stored in one manifest per shard."""
    from lhotse.serialization import load_jsonl
    from lhotse.shar.writers import JsonlShardWriter

    json_p, tar_p = nemo_tarred_manifest_path

    json_dir = json_p.parent / "shard_manifests"
    json_dir.mkdir(exist_ok=True)
    with JsonlShardWriter(f"{json_dir}/manifest_%d.jsonl", shard_size=5) as mft_writer:
        for item in load_jsonl(json_p):
            mft_writer.write(item)
    return f"{json_dir}/manifest__OP_0..1_CL_.jsonl", tar_p


@pytest.fixture(scope="session")
def nemo_tarred_manifest_subset_path(nemo_tarred_manifest_path: Tuple[str, str]) -> Tuple[str, str]:
    """Create a shard manifests with randomly chosen 50% percent of tarred contents."""
    from lhotse.serialization import load_jsonl
    from lhotse.shar.writers import JsonlShardWriter

    json_p, tar_p = nemo_tarred_manifest_path
    json_dir = json_p.parent / "shard_manifests"
    json_dir.mkdir(exist_ok=True)
    all_items = list(load_jsonl(json_p))
    tarr_0_data = all_items[:5]
    tarr_1_data = all_items[5:]

    subset_items = tarr_0_data[-3:] + tarr_1_data[-3:]
    with JsonlShardWriter(f"{json_dir}/manifest_%d.jsonl", shard_size=3) as mft_writer:
        for item in subset_items:
            mft_writer.write(item)
    return f"{json_dir}/manifest__OP_0..1_CL_.jsonl", tar_p, subset_items


class UnsupervisedAudioDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts: lhotse.CutSet) -> Dict[str, torch.Tensor]:
        audio, audio_lens = lhotse.dataset.collation.collate_audio(cuts)
        return {"audio": audio, "audio_lens": audio_lens, "ids": [c.id for c in cuts]}


def test_dataloader_from_lhotse_cuts(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": cutset_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [batch for batch in dl]
    assert len(batches) == 4

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 1


def test_dataloader_from_lhotse_cuts_truncate(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": cutset_path,
            "truncate_duration": 0.5,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [b for b in dl]
    assert len(batches) == 3
    # 0.5s = 8000 samples, note the constant duration and batch size except for last batch
    assert batches[0]["audio"].shape == (4, 8000)
    assert batches[1]["audio"].shape == (4, 8000)
    assert batches[2]["audio"].shape == (2, 8000)
    # exactly 10 cuts were used


def test_dataloader_from_lhotse_cuts_cut_into_windows(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": cutset_path,
            "cut_into_windows_duration": 0.5,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [b for b in dl]
    assert len(batches) == 5
    # 0.5s = 8000 samples, note the constant duration and batch size
    assert batches[0]["audio"].shape == (4, 8000)
    assert batches[1]["audio"].shape == (4, 8000)
    assert batches[2]["audio"].shape == (4, 8000)
    assert batches[3]["audio"].shape == (4, 8000)
    assert batches[4]["audio"].shape == (4, 8000)
    # exactly 20 cuts were used because we cut 10x 1s cuts into 20x 0.5s cuts


def test_dataloader_from_lhotse_cuts_pad_min_duration(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": cutset_path,
            "pad_min_duration": 21.0,
            "pad_direction": "left",
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 1,
            "seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    batch = next(iter(dl))
    (cut,) = batch
    assert cut.duration == 21.0
    assert isinstance(cut, MixedCut)
    assert len(cut.tracks) == 2
    assert isinstance(cut.tracks[0].cut, PaddingCut)
    assert isinstance(cut.tracks[1].cut, MonoCut)


def test_dataloader_from_lhotse_cuts_channel_selector(mc_cutset_path: Path):
    # Dataloader without channel selector
    config = OmegaConf.create(
        {
            "cuts_path": mc_cutset_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )
    batches = [b for b in dl]
    assert len(batches) == 3

    # 1.0s = 16000 samples, two channels, note the constant duration and batch size
    assert batches[0]["audio"].shape == (4, 2, 16000)
    assert batches[1]["audio"].shape == (4, 2, 16000)
    assert batches[2]["audio"].shape == (2, 2, 16000)
    # exactly 10 cuts were used

    # Apply channel selector
    for channel_selector in [None, 0, 1]:

        config_cs = OmegaConf.create(
            {
                "cuts_path": mc_cutset_path,
                "channel_selector": channel_selector,
                "sample_rate": 16000,
                "shuffle": True,
                "use_lhotse": True,
                "num_workers": 0,
                "batch_size": 4,
                "seed": 0,
            }
        )

        dl_cs = get_lhotse_dataloader_from_config(
            config=config_cs, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
        )

        for n, b_cs in enumerate(dl_cs):
            if channel_selector is None:
                # no channel selector, needs to match the original dataset
                assert torch.equal(b_cs["audio"], batches[n]["audio"])
            else:
                # channel selector, needs to match the selected channel
                assert torch.equal(b_cs["audio"], batches[n]["audio"][:, channel_selector, :])


def test_dataloader_from_lhotse_shar_cuts(cutset_shar_path: Path):
    config = OmegaConf.create(
        {
            "shar_path": cutset_shar_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    # Note: we use islice here because with Lhotse Shar the dataloader will always be infinite.
    batches = [batch for batch in islice(dl, 4)]
    assert len(batches) == 4

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3


def test_dataloader_from_lhotse_shar_cuts_via_fields(cutset_shar_path: Path):
    config = OmegaConf.create(
        {
            "shar_path": {
                "cuts": f"{cutset_shar_path}/cuts._OP_000000..000001_CL_.jsonl.gz",
                "recording": f"{cutset_shar_path}/recording._OP_000000..000001_CL_.tar",
            },
            "sample_rate": 16000,
            "num_workers": 0,
            "shuffle": False,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    batch = next(iter(dl))
    assert len(batch) == 4
    audio = batch[0].load_audio()
    assert isinstance(audio, np.ndarray)


def test_dataloader_from_lhotse_shar_cuts_add_new_field(tmp_path_factory, cutset_shar_path: Path):

    # We're creating a new field called "wer" that will be dynamically attached to Lhotse Shar cuts.
    # Each "wer" shard is a jsonl manifest that has to match the "cuts" sharded manifest.
    # It must have a "cut_id" field used for runtime check that the user provided correct paths.
    # "wer" will be attached to each cut under `cut.wer` / cut.custom["wer"].
    wer_dir = tmp_path_factory.mktemp("wer_dir")
    with JsonlShardWriter(f"{wer_dir}/wer.%06d.jsonl.gz", shard_size=5) as writer:
        for i in range(10):
            writer.write({"cut_id": "dummy-mono-cut-%04d" % i, "wer": 0.5})

    config = OmegaConf.create(
        {
            "shar_path": {
                "cuts": f"{cutset_shar_path}/cuts._OP_000000..000001_CL_.jsonl.gz",
                "recording": f"{cutset_shar_path}/recording._OP_000000..000001_CL_.tar",
                "wer": f"{wer_dir}/wer._OP_000000..000001_CL_.jsonl.gz",
            },
            "sample_rate": 16000,
            "num_workers": 0,
            "shuffle": False,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    batch = next(iter(dl))
    assert len(batch) == 4
    assert batch[0].wer == 0.5


def test_dataloader_from_nemo_manifest(nemo_manifest_path: Path):
    config = OmegaConf.create(
        {
            "manifest_filepath": nemo_manifest_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [batch for batch in dl]
    assert len(batches) == 4

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 1


class _Identity:
    def __getitem__(self, cuts):
        return cuts


def test_dataloader_from_nemo_manifest_has_custom_fields(nemo_manifest_path: Path):
    config = OmegaConf.create(
        {
            "manifest_filepath": nemo_manifest_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": False,
            "batch_duration": 4.0,  # seconds
            "shuffle_buffer_size": 10,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())

    batch = next(iter(dl))
    for cut in batch:
        assert isinstance(cut.custom, dict)
        assert "my-custom-field" in cut.custom


def test_dataloader_from_tarred_nemo_manifest(nemo_tarred_manifest_path: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [batch for batch in islice(dl, 4)]
    assert len(batches) == 4

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3


def test_dataloader_from_tarred_nemo_manifest_weighted_combination(nemo_tarred_manifest_path: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path
    config = OmegaConf.create(
        {
            "manifest_filepath": [[json_mft, 0.8], [json_mft, 0.2]],
            "tarred_audio_filepaths": [[tar_mft], [tar_mft]],
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    b = next(iter(dl))
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3


def test_dataloader_from_tarred_nemo_manifest_multi(nemo_tarred_manifest_path_multi: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path_multi
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [batch for batch in islice(dl, 4)]
    assert len(batches) == 4

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3


def test_dataloader_from_tarred_nemo_manifest_multi_max_open_streams(nemo_tarred_manifest_path_multi: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path_multi
    config = OmegaConf.create(
        {
            "manifest_filepath": [[json_mft], [json_mft]],
            "tarred_audio_filepaths": [[tar_mft], [tar_mft]],
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "max_open_streams": 1,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    _ = next(iter(dl))


def test_dataloader_from_tarred_nemo_manifest_concat(nemo_tarred_manifest_path: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "concatenate_samples": True,
            "concatenate_duration_factor": 3.0,
            "batch_duration": 4.0,
            "quadratic_duration": 15.0,  # seconds
            "use_bucketing": False,
            "drop_last": False,
            "shuffle_buffer_size": 10,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [batch for batch in islice(dl, 4)]

    assert len(batches) == 4

    # the first element has been concatenated: 2x16000 speech (2x1s) + 1600 gap (0.1s)
    expected_audio_lens = torch.tensor([33600, 16000], dtype=torch.int32)

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 2
    torch.testing.assert_close(b["audio_lens"], expected_audio_lens)

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 2
    torch.testing.assert_close(b["audio_lens"], expected_audio_lens)

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 2
    torch.testing.assert_close(b["audio_lens"], expected_audio_lens)

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 2
    torch.testing.assert_close(b["audio_lens"], expected_audio_lens)


def test_dataloader_from_lhotse_shar_cuts_combine_datasets_unweighted(
    cutset_shar_path: Path, cutset_shar_path_other: Path
):
    """
    Note: if we iterated more mini-batches in this test, in the expectation there
    will be 50-50 % mini-batch occupancy of examples from both datasets.
    """
    config = OmegaConf.create(
        {
            "shar_path": [cutset_shar_path, cutset_shar_path_other],
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    # Note: we use islice here because with Lhotse Shar the dataloader will always be infinite.
    batches = [batch for batch in islice(dl, 4)]
    assert len(batches) == 4

    b = batches[0]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 1  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 2  # dataset 2

    b = batches[1]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 0  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 3  # dataset 2

    b = batches[2]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 2  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 1  # dataset 2

    b = batches[3]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 1  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 2  # dataset 2


def test_dataloader_from_lhotse_shar_cuts_combine_datasets_weighted(
    cutset_shar_path: Path, cutset_shar_path_other: Path
):
    """
    Note: if we iterated more mini-batches in this test, in the expectation there
    will be 90-10 % mini-batch occupancy of examples from both datasets.
    """
    config = OmegaConf.create(
        {
            "shar_path": [[cutset_shar_path, 90], [cutset_shar_path_other, 10]],
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    # Note: we use islice here because with Lhotse Shar the dataloader will always be infinite.
    batches = [batch for batch in islice(dl, 6)]
    assert len(batches) == 6

    b = batches[0]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[1]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 1  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 2  # dataset 2

    b = batches[2]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 2  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 1  # dataset 2

    b = batches[3]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[4]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[5]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2


class TextDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts: lhotse.CutSet) -> List[str]:
        return [c.supervisions[0].text for c in cuts]


@pytest.mark.parametrize(["text_field", "text_value"], [(None, "irrelevant"), ("text-other", "not relevant")])
def test_dataloader_from_nemo_manifest_with_text_field(nemo_manifest_path: Path, text_field: str, text_value: str):
    kwarg = {"text_field": text_field} if text_field is not None else {}
    config = OmegaConf.create(
        {
            "manifest_filepath": nemo_manifest_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 2,
            # lhotse specific
            "use_bucketing": False,
            **kwarg,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=TextDataset())
    b = next(iter(dl))
    assert b == [text_value] * 2


class LangDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts: lhotse.CutSet) -> List[str]:
        return [c.supervisions[0].language for c in cuts]


@pytest.mark.parametrize(["lang_field", "lang_value"], [(None, "en"), ("custom-lang", "pl")])
def test_dataloader_from_nemo_manifest_with_lang_field(nemo_manifest_path: Path, lang_field: str, lang_value: str):
    kwarg = {"lang_field": lang_field} if lang_field is not None else {}
    config = OmegaConf.create(
        {
            "manifest_filepath": nemo_manifest_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 2,
            # lhotse specific
            "use_bucketing": False,
            **kwarg,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=LangDataset())
    b = next(iter(dl))
    assert b == [lang_value] * 2


def test_lazy_nemo_iterator_with_offset_field(tmp_path: Path):
    import numpy as np
    import soundfile as sf

    from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator

    # Have to generate as INT16 to avoid quantization error after saving to 16-bit WAV
    INT16MAX = 2**15
    expected_audio = np.random.randint(low=-INT16MAX - 1, high=INT16MAX, size=(16000,)).astype(np.float32) / INT16MAX
    audio_path = str(tmp_path / "dummy.wav")
    sf.write(audio_path, expected_audio, 16000)

    manifest_path = str(tmp_path / "manifest.json")
    lhotse.serialization.save_to_jsonl(
        [
            {"audio_filepath": audio_path, "offset": 0.0, "duration": 0.5, "text": "irrelevant"},
            {"audio_filepath": audio_path, "offset": 0.5, "duration": 0.5, "text": "irrelevant"},
        ],
        manifest_path,
    )

    cuts = lhotse.CutSet(LazyNeMoIterator(manifest_path))

    cut = cuts[0]
    assert isinstance(cut, lhotse.MonoCut)
    assert cut.start == 0.0
    assert cut.duration == 0.5
    assert cut.sampling_rate == 16000
    assert cut.num_samples == 8000
    assert cut.supervisions[0].text == "irrelevant"
    audio = cut.load_audio()
    assert audio.shape == (1, 8000)
    np.testing.assert_equal(audio[0], expected_audio[:8000])

    cut = cuts[1]
    assert isinstance(cut, lhotse.MonoCut)
    assert cut.start == 0.5
    assert cut.duration == 0.5
    assert cut.sampling_rate == 16000
    assert cut.num_samples == 8000
    assert cut.supervisions[0].text == "irrelevant"
    audio = cut.load_audio()
    assert audio.shape == (1, 8000)
    np.testing.assert_allclose(audio[0], expected_audio[8000:], atol=5e-5)

    assert cuts[0].id != cuts[1].id


def test_lazy_nemo_iterator_with_relative_paths(tmp_path: Path):
    import numpy as np
    import soundfile as sf

    from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator

    # Have to generate as INT16 to avoid quantization error after saving to 16-bit WAV
    INT16MAX = 2**15
    expected_audio = np.random.randint(low=-INT16MAX - 1, high=INT16MAX, size=(16000,)).astype(np.float32) / INT16MAX
    audio_path = str(tmp_path / "dummy.wav")
    sf.write(audio_path, expected_audio, 16000)

    manifest_path = str(tmp_path / "manifest.json")
    lhotse.serialization.save_to_jsonl(
        [
            # note: relative path
            {"audio_filepath": "dummy.wav", "offset": 0.0, "duration": 0.5, "text": "irrelevant"},
        ],
        manifest_path,
    )

    cuts = lhotse.CutSet(LazyNeMoIterator(manifest_path))
    cut = cuts[0]
    audio = cut.load_audio()

    assert isinstance(cut, lhotse.MonoCut)
    assert cut.start == 0.0
    assert cut.duration == 0.5
    assert cut.sampling_rate == 16000
    assert cut.num_samples == 8000
    assert cut.supervisions[0].text == "irrelevant"
    assert audio.shape == (1, 8000)
    np.testing.assert_equal(audio[0], expected_audio[:8000])


def test_lhotse_cuts_resolve_relative_paths(tmp_path: Path):
    cuts_path = tmp_path / "cuts.jsonl.gz"
    audio_path = tmp_path / "_relative_test_audio_.wav"
    lhotse.audio.save_audio(audio_path, np.random.rand(16000) - 0.5, 16000)
    cut = Recording.from_file(audio_path).to_cut()
    cut.recording.sources[0].source = str(audio_path.name)  # make the path relative
    cut.target_recording = cut.recording  # assign a custom field with relative path
    with NumpyFilesWriter(tmp_path) as w:
        cut.some_array = w.store_array(cut.id, np.random.randn(32))
        cut.some_array.storage_path = ""  # relative path

    with pytest.raises(AudioLoadingError):
        cut.load_audio()  # Lhotse doesn't know about what the path should be relative to
        cut.load_target_recording()

    CutSet([cut]).to_file(cuts_path)

    config = OmegaConf.create(
        {
            "cuts_path": cuts_path,
            "sample_rate": 16000,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 2,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())

    batches = [batch for batch in dl]
    assert len(batches) == 1

    for cut in batches[0]:
        assert cut.has_recording
        cut.load_audio()  # works
        assert cut.has_custom("target_recording")
        cut.load_target_recording()
        assert cut.has_custom("some_array")
        cut.load_some_array()


class Identity(torch.utils.data.Dataset):
    def __getitem__(self, cuts: lhotse.CutSet) -> lhotse.CutSet:
        return cuts


def test_extended_data_input_cfg(cutset_shar_path, nemo_tarred_manifest_path_multi):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "nemo_tarred",
                    "manifest_filepath": nemo_tarred_manifest_path_multi[0],
                    "tarred_audio_filepaths": nemo_tarred_manifest_path_multi[1],
                    "weight": 0.5,
                    "tags": {
                        "language": "en",
                        "modality": "audio",
                        "dataset_name": "D1",
                    },
                },
                {
                    "type": "lhotse_shar",
                    "shar_path": cutset_shar_path,
                    "weight": 0.5,
                    "tags": {
                        "language": "en",
                        "modality": "audio",
                        "dataset_name": "D2",
                    },
                },
            ],
            "sample_rate": 16000,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert all(c.custom["language"] == "en" for c in b)
    assert all(c.custom["modality"] == "audio" for c in b)
    assert sum(c.custom["dataset_name"] == "D1" for c in b) == 2
    assert sum(c.custom["dataset_name"] == "D2" for c in b) == 2

    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert all(c.custom["language"] == "en" for c in b)
    assert all(c.custom["modality"] == "audio" for c in b)
    assert sum(c.custom["dataset_name"] == "D1" for c in b) == 1
    assert sum(c.custom["dataset_name"] == "D2" for c in b) == 3


def test_extended_data_input_cfg_subgroup(cutset_shar_path, nemo_tarred_manifest_path_multi):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "group",
                    "input_cfg": [
                        {
                            "type": "nemo_tarred",
                            "manifest_filepath": nemo_tarred_manifest_path_multi[0],
                            "tarred_audio_filepaths": nemo_tarred_manifest_path_multi[1],
                            "weight": 0.5,
                            "tags": {
                                "language": "en",
                                "modality": "audio",
                                "dataset_name": "D1",
                            },
                        },
                        {
                            "type": "lhotse_shar",
                            "shar_path": cutset_shar_path,
                            "weight": 0.5,
                            "tags": {
                                "language": "en",
                                "modality": "audio",
                                "dataset_name": "D2",
                            },
                        },
                    ],
                    "weight": 0.2,
                    "tags": {
                        "group_name": "G1",
                    },
                },
                {
                    "type": "group",
                    "weight": 0.8,
                    "input_cfg": [
                        {
                            "type": "nemo_tarred",
                            "manifest_filepath": nemo_tarred_manifest_path_multi[0],
                            "tarred_audio_filepaths": nemo_tarred_manifest_path_multi[1],
                            "weight": 0.5,
                            "tags": {
                                "language": "en",
                                "modality": "audio",
                                "dataset_name": "D3",
                            },
                        },
                        {
                            "type": "lhotse_shar",
                            "shar_path": cutset_shar_path,
                            "weight": 0.5,
                            "tags": {
                                "language": "en",
                                "modality": "audio",
                                "dataset_name": "D4",
                            },
                        },
                    ],
                    "tags": {
                        "group_name": "G2",
                    },
                },
            ],
            "sample_rate": 16000,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 32,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Sample 100 mini-batches and test statistical properties
    group_occurrences = Counter()
    dataset_occurrences = Counter()
    for batch in islice(dl, 100):
        for cut in batch:
            group_occurrences[cut.group_name] += 1
            dataset_occurrences[cut.dataset_name] += 1

    tot = sum(group_occurrences.values())
    for k in group_occurrences:
        group_occurrences[k] /= tot
    for k in dataset_occurrences:
        dataset_occurrences[k] /= tot

    def almost(number):
        return pytest.approx(number, abs=0.02)

    assert group_occurrences["G1"] == almost(0.2)  # group weight: 0.2
    assert group_occurrences["G2"] == almost(0.8)  # group weight: 0.8
    assert dataset_occurrences["D1"] == almost(0.1)  # group weight: 0.2 * dataset weight 0.5 => 0.1
    assert dataset_occurrences["D2"] == almost(0.1)  # group weight: 0.2 * dataset weight 0.5 => 0.1
    assert dataset_occurrences["D3"] == almost(0.4)  # group weight: 0.8 * dataset weight 0.5 => 0.4
    assert dataset_occurrences["D4"] == almost(0.4)  # group weight: 0.8 * dataset weight 0.5 => 0.4


def test_extended_data_input_cfg_yaml_path(tmp_path, cutset_shar_path, nemo_tarred_manifest_path_multi):
    input_cfg = [
        {
            "type": "nemo_tarred",
            "manifest_filepath": str(nemo_tarred_manifest_path_multi[0]),
            "tarred_audio_filepaths": str(nemo_tarred_manifest_path_multi[1]),
            "weight": 0.5,
            "tags": {
                "language": "en",
                "modality": "audio",
                "dataset_name": "D1",
            },
        },
        {
            "type": "lhotse_shar",
            "shar_path": str(cutset_shar_path),
            "weight": 0.5,
            "tags": {
                "language": "en",
                "modality": "audio",
                "dataset_name": "D2",
            },
        },
    ]

    yaml_path = tmp_path / "input_cfg.yaml"
    lhotse.serialization.save_to_yaml(input_cfg, yaml_path)

    config = OmegaConf.create(
        {
            "input_cfg": input_cfg,
            "sample_rate": 16000,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 32,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    batch = next(iter(dl))
    assert isinstance(batch, lhotse.CutSet)
    for cut in batch:
        assert cut.dataset_name in ("D1", "D2")


@pytest.fixture(scope="session")
def txt_en_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("text_data")
    en_path = tmp_path / "text.en"
    en_path.write_text(
        """Example text in English.
Another sentence.
        """
    )
    return en_path


@pytest.fixture(scope="session")
def txt_es_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("text_data")
    es_path = tmp_path / "text.es"
    es_path.write_text(
        """Otro texto en ingles.
Otra frase."""
    )
    return es_path


@pytest.fixture(scope="session")
def questions_path(tmp_path_factory) -> str:
    tmpdir = tmp_path_factory.mktemp("questions")
    qp = tmpdir / "questions.txt"
    qp.write_text("translate the following to spanish")
    return str(qp)


def test_text_file_input(txt_en_path, txt_es_path):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "txt",
                    "paths": txt_en_path,
                    "language": "en",
                },
            ],
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    # Note: this test does not need to pass a tokenizer because we use static batch sizes
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, TextExample) for c in b)
    assert all(c.language == "en" for c in b)

    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, TextExample) for c in b)
    assert all(c.language == "en" for c in b)


def test_text_file_pairs_input(txt_en_path, txt_es_path, questions_path):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "txt_pair",
                    "source_paths": txt_en_path,
                    "target_paths": txt_es_path,
                    "questions_path": questions_path,
                    "source_language": "en",
                    "target_language": "es",
                    "questions_language": "en",
                },
            ],
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    # Note: this test does not need to pass a tokenizer because we use static batch sizes
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, SourceTargetTextExample) for c in b)
    assert all(c.source.language == "en" for c in b)
    assert all(c.target.language == "es" for c in b)

    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, SourceTargetTextExample) for c in b)
    assert all(c.source.language == "en" for c in b)
    assert all(c.target.language == "es" for c in b)


@pytest.fixture(scope="session")
def txt_pair_paths_shards(tmp_path_factory, txt_en_path, txt_es_path):
    tmp_path = tmp_path_factory.mktemp("text_data_shards")

    en_text = txt_en_path.read_text().splitlines()
    (tmp_path / "en_0.txt").write_text("\n".join(en_text[:5]))
    (tmp_path / "en_1.txt").write_text("\n".join(en_text[5:]))

    es_text = txt_es_path.read_text().splitlines()
    (tmp_path / "es_0.txt").write_text("\n".join(es_text[:5]))
    (tmp_path / "es_1.txt").write_text("\n".join(es_text[5:]))

    return f"{tmp_path}/en__OP_0..1_CL_.txt", f"{tmp_path}/es__OP_0..1_CL_.txt"


def test_text_file_pairs_shards_input(txt_pair_paths_shards: tuple[str, str], questions_path):
    en_paths, es_paths = txt_pair_paths_shards

    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "txt_pair",
                    "source_paths": en_paths,
                    "target_paths": es_paths,
                    "questions_path": questions_path,
                    "source_language": "en",
                    "target_language": "es",
                    "questions_language": "en",
                },
            ],
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    # Note: this test does not need to pass a tokenizer because we use static batch sizes
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, SourceTargetTextExample) for c in b)
    assert all(c.source.language == "en" for c in b)
    assert all(c.target.language == "es" for c in b)

    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, SourceTargetTextExample) for c in b)
    assert all(c.source.language == "en" for c in b)
    assert all(c.target.language == "es" for c in b)


@pytest.fixture(scope="session")
def en_es_tokenizer(tmp_path_factory, txt_en_path, txt_es_path) -> SentencePieceTokenizer:
    tmpdir = tmp_path_factory.mktemp("en_es_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text(txt_en_path.read_text() + "\n" + txt_es_path.read_text())
    create_spt_model(text_path, vocab_size=128, sample_size=-1, do_lower_case=False, output_dir=str(tmpdir))
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


def test_multimodal_text_audio_dataloading(
    txt_pair_paths_shards: tuple[str, str],
    nemo_tarred_manifest_path_multi: tuple[str, str],
    en_es_tokenizer: SentencePieceTokenizer,
    questions_path: str,
):
    en_paths, es_paths = txt_pair_paths_shards
    manifest_filepath, tarred_audio_filepaths = nemo_tarred_manifest_path_multi
    QF, BT = 50, 1024
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "txt_pair",
                    "source_paths": en_paths,
                    "target_paths": es_paths,
                    "source_language": "en",
                    "target_language": "es",
                    "questions_path": questions_path,
                    "questions_language": "en",
                    "tags": {
                        "modality": "text",
                    },
                },
                {
                    "type": "nemo_tarred",
                    "manifest_filepath": manifest_filepath,
                    "tarred_audio_filepaths": tarred_audio_filepaths,
                    "tags": {
                        "modality": "audio",
                    },
                },
            ],
            "shuffle": True,
            "num_workers": 0,
            "use_multimodal_sampling": True,
            "prompt_format": "plain",
            "batch_tokens": BT,
            # How to set token equivalent duration in actual training?
            #   assuming fbank frames: 0.01 is the base due to frame shift;
            #       + subsampling x8 gives us 0.08
            #   assuming discrete audio tokens, with frame rate 50Hz,
            #       we'd get 0.02
            #   in this test we'll just use 0.1 for simplicity
            "token_equivalent_duration": 0.1,
            "quadratic_factor": QF,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=en_es_tokenizer,
    )

    b = next(iter(dl))
    assert isinstance(b, lhotse.CutSet)
    assert len(b)
    assert any(isinstance(ex, Cut) for ex in b)
    assert any(isinstance(ex, SourceTargetTextExample) for ex in b)
    # Batch tokens is not exceeded after applying the quadratic factor correction
    assert sum(ex.num_tokens**2 / QF for ex in b) <= BT
    for ex in b:
        if isinstance(ex, Cut):
            assert ex.modality == "audio"
            assert isinstance(ex.load_audio(), np.ndarray)
            assert isinstance(ex.supervisions[0].text, str)
        if isinstance(ex, SourceTargetTextExample):
            assert ex.modality == "text"
            assert ex.source.language == "en"
            assert ex.target.language == "es"
            assert isinstance(ex.source.text, str)
            assert isinstance(ex.target.text, str)
            assert isinstance(ex.question.text, str)
            assert torch.is_tensor(ex.input_ids)
            assert torch.is_tensor(ex.context_ids)
            assert torch.is_tensor(ex.answer_ids)
            assert torch.is_tensor(ex.mask)


def test_multimodal_text_audio_dataloading_zip_strategy(
    txt_pair_paths_shards: tuple[str, str],
    nemo_tarred_manifest_path_multi: tuple[str, str],
    en_es_tokenizer: SentencePieceTokenizer,
    questions_path: str,
):
    en_paths, es_paths = txt_pair_paths_shards
    manifest_filepath, tarred_audio_filepaths = nemo_tarred_manifest_path_multi
    QF, BT = 50, 64
    config = OmegaConf.create(
        {
            "multi_config": True,
            "sampler_fusion": "zip",  # <---- !!! this option is being tested here !!!
            "seed": 0,
            "shard_seed": 0,
            "shuffle": True,
            "num_workers": 0,
            "audio": {
                "input_cfg": [
                    {
                        "type": "nemo_tarred",
                        "manifest_filepath": manifest_filepath,
                        "tarred_audio_filepaths": tarred_audio_filepaths,
                        "tags": {
                            "modality": "audio",
                        },
                    },
                ],
                "prompt_format": "plain",
                "use_multimodal_sampling": True,
                "batch_tokens": BT,
                # How to set token equivalent duration in actual training?
                #   assuming fbank frames: 0.01 is the base due to frame shift;
                #       + subsampling x8 gives us 0.08
                #   assuming discrete audio tokens, with frame rate 50Hz,
                #       we'd get 0.02
                #   in this test we'll just use 0.1 for simplicity
                "token_equivalent_duration": 0.1,
                "quadratic_factor": QF,
            },
            "text": {
                "input_cfg": [
                    {
                        "type": "txt_pair",
                        "source_paths": en_paths,
                        "target_paths": es_paths,
                        "source_language": "en",
                        "target_language": "es",
                        "questions_path": questions_path,
                        "questions_language": "en",
                        "tags": {
                            "modality": "text",
                        },
                    },
                ],
                "use_multimodal_sampling": True,
                "prompt_format": "plain",
                "batch_tokens": 64,
                # How to set token equivalent duration in actual training?
                #   assuming fbank frames: 0.01 is the base due to frame shift;
                #       + subsampling x8 gives us 0.08
                #   assuming discrete audio tokens, with frame rate 50Hz,
                #       we'd get 0.02
                #   in this test we'll just use 0.1 for simplicity
                "token_equivalent_duration": 0.1,
                "quadratic_factor": 50,
            },
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=en_es_tokenizer,
    )

    assert isinstance(dl.dataset.sampler, ZipSampler)

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert len(b)
    assert any(isinstance(ex, Cut) for ex in b)
    assert any(isinstance(ex, SourceTargetTextExample) for ex in b)
    # Batch tokens is not exceeded after applying the quadratic factor correction
    # Note: zip samples stitches together two batches hence * 2
    assert sum(ex.num_tokens**2 / QF for ex in b) <= BT * 2
    for ex in b:
        if isinstance(ex, Cut):
            assert ex.modality == "audio"
            assert isinstance(ex.load_audio(), np.ndarray)
            assert isinstance(ex.supervisions[0].text, str)
        if isinstance(ex, SourceTargetTextExample):
            assert ex.modality == "text"
            assert ex.source.language == "en"
            assert ex.target.language == "es"
            assert torch.is_tensor(ex.input_ids)
            assert torch.is_tensor(ex.context_ids)
            assert torch.is_tensor(ex.answer_ids)
            assert torch.is_tensor(ex.mask)

    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert len(b)
    assert any(isinstance(ex, Cut) for ex in b)
    assert any(isinstance(ex, SourceTargetTextExample) for ex in b)
    # Batch tokens is not exceeded after applying the quadratic factor correction
    # Note: zip samples stitches together two batches hence * 2
    assert sum(ex.num_tokens**2 / QF for ex in b) <= BT * 2
    for ex in b:
        if isinstance(ex, Cut):
            assert ex.modality == "audio"
            assert isinstance(ex.load_audio(), np.ndarray)
            assert isinstance(ex.supervisions[0].text, str)
        if isinstance(ex, SourceTargetTextExample):
            assert ex.modality == "text"
            assert ex.source.language == "en"
            assert ex.target.language == "es"
            assert torch.is_tensor(ex.input_ids)
            assert torch.is_tensor(ex.context_ids)
            assert torch.is_tensor(ex.answer_ids)
            assert torch.is_tensor(ex.mask)


def test_multimodal_text_audio_dataloading_round_robin_strategy(
    txt_pair_paths_shards: tuple[str, str],
    nemo_tarred_manifest_path_multi: tuple[str, str],
    en_es_tokenizer: SentencePieceTokenizer,
    questions_path: str,
):
    en_paths, es_paths = txt_pair_paths_shards
    manifest_filepath, tarred_audio_filepaths = nemo_tarred_manifest_path_multi
    QF, BT = 50, 64
    config = OmegaConf.create(
        {
            "multi_config": True,
            "sampler_fusion": "round_robin",  # <---- !!! this option is being tested here !!!
            "seed": 0,
            "shard_seed": 0,
            "shuffle": True,
            "num_workers": 0,
            "audio": {
                "input_cfg": [
                    {
                        "type": "nemo_tarred",
                        "manifest_filepath": manifest_filepath,
                        "tarred_audio_filepaths": tarred_audio_filepaths,
                        "tags": {
                            "modality": "audio",
                        },
                    },
                ],
                "use_multimodal_sampling": True,
                "prompt_format": "plain",
                "batch_tokens": BT,
                # How to set token equivalent duration in actual training?
                #   assuming fbank frames: 0.01 is the base due to frame shift;
                #       + subsampling x8 gives us 0.08
                #   assuming discrete audio tokens, with frame rate 50Hz,
                #       we'd get 0.02
                #   in this test we'll just use 0.1 for simplicity
                "token_equivalent_duration": 0.1,
                "quadratic_factor": QF,
            },
            "text": {
                "input_cfg": [
                    {
                        "type": "txt_pair",
                        "source_paths": en_paths,
                        "target_paths": es_paths,
                        "source_language": "en",
                        "target_language": "es",
                        "questions_path": questions_path,
                        "questions_language": "en",
                        "tags": {
                            "modality": "text",
                        },
                    },
                ],
                "prompt_format": "plain",
                "use_multimodal_sampling": True,
                "batch_tokens": BT,
                # How to set token equivalent duration in actual training?
                #   assuming fbank frames: 0.01 is the base due to frame shift;
                #       + subsampling x8 gives us 0.08
                #   assuming discrete audio tokens, with frame rate 50Hz,
                #       we'd get 0.02
                #   in this test we'll just use 0.1 for simplicity
                "token_equivalent_duration": 0.1,
                "quadratic_factor": QF,
            },
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=en_es_tokenizer,
    )

    assert isinstance(dl.dataset.sampler, RoundRobinSampler)

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    # Batch 0 is audio-only
    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert len(b)
    assert all(isinstance(ex, Cut) for ex in b)
    # Batch tokens is not exceeded after applying the quadratic factor correction
    assert sum(ex.num_tokens**2 / QF for ex in b) <= BT
    for ex in b:
        assert ex.modality == "audio"
        assert isinstance(ex.load_audio(), np.ndarray)
        assert isinstance(ex.supervisions[0].text, str)

    # Batch 1 is text-only
    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert len(b)
    assert all(isinstance(ex, SourceTargetTextExample) for ex in b)
    # Batch tokens is not exceeded after applying the quadratic factor correction
    assert sum(ex.num_tokens**2 / QF for ex in b) <= BT
    for ex in b:
        assert ex.modality == "text"
        assert ex.source.language == "en"
        assert ex.target.language == "es"
        assert torch.is_tensor(ex.input_ids)
        assert torch.is_tensor(ex.context_ids)
        assert torch.is_tensor(ex.answer_ids)
        assert torch.is_tensor(ex.mask)


def test_multimodal_text_audio_dataloading_randomized_round_robin_strategy(
    deterministic_rng,
    txt_pair_paths_shards: tuple[str, str],
    nemo_tarred_manifest_path_multi: tuple[str, str],
    en_es_tokenizer: SentencePieceTokenizer,
    questions_path: str,
):
    en_paths, es_paths = txt_pair_paths_shards
    manifest_filepath, tarred_audio_filepaths = nemo_tarred_manifest_path_multi
    QF, BT = 50, 64
    config = OmegaConf.create(
        {
            "multi_config": True,
            "sampler_fusion": "randomized_round_robin",  # <---- !!! this option is being tested here !!!
            "sampler_weights": {
                "audio": 0.5,
                "text": 0.5,
            },
            "seed": 0,
            "shard_seed": 0,
            "shuffle": True,
            "num_workers": 0,
            "audio": {
                "input_cfg": [
                    {
                        "type": "nemo_tarred",
                        "manifest_filepath": manifest_filepath,
                        "tarred_audio_filepaths": tarred_audio_filepaths,
                        "tags": {
                            "modality": "audio",
                        },
                    },
                ],
                "use_multimodal_sampling": True,
                "prompt_format": "plain",
                "batch_tokens": BT,
                # How to set token equivalent duration in actual training?
                #   assuming fbank frames: 0.01 is the base due to frame shift;
                #       + subsampling x8 gives us 0.08
                #   assuming discrete audio tokens, with frame rate 50Hz,
                #       we'd get 0.02
                #   in this test we'll just use 0.1 for simplicity
                "token_equivalent_duration": 0.1,
                "quadratic_factor": QF,
            },
            "text": {
                "input_cfg": [
                    {
                        "type": "txt_pair",
                        "source_paths": en_paths,
                        "target_paths": es_paths,
                        "source_language": "en",
                        "target_language": "es",
                        "questions_path": questions_path,
                        "questions_language": "en",
                        "tags": {
                            "modality": "text",
                        },
                    },
                ],
                "prompt_format": "plain",
                "use_multimodal_sampling": True,
                "batch_tokens": BT,
                # How to set token equivalent duration in actual training?
                #   assuming fbank frames: 0.01 is the base due to frame shift;
                #       + subsampling x8 gives us 0.08
                #   assuming discrete audio tokens, with frame rate 50Hz,
                #       we'd get 0.02
                #   in this test we'll just use 0.1 for simplicity
                "token_equivalent_duration": 0.1,
                "quadratic_factor": QF,
            },
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=en_es_tokenizer,
    )

    assert isinstance(dl.dataset.sampler, RoundRobinSampler)

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    # Batch 0 is audio-only
    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert len(b)
    assert all(isinstance(ex, Cut) for ex in b)
    # Batch tokens is not exceeded after applying the quadratic factor correction
    assert sum(ex.num_tokens**2 / QF for ex in b) <= BT
    for ex in b:
        assert ex.modality == "audio"
        assert isinstance(ex.load_audio(), np.ndarray)
        assert isinstance(ex.supervisions[0].text, str)

    # Batch 1 is text-only
    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert len(b)
    assert all(isinstance(ex, SourceTargetTextExample) for ex in b)
    # Batch tokens is not exceeded after applying the quadratic factor correction
    assert sum(ex.num_tokens**2 / QF for ex in b) <= BT
    for ex in b:
        assert ex.modality == "text"
        assert ex.source.language == "en"
        assert ex.target.language == "es"
        assert torch.is_tensor(ex.input_ids)
        assert torch.is_tensor(ex.context_ids)
        assert torch.is_tensor(ex.answer_ids)
        assert torch.is_tensor(ex.mask)


def test_dataloader_with_noise_nemo_json(cutset_path: Path, nemo_manifest_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "noise_path": str(nemo_manifest_path),
            "noise_mix_prob": 1.0,
            "noise_snr": [-5.0, 5.0],
            "batch_size": 2,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 2
    cut = batch[0]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0
    cut = batch[1]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0


def test_dataloader_with_noise_nemo_json(cutset_path: Path, nemo_manifest_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "noise_path": str(nemo_manifest_path),
            "noise_mix_prob": 1.0,
            "noise_snr": [-5.0, 5.0],
            "batch_size": 2,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 2
    cut = batch[0]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0
    cut = batch[1]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0


def test_dataloader_with_noise_lhotse_jsonl(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "noise_path": str(cutset_path),
            "noise_mix_prob": 1.0,
            "noise_snr": [-5.0, 5.0],
            "batch_size": 2,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 2
    cut = batch[0]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0
    cut = batch[1]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0


def test_dataloader_with_noise_nemo_tar(cutset_path: Path, nemo_tarred_manifest_path_multi: Path):
    noise_json, noise_tar = nemo_tarred_manifest_path_multi
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "noise_path": {
                "manifest_filepath": noise_json,
                "tarred_audio_filepaths": noise_tar,
            },
            "noise_mix_prob": 1.0,
            "noise_snr": [-5.0, 5.0],
            "batch_size": 2,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 2
    cut = batch[0]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0
    cut = batch[1]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0


def test_dataloader_with_synth_rir(cutset_path: Path):
    from lhotse.augmentation import ReverbWithImpulseResponse

    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "rir_enabled": True,
            "rir_prob": 0.5,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 4
    cut = batch[0]
    assert isinstance(cut, MonoCut)
    assert cut.recording.transforms is None
    cut = batch[1]
    assert isinstance(cut, MonoCut)
    assert cut.recording.transforms is None
    cut = batch[2]
    assert isinstance(cut, MonoCut)
    assert isinstance(cut.recording.transforms, list) and len(cut.recording.transforms) == 1
    tfnm = cut.recording.transforms[0]
    if isinstance(tfnm, dict):  # lhotse<=1.23.0
        assert tfnm["name"] == "ReverbWithImpulseResponse"
    else:  # lhotse>=1.24.0
        assert isinstance(tfnm, ReverbWithImpulseResponse)
    cut = batch[3]
    assert isinstance(cut, MonoCut)
    assert isinstance(cut.recording.transforms, list) and len(cut.recording.transforms) == 1
    tfnm = cut.recording.transforms[0]
    if isinstance(tfnm, dict):  # lhotse<=1.23.0
        assert tfnm["name"] == "ReverbWithImpulseResponse"
    else:  # lhotse>=1.24.0
        assert isinstance(tfnm, ReverbWithImpulseResponse)


def test_dataloader_bucket_batch_size(nemo_tarred_manifest_path_multi: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path_multi
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            # Note: all input cuts belong to the first bucket so the batch size will always be 2.
            "bucket_duration_bins": [2.0, 4.0],
            "bucket_batch_size": [2, 1],
            "drop_last": False,
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    for b in islice(dl, 10):
        assert len(b) == 2


def test_dataloader_2d_bucketing(nemo_tarred_manifest_path_multi: tuple[str, str], en_es_tokenizer):
    json_mft, tar_mft = nemo_tarred_manifest_path_multi
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            # Here each bin has the format: [audio_duration, token_sequence_length]
            "bucket_duration_bins": [[0.5, 1], [0.5, 2], [2.0, 5], [2.0, 15], [4.0, 10], [4.0, 20]],
            "bucket_batch_size": [7, 6, 5, 4, 3, 2],
            "drop_last": False,
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=Identity(), tokenizer=en_es_tokenizer
    )

    # All of our data have duration 1.0 and 10 tokens so they will fall to bin[3] with batch_size=4
    for b in islice(dl, 10):
        assert len(b) == 4


@pytest.fixture(scope="session")
def questions_path(tmp_path_factory) -> Path:
    """A text file with 10 lines containing question values"""
    qdir = tmp_path_factory.mktemp("questions")
    path = qdir / "questions.txt"
    path.write_text("\n".join(f"some question number {i}" for i in range(10)))
    return path


def test_dataloader_from_nemo_nontarred_manifest_with_extra_questions_field_iter(
    nemo_manifest_path: Path, questions_path: Path
):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "manifest_filepath": nemo_manifest_path,
                    "type": "nemo",
                    "extra_fields": [
                        {
                            "type": "text_iter",
                            "name": "question",
                            "path": questions_path,
                        }
                    ],
                },
            ],
            "sample_rate": 16000,
            "shuffle": False,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 2,
            "use_bucketing": False,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    b = next(iter(dl))
    c = b[0]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 0"
    c = b[1]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 1"


def test_dataloader_from_nemo_manifest_with_extra_questions_field_iter(
    nemo_tarred_manifest_path: tuple, questions_path: Path
):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "manifest_filepath": nemo_tarred_manifest_path[0],
                    "tarred_audio_filepaths": nemo_tarred_manifest_path[1],
                    "type": "nemo_tarred",
                    "extra_fields": [
                        {
                            "type": "text_iter",
                            "name": "question",
                            "path": questions_path,
                        }
                    ],
                },
            ],
            "sample_rate": 16000,
            "shuffle": False,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 2,
            "use_bucketing": False,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    b = next(iter(dl))
    c = b[0]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 0"
    c = b[1]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 1"


def test_dataloader_from_nemo_manifest_with_extra_questions_field_sample(
    nemo_tarred_manifest_path: tuple, questions_path: Path
):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "manifest_filepath": nemo_tarred_manifest_path[0],
                    "tarred_audio_filepaths": nemo_tarred_manifest_path[1],
                    "type": "nemo_tarred",
                    "extra_fields": [
                        {
                            "type": "text_sample",
                            "name": "question",
                            "path": questions_path,
                        }
                    ],
                },
            ],
            "sample_rate": 16000,
            "shuffle": False,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 5,
            "seed": 0,
            "shard_seed": 0,
            "use_bucketing": False,
        }
    )

    # Note: despite shuffle=True, it is sampling lines from questions_path because of type: "text_sample"
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    b = next(iter(dl))
    c = b[0]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 6"
    c = b[1]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 6"
    c = b[2]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 0"
    c = b[3]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 4"
    c = b[4]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 8"


@pytest.fixture(scope="session")
def nemo_tarred_manifest_path_with_offset(tmp_path_factory) -> Tuple[str, str]:
    """10 utterances of length 1s as a NeMo tarred manifest."""
    from lhotse.serialization import SequentialJsonlWriter
    from lhotse.shar.writers import TarWriter

    root = tmp_path_factory.mktemp("nemo_tar_offset")
    root.mkdir(exist_ok=True)
    recording = dummy_recording(0, duration=10.0, with_data=True)

    with (
        TarWriter(f"{root}/audios_0.tar", shard_size=None) as tar_writer,
        SequentialJsonlWriter(root / "tarred_audio_filepaths.jsonl") as mft_writer,
    ):

        def audio_path(n: int = None):
            return recording.id + ("" if n is None else f"-sub{n}") + ".wav"

        tar_writer.write(audio_path(), BytesIO(recording.sources[0].source))
        mft_writer.write(
            {  # segment 0-3s
                "audio_filepath": audio_path(),
                "offset": 0.0,
                "duration": 3.0,
                "text": "irrelevant",
                "lang": "en",
                "shard_id": 0,
            }
        )
        mft_writer.write(
            {  # segment 4-9s
                "audio_filepath": audio_path(1),
                "offset": 4.0,
                "duration": 5.0,
                "text": "irrelevant-2",
                "lang": "en",
                "shard_id": 0,
            }
        )
        mft_writer.write(
            {  # full recording - for reference
                "audio_filepath": audio_path(2),
                "offset": 0.0,
                "duration": 10.0,
                "text": "irrelevant irrelevant-2",
                "lang": "en",
                "shard_id": 0,
            }
        )
    return mft_writer.path, tar_writer.output_paths[0]


def test_dataloader_from_tarred_nemo_manifest_with_offset(nemo_tarred_manifest_path_with_offset: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path_with_offset
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": False,
            "num_workers": 0,
            "batch_size": 3,
            "seed": 0,
            "shard_seed": 0,
            "force_finite": True,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Loads all three examples in a single mini-batch (that's why batch_size=3).
    batches = [b for b in dl]
    assert len(batches) == 1
    (batch,) = batches
    assert len(batch) == 3

    # Validate example containing full 10s recording.
    full_cut = batch[1]
    assert full_cut.start == 0.0
    assert full_cut.duration == 10.0
    assert full_cut.supervisions[0].text == "irrelevant irrelevant-2"
    assert full_cut.supervisions[0].language == "en"
    full_audio = full_cut.load_audio()
    assert full_audio.shape[1] == full_cut.num_samples == 160000  # 10s * 16kHz

    # Validate segment 0-3s.
    cut = batch[2]
    assert cut.start == 0.0
    assert cut.duration == 3.0
    assert cut.supervisions[0].text == "irrelevant"
    assert cut.supervisions[0].language == "en"
    audio = cut.load_audio()
    assert audio.shape[1] == cut.num_samples
    # Check the audio for the segment is identical to a slice of the full audio.
    np.testing.assert_equal(audio, full_audio[:, : compute_num_samples(cut.duration, cut.sampling_rate)])

    # Validate segment 4-9s.
    # Note: LazyNeMoTarredIterator removes the offset information, as it creates a new recording
    # that's a "subset" of the original recording as a memory saving optimization.
    # Hence, we will not see cut.start == 4.0.
    cut = batch[0]
    assert cut.start == 0.0
    assert cut.duration == 5.0
    assert cut.supervisions[0].text == "irrelevant-2"
    assert cut.supervisions[0].language == "en"
    audio = cut.load_audio()
    assert audio.shape[1] == cut.num_samples
    # Check the audio for the segment is identical to a slice of the full audio.
    np.testing.assert_equal(
        audio, full_audio[:, compute_num_samples(4.0, cut.sampling_rate) : compute_num_samples(9.0, cut.sampling_rate)]
    )


def test_force_iterable_dataset(cutset_path: Path):
    config = OmegaConf.create({"cuts_path": cutset_path, "batch_size": 2, "num_workers": 2})
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    batches_map = [b for b in dl]

    config = OmegaConf.create(
        {"cuts_path": cutset_path, "batch_size": 2, "num_workers": 2, "force_iterable_dataset": True}
    )
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    batches_iter = [b for b in dl]

    # 2x duplicated data due to iterable dataset lack of deduplication
    assert len(batches_iter) == 2 * len(batches_map)
    # assertion that this is in fact the same data (same ids)
    assert set(c.id for b in batches_iter for c in b) == set(c.id for b in batches_map for c in b)


def test_force_map_dataset(cutset_shar_path: Path):
    config = OmegaConf.create({"shar_path": cutset_shar_path, "batch_size": 2, "num_workers": 2, "force_finite": True})
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    batches_iter = [b for b in dl]

    config = OmegaConf.create(
        {
            "shar_path": cutset_shar_path,
            "batch_size": 2,
            "num_workers": 2,
            "force_map_dataset": True,
            "force_finite": True,
        }
    )
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    batches_map = [b for b in dl]

    # 2x duplicated data due to iterable dataset lack of deduplication
    assert len(batches_iter) == 2 * len(batches_map)
    # assertion that this is in fact the same data (same ids)
    assert set(c.id for b in batches_iter for c in b) == set(c.id for b in batches_map for c in b)


def test_dataloader_from_tarred_nemo_subset_manifest(nemo_tarred_manifest_subset_path: tuple[str, str]):
    json_mft, tar_mft, subset_items = nemo_tarred_manifest_subset_path
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
            "tarred_random_access": True,
            "force_finite": True,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )
    seen_ids = list()
    for batch in dl:
        current_ids = batch["ids"]
        seen_ids += current_ids

    expected_ids = set([data['audio_filepath'] for data in subset_items])
    seen_ids_set = set(seen_ids)
    assert len(seen_ids_set) == len(seen_ids), "Duplicate IDs found in the batch."
    assert seen_ids_set == expected_ids, "The set of IDs in the batches does not match the input JSON manifests."


def test_dataloader_from_nemo_manifest_with_skipme(nemo_manifest_with_skipme_path: Path):
    config = OmegaConf.create(
        {
            "manifest_filepath": nemo_manifest_with_skipme_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 1,
            # lhotse specific
            "use_bucketing": False,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())
    batches = [batch for batch in dl]
    skipme_s = [cut.custom.get('_skipme', 0) for batch in batches for cut in batch]

    assert len(batches) == 8
    assert not any(skipme_s)


def test_dataloader_from_tarred_nemo_manifest_with_skipme(nemo_tarred_manifest_with_skipme_path: tuple[Path, str]):
    json_mft, tar_mft = nemo_tarred_manifest_with_skipme_path
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 1,
            # lhotse specific
            "use_bucketing": False,
            "force_finite": True,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())
    batches = [batch for batch in dl]
    skipme_s = [cut.custom.get('_skipme', 0) for batch in batches for cut in batch]

    assert len(batches) == 8
    assert not any(skipme_s)


def test_dataloader_from_data_input_cfg_yaml_path_with_skipme(cutset_shar_path, nemo_tarred_manifest_with_skipme_path):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "nemo_tarred",
                    "manifest_filepath": nemo_tarred_manifest_with_skipme_path[0],
                    "tarred_audio_filepaths": nemo_tarred_manifest_with_skipme_path[1],
                    "weight": 0.5,
                    "tags": {
                        "language": "en",
                        "modality": "audio",
                        "dataset_name": "D1",
                    },
                },
                {
                    "type": "lhotse_shar",
                    "shar_path": cutset_shar_path,
                    "weight": 0.5,
                    "tags": {
                        "language": "en",
                        "modality": "audio",
                        "dataset_name": "D2",
                    },
                },
            ],
            "sample_rate": 16000,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
            "force_finite": True,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    batches = [batch for batch in dl]
    skipme_s = [cut.custom.get('_skipme', 0) for batch in batches for cut in batch]

    assert not any(skipme_s)
