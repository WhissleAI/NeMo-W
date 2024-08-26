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

from typing import List, Optional, Tuple, Union

import editdistance
import jiwer
import torch
from torchmetrics import Metric

from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.utils import logging

import regex as re

__all__ = ['AV_WER']

def move_dimension_to_the_front(tensor, dim_index):
    all_dims = list(range(tensor.ndim))
    return tensor.permute(*([dim_index] + all_dims[:dim_index] + all_dims[dim_index + 1 :]))


class AV_WER(Metric):
    """
    This metric computes numerator and denominator for Overall Word Error Rate (WER) between prediction and reference
    texts. When doing distributed training/evaluation the result of ``res=WER(predictions, predictions_lengths, targets, target_lengths)``
    calls will be all-reduced between all workers using SUM operations. Here ``res`` contains three numbers
    ``res=[wer, total_levenstein_distance, total_number_of_words]``.
    
    This also has options to compute WER with tags, without tags and accuracy for tag prediction too.
    TODO @Balu: Can also integrate spans of the tag predicted.

    If used with PytorchLightning LightningModule, include wer_numerator and wer_denominators inside validation_step
    results. Then aggregate (sum) then at the end of validation epoch to correctly compute validation WER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            wer_num, wer_denom = self.__wer(predictions, predictions_len, transcript, transcript_len)
            self.val_outputs = {'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom}
            return self.val_outputs

        def on_validation_epoch_end(self):
            ...
            wer_num = torch.stack([x['val_wer_num'] for x in self.val_outputs]).sum()
            wer_denom = torch.stack([x['val_wer_denom'] for x in self.val_outputs]).sum()
            tensorboard_logs = {'validation_loss': val_loss_mean, 'validation_avg_wer': wer_num / wer_denom}
            self.val_outputs.clear()  # free memory
            return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    Args:
        decoding: An instance of CTCDecoding or RNNTDecoding.
        use_cer: Whether to use Character Error Rate instead of Word Error Rate.
        log_prediction: Whether to log a single decoded sample per call.
        batch_dim_index: Index corresponding to batch dimension. (For RNNT.)
        dist_dync_on_step: Whether to perform reduction on forward pass of metric.
        labelled_manifest: Whether the manifest has labels or not.

    Returns:
        res: a tuple of 3 zero dimensional float32 ``torch.Tensor` objects: a WER score, a sum of Levenstein's
            distances for all prediction - reference pairs, total number of words in all references.
    """

    full_state_update: bool = True

    def __init__(
        self,
        decoding: Union[AbstractCTCDecoding, AbstractRNNTDecoding, AbstractMultiTaskDecoding],
        use_cer=False, 
        log_prediction=True,
        fold_consecutive=True,
        batch_dim_index=0,
        dist_sync_on_step=False,
        labelled_manifest=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.decoding = decoding
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive
        self.batch_dim_index = batch_dim_index

        self.has_spl_tokens = False
        self.decode = None
        if isinstance(self.decoding, AbstractRNNTDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, targets: self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=predictions, encoded_lengths=predictions_lengths
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, targets: self.decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=predictions,
                decoder_lengths=predictions_lengths,
                fold_consecutive=self.fold_consecutive,
            )
        elif isinstance(self.decoding, AbstractMultiTaskDecoding):
            self.has_spl_tokens = True
            self.decode = lambda predictions, prediction_lengths, predictions_mask, input_ids, targets: self.decoding.decode_predictions_tensor(
                encoder_hidden_states=predictions,
                encoder_input_mask=predictions_mask,
                decoder_input_ids=input_ids,
                return_hypotheses=False,
            )
        else:
            raise TypeError(f"WER metric does not support decoding of type {type(self.decoding)}")

        self.add_state("scores_labelled", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("words_labelled", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("scores_unlabelled", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("words_unlabelled", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("correct_label_count", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        
        self.labelled_manifest = labelled_manifest
        
    def get_words_and_scores(self, hypotheses: List[str], references: List[str], labelled_data: str):
        words = 0
        scores = 0
        
        for h, r in zip(hypotheses, references):
            if self.use_cer:
                h_list = list(h)
                r_list = list(r)
            else:
                h_list = h.split()
                r_list = r.split()
            words += len(r_list)
            # Compute Levenstein's distance
            scores += editdistance.eval(h_list, r_list)
            
        if labelled_data:
            self.scores_labelled = torch.tensor(scores, device=self.scores_labelled.device, dtype=self.scores_labelled.dtype)
            self.words_labelled = torch.tensor(words, device=self.words_labelled.device, dtype=self.words_labelled.dtype)
        else:
            self.scores_unlabelled = torch.tensor(scores, device=self.scores_unlabelled.device, dtype=self.scores_unlabelled.dtype)
            self.words_unlabelled = torch.tensor(words, device=self.words_unlabelled.device, dtype=self.words_unlabelled.dtype)
    
    def seperate_labels_from_labelled_data(self, hypotheses: List[str], references: List[str]):
        # labels are in the text of form <N1>...text...<N2>, note it is not only <N1> and <N2> but can be any number of tag marked by <>
        unlabelled_hypotheses = []
        unlabelled_references = []
        labels_hypotheses = []
        labels_references = []
        correct_label_count = 0
        
        for h, r in zip(hypotheses, references):
            # identify the tags with <> 
            # h_tags = [h[i:j+1] for i in range(len(h)) for j in range(i, len(h)) if h[i] == '<' and h[j] == '>']
            # r_tags = [r[i:j+1] for i in range(len(r)) for j in range(i, len(r)) if r[i] == '<' and r[j] == '>']
            r_tags = re.findall(r'<N\d+>', r)
            h_tags = re.findall(r'<N\d+>', h)
            # assert len(r_tags) == 2, f"Reference tags are not 2, they are {r_tags} for {r}" # Note we are only considering for single label.
            # Above assert doesnt apply when ps audio is of 15 seconds but words are in only first 4 seconds and noise occurs from 8 to 10 secs.
            # Replace all tags in the hypothesis and reference
            unlabelled_h = h
            unlabelled_r = r
            for tag in r_tags:
                unlabelled_h = unlabelled_h.replace(tag, '')
                unlabelled_r = unlabelled_r.replace(tag, '')
            
            unlabelled_hypotheses.append(unlabelled_h)
            unlabelled_references.append(unlabelled_r)
            labels_hypotheses.append(h_tags)
            # FOR IT1
            # if len(r_tags) == 2: 
            #     labels_references.append(r_tags[0])
            # else:
            #     labels_references.append([])
            # if len(h_tags) == 2 and len(r_tags) == 2 and h_tags[0] == r_tags[0]:
            #     correct_label_count += 1
            
            # FOR IT2
            if len(r_tags) == 1:
                labels_references.append(r_tags[0])
            else:
                labels_references.append([])
            if len(h_tags) == 1 and len(r_tags) == 1 and h_tags[0] == r_tags[0]:
                correct_label_count += 1
            
        return unlabelled_hypotheses, unlabelled_references, labels_hypotheses, labels_references, correct_label_count
            
    
    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        predictions_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        """
        Updates metric state.
        Args:
            predictions: an integer torch.Tensor of shape ``[Batch, Time, {Vocabulary}]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            prediction_lengths: an integer torch.Tensor of shape ``[Batch]``
            targets: an integer torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            target_lengths: an integer torch.
            predictions_lengths: an integer torch.Tensor of shape ``[Batch]``
        """
        references = []
        with torch.no_grad():
            tgt_lenths_cpu_tensor = targets_lengths.long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            # check batch_dim_index is first dim
            if self.batch_dim_index != 0:
                targets_cpu_tensor = move_dimension_to_the_front(targets_cpu_tensor, self.batch_dim_index)
            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[0]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decoding.decode_tokens_to_str(target)
                references.append(reference)
            hypotheses, _ = self.decode(predictions, predictions_lengths, predictions_mask, input_ids, targets)

            if self.has_spl_tokens:
                hypotheses = [self.decoding.strip_special_tokens(hyp) for hyp in hypotheses]
                references = [self.decoding.strip_special_tokens(ref) for ref in references]

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference:{references[0]}")
            logging.info(f"predicted:{hypotheses[0]}")
            logging.info(f"\n")

        unlabelled_hypotheses, unlabelled_references, labels_hypotheses, labels_references, correct_label_count = self.seperate_labels_from_labelled_data(hypotheses, references)
        self.get_words_and_scores(unlabelled_hypotheses, unlabelled_references, labelled_data=False)
        self.get_words_and_scores(hypotheses, references, labelled_data=True)
        self.correct_label_count = torch.tensor(correct_label_count, device=self.correct_label_count.device, dtype=self.correct_label_count.dtype)
        self.num_samples = torch.tensor(len(references), device=self.num_samples.device, dtype=self.num_samples.dtype)
        

    def compute(self):
        if self.labelled_manifest:
            scores_labelled = self.scores_labelled.detach().float()
            words_labelled = self.words_labelled.detach().float()
            labelled_wer = scores_labelled / words_labelled
        else:
            scores_labelled = None
            words_labelled = None
            labelled_wer = None
        scores_unlabelled = self.scores_unlabelled.detach().float()
        words_unlabelled = self.words_unlabelled.detach().float()
        unlabelled_wer = scores_unlabelled / words_unlabelled
        correct_label_count = self.correct_label_count.detach().float()
        num_samples = self.num_samples.detach().float()

        return labelled_wer, unlabelled_wer, correct_label_count/num_samples, scores_unlabelled, words_unlabelled
