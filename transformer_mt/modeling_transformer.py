#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 Vladislav Lialin and Namrata Shivagunde 
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
import os
import json
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_mt.modeling_attention import MultiHeadAttention
from transformer_mt.utils import pad


Hypothesis = namedtuple("Hypothesis", ["value", "score"])


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden, num_heads, fcn_hidden, dropout=0.0, causal=False):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            input_size=hidden,
            hidden=hidden,
            num_heads=num_heads,
            causal=causal,
        )
        self.att_layer_norm = nn.LayerNorm(hidden)

        self.fcn = nn.Sequential(
            nn.Linear(hidden, fcn_hidden),
            nn.ReLU(),
            nn.Linear(fcn_hidden, hidden),
        )
        self.fcn_layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        """Self-Attention -> residual -> LayerNorm -> FCN -> residual -> LayerNorm
        
        Args:
            x: FloatTensor[batch_size, seq_len, input_size]
        
        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """
        residual = x
        x = self.self_attention(x, key_padding_mask=key_padding_mask)
        x = self.att_layer_norm(x + residual)

        residual = x
        x = self.fcn(x)
        x = self.dropout(x)
        x = self.fcn_layer_norm(x + residual)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden, num_heads, fcn_hidden, dropout=0.0):
        super().__init__()

        # Task 2.1 (1 point)
        # Create layers needed for Transformer Decoder Layer
        # 1. Create self.self_attention layer using MultiHeadAttention
        # 2. Create self.cross_attention layer using MultiHeadAttention
        # 2a. Which one of self_attention or cross_attention should have causal=True? Set it there.
        # 3. Create self.att_layer_norm, self.cross_att_layer_norm, and self.fcn_layer_norm layers using LayerNorm
        # 4. Create self.fcn network using nn.Sequential, nn.ReLU and nn.Linear
        # 5. Create self.dropout layer using nn.Dropout
        # YOUR CODE STARTS HERE  (our implementation is about 5-8 lines) 

        self.self_attention = MultiHeadAttention(input_size=hidden, hidden=hidden, num_heads=num_heads)
        self.cross_attention = MultiHeadAttention(input_size=hidden, hidden=hidden, num_heads=num_heads, causal=True)

        self.self_att_layer_norm = nn.LayerNorm(hidden)
        self.cross_att_layer_norm = nn.LayerNorm(hidden)

        self.fcn = nn.Sequential(
          nn.Linear(hidden, fcn_hidden),
          nn.ReLU(),
          nn.Linear(fcn_hidden, hidden),
          nn.ReLU()
        )
        self.fcn_layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # YOUR CODE ENDS HERE 
    
    def forward(self, decoder_hidden_states, encoder_hidden_states, key_padding_mask=None):
        """Transformer Decoder Layer

        Args:
            decoder_hidden_states: FloatTensor[batch_size, query_seq_len, hidden]
            encoder_hidden_states: FloatTensor[batch_size, kv_seq_len, hidden]
            key_padding_mask: ByteTensor[batch_size, kv_seq_len] with 1 for padded tokens and 0 for regular tokens

        Returns:
            FloatTensor[batch_size, query_seq_len, hidden]
        """

        # Task 2.2 (1 point)
        # Implement Transformer decoder block
        # Remember that transformer decoder block is composed of:
        # 1. Self-Attention
        # 2. Residual connection
        # 3. LayerNorm
        # 4. Cross-Attention
        # 5. Residual connection
        # 6. LayerNorm
        # 7. Fully-Connected Layer
        # 8. Dropout
        # 9. Residual connection
        # 10. LayerNorm
        # Note : Please write shape of the tensor for each line of code
        # YOUR CODE STARTS HERE (our implementation is about 10 lines)

        h = self.self_attention(decoder_hidden_states, encoder_hidden_states, key_padding_mask) #shape [batch_size, target_seq_len, hidden]
        h = h + decoder_hidden_states #shape [batch_size, target_seq_len, hidden]
        h = self.self_att_layer_norm(h) #shape [batch_size, target_seq_len, hidden]
        h2 = self.cross_attention(h, encoder_hidden_states, key_padding_mask) #shape [batch_size, target_seq_len, hidden]
        h2 = h2 + h #shape [batch_size, target_seq_len, hidden]
        h2 = self.cross_att_layer_norm(h2) #shape [batch_size, target_seq_len, hidden]
        h3 = self.fcn(h2) #shape [batch_size, target_seq_len, hidden]
        h3 = self.dropout(h3) #shape [batch_size, target_seq_len, hidden]
        h3 = h3 + h2 #shape [batch_size, target_seq_len, hidden]
        x = self.fcn_layer_norm(h3) #shape [batch_size, target_seq_len, hidden]
        
        ##YOUR CODE ENDS HERE##
        return x


class TransfomerEncoderDecoderModel(nn.Module):
    def __init__(
        self,
        *,
        num_layers,
        hidden,
        num_heads,
        fcn_hidden,
        max_seq_len,
        src_vocab_size,
        tgt_vocab_size,
        dropout=0.1,
    ):
        """A minimal implementation of Transformer Encoder Decoder Model
        
        Args:
            num_layer: number of layers for encoder and decoder (in total, model will have 2 * num_layers layers)
            hidden : embedding size and hidden size of attentions
            fcn_hidden: hidden size of fully-connected networks inside transformer layers
            vocab_size: size of vocabulary
            max_seq_len: maximum length of input, target sequence whichever is higher number
            src_vocab_size : source voacb size
            tgt_vocab_size : target voab size
        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_layers = num_layers
        self.hidden = hidden
        self.num_heads = num_heads
        self.fcn_hidden = fcn_hidden
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len

        # Task 2.3 (1 point)
        # 1. Create encoder, decoder and positional embedding layer
        # Use nn.Embedding for that and make sure to include source and target vocabulary size
        # 2. Create a linear layer out_proj that will project contextualized representations
        # of size hidden to your target vocabulary size.
        # 3. Create a dropout layer
        # YOUR CODE STARTS HERE (our implementation is about 5 lines)

        self.encoder_embeddings = nn.Embedding(self.src_vocab_size, self.hidden)
        self.decoder_embeddings = nn.Embedding(self.tgt_vocab_size, self.hidden)
        self.positional_emb = nn.Embedding(self.max_seq_len, self.hidden)

        self.out_proj = nn.Linear(self.hidden, self.tgt_vocab_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        # YOUR CODE ENDS HERE

        # Task 2.4 (1 point)
        # 1. Create a list of encoder Layers
        # 2. Create a list of decoder Layers
        #
        # Note that you need to wrap it with nn.ModuleList,
        # so that the parameters of the layers would be counted as the paramertes of the model
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        # Read more about ModuleList here:
        # https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict
        # You can use for-loop of python list comprehension to create the list of layers
        #
        # YOUR CODE STARTS HERE (our implementation is 3-6 lines)
        
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(
                self.hidden, 
                self.num_heads, 
                self.fcn_hidden, 
                self.dropout_rate) for i in range(self.num_layers)])

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(
                self.hidden, 
                self.num_heads, 
                self.fcn_hidden, 
                self.dropout_rate) for i in range(self.num_layers)])
        
        # YOUR CODE ENDS HERE

    def _add_positions(self, sequence_tensor):
        """Adds positional embeddings to the input tensor.
        Args:
            sequence_tensor: FloatTensor[batch_size, seq_len, hidden]
        """
        seq_len = sequence_tensor.shape[1]
        positions = torch.arange(seq_len, device=sequence_tensor.device)
        positional_emb = self.positional_emb(positions)
        output = sequence_tensor + positional_emb
        return output

    def forward(
        self,
        input_ids=None,
        encoder_hidden_states=None,
        decoder_input_ids=None,
        key_padding_mask=None,
    ):
        """
        input_ids -> encoder_emb -> encoder -> 
                                                -->  decoder(encoder_output, decoder_emb) -> logits
        decoder_input_ids -> decoder_emb ---->

        Model accepts either input_ids or encoder_hidden_states.
        The former is used for training, the latter is used for inference, because during inference
        we don't have the target sequence and want to forward the decoder multiple times.
        To make the inference more efficient, we can only compute encoder output once and reuse it
        for all decoder steps.

        Meaning during training you should forward the model like this:
            model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        but during inference (generating translation) you should forward the model like this:
            model(encoder_hidden_states=encoder_hidden_states, decoder_input_ids=decoder_input_ids)

        Args:
            input_ids (LongTensor): Encoder input sequence of size (batch_size, seq_len)
            encoder_hidden_states (FloatTensor): Encoder hidden states of size (batch_size, seq_len, hidden)
            decoder_input_ids (LongTensor) : Decoder input sequence of size (batch_size, out_seq_len)
            key_padding_mask (ByteTensor): Mask of size (batch_size, seq_len) where 1 means that the token is padding

        Return:
            logits (FloatTensor): Logits for output sequence of size (batch_size, out_seq_len, dec_vocab_size)

        """
        if input_ids is None and encoder_hidden_states is None:
            raise ValueError("You should provide either input_ids or encoder_hidden_states")

        if encoder_hidden_states is None:
            encoder_hidden_states = self._encode(input_ids, key_padding_mask)

        logits = self._decode(encoder_hidden_states, decoder_input_ids, key_padding_mask)

        return logits

    def _encode(self, input_ids, key_padding_mask):
        # Task 2.5 (2 points)
        # 1. Get source embeddings using self.encoder_embeddings
        # 2. Add positional embedding to encoder embeddings using _add_positions
        # 3. Pass source embeddings through the encoder layers, name them encoder_hidden_states
        # 3a. Remember to use key_padding_mask to mask out padding tokens
        # YOUR CODE STARTS HERE

        encoder_hidden_states = self.encoder_embeddings(input_ids)
        encoder_hidden_states = self._add_positions(encoder_hidden_states)
        for i in range(len(self.encoder_layers)):
            encoder_hidden_states = self.encoder_layers[i](
                x=encoder_hidden_states,
                key_padding_mask=key_padding_mask)

        # YOUR CODE ENDS HERE

        return encoder_hidden_states
    
    def _decode(self, encoder_hidden_states, decoder_input_ids, key_padding_mask):
        # TASK 2.6 (2 points)
        # 1. Get decoder embeddings using self.decoder_embeddings
        # 2. Add positional embedding to target embeddings using _add_positions
        # 3.Use decoder embeddings and encoder_hidden_states for the decoder input
        # (please use keyword arguments instead of positional arguments to minimize a chance of a bug)
        # 3a. Remember to use key_padding_mask to mask out padding tokens for the encoder inputs
        # 4. use self.out_proj to get output logits, a.k.a log-probabilies of the next translation tokens
        # YOUR CODE STARTS HERE

        decoder_embeddings = self.decoder_embeddings(decoder_input_ids)
        decoder_embeddings = self._add_positions(decoder_embeddings)
        for i in range(len(self.decoder_layers)):
            decoder_embeddings = self.decoder_layers[i](
                decoder_hidden_states=decoder_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                key_padding_mask=key_padding_mask)
        logits = self.out_proj(decoder_embeddings)

        ## YOUR CODE ENDS HERE
        return logits

    ##############################################################################
    # Don't worry about any of the code below this line, but feel free to take a look
    # if you are interested in generation or model saving/loading.
    ##############################################################################
    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        *,
        bos_token_id,
        eos_token_id,
        pad_token_id=None,
        key_padding_mask=None,
        max_length=50,
        beam_size=5,
        kind="beam_search",
    ):
        """
        Generate a translation given an input sequence.

        Args:
            input_ids (LongTensor): Encoder input sequence of size (batch_size, seq_len)
            bos_token_id (int): Beginning of sentence token id
            eos_token_id (int): End of sentence token id
            pad_token_id (int): Padding token id, required if doing beam search
            key_padding_mask (ByteTensor): Mask of size (batch_size, seq_len) where 1 means that the token is padding
            max_length (int): Maximum length of the generated sequence
            beam_size (int): Beam size for beam search
            kind (str): Can be either "greedy" or "beam_search"

        Return:
            decoded_ids (LongTensor): Decoder output sequence of size (batch_size, seq_len)
        """
        if kind not in ["greedy", "beam_search"]:
            raise ValueError("Unknown kind of generation: {}".format(kind))
        if kind == "beam_search" and pad_token_id is None:
            raise ValueError("Beam search requires a pad_token_id to be provided")

        if kind == "greedy":
            return self._generate_greedy(
                input_ids=input_ids,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                key_padding_mask=key_padding_mask,
                max_length=max_length,
            )
        
        # beam search only supports batch size 1
        beam_search_generations = []
        for i in range(input_ids.size(0)):
            _input_ids = input_ids[i].unsqueeze(0)
            _key_padding_mask = key_padding_mask[i].unsqueeze(0) if key_padding_mask is not None else None

            generated = self._generate_beam_search(
                input_ids=_input_ids,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                key_padding_mask=_key_padding_mask,
                max_length=max_length,
                beam_size=beam_size,
            )

            beam_search_generations.append(generated[0].detach().cpu().tolist())
        
        return pad(beam_search_generations, pad_id=eos_token_id)

    @torch.inference_mode()
    def _generate_greedy(
        self,
        input_ids,
        *,
        bos_token_id,
        eos_token_id,
        key_padding_mask=None,
        max_length=50,
    ):
        """
        Greedy generation of translation. Selects most likely word on every step.

        Args:
            input_ids (LongTensor): Encoder input sequence of size (batch_size, seq_len)
            max_length (int): Maximum length of the generated sequence
            bos_token_id (int): Beginning of sentence token id
            eos_token_id (int): End of sequence token id

        Return:
            translation (LongTensor): Decoder output sequence of size (batch_size, out_seq_len)
                where out_seq_len <= max_length
        """
        encoder_hidden_states = self._encode(input_ids, key_padding_mask)

        decoder_input_ids = torch.full((input_ids.shape[0], 1), bos_token_id, dtype=torch.long, device=input_ids.device)
        translation = torch.zeros((input_ids.shape[0], 0), dtype=torch.long, device=input_ids.device)

        eos_flags = torch.zeros((input_ids.shape[0],), dtype=torch.uint8, device=input_ids.device)

        for _ in range(max_length):
            logits = self._decode(encoder_hidden_states, decoder_input_ids, key_padding_mask)
            logits = logits[:, -1, :]

            next_token_id = torch.argmax(logits, dim=-1)

            decoder_input_ids = torch.cat((decoder_input_ids, next_token_id.unsqueeze(1)), dim=1)
            translation = torch.cat((translation, next_token_id.unsqueeze(1)), dim=1)

            eos_flags |= (next_token_id == eos_token_id)

            if eos_flags.all():
                break

        return translation

    @torch.inference_mode()
    def _generate_beam_search(
        self,
        input_ids,
        *,
        bos_token_id,
        eos_token_id,
        key_padding_mask=None,
        beam_size=5,
        max_length=50,
    ):
        """
        Beam search generation of translation.
        Heavily inspired by https://github.com/pcyin/pytorch_basic_nmt

        Args:
            input_ids (LongTensor): Encoder input sequence of size (batch_size, seq_len)
            max_length (int): Maximum length of the generated sequence
            bos_token_id (int): Beginning of sentence token id
            eos_token_id (int): End of sequence token id

        Return:
            translation (LongTensor): Decoder output sequence of size (batch_size, out_seq_len)
                where out_seq_len <= max_length
        """
        assert len(input_ids) == 1, "Beam search is only supported for a single input sequence"
        encoder_hidden_states = self._encode(input_ids, key_padding_mask)
        device = input_ids.device

        hypotheses = [[bos_token_id]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device)
        completed_hypotheses = []

        for _ in range(max_length):
            if len(completed_hypotheses) >= beam_size:
                break

            hyp_num = len(hypotheses)
            expanded_encoder_hidden_states = encoder_hidden_states.expand(
                hyp_num,
                encoder_hidden_states.size(1),
                encoder_hidden_states.size(2),
            )

            # [batch_size*hyp_num=1*hyp_num, seq_len, hidden]
            hypotheses_tensor = torch.tensor(hypotheses, dtype=torch.int64, device=device)
            logits = self._decode(expanded_encoder_hidden_states, hypotheses_tensor, key_padding_mask)
            logits = logits[:, -1, :]  # [vocab_size]

            log_p_t = F.log_softmax(logits, dim=-1)
            live_hyp_num = beam_size - len(completed_hypotheses)

            # [hyp_num] -> [1, hyp_num] -> [hyp_num, vocab_size] -> [hyp_num * vocab_size]
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            # [live_hyp_num], [live_hyp_num]
            # for indices, the values range from 0 to hyp_num * vocab_size
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)

            # hypotheses ids in hyp_scores tensor [hyp_num,]
            prev_hyp_ids = torch.div(top_new_hyp_pos, self.tgt_vocab_size, rounding_mode='floor')

            # ids of the next words for each hypothesis
            token_ids = top_new_hyp_pos % self.tgt_vocab_size

            new_hypotheses = []
            new_hyp_scores = []

            # iterate live_hyp_num times
            for prev_hyp_id, hyp_token_id, cand_new_hyp_score in zip(prev_hyp_ids, token_ids, top_new_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_token_id = hyp_token_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_token_id]
                if hyp_token_id == eos_token_id:
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1], score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:], score=hyp_scores[0].item()))
        
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return torch.LongTensor(completed_hypotheses[0].value).unsqueeze(0)

    def save_pretrained(self, save_path):
        """Save the model weights to a directory

        Args:
            save_path: directory to save the model
        """
        config = {
            "num_layers": self.num_layers,
            "hidden": self.hidden,
            "num_heads": self.num_heads,
            "fcn_hidden": self.fcn_hidden,
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout_rate,
        }

        with open(os.path.join(save_path, "model_config.json"), "w") as f:
           json.dump(config, f)

        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_path, "model.pt"))
    
    @classmethod
    def from_pretrained(cls, save_path, map_location=None):
        """Load the model weights from a directory

        Args:
            save_path: directory to load the model
        """
        if map_location is None and not torch.cuda.is_available():
            map_location = "cpu"

        with open(os.path.join(save_path, "model_config.json"), "r") as f:
            config = json.load(f)

        model = cls(**config)
        state_dict = torch.load(os.path.join(save_path, "model.pt"), map_location=map_location)
        model.load_state_dict(state_dict)
        return model
