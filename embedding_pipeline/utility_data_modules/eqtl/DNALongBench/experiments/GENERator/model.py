import torch
import torch.nn as nn
from transformers import AutoModel

class RegulatorySignalPredictor(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        max_subsequence_length: int,
        num_subsequences: int,
        output_bins: int,
        output_tracks: int,
        gradient_checkpointing: bool = True
    ):
        super().__init__()
        # 1. backbone
        self.base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
        
        # 2. bookkeeping
        self.max_subsequence_length = max_subsequence_length
        self.num_subsequences    = num_subsequences
        self.output_bins         = output_bins
        self.output_tracks       = output_tracks
        
        # 3. pooling + small head (instead of one giant regression matrix)
        hidden_size = self.base_model.config.hidden_size
   
        # self.head = nn.Linear(hidden_size, self.output_tracks) 
        self.head = nn.Sequential(
            nn.Linear(hidden_size * num_subsequences, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_bins * output_tracks),
        )


    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        """
        input_ids:       [B, L]
        attention_mask:  [B, L]
        returns preds:   [B, output_bins, output_tracks]
        """
        # 1) run each chunk through the backbone and collect all token embeddings
        hidden_states = []
        for i in range(self.num_subsequences):
            start_idx = i * self.max_subsequence_length
            end_idx = (i + 1) * self.max_subsequence_length
            sub_input_ids = input_ids[:, start_idx:end_idx]
            sub_attention_mask = attention_mask[:, start_idx:end_idx]

            outputs = self.base_model(input_ids=sub_input_ids, attention_mask=sub_attention_mask)
            last_hidden_state = outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, -1, :]
            hidden_states.append(cls_embedding)
        
        # 2) stitch back into one long sequence → [B, total_L, H]
        combined_hidden_states = torch.cat(hidden_states, dim=-1) # [4, 16384]
        
        
        preds = self.head(combined_hidden_states )
        
        return {"logits": preds}

class LongSequenceRegressionModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        num_labels: int = 2,
        max_subsequence_length: int = 9375,
        num_subsequences: int = 8,
        gradient_checkpointing: bool = True
    ):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        hidden_size = self.base_model.config.hidden_size
        self.regression_head = nn.Linear(num_subsequences * hidden_size, num_labels, bias=False)

        self.max_subsequence_length = max_subsequence_length
        self.num_subsequences      = num_subsequences

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        input_ids:        [B, L]
        attention_mask:   [B, L] (optional — if None, we treat it as all 1’s)
        """
        # if no mask provided, assume everything is valid
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        hidden_states = []
        for i in range(self.num_subsequences):
            start_idx = i * self.max_subsequence_length
            end_idx   = (i + 1) * self.max_subsequence_length

            sub_input_ids      = input_ids[:, start_idx:end_idx]
            sub_attention_mask = attention_mask[:, start_idx:end_idx]

            outputs = self.base_model(
                input_ids=sub_input_ids,
                attention_mask=sub_attention_mask
            )
            # take the CLS token embedding
            cls_embedding = outputs.last_hidden_state[:, -1, :]
            hidden_states.append(cls_embedding)

        combined = torch.cat(hidden_states, dim=-1)   # [B, num_subseq * H]
        logits   = self.regression_head(combined)     # [B, num_labels]
        return {"logits": logits}

class LongSequenceClassificationModel(nn.Module):
    def __init__(self, base_model_name, num_labels=2, max_subsequence_length=9375, num_subsequences=8, gradient_checkpointing=True):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        self.classification_head = nn.Linear(num_subsequences * self.base_model.config.hidden_size, num_labels, bias=False)
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
        self.max_subsequence_length = max_subsequence_length
        self.num_subsequences = num_subsequences

    # def forward(self, input_ids, attention_mask, labels=None):
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        hidden_states = []

        for i in range(self.num_subsequences):
            start_idx = i * self.max_subsequence_length
            end_idx = (i + 1) * self.max_subsequence_length
            sub_input_ids = input_ids[:, start_idx:end_idx]
            sub_attention_mask = attention_mask[:, start_idx:end_idx]

            outputs = self.base_model(input_ids=sub_input_ids, attention_mask=sub_attention_mask)
            last_hidden_state = outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, -1, :]
            hidden_states.append(cls_embedding)

        combined_hidden_states = torch.cat(hidden_states, dim=-1)
        logits = self.classification_head(combined_hidden_states)

        return {"logits": logits}

class EqtlModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        num_labels: int = 2,
        max_subsequence_length: int = 9375,
        num_subsequences: int = 8,
        gradient_checkpointing: bool = True
    ):
        super().__init__()
        # shared encoder
        self.base_model = AutoModel.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        self.max_sub_len = max_subsequence_length
        self.num_subseqs = num_subsequences
        hidden_size = self.base_model.config.hidden_size * self.num_subseqs

        # [allele; ref; |allele–ref|] → logits
        self.classification_head = nn.Linear(3 * hidden_size, num_labels, bias=False)

    def _encode(self, input_ids: torch.LongTensor):
        """
        Break into chunks, encode each, grab final token embedding,
        concat along seq‐chunks.
        """
        seq_states = []
        for i in range(self.num_subseqs):
            start = i * self.max_sub_len
            end   = (i + 1) * self.max_sub_len

            chunk_ids = input_ids[:, start:end]
            # create a full‐ones mask so every token is attended
            chunk_mask = torch.ones_like(chunk_ids)

            out = self.base_model(input_ids=chunk_ids, attention_mask=chunk_mask)
            # final token as CLS proxy
            cls_emb = out.last_hidden_state[:, -1, :]  # [B, hidden]
            seq_states.append(cls_emb)

        return torch.cat(seq_states, dim=-1)  # [B, num_subseqs*hidden]

    def forward(
        self,
        x_alt: torch.LongTensor,   # your “allele” seqs
        x_ref: torch.LongTensor,   # your “reference” seqs
    ):
        emb_alt = self._encode(x_alt)
        emb_ref = self._encode(x_ref)

        delta = torch.abs(emb_alt - emb_ref)
        features = torch.cat([emb_alt, emb_ref, delta], dim=-1)  # [B, 3*H]
        logits = self.classification_head(features)
        return {"logits": logits}