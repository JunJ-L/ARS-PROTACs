"""
ARS-PROTACs: an asymmetric multimodal network for binary PROTAC degradation prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from se3_transformer_pytorch import SE3Transformer

class ESMWrapper(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.down_proj = nn.Identity()

    def forward(self, x):
        x = self.down_proj(x)
        return x


class GraphTransformer(nn.Module):
    """SE(3) Transformer block to process 3D graph data."""
    def __init__(self, num_embeddings, dim=128,
                 depth=1, heads=9, dim_head=8, num_degrees=1):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, dim)
        self.transformer = SE3Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            num_degrees=num_degrees
        )

    def forward(self, data):
        feats, coors, batch = data.x, data.pos, data.batch
        feats = self.embed(feats.squeeze(-1))
        dense_feats, node_mask = to_dense_batch(feats, batch)
        dense_coors, _ = to_dense_batch(coors, batch)
        transformed_feats_dict = self.transformer(dense_feats, dense_coors, mask=node_mask)
        if isinstance(transformed_feats_dict, dict):
            scalar_feats = transformed_feats_dict['0']
        else:
            scalar_feats = transformed_feats_dict
        scalar_feats = scalar_feats.masked_fill(~node_mask[..., None], 0.)
        return scalar_feats


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_hiddens = num_hiddens
        self.register_buffer("P", self._build_encoding(num_hiddens, max_len), persistent=False)

    @staticmethod
    def _build_encoding(num_hiddens, max_len):
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_hiddens, 2, dtype=torch.float32) * (-math.log(10000.0) / num_hiddens))
        encoding = torch.zeros(1, max_len, num_hiddens)
        encoding[:, :, 0::2] = torch.sin(positions * div_term)
        encoding[:, :, 1::2] = torch.cos(positions * div_term)
        return encoding

    def forward(self, x):
        if x.size(1) > self.P.size(1):
            self.P = self._build_encoding(self.num_hiddens, x.size(1)).to(x.device)
        x = x + self.P[:, :x.size(1)].to(x.device)
        return self.dropout(x)


class E3MoKA(nn.Module):
    """
    MoKA (Molecule-Conditioned Kernel Adaptation) extracts local motifs from short
    E3-ligase sequences with multiscale CNNs and adapts dynamic convolution kernels
    using molecular context.
    Stage 1: encode local protein-sequence motifs with multiscale convolutions.
    Stage 2: integrate molecular context through cross-attention and dynamic kernels.
    """
    def __init__(self, vocab_size, embed_dim=128, proj_dim=128, kernel_size=3, drop_out=0.2):
        super().__init__()
        self.dim = proj_dim
        self.kernel_size = kernel_size

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, drop_out)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, proj_dim, kernel_size=2, padding=1),
            nn.Conv1d(embed_dim, proj_dim, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, proj_dim, kernel_size=4, padding=2),
            nn.Conv1d(embed_dim, proj_dim, kernel_size=5, padding=2),
        ])
        self.out_proj = nn.Linear(proj_dim * len(self.convs), proj_dim)
        self.cnn_dropout = nn.Dropout(drop_out)
        self.relu = nn.ReLU()

        self.sequence_proj = nn.Linear(proj_dim, proj_dim)
        self.compound_key = nn.Linear(proj_dim, proj_dim)
        self.compound_value = nn.Linear(proj_dim, proj_dim)
        self.kernel_generator = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim * kernel_size),
        )
        self.fusion = nn.Linear(proj_dim * 3, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.moka_dropout = nn.Dropout(drop_out)

    def forward(self, tokens, compound_feats, mask=None):
        x = self.embed(tokens) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_encoding(x)
        x = x.transpose(1, 2)
        seq_len = tokens.size(1)

        feats = []
        for conv in self.convs:
            feat = self.relu(conv(x))
            if feat.size(-1) > seq_len:
                feat = feat[..., :seq_len]
            elif feat.size(-1) < seq_len:
                feat = F.pad(feat, (0, seq_len - feat.size(-1)))
            feats.append(feat)
        x = torch.cat(feats, dim=1).transpose(1, 2)
        sequence_feats = self.cnn_dropout(self.out_proj(x))

        if mask is not None:
            sequence_feats = sequence_feats * mask

        query = self.sequence_proj(sequence_feats)
        key = self.compound_key(compound_feats)
        value = self.compound_value(compound_feats)

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        compound_context = torch.matmul(attn_weights, value)

        compound_summary = compound_feats.mean(dim=1)
        dynamic_kernel = self.kernel_generator(compound_summary).view(-1, self.dim, self.kernel_size)
        dynamic_kernel = torch.softmax(dynamic_kernel, dim=-1)

        seq_channels = sequence_feats.transpose(1, 2)
        pad = self.kernel_size // 2
        seq_windows = F.pad(seq_channels, (pad, pad), mode="replicate").unfold(2, self.kernel_size, 1)
        local_response = torch.einsum("bdlk,bdk->bdl", seq_windows, dynamic_kernel).transpose(1, 2)

        fused = torch.cat([sequence_feats, compound_context, local_response], dim=-1)
        fused = self.moka_dropout(self.fusion(fused))

        out = self.norm(sequence_feats + fused)  # [B, N_l, dim]
        if mask is not None:
            out = out * mask
        return out


class DrugConditionedTargetFocus(nn.Module):
    """
    Use warhead features to focus and denoise target-protein residue representations.
    """
    def __init__(self, dim=128):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(dim))

    def forward(self, target_feats, warhead_feats, mask=None):
        warhead_summary = warhead_feats.mean(dim=1)
        query = self.query_proj(warhead_summary)
        keys = self.key_proj(target_feats)

        scores = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1)
        scores = scores / self.temperature

        focus_weights = torch.sigmoid(scores).unsqueeze(-1)

        if mask is not None:
            focus_weights = focus_weights * mask

        focused_target = target_feats * (0.5 + focus_weights)
        return focused_target


class RolePairwiseInteraction(nn.Module):
    """
    Role-Specific Pairwise Interaction (RPI)
    Couple the target and E3 streams through additive pairwise cross-attention,
    then modulate their attention-based reweighting with cooperativity gains.

    This operation represents cooperative recognition signals between the two
    proteins when bridged by a PROTAC in a ternary complex.
    """
    def __init__(self, dim=128, gain_scale=0.1):
        super().__init__()
        self.attention_layer = nn.Linear(dim, dim)
        self.target_attention_layer = nn.Linear(dim, dim)
        self.ligase_attention_layer = nn.Linear(dim, dim)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.gain_scale = gain_scale
        self.alpha_t = nn.Parameter(torch.zeros(1))
        self.alpha_l = nn.Parameter(torch.zeros(1))

    def forward(self, target_stream, e3_stream):
        """
        Args:
            target_stream: [B, N_t+N_1+N_2, dim], target-side sequence (ESM target + warhead + linker)
            e3_stream: [B, N_l+N_0+N_2, dim], E3-side sequence (ESM E3 + E3 ligand + linker)
        Returns:
            pooled_target_3d: [B, dim], reweighted and pooled target representation
            pooled_e3_3d: [B, dim], reweighted and pooled E3 representation
            e3_attn_weights: [B, N_l+N_0+N_2, dim], E3 attention weights used by AGF
        """
        gain_t = 1.0 + self.gain_scale * torch.tanh(self.alpha_t)
        gain_l = 1.0 + self.gain_scale * torch.tanh(self.alpha_l)

        target_proj = self.target_attention_layer(target_stream)  # [B, N_t+N_1+N_2, dim]
        e3_proj = self.ligase_attention_layer(e3_stream)  # [B, N_l+N_0+N_2, dim]

        combined_proj = target_proj.unsqueeze(2) + e3_proj.unsqueeze(1)  # [B, N_t+N_1+N_2, N_l+N_0+N_2, dim]
        pairwise_attn = self.attention_layer(self.relu(combined_proj))

        target_attn_weights = self.sigmoid(pairwise_attn.mean(2))  # [B, N_t+N_1+N_2, dim]
        e3_attn_weights = self.sigmoid(pairwise_attn.mean(1))  # [B, N_l+N_0+N_2, dim]

        target_stream = target_stream * (0.5 + gain_t * target_attn_weights)
        e3_stream = e3_stream * (0.5 + gain_l * e3_attn_weights)

        pooled_target_3d = torch.sum(target_stream, 1)  # [B, dim]
        pooled_e3_3d = torch.sum(e3_stream, 1)  # [B, dim]

        return pooled_target_3d, pooled_e3_3d, e3_attn_weights


class AsymmetricGatedFusion(nn.Module):
    """
    AGF (Asymmetric Gated Fusion)
    Adaptively fuse E3-side 3D spatial features with 1D local-motif features.
    The 3D stream forms the backbone, while a learnable gate adds local information as needed.
    """
    def __init__(self, dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, pooled_3d, e3_1d_feats, e3_attn_weights):
        """
        Args:
            pooled_3d: pooled E3-side 3D features [B, dim]
            e3_1d_feats: 1D motif features from MoKA [B, N_l, dim]
            e3_attn_weights: E3-side attention weights from RPI [B, N_l+N_0+N_2, dim]
        Returns:
            pooled_e3: fused E3-side representation [B, dim]
        """
        # Use the first N_l attention positions, corresponding to the E3 sequence, as spatial weights.
        e3_seq_len = e3_1d_feats.size(1)  # N_l
        spatial_weights = e3_attn_weights[:, :e3_seq_len, :]  # [B, N_l, dim]

        focused_e3_1d = e3_1d_feats * spatial_weights  # [B, N_l, dim]
        pooled_1d = torch.sum(focused_e3_1d, 1)  # [B, dim]

        gate_input = torch.cat([pooled_3d, pooled_1d], dim=-1)  # [B, dim*2]
        g = self.gate(gate_input)  # [B, dim]
        return pooled_3d + g * pooled_1d  # [B, dim]


class PhysChemEncoder(nn.Module):
    """
    Component-aware chemical-descriptor branch.

    Split the 402-dimensional descriptor into three component tokens (warhead,
    linker, and E3 ligand). Project fingerprints and physicochemical properties
    separately, normalize the latter component-wise with [3, 6] mean and standard
    deviation buffers, model inter-component relationships with self-attention,
    and mean-pool the result to [B, desc_dim].
    """
    def __init__(self, fp_dim=128, num_physchem=6, num_components=3, desc_dim=64, drop_out=0.2):
        super().__init__()
        self.fp_dim = fp_dim
        self.num_physchem = num_physchem
        self.num_components = num_components
        self.desc_dim = desc_dim
        self.chunk_size = fp_dim + num_physchem  # 134

        comp_dim = desc_dim

        # Component-wise physicochemical normalization parameters [3, 6].
        self.register_buffer('physchem_mean', torch.zeros(num_components, num_physchem))
        self.register_buffer('physchem_std', torch.ones(num_components, num_physchem))

        self.fp_proj = nn.Sequential(
            nn.Linear(fp_dim, comp_dim // 2),  # 128 → 32
            nn.ReLU(),
        )

        self.pc_proj = nn.Sequential(
            nn.Linear(num_physchem, comp_dim // 2),  # 6 → 32
            nn.ReLU(),
        )

        self.comp_position = nn.Parameter(torch.randn(1, num_components, comp_dim) * 0.02)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=comp_dim, num_heads=2, dropout=drop_out, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(comp_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(comp_dim, desc_dim),
            nn.LayerNorm(desc_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
        )

    def forward(self, descriptors):
        """
        Args:
            descriptors: [B, 402] = [warhead(134) | linker(134) | e3_ligand(134)]
        Returns:
            desc_embed: [B, desc_dim]
        """
        B = descriptors.size(0)

        chunks = descriptors.view(B, self.num_components, self.chunk_size)  # [B, 3, 134]
        fp = chunks[:, :, :self.fp_dim]           # [B, 3, 128]
        physchem = chunks[:, :, self.fp_dim:]     # [B, 3, 6]

        # Normalize absolute physicochemical properties component-wise.
        physchem = (physchem - self.physchem_mean) / (self.physchem_std + 1e-8)

        fp_feat = self.fp_proj(fp)       # [B, 3, 32]
        pc_feat = self.pc_proj(physchem) # [B, 3, 32]

        tokens = torch.cat([fp_feat, pc_feat], dim=-1)  # [B, 3, 64]
        tokens = tokens + self.comp_position            # Add component-position encoding.

        attn_out, _ = self.self_attn(tokens, tokens, tokens)  # [B, 3, 64]
        tokens = self.attn_norm(tokens + attn_out)            # Residual connection and LayerNorm.

        pooled = tokens.mean(dim=1)  # [B, 64]

        desc_embed = self.out_proj(pooled)  # [B, desc_dim]
        return desc_embed


class Model(nn.Module):
    """
    ARS-PROTACs main model with asymmetric processing and auxiliary chemical descriptors.

    Fuse three SE(3)-encoded molecular components (warhead, linker, and E3 ligand)
    with ESM protein features. Drug-conditioned target focus, pairwise interaction,
    and AGF produce the deep representation. Morgan fingerprints and physicochemical
    descriptors provide complementary topology and property information before binary classification.
    """
    def __init__(self,
                 e3_ligand_model,
                 ligase_esm_wrapper,
                 warhead_model,
                 target_esm_wrapper,
                 linker_model,
                 dim=320,
                 proj_dim=128,
                 protein_vocab_size=26,
                 desc_dim=64,
                 fp_dim=128,
                 drop_out=0.2,
                 gain_scale=0.1):
        super().__init__()

        self.ligase_ligand_model = e3_ligand_model      # E3-ligand 3D encoder
        self.ligase_model = ligase_esm_wrapper          # Parameter-free ESMWrapper(Identity)
        self.target_ligand_model = warhead_model        # Warhead 3D encoder
        self.target_model = target_esm_wrapper          # Parameter-free ESMWrapper(Identity)
        self.linker_model = linker_model                # Linker 3D encoder

        self.proj = nn.Linear(dim, proj_dim)  # ESM 320 → 128

        self.e3_moka = E3MoKA(
            protein_vocab_size, proj_dim, proj_dim, kernel_size=3, drop_out=drop_out
        )

        self.target_focus = DrugConditionedTargetFocus(dim=proj_dim)

        self.rpi = RolePairwiseInteraction(dim=proj_dim, gain_scale=gain_scale)

        self.agf = AsymmetricGatedFusion(dim=proj_dim)

        self.physchem = PhysChemEncoder(
            fp_dim=fp_dim,
            num_physchem=6,
            num_components=3,
            desc_dim=desc_dim,
            drop_out=drop_out
        )

        classifier_input_dim = proj_dim * 2 + desc_dim  # 128*2 + 64 = 320
        self.relu = nn.LeakyReLU()

        self.dropout2 = nn.Dropout(drop_out)
        self.dropout3 = nn.Dropout(drop_out)

        self.fc1 = nn.Linear(classifier_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 2)

    def forward(self,
                target_embed,
                target_tokens,
                warhead_graph,
                linker_graph,
                e3_ligand_graph,
                ligase_embed,
                ligase_tokens,
                mol_descriptors):
        """
        Args:
            target_embed: target-protein ESM embeddings [B, N_t, 320]
            target_tokens: target-protein token sequence [B, N_t]
            warhead_graph: warhead 3D graph
            linker_graph: linker 3D graph
            e3_ligand_graph: E3-ligand 3D graph
            ligase_embed: E3-ligase ESM embeddings [B, N_l, 320]
            ligase_tokens: E3-ligase token sequence [B, N_l]
            mol_descriptors: chemical descriptors [B, 402]
        """

        target_mask = target_tokens.ne(0).unsqueeze(-1)  # [B, N_t, 1]
        e3_ligase_mask = ligase_tokens.ne(0).unsqueeze(-1)  # [B, N_l, 1]

        v_target_esm = self.proj(target_embed) * target_mask  # [B, N_t, 128]
        v_e3_esm = self.proj(ligase_embed) * e3_ligase_mask  # [B, N_l, 128]

        v_e3_ligand = self.ligase_ligand_model(e3_ligand_graph)   # E3 ligand [B, N_0, 128]
        v_warhead = self.target_ligand_model(warhead_graph)    # Warhead [B, N_1, 128]
        v_linker = self.linker_model(linker_graph)                   # Linker [B, N_2, 128]

        v_target_esm = self.target_focus(v_target_esm, v_warhead, mask=target_mask)  # [B, N_t, 128]

        e3_1d_feats = self.e3_moka(ligase_tokens, torch.cat([v_e3_ligand, v_linker], dim=1), mask=e3_ligase_mask)  # [B, N_l, 128]

        target_stream = torch.cat([v_target_esm, v_warhead, v_linker], dim=1)  # [B, N_t+N_1+N_2, 128]
        e3_stream = torch.cat([v_e3_esm, v_e3_ligand, v_linker], dim=1)  # [B, N_l+N_0+N_2, 128]

        pooled_target_3d, pooled_e3_3d, e3_attn_weights = self.rpi(
            target_stream, e3_stream
        )  # pooled_*: [B, 128]; e3_attn_weights: [B, N_l+N_0+N_2, 128]

        pooled_e3 = self.agf(pooled_e3_3d, e3_1d_feats, e3_attn_weights)  # [B, 128]

        desc_embed = self.physchem(mol_descriptors)  # [B, 64]

        deep_feats = torch.cat([pooled_target_3d, pooled_e3], dim=1)  # [B, 256]
        combined = torch.cat([deep_feats, desc_embed], dim=1)  # [B, 320]
        
        fully1 = self.relu(self.fc1(combined))  # [B, 512]
        fully1 = self.dropout2(fully1)  # [B, 512]
        fully2 = self.relu(self.fc2(fully1))  # [B, 256]
        fully2 = self.dropout3(fully2)  # [B, 256]
        fully3 = self.relu(self.fc3(fully2))  # [B, 128]

        predict = self.out(fully3)  # [B, 2]
        return predict

