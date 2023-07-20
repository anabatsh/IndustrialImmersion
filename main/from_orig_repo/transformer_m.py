import torch
import torch.nn as nn
import utils
from transformer_m_encoder import TransformerMEncoder


class TransformerM():
    
    def __init__(self, args):
        self.max_positions = args.max_positions
        self.molecule_encoder = TransformerMEncoder(
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_init=args.apply_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
            sandwich_ln=args.sandwich_ln,
            droppath_prob=args.droppath_prob,
            add_3d=args.add_3d,
            num_3d_bias_kernel=args.num_3d_bias_kernel,
            no_2d=args.no_2d,
            mode_prob=args.mode_prob,
        )

        self.embed_out = None
        self.proj_out = None

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = nn.LayerNorm(args.encoder_embed_dim, eps=1e-5, elementwise_affine=True)

        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.embed_out = nn.Linear(
            args.encoder_embed_dim, 1, bias=False
        )


    def forward(self, batched_data, perturb=None, segment_labels=None, 
                masked_tokens=None, **unused):

        inner_states, atom_output = self.molecule_encoder(
            batched_data,
            segment_labels=segment_labels,
            perturb=perturb,
        )
        x = inner_states[-1].transpose(0, 1)
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        x = self.embed_out(x)
        x = x + self.lm_output_learned_bias
        return x, atom_output, {
            "inner_states": inner_states,
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions