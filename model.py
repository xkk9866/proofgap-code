"""ERC-Net: Evidence Requirement Circuit networks and controlled set-function baselines.

Every model shares the same evidence-state-conditioned cross-encoder.  They differ only in
the *set function* that turns candidate representations into the completeness probability
log C(S) (or, for the pointwise baselines, per-document scores).

  head="erc"       C(S) = prod_j [1 - z_j prod_{d in S} (1 - a_jd)]   (AND over requirements of OR over witnesses)
  head="additive"  logit C(S) = b + sum_{d in S} v_d                    (order-1 set function)
  head="pairwise"  logit C(S) = b + sum v_d + sum_{d<d'} w_dd'          (order-2 set function)
  head="point"     p(d in G \\ S | q, S)                               (Beam-Retrieval style next-evidence classifier)
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

EPS = 1e-6
ENC_CHUNK = int(os.environ.get("ENC_CHUNK", "64"))  # sequences per encoder call at inference (bounds memory on large pools)
R_DEFICIT = 4  # deficit ladder rungs 0, 1, 2, >=3
R_GAP = 5  # proof-gap field: r in {0,1,2,3,4} exactly (every question has at most four gold passages)
NRUNG = R_DEFICIT - 1  # informative cumulative logits u_0, u_1, u_2 (the last class r >= 3 is the tail)

# Official DeBERTa-v3-base (the backbone of Beam Retrieval).  Older checkpoints were trained from an NLI-tuned
# copy that no longer exists on the server; their state dicts contain the full encoder, so loading them only
# needs the (identical) architecture and tokenizer of the official model.
ENC_DEFAULT = "/data/yaokang/models/dv3/deberta-v3-base"


def resolve_enc(path):
    return path if path and os.path.exists(os.path.join(path, "config.json")) and (
        os.path.exists(os.path.join(path, "pytorch_model.bin")) or os.path.exists(os.path.join(path, "model.safetensors"))
    ) else ENC_DEFAULT


def cum_logits(o):
    """Monotone cumulative logits u_0 <= u_1 <= ... from raw head outputs o [n, R]: u_h = o_0 + sum_{i<=h} softplus(o_i)."""
    inc = F.softplus(o[:, 1:]) + 1e-3
    return torch.cat([o[:, :1], o[:, :1] + torch.cumsum(inc, 1)], 1)


def rung_stats(u):
    """From cumulative logits u [n, >=NRUNG]: log P(r=0), log P(r<=1), expected truncated deficit E[min(r, R-1)]."""
    u = u[:, :NRUNG].float()
    return F.logsigmoid(u[:, 0]), F.logsigmoid(u[:, 1]), torch.sigmoid(-u).sum(1)


def log1mexp(x):
    """log(1 - exp(x)) for x <= 0, numerically stable."""
    x = x.clamp(max=-EPS)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


class SetReasoner(nn.Module):
    """Transformer over candidate vectors with a selected/unselected state embedding."""

    def __init__(self, hidden, layers=2, heads=8):
        super().__init__()
        self.state_emb = nn.Embedding(2, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, h, sel):  # h [B,N,H], sel [B,N] long, pad mask [B,N]
        return self.enc(h + self.state_emb(sel))


class ERCHead(nn.Module):
    """Requirement decoder + probabilistic AND-OR completeness circuit."""

    def __init__(self, hidden, m=4, layers=2, heads=8):
        super().__init__()
        self.m = m
        self.queries = nn.Parameter(torch.randn(m, hidden) * 0.02)
        self.g_proj = nn.Linear(hidden, hidden)
        layer = nn.TransformerDecoderLayer(hidden, heads, 2 * hidden, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.wr = nn.Linear(hidden, hidden)
        self.wh = nn.Linear(hidden, hidden)
        self.z = nn.Linear(hidden, 1)
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, h, mask):
        # h [B,N,H]; mask [B,N] bool valid
        B = h.size(0)
        g = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        q = self.queries.unsqueeze(0).expand(B, -1, -1) + self.g_proj(g).unsqueeze(1)
        r = self.dec(q, h, memory_key_padding_mask=~mask)  # [B,m,H]
        logit_a = torch.einsum("bmh,bnh->bmn", self.wr(r), self.wh(h)) / math.sqrt(h.size(-1)) + self.bias
        logit_a = logit_a.masked_fill(~mask.unsqueeze(1), -30.0)
        logit_z = self.z(r).squeeze(-1)  # [B,m]
        return logit_a, logit_z

    @staticmethod
    def completeness(logit_a, logit_z, sel):
        """Return log C(S) [B] and log C(S + {d}) [B,N] from one circuit evaluation."""
        log1m_a = F.logsigmoid(-logit_a)  # log(1-a) [B,m,N]
        logP = (log1m_a * sel.unsqueeze(1).float()).sum(-1)  # [B,m]
        logz = F.logsigmoid(logit_z)
        logC = log1mexp(logz + logP).sum(-1)  # [B]
        logP_add = logP.unsqueeze(-1) + log1m_a  # [B,m,N]
        logC_add = log1mexp(logz.unsqueeze(-1) + logP_add).sum(1)  # [B,N]
        return logC, logC_add


class FactorizedERCHead(nn.Module):
    """Witness-requirement factorised circuit: a_jd = p_d * pi_jd.

    p_d   = sigma(w^T h_d): d is evidence for *some* requirement (read from the state-conditioned encoder)
    pi_jd = softmax_j(<W_r r_j, W_h h_d>): which requirement d witnesses (requirement decoder)
    z_j   = requirement activation.  The AND-OR circuit itself is unchanged.
    """

    def __init__(self, hidden, m=4, layers=2, heads=8):
        super().__init__()
        self.m = m
        self.wit = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))
        self.queries = nn.Parameter(torch.randn(m, hidden) * 0.02)
        self.g_proj = nn.Linear(hidden, hidden)
        layer = nn.TransformerDecoderLayer(hidden, heads, 2 * hidden, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.wr = nn.Linear(hidden, hidden)
        self.wh = nn.Linear(hidden, hidden)
        self.z = nn.Linear(hidden, 1)

    def forward(self, h, mask):
        B = h.size(0)
        g = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        q = self.queries.unsqueeze(0).expand(B, -1, -1) + self.g_proj(g).unsqueeze(1)
        r = self.dec(q, h, memory_key_padding_mask=~mask)
        logit_p = self.wit(h).squeeze(-1).masked_fill(~mask, -30.0)  # [B,N]
        pi_logit = torch.einsum("bmh,bnh->bmn", self.wr(r), self.wh(h)) / math.sqrt(h.size(-1))
        log_pi = F.log_softmax(pi_logit, dim=1)  # over requirements
        log_a = F.logsigmoid(logit_p).unsqueeze(1) + log_pi  # [B,m,N]
        log1m_a = log1mexp(log_a).masked_fill(~mask.unsqueeze(1), 0.0)
        return logit_p, log_pi, log1m_a, self.z(r).squeeze(-1)


def circuit(log1m_a, logit_z, sel):
    logP = (log1m_a * sel.unsqueeze(1).float()).sum(-1)
    logz = F.logsigmoid(logit_z)
    logC = log1mexp(logz + logP).sum(-1)
    logC_add = log1mexp(logz.unsqueeze(-1) + logP.unsqueeze(-1) + log1m_a).sum(1)
    return logC, logC_add


class ERCNet(nn.Module):
    def __init__(self, enc_name, head="erc", m=4, set_layers=2, use_set=True, grad_ckpt=False, hop_heads=False, n_rung=R_DEFICIT):
        super().__init__()
        self.R = n_rung  # deficit classes 0, 1, ..., R-2 and the tail ">= R-1" (v3: R = 5, exact for |G| <= 4)
        self.encoder = AutoModel.from_pretrained(resolve_enc(enc_name))
        if grad_ckpt:
            self.encoder.gradient_checkpointing_enable()
        H = self.encoder.config.hidden_size
        self.head_type = head
        self.use_set = use_set and head not in ("point_noset", "prover", "deficit", "ladder", "gap")
        if self.use_set:
            self.setr = SetReasoner(H, layers=set_layers)
        if head == "erc":
            self.head = ERCHead(H, m=m)
        elif head == "erc2":
            self.head = FactorizedERCHead(H, m=m)
        elif head == "gap":  # one proof-gap field read from the joint encoding of a whole set [q; S]
            self.ver = nn.Sequential(nn.Dropout(0.1), nn.Linear(H, R_GAP))
            self.target = "ordinal"
        elif head in ("additive", "pairwise"):
            self.v = nn.Linear(H, 1)
            self.b = nn.Parameter(torch.tensor(-3.0))
            if head == "pairwise":
                self.p1 = nn.Linear(H, 256)
                self.p2 = nn.Linear(H, 256)
        else:  # point / point_noset / prover
            self.cls = nn.Sequential(nn.Dropout(0.1), nn.Linear(H, 1))
            if hop_heads:  # official Beam Retrieval: separate classifiers for the first hop and for later hops
                self.cls1 = nn.Sequential(nn.Dropout(0.1), nn.Linear(H, 1))
                self.sep_id = 2
            if head in ("prover", "deficit"):  # completeness verdict read from one joint encoding of [q; S]
                self.ver = nn.Sequential(nn.Dropout(0.1), nn.Linear(H, 1))
            if head == "ladder":  # vector deficit ladder: one logit per rung, made monotone by construction
                self.ver = nn.Sequential(nn.Dropout(0.1), nn.Linear(H, n_rung))
            if head == "deficit":
                # ordered cut-points theta_0 < theta_1 < ... < theta_{R-1} of the proof-deficit ladder
                self.theta0 = nn.Parameter(torch.tensor(0.0))
                self.dtheta = nn.Parameter(torch.zeros(R_DEFICIT - 1))

    def cutpoints(self):
        return self.theta0 + torch.cat([self.theta0.new_zeros(1), torch.cumsum(F.softplus(self.dtheta) + 1e-3, 0)])

    def deficit_energy(self, input_ids, attn):
        """Proof-deficit energy f(q, S): P(|G minus S| <= h | q, S) = sigmoid(theta_h - f)."""
        h = self.encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state[:, 0].float()
        return self.ver(h).squeeze(-1)

    ladder_kind = "ordinal"  # ordinal | binary (only rung 0 trained) | multiclass (unordered softmax, ablation)

    def to_cum(self, o):
        """Cumulative logits u_h = logit P(r <= h) from raw head outputs (all kinds expose the same interface)."""
        if self.ladder_kind == "multiclass":
            cdf = torch.cumsum(torch.softmax(o[:, : self.R], 1), 1)[:, : self.R - 1].clamp(1e-6, 1 - 1e-6)
            return torch.cat([torch.log(cdf) - torch.log1p(-cdf), torch.full_like(cdf[:, :1], 20.0)], 1)
        return cum_logits(o)

    def set_reading(self, input_ids, attn):
        """Raw deficit-head outputs of the holistic reading [CLS] q [SEP] s_1 [SEP] ... s_k [SEP]."""
        h = self.encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state[:, 0].float()
        return self.ver(h)

    def view_heads(self, input_ids, attn):
        """Both readings of a member-focused view [CLS] q [SEP] T minus t [SEP] t [SEP]: the raw deficit-head outputs of
        the set (R numbers) and the membership logit of the focused passage t (does t belong to the proof?)."""
        h = self.encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state[:, 0].float()
        return self.ver(h), self.cls(h).squeeze(-1)

    def ladder_logits(self, input_ids, attn):
        """logit P(|G minus S| <= h | q, S) for h = 0..R-1: u_h = u_0 + sum_{i<=h} softplus(delta_i)."""
        return self.to_cum(self.set_reading(input_ids, attn))

    def ladder_from_h(self, h):
        """Deficit ladder of the set serialised in the encoding whose [CLS] vectors are h [..., H].

        Applied to the proposer's encoding [q; S; d] this is the *incremental* reading of the child S + {d};
        applied to [q; s_1; ...; s_k] it is the *holistic* reading of the set.  Same head, same quantity."""
        shp = h.shape[:-1]
        return self.to_cum(self.ver(h.reshape(-1, h.size(-1)).float())).view(*shp, -1)

    def expected_deficit(self, input_ids, attn):
        """E[r(S)] = sum_h (1 - P(r <= h)), the last rung counting r >= R-1."""
        return (1 - torch.sigmoid(self.ladder_logits(input_ids, attn))).sum(1)

    def gap_field(self, input_ids, attn):
        """Proof-gap field of a set: (log P(r(S)=0), E[r(S)], log pmf [n,R_GAP]) with r(S) = |G minus S|.

        ordinal   : cumulative logits u_0 < u_1 < u_2 < u_3, P(r <= h) = sigma(u_h), P(r <= 4) = 1 (exact, no tail rung)
        multiclass: unordered softmax over the five gap values          (ablation)
        binary    : only P(r = 0) is modelled, E[r] := 1 - P(r = 0)      (ablation: completeness verdict)
        regress   : r_hat = softplus(o_0) regressed on r, E[r] := r_hat (ablation; log P0 := -r_hat is a score)
        """
        h = self.encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state[:, 0].float()
        o = self.ver(h)
        if self.target == "ordinal":
            u = torch.cat([o[:, :1], o[:, :1] + torch.cumsum(F.softplus(o[:, 1:R_GAP - 1]) + 1e-3, 1)], 1)
            cdf = torch.cat([torch.sigmoid(u), torch.ones_like(u[:, :1])], 1)
            pmf = cdf - torch.cat([torch.zeros_like(cdf[:, :1]), cdf[:, :-1]], 1)
            return F.logsigmoid(u[:, 0]), (1 - cdf[:, :-1]).sum(1), torch.log(pmf.clamp_min(1e-8))
        if self.target == "multiclass":
            lp = F.log_softmax(o, 1)
            return lp[:, 0], (lp.exp() * torch.arange(R_GAP, device=o.device)).sum(1), lp
        if self.target == "binary":
            return F.logsigmoid(o[:, 0]), 1 - torch.sigmoid(o[:, 0]), o[:, :1]
        rh = F.softplus(o[:, 0])
        return -rh, rh, o[:, :1]

    def verdict(self, input_ids, attn):
        """Logit of P(G subset of S | q, S) from the joint encoding [CLS] q [SEP] s_1 [SEP] ... s_k [SEP].
        For the deficit head this is the first rung of the ladder, theta_0 - f."""
        if self.head_type == "ladder":
            return self.ladder_logits(input_ids, attn)[:, 0]
        if self.head_type == "gap":
            lp0 = self.gap_field(input_ids, attn)[0]
            return lp0 - log1mexp(lp0)
        f = self.deficit_energy(input_ids, attn)
        return self.cutpoints()[0] - f if self.head_type == "deficit" else f

    def encode(self, input_ids, attn, B, N):
        c = ENC_CHUNK if not torch.is_grad_enabled() else input_ids.size(0)
        out = torch.cat([self.encoder(input_ids=input_ids[i: i + c], attention_mask=attn[i: i + c]).last_hidden_state[:, 0]
                         for i in range(0, input_ids.size(0), c)])
        return out.view(B, N, -1)

    def forward(self, input_ids, attn, mask, sel):
        """input_ids [B*N,L]; mask [B,N] bool; sel [B,N] long (1 = already selected)."""
        B, N = mask.shape
        h = self.encode(input_ids, attn, B, N).float()
        if self.use_set:
            h = self.setr(h, sel)
        out = {}
        if self.head_type == "erc":
            la, lz = self.head(h, mask)
            out["logit_a"], out["logit_z"] = la, lz
            out["logC"], out["logC_add"] = ERCHead.completeness(la, lz, sel)
        elif self.head_type == "erc2":
            lp, lpi, l1a, lz = self.head(h, mask)
            out.update(logit_p=lp, log_pi=lpi, log1m_a=l1a, logit_z=lz)
            out["logC"], out["logC_add"] = circuit(l1a, lz, sel)
        elif self.head_type in ("additive", "pairwise"):
            v = self.v(h).squeeze(-1).masked_fill(~mask, 0.0)  # [B,N]
            s = sel.float()
            base = self.b + (v * s).sum(-1)  # [B]
            add = base.unsqueeze(-1) + v  # [B,N]
            if self.head_type == "pairwise":
                W = torch.einsum("bnk,bmk->bnm", self.p1(h), self.p2(h)) / 16.0
                W = 0.5 * (W + W.transpose(1, 2))
                W = W * (mask.unsqueeze(1) & mask.unsqueeze(2)).float()
                W = W - torch.diag_embed(torch.diagonal(W, dim1=1, dim2=2))
                pair_S = 0.5 * torch.einsum("bn,bnm,bm->b", s, W, s)
                pair_add = torch.einsum("bnm,bm->bn", W, s)  # interaction of d with S
                base = base + pair_S
                add = base.unsqueeze(-1) + v + pair_add
            out["logC"] = F.logsigmoid(base)
            out["logC_add"] = F.logsigmoid(add)
        else:
            sc = self.cls(h).squeeze(-1)
            if hasattr(self, "cls1"):  # first hop <=> the sequence is [CLS] q [SEP] d [SEP] (two separators)
                first = (input_ids.view(B, N, -1) == self.sep_id).sum(-1) <= 2
                sc = torch.where(first, self.cls1(h).squeeze(-1), sc)
            out["score"] = sc.masked_fill(~mask, -30.0)
            out["h"] = h  # [B,N,H] [CLS] of [q; S; d]: the ladder head reads the child S + {d} from it
        return out
