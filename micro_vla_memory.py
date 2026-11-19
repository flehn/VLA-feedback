"""
micro_vla_memory.py: MicroVLA with switchable *continuous* feedback pathways.

Built on top of micro_vla.py (env, tokenizer, patch embedding and Transformer blocks are reused).

Finding 2 in docs/findings.md showed that a 1-bit gate driven by h_{t-1} cannot carry spatial
information (e.g. where an occluded target was). This file tests three richer feedback signals,
each of which can be switched on independently (and combined):

  prev_action  (+2 tokens)   The action executed at t-1 is fed back as two extra tokens
                             [PREV_X_BIN, PREV_Y_BIN] (or [NO_PREV, NO_PREV] at t=0), reusing the
                             action-bin embeddings. Cheapest form of proprioceptive feedback.
  prev_frame   (+64 tokens)  The previous camera frame is fed as 64 extra patch tokens (shared patch
                             embedding + a learned frame-index embedding). CronusVLA-style multi-frame.
  memory       (+1 token)    A continuous recurrent memory vector m_{t-1} in R^d is fed as a token at
                             the very start of the sequence (every position can attend to it). The new
                             memory m_t = tanh(W h_t[ACT_START]) is read out from the final hidden state.
                             Trained with full-episode BPTT. A tiny ReMem-VLA / AVA-VLA style recurrent state.

Unified sequence (optional parts in brackets):
  [MEM] | [PREV_PATCH_0..63] | PATCH_0..63 | [PREV_X, PREV_Y] | "reach" <color> ACT_START [ACT_X]

Training uses the executed (expert) previous action and previous frame (teacher forcing); at
inference the model's own executed actions and observed frames are fed back.
"""

import math
import statistics
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from micro_vla import (
    DEVICE,
    IMG_SIZE,
    MAX_STEP_PX,
    NUM_BINS,
    PATCH_SIZE,
    VOCAB_SIZE,
    X_OFFSET,
    Y_OFFSET,
    MicroPatchEmbed,
    TransformerBlock,
    bin_to_action,
    hidden_colors_at,
    render_env,
    step_targets,
    stoi,
)

NO_PREV = VOCAB_SIZE  # extra token id meaning "no previous action" (t = 0)

VARIANTS = {
    # name:          (prev_action, prev_frame, memory)
    "baseline": (False, False, False),
    "prev_action": (True, False, False),
    "prev_frame": (False, True, False),
    "memory": (False, False, True),
    "memory+prev_action": (True, False, True),
}


class MemoryMicroVLA(nn.Module):
    def __init__(
        self,
        prev_action: bool = False,
        prev_frame: bool = False,
        memory: bool = False,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
    ):
        super().__init__()
        self.use_prev_action = prev_action
        self.use_prev_frame = prev_frame
        self.use_memory = memory
        self.d_model = d_model
        self.patch_embed = MicroPatchEmbed(IMG_SIZE, PATCH_SIZE, d_model)
        self.token_embed = nn.Embedding(VOCAB_SIZE + (1 if prev_action else 0), d_model)
        self.token_pos_emb = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
        if prev_frame:
            self.frame_emb = nn.Parameter(torch.randn(2, d_model) * 0.02)  # [previous, current]
        if memory:
            self.mem_init = nn.Parameter(torch.zeros(d_model))
            self.mem_in = nn.Linear(d_model, d_model)
            self.mem_out = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, VOCAB_SIZE, bias=False)

    @property
    def is_recurrent(self) -> bool:
        return self.use_memory

    def initial_memory(self, batch_size: int):
        if not self.use_memory:
            return None
        return self.mem_init.unsqueeze(0).expand(batch_size, -1)

    def forward(self, img, token_ids, prev_img=None, prev_act=None, mem=None, targets=None):
        """img [B,32,32,3]; token_ids [B,P] prompt ("reach", color, ACT_START[, ACT_X]);
        prev_img [B,32,32,3]; prev_act [B,2] token ids; mem [B,d]; targets [B,P].
        Returns (logits over the P prompt positions [B,P,V], loss, next memory)."""
        B, P = token_ids.shape
        parts = []
        if self.use_memory:
            parts.append(self.mem_in(mem).unsqueeze(1))
        cur = self.patch_embed(img)
        if self.use_prev_frame:
            parts.append(self.patch_embed(prev_img) + self.frame_emb[0])
            cur = cur + self.frame_emb[1]
        parts.append(cur)
        text = torch.cat([prev_act, token_ids], dim=1) if self.use_prev_action else token_ids
        parts.append(self.token_embed(text) + self.token_pos_emb[:, : text.size(1)])
        x = torch.cat(parts, dim=1)

        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)
        logits = self.lm_head(h[:, -P:])  # predictions at the prompt positions only

        mem_next = None
        if self.use_memory:
            act_start_idx = x.size(1) - P + 2
            mem_next = torch.tanh(self.mem_out(h[:, act_start_idx]))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss, mem_next


# ------------------------------------------------------------------------------
# Data plumbing
# ------------------------------------------------------------------------------
def to_streams(tensors, episode_len: int):
    """Reshape flat micro_vla data [E*L*3, ...] into per-(episode, prompt) streams [E*3, L, ...]."""
    out = []
    for t in tensors:
        E = t.size(0) // (episode_len * 3)
        t = t.view(E, episode_len, 3, *t.shape[1:]).transpose(1, 2).contiguous()
        out.append(t.view(E * 3, episode_len, *t.shape[3:]))
    return out


def prev_action_tokens(exec_bins: torch.Tensor) -> torch.Tensor:
    """exec_bins [S, L, 2] executed bins per step -> previous-action token ids [S, L, 2]."""
    toks = torch.stack([exec_bins[..., 0] + X_OFFSET, exec_bins[..., 1] + Y_OFFSET], dim=-1)
    first = torch.full_like(toks[:, :1], NO_PREV)
    return torch.cat([first, toks[:, :-1]], dim=1)


def prev_frames(images: torch.Tensor) -> torch.Tensor:
    """images [S, L, H, W, 3] -> previous frame per step (frame 0 repeats itself)."""
    return torch.cat([images[:, :1], images[:, :-1]], dim=1)


# ------------------------------------------------------------------------------
# Training (shared episode batches; BPTT through the memory token when enabled)
# ------------------------------------------------------------------------------
def episode_ce(model, imgs, inps, tgts, pacts, reduction="mean"):
    """Run a batch of streams [S, L, ...]. Returns mean CE (or per-sample CE [S, L])."""
    S, L = imgs.shape[:2]
    pimgs = prev_frames(imgs)
    if not model.is_recurrent:  # no state across steps -> one flat batch
        logits, loss, _ = model(
            imgs.flatten(0, 1),
            inps.flatten(0, 1),
            pimgs.flatten(0, 1),
            pacts.flatten(0, 1),
            None,
            tgts.flatten(0, 1),
        )
        if reduction == "mean":
            return loss
        return per_sample_ce(logits, tgts.flatten(0, 1)).view(S, L)
    mem = model.initial_memory(S)
    losses, per = 0.0, []
    for t in range(L):
        logits, loss, mem = model(imgs[:, t], inps[:, t], pimgs[:, t], pacts[:, t], mem, tgts[:, t])
        losses = losses + loss
        if reduction != "mean":
            per.append(per_sample_ce(logits, tgts[:, t]))
    return losses / L if reduction == "mean" else torch.stack(per, dim=1)


def per_sample_ce(logits, targets):
    """Mean CE of the two action tokens (X predicted at ACT_START, Y at ACT_X) per sample."""
    lx = F.cross_entropy(logits[:, 2], targets[:, 2], reduction="none")
    ly = F.cross_entropy(logits[:, 3], targets[:, 3], reduction="none")
    return 0.5 * (lx + ly)


def train_memory_vla(
    model,
    streams,
    steps: int = 520,
    batch_schedule=None,
    eps_per_batch: int = 8,
    lr: float = 4e-3,
    log_every: int = 0,
    device: torch.device = DEVICE,
):
    """streams = (images, inputs, targets, prev_act_tokens), each [E*3, L, ...]. Same AdamW + LR decay
    as micro_vla.train_vla. All 3 counterfactual prompt streams of an episode are kept together."""
    model.to(device)
    imgs_all, inps_all, tgts_all, pacts_all = (x.to(device) for x in streams)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    num_episodes = imgs_all.size(0) // 3
    decay_start = int(steps * 0.71)
    history = []
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        if step > decay_start:
            frac = (step - decay_start) / max(1, steps - decay_start)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * max(0.1, 1.0 - 0.9 * frac)
        if batch_schedule is not None:
            ep_idx = batch_schedule[step - 1].to(device) * 3
        else:
            ep_idx = torch.randint(0, num_episodes, (eps_per_batch,), device=device) * 3
        idx = torch.cat([ep_idx, ep_idx + 1, ep_idx + 2])
        loss = episode_ce(model, imgs_all[idx], inps_all[idx], tgts_all[idx], pacts_all[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.item()))
        if log_every and (step % log_every == 0 or step == 1 or step == steps):
            print(f"step {step:4d}/{steps} | ce: {loss.item():.4f} | {time.time() - t0:5.1f}s")
    return history


# ------------------------------------------------------------------------------
# Batched closed-loop evaluation with feedback from the model's own actions
# ------------------------------------------------------------------------------
@torch.no_grad()
def closed_loop(model, env_states, hidden_windows, target_vels_list, steps: int, success_px: float):
    """Roll out every (env_state x 3 prompts) in one batch. Returns aggregate metrics and the
    per-rollout trajectories. hidden_windows[i] is {color: set_of_timesteps} for env_states[i]."""
    model.eval()
    dev = next(model.parameters()).device
    colors = ["red", "green", "blue"]
    M = len(env_states) * 3
    grippers = [list(s[0]) for s in env_states for _ in colors]
    targets = [{c: list(p) for c, p in s[1].items()} for s in env_states for _ in colors]
    vels = [v for v in target_vels_list for _ in colors]
    hiddens = [h for h in hidden_windows for _ in colors]
    tcolors = [c for _ in env_states for c in colors]
    prefix = torch.tensor(
        [[stoi["reach"], stoi[c], stoi["ACT_START"]] for c in tcolors], dtype=torch.long, device=dev
    )
    prev_act = torch.full((M, 2), NO_PREV, dtype=torch.long, device=dev)
    prev_img = None
    mem = model.initial_memory(M)
    dists, actions, trajs = [], [], [[list(g)] for g in grippers]

    for t in range(steps):
        img = torch.stack(
            [
                render_env(g, tg, hidden_colors=hidden_colors_at(hd, t, tc))
                for g, tg, hd, tc in zip(grippers, targets, hiddens, tcolors)
            ]
        ).to(dev)
        if prev_img is None:
            prev_img = img
        logits_x, _, mem_next = model(img, prefix, prev_img, prev_act, mem)
        xb = logits_x[:, 2, X_OFFSET : X_OFFSET + NUM_BINS].argmax(-1)
        inp_y = torch.cat([prefix, (X_OFFSET + xb).unsqueeze(1)], dim=1)
        logits_y, _, _ = model(img, inp_y, prev_img, prev_act, mem)
        yb = logits_y[:, 3, Y_OFFSET : Y_OFFSET + NUM_BINS].argmax(-1)
        mem, prev_img = mem_next, img
        prev_act = torch.stack([X_OFFSET + xb, Y_OFFSET + yb], dim=1)

        step_d, step_a = [], []
        for i, (x, y) in enumerate(zip(xb.tolist(), yb.tolist())):
            dx, dy = bin_to_action(x), bin_to_action(y)
            grippers[i] = [
                max(2.0, min(29.0, grippers[i][0] + dx * MAX_STEP_PX)),
                max(2.0, min(29.0, grippers[i][1] + dy * MAX_STEP_PX)),
            ]
            targets[i] = step_targets(targets[i], vels[i])
            goal = targets[i][tcolors[i]]
            step_d.append(math.hypot(goal[0] - grippers[i][0], goal[1] - grippers[i][1]))
            step_a.append((dx, dy))
            trajs[i].append(list(grippers[i]))
        dists.append(step_d)
        actions.append(step_a)

    final = dists[-1]
    # Jitter: mean ||a_t - a_{t-1}|| over consecutive actions (lower = smoother control)
    jitter = [
        statistics.mean(
            math.hypot(actions[t][i][0] - actions[t - 1][i][0], actions[t][i][1] - actions[t - 1][i][1])
            for t in range(1, steps)
        )
        for i in range(M)
    ]
    # Time to reach: first step within success_px (steps + 1 if never)
    ttr = [next((t + 1 for t in range(steps) if dists[t][i] < success_px), steps + 1) for i in range(M)]
    # Grounding error: gripper ends closer to a non-prompted target than to the prompted one
    wrong = [
        float(
            min(
                math.hypot(targets[i][c][0] - grippers[i][0], targets[i][c][1] - grippers[i][1])
                for c in colors
                if c != tcolors[i]
            )
            < final[i]
        )
        for i in range(M)
    ]
    metrics = {
        "final_dist_px": statistics.mean(final),
        "success_rate": statistics.mean(float(d < success_px) for d in final),
        "time_to_reach": statistics.mean(ttr),
        "jitter": statistics.mean(jitter),
        "wrong_target": statistics.mean(wrong),
    }
    return metrics, {"trajectories": trajs, "dists": dists, "tcolors": tcolors}
