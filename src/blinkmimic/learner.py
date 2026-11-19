"""Replay-based BlinkMimic updates shared by native and external simulation collectors."""

from dataclasses import dataclass

import torch
from torch import nn

from .mdp import flatten_state, unflatten_state
from .objective import actor_objective


class WorldModelLearner:
    """Fit dynamics on masked replay windows, with optional mirrored samples."""

    def __init__(self, model, optimizer, normalize_context, *, num_joints,
                 window, batch_size, mirror=None, mirror_fraction=0.0, compile_kernels=False):
        self.model = model
        self.optimizer = optimizer
        self.normalize_context = normalize_context
        self.window = window
        self.batch_size = batch_size
        self.mirror = mirror
        self.mirror_fraction = mirror_fraction
        self.error_floor = model.inc_norm.mean.new_tensor(
            [0.003] * 3 + [0.005] * 3 + [0.02] * 3 + [0.05] * 3
            + [0.003] * num_joints + [0.05] * num_joints
        )
        self.rollout = torch.compile(self._rollout) if compile_kernels else self._rollout
        self.error = torch.compile(self._error) if compile_kernels else self._error

    def _rollout(self, s, a, a_prev, xi):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(self.window):
            S = self.model.step(S, a[:, j], a_prev[:, j], self.normalize_context(xi[:, j]))
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)

    def _error(self, S_hat, s1, mask):
        std = torch.maximum(self.model.inc_norm.std(), self.error_floor)
        S1 = unflatten_state(s1)
        e = (self.model.state_error(S_hat, S1) / std) ** 2  # (B, k, D)
        e = e.mean(-1)
        return (e * mask).sum() / s1.shape[0], e[:, -1]

    def _mirror_windows(self, w: dict, n: int) -> dict:
        """Replace the first n windows by their mirror images (states, actions, sim context)."""
        out = dict(w)
        ms = lambda x: flatten_state(self.mirror.state(unflatten_state(x)))  # noqa: E731
        for k, fn in (("s", ms), ("s1", ms), ("a", self.mirror.joints), ("a_prev", self.mirror.joints), ("xi", self.mirror.xi)):
            out[k] = torch.cat((fn(w[k][:n]), w[k][n:]))
        return out

    def update(self, buffer, *, exclude_cuts=False, check_finite=False):
        w = buffer.sample_windows(self.batch_size, self.window)
        if self.mirror is not None:
            w = self._mirror_windows(w, int(self.batch_size * self.mirror_fraction))
        S_hat = self.rollout(w["s"], w["a"], w["a_prev"], w["xi"])
        mask = w["mask"] * (~w["cut"]) if exclude_cuts else w["mask"]
        loss, e_last = self.error(S_hat, w["s1"], mask)
        if check_finite and not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite dynamics loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=check_finite)
        self.optimizer.step()
        m = mask[:, -1]
        return loss.detach() / self.window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


@dataclass(frozen=True)
class PolicyUpdateCfg:
    hist: int
    goal_frames: int
    horizon: int
    pi_batch: int
    gamma: float
    lam: float
    contact_reward: float
    critic_epochs: int
    critic_minibatches: int
    pi_clip_value: float
    pi_ema: float
    no_sync: bool = False
    cuda_graph: bool = False


class PolicyLearner:
    """Optimize the actor through imagined dynamics, then fit the critic on GAE targets."""

    def __init__(self, cfg, actor, critic, dynamics, actor_optimizer, critic_optimizer,
                 mdp, motion, normalize_context, *, average=None, sensing=None,
                 control_dt=None, compile_kernels=False, compile_bootstrap=False,
                 compile_critic=False, bounded_starts=False, keep_partial_minibatches=False, check_finite=False):
        self.cfg = cfg
        self.pi, self.critic, self.wm = actor, critic, dynamics
        self.opt_pi, self.opt_v = actor_optimizer, critic_optimizer
        self.mdp, self.motion = mdp, motion
        self.normalize_context = normalize_context
        self.pi_avg, self.pipe, self.control_dt = average, sensing, control_dt
        self.bounded_starts = bounded_starts
        self.keep_partial_minibatches = keep_partial_minibatches
        self.check_finite = check_finite
        self.nonfinite = torch.zeros((), dtype=torch.long, device=next(actor.parameters()).device)
        self.stateful = sensing is not None and sensing.stateful

        def kernel(fn):
            return torch.compile(fn) if compile_kernels else fn

        self.imag_step = kernel(self._imag_step)
        self.imag_step_sensed = kernel(self._imag_step_sensed)
        self.reward_term = kernel(self._reward_term)
        self.critic_input = kernel(self._critic_input)
        self.bootstrap_input = self.critic_input if compile_bootstrap else self._critic_input
        self.gae_targets = kernel(self._gae_targets) if compile_critic else self._gae_targets
        self.critic_loss = kernel(self._critic_loss) if compile_critic else self._critic_loss

    def _imag_step(self, S, o_hist, a_prev, f, xi, xi_n, g, clip_end=None):
        a = self.pi(o_hist, g)
        S1 = self.wm.step(S, a, a_prev, xi_n)
        f1 = f + 1
        if clip_end is not None:
            f1 = f1.minimum(clip_end)
        o1 = self.mdp.add_noise(self.mdp.obs(S1, a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, self.motion.goal(f1, self.cfg.goal_frames)

    def _imag_step_sensed(self, S, o_hist, a_prev, f, xi, xi_n, g, st):
        """Imagination step with the robot's stateful signal chain (filters, delay) at the control rate."""
        a = self.pi(o_hist, g)
        S1 = self.wm.step(S, a, a_prev, xi_n)
        f1 = f + 1
        gyro, accel = self.pipe.imu_from_states(S, S1, self.control_dt) if self.pipe.cfg.imu is not None else (None, None)
        st = self.pipe.filter_update(st, S1[5], gyro, accel, xi, self.control_dt, noise=True, ctrl_rate=True)
        o1, st = self.pipe.observe(st, self.pipe.raw_obs(st, S1, a, xi, f1), noise=True, delay=True)
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, self.motion.goal(f1, self.cfg.goal_frames), st

    def _reward_term(self, S, f, a, a_prev):
        return self.mdp.reward(S, f, a, a_prev, w_contact=self.cfg.contact_reward), self.mdp.soft_termination(S, f)

    def _gae_targets(self, V, rews, dones):
        """GAE(lambda) returns over the imagined transitions; V is (H+1, B), bootstrapped from V[-1]."""
        R = V[-1]
        targets = []
        for t in range(self.cfg.horizon - 1, -1, -1):
            R = rews[t] + self.cfg.gamma * (1 - dones[t]) * ((1 - self.cfg.lam) * V[t + 1] + self.cfg.lam * R)
            targets.append(R)
        return torch.stack(targets[::-1])

    def _critic_loss(self, X, Y):
        return ((self.critic(X) - Y) ** 2).mean()

    def _critic_input(self, S, f, xi_n, g):
        return torch.cat((self.mdp.critic_features(S, f, xi_n), g), -1)

    def _sample_valid_starts(self, buffer, B: int):
        if self.cfg.no_sync or self.bounded_starts:
            # fixed-size selection without reading the valid count on the host: stable-sort the valid
            # candidates first (only starts within H frames of a clip end are invalid, < 1%)
            s, h = buffer.sample_starts(2 * B, self.cfg.hist)
            f = s["f"]
            ok = f + self.cfg.horizon <= self.motion.clip_end[self.motion.clip_id[f]]
            idx = torch.sort((~ok).to(torch.uint8), stable=True).indices[:B]
            return {k: v[idx] for k, v in s.items()}, {k: v[idx] for k, v in h.items()}
        starts, hists, n = [], [], 0
        while n < B:
            s, h = buffer.sample_starts(2 * B, self.cfg.hist)
            f = s["f"]
            ok = f + self.cfg.horizon <= self.motion.clip_end[self.motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
            starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
            n += len(idx)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
        return cat(starts), cat(hists)

    @torch.no_grad()
    def _ema_update(self):
        """The EMA step of ``AveragedModel.update_parameters`` (same lerp) without its host-side first-call test; the
        host-side counter n_averaged is advanced by the training loop."""
        avg = list(self.pi_avg.module.parameters()) + list(self.pi_avg.module.buffers())
        live = [t.detach() for t in list(self.pi.parameters()) + list(self.pi.buffers())]
        assert all(t.is_floating_point() for t in avg)
        torch._foreach_lerp_(avg, live, 1 - self.cfg.pi_ema)

    def _update_actor(self, loss):
        if self.check_finite and not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite actor loss")
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        if self.cfg.pi_clip_value > 0:
            nn.utils.clip_grad_value_(self.pi.parameters(), self.cfg.pi_clip_value)
        if self.cfg.no_sync:
            # skip a non-finite update without a host read: zero the gradients on the GPU and gate the optimizer step (parameters
            # and all optimizer state unchanged; the plain Adam path without --fused_adam only zeroes the gradients, so its
            # momenta still move the parameters); counted in the log column nonfinite
            finite = torch.isfinite(gn)
            self.nonfinite.add_((~finite).long())
            for p in self.pi.parameters():
                if p.grad is not None:
                    p.grad.copy_(torch.where(finite, p.grad, 0.0))
            if isinstance(self.opt_pi, torch.optim.Optimizer) and not self.opt_pi.defaults.get("fused"):
                self.opt_pi.step()
            elif isinstance(self.opt_pi, torch.optim.Optimizer):
                self.opt_pi.found_inf = (~finite).float()  # the fused kernel skips the step and undoes the step count
                try:
                    self.opt_pi.step()
                finally:
                    self.opt_pi.found_inf = None
            else:
                self.opt_pi.step(ok=finite)
        elif torch.isfinite(gn):
            self.opt_pi.step()
        elif self.check_finite:
            self.opt_pi.zero_grad(set_to_none=True)
        if self.pi_avg is not None:
            if self.cfg.cuda_graph and int(self.pi_avg.n_averaged) > 0:
                self._ema_update()
            else:
                self.pi_avg.update_parameters(self.pi)
        return gn.detach()

    def _fit_critic(self, X, targets):
        v_loss = torch.zeros(())
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        for _ in range(self.cfg.critic_epochs):
            perm = torch.randperm(n, device=X.device)
            if self.keep_partial_minibatches:
                batches = perm.tensor_split(min(n, self.cfg.critic_minibatches))
            else:
                mb = n // self.cfg.critic_minibatches
                batches = (perm[i * mb:(i + 1) * mb] for i in range(self.cfg.critic_minibatches))
            for idx in batches:
                v_loss = self.critic_loss(Xf[idx], Yf[idx])
                if self.check_finite and not torch.isfinite(v_loss):
                    raise FloatingPointError("Nonfinite critic loss")
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0, error_if_nonfinite=self.check_finite)
                self.opt_v.step()
        return v_loss.detach()

    def update(self, buffer):
        s, h = self._sample_valid_starts(buffer, self.cfg.pi_batch)
        S = unflatten_state(s["s"])
        xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
        xi_n = self.normalize_context(xi)
        if self.stateful:  # the observations the policy actually saw, and the sensing state after the last one
            o_hist, st = h["o"], self.pipe.unpack(s["obs_state"])
        else:
            with torch.no_grad():
                o_hist = self.mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
            o_hist = self.mdp.add_noise(o_hist)
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            B = self.cfg.pi_batch
            g0 = self.motion.goal(f, self.cfg.goal_frames)
            g = g0
            S0 = S
            Ss, As, Aps, Fs, Gs = [], [], [], [], []
            clip_end = self.motion.clip_end[self.motion.clip_id[f]] if self.bounded_starts else None
            for _ in range(self.cfg.horizon):
                Aps.append(a_prev)
                if self.stateful:
                    S, a_prev, o_hist, f, g, st = self.imag_step_sensed(S, o_hist, a_prev, f, xi, xi_n, g, st)
                else:
                    S, a_prev, o_hist, f, g = self.imag_step(S, o_hist, a_prev, f, xi, xi_n, g, clip_end)
                Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
            flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
            S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
            f_all = flat(Fs)
            r, d = self.reward_term(S_all, f_all, flat(As), flat(Aps))
            r, d = r.view(self.cfg.horizon, B), d.view(self.cfg.horizon, B)
            if clip_end is not None:
                prior_frames = torch.stack([s["f"], *Fs[:-1]])
                d = torch.where(prior_frames >= clip_end, 1.0, d)
            v_boot = self.critic(self.bootstrap_input(S, f, xi_n, g))
            # --cuda_graph: survival products as a loop (cumprod's backward reads the host)
            J, c_t, c = actor_objective(r, d, v_boot, self.cfg.gamma, loop_cumprod=self.cfg.cuda_graph)
            r_sum = (c_t * r).detach().sum(0)
            rews, dones = r.detach(), d.detach()
            loss = -J.mean()
            gn = self._update_actor(loss)
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(self.cfg.horizon + 1, -1, -1).flatten(0, 1)
            X = self.critic_input(S_c, f_c, xi_c, g_c).view(self.cfg.horizon + 1, B, -1)
        self.critic.norm.update(X)
        with torch.no_grad():
            targets = self.gae_targets(self.critic(X), rews, dones)  # (H, B)
        v_loss = self._fit_critic(X, targets)
        return loss.detach(), v_loss, (r_sum / self.cfg.horizon / 0.1).mean(), gn, c.mean().detach()
