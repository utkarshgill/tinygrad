import numpy as np
import math, contextlib, random, os, multiprocessing as _mp, types
import time  # added for profiling utilities

# -----------------------------------------------------------------------------
# tinygrad import (after env vars are set)
# -----------------------------------------------------------------------------
os.environ.setdefault('DEVICE', 'CPU')  # force CPU to eliminate GPU⇄CPU overhead
from tinygrad import Tensor, nn, TinyJit
import tinygrad.nn.optim as _tng_optim
from tinygrad.dtype import dtypes

# -----------------------------------------------------------------------------
# Global setup: run everything in float16 for ~2× speed on GPU/CPU
# -----------------------------------------------------------------------------
Tensor.default_dtype = dtypes.float16

# tinygrad optimizers require training flag
Tensor.training = True

# -----------------------------------------------------------------------------
# tinygrad-only helper utilities (replaces the previous PyTorch shim)
# -----------------------------------------------------------------------------

def as_tensor(x, dtype=None):
    if dtype is not None:
        x = np.asarray(x, dtype=dtype)
    return x if isinstance(x, Tensor) else Tensor(x)

def ones_like(t):            return Tensor.ones(*t.shape)
def zeros_like(t):           return Tensor.zeros(*t.shape)

Tensor.size = lambda self, dim=None: (self.shape if dim is None else self.shape[dim])  # type: ignore
Tensor.dim  = lambda self: self.ndim                                                   # type: ignore
Tensor.cpu  = lambda self: self                                                        # type: ignore

class F:
    relu = staticmethod(lambda x: x.relu())
    @staticmethod
    def smooth_l1_loss(input, target, reduction='none'):
        diff = (input - target).abs()
        mask = (diff < 1)
        loss = mask.where(0.5 * diff * diff, diff - 0.5)
        if reduction == 'none': return loss
        if reduction == 'mean': return loss.mean()
        if reduction == 'sum' : return loss.sum()
        raise ValueError(f'Unsupported reduction {reduction}')

# tinygrad currently lacks a trivial Identity layer
class _Identity:
    def __call__(self, x): return x
nn.Identity = _Identity

def _orthogonal_(tensor, gain=1.0):
    tensor.assign(Tensor.glorot_uniform(*tensor.shape) * gain)
    return tensor

def _constant_(tensor, val):
    tensor.assign(Tensor.ones(*tensor.shape) * val)
    return tensor

nn.init = types.SimpleNamespace(orthogonal_=_orthogonal_, constant_=_constant_)

def _wrap_optimizer(opt, params, lr_val):
    class _ParamGroup(dict):
        def __setitem__(self, key, value):
            if key == 'lr' and hasattr(opt, 'lr'):
                opt.lr.assign([value])
            super().__setitem__(key, value)
    pg = _ParamGroup({'params': params, 'lr': float(lr_val)})
    opt.param_groups = [pg]
    return opt

# Use fused optimizers by default for significant speed-ups on tinygrad.
# Users can override by passing fused=False in kwargs.
def Adam(params, lr=0.001, fused=True, **kw):
    return _wrap_optimizer(_tng_optim.Adam(params, lr=lr, fused=fused, **kw), params, lr)

optim = types.SimpleNamespace(Adam=Adam)

class MultivariateNormal:
    def __init__(self, mean: Tensor, std: Tensor):
        self.mean, self.std = mean, std
        self.var = self.std * self.std
        self._log_coeff = -0.5 * math.log(2 * math.pi)
    def sample(self):
        return self.mean + Tensor.randn(*self.mean.shape) * self.std
    def log_prob(self, value):
        diff = value - self.mean
        return ((-0.5 * diff * diff / self.var) - self.std.log() + self._log_coeff).sum(-1)
    def entropy(self):
        return (0.5 + 0.5 * math.log(2 * math.pi) + self.std.log()).sum(-1)

no_grad = contextlib.nullcontext

# Hyperparameters
env_name = 'LunarLanderContinuous-v2'
state_dim = 8
action_dim = 2
# -----------------------------------------------------------------------------
# Parallel environment count – default to 64 to better saturate the GPU.  You
# can still override via the NUM_ENVS environment variable at runtime.
# -----------------------------------------------------------------------------
num_envs = int(os.getenv('NUM_ENVS', 64))
max_episodes = 5000
max_timesteps = 1000
# NOTE: update_timestep is recomputed inside `train()`, but keep this placeholder
# somewhat consistent with the runtime value for clarity.
update_timestep = 32768
log_interval = 20
hidden_dim = 256
lr_actor = 1e-4  # tuned actor learning rate
lr_critic = 4e-4  # slightly lower critic learning rate for stability
gamma = 0.99
K_epochs = 2  # fewer optimisation epochs per update (larger batch)
eps_clip = 0.2  # wider clip range for two-epoch regime
action_std = 0.8  # initial std for Gaussian policy
gae_lambda = 0.97
ppo_loss_coef = 1.0
critic_loss_coef = 1.0
entropy_coef = 0.03  # starting entropy coefficient (will decay)
batch_size = 1024

# -----------------------------------------------------------------------------
# Configuration flags controlled via environment variables
#   PLOT=0   -> disable matplotlib plotting (useful in headless environments)
#   RENDER=0 -> disable Gym rendering
# -----------------------------------------------------------------------------
PLOT = bool(int(os.getenv('PLOT', '0')))
RENDER = bool(int(os.getenv('RENDER', '0')))

# Conditional import to avoid issues on machines without display back-end
if PLOT:
    import matplotlib.pyplot as plt

# How often (episodes) to run a human-rendered evaluation when RENDER=1
eval_interval = int(os.getenv('EVAL_INTERVAL', 200))

# Entropy decay schedule
entropy_coef_final = 0.005  # keep small entropy floor to prevent premature convergence
entropy_decay_steps = 500_000  # faster entropy annealing

# -----------------------------------------------------------------------------
# Reward normalisation can be disabled to keep absolute reward scale
# Set NORMALIZE_REWARDS=1 to enable original RunningNorm behaviour
# -----------------------------------------------------------------------------
NORMALIZE_REWARDS = bool(int(os.getenv('NORMALIZE_REWARDS', '0')))

# seed & determinism helpers

class ActorCritic:
    def __init__(self, state_dim, action_dim, hidden_dim):
        # Shared feature extractor
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor head
        self.actor_out = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic_out = nn.Linear(hidden_dim, 1)

        # Orthogonal weight initialisation (recommended for PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Last layer smaller init
        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_out.weight, gain=1.0)

        # JIT-compiled inference for fast actor/critic evaluation (no grads)
        self.forward_jit = TinyJit(self.forward)

        # ---- cached NumPy parameters for ultra-fast rollout inference ----
        self._np_cache_version = -1  # to trigger initial sync

    def forward(self, state):
        # Actor network forward pass (used for training path – keeps grads)
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        action_mean = self.actor_out(x).tanh()
        value = self.critic_out(x)
        return action_mean, value

    def modules(self):
        return [self.shared_fc1, self.shared_fc2, self.actor_out, self.critic_out]

    def __call__(self, *args, **kw):
        return self.forward(*args, **kw)

    def named_parameters(self):
        from tinygrad.nn.state import get_state_dict
        return get_state_dict(self).items()

    # --- tinygrad model persistence helpers ---
    def state_dict(self):
        from tinygrad.nn.state import get_state_dict
        return get_state_dict(self)

    def load_state_dict(self, state_dict):
        from tinygrad.nn.state import load_state_dict as _lsd
        _lsd(self, state_dict)

    # mimic torch's .eval() for compatibility – no-op in tinygrad
    def eval(self):
        return self

    # --- fast NumPy-only forward (no tinygrad) ---------------------------------
    def _sync_numpy_params(self):
        """Refresh cached NumPy weights whenever tinygrad params update."""
        # Use a simple counter based on global PPO step to know when to refresh. Caller passes in a version.
        self.W1 = self.shared_fc1.weight.detach().to("CPU").numpy().astype(np.float32)
        self.b1 = self.shared_fc1.bias.detach().to("CPU").numpy().astype(np.float32)
        self.W2 = self.shared_fc2.weight.detach().to("CPU").numpy().astype(np.float32)
        self.b2 = self.shared_fc2.bias.detach().to("CPU").numpy().astype(np.float32)
        self.Wa = self.actor_out.weight.detach().to("CPU").numpy().astype(np.float32)
        self.ba = self.actor_out.bias.detach().to("CPU").numpy().astype(np.float32)
        self.Wc = self.critic_out.weight.detach().to("CPU").numpy().astype(np.float32)
        self.bc = self.critic_out.bias.detach().to("CPU").numpy().astype(np.float32)

    def forward_np(self, state_np: np.ndarray, version:int):
        """Fast NumPy forward. Caller ensures state_np is 2-D (batch, state_dim)."""
        if version != self._np_cache_version:
            self._sync_numpy_params(); self._np_cache_version = version

        x = state_np @ self.W1.T + self.b1
        x = np.maximum(x, 0)
        x = x @ self.W2.T + self.b2
        x = np.maximum(x, 0)
        action_mean = np.tanh(x @ self.Wa.T + self.ba)
        value = x @ self.Wc.T + self.bc
        return action_mean.astype(np.float32), value.astype(np.float32)

class Memory:
    """Lightweight replay buffer using Python lists to avoid tinygrad slice assignments."""

    def __init__(self, max_steps:int, state_dim:int, action_dim:int):
        # Store raw NumPy arrays to minimise tinygrad overhead
        self.states   : list[np.ndarray] = []
        self.actions  : list[np.ndarray] = []
        self.logps    : list[np.ndarray] = []
        self.rewards  : list[np.ndarray] = []
        self.terminals: list[np.ndarray] = []
        self.capacity = max_steps  # fixed rollout length for padding

    # ---- store API ----
    def store(self, state:Tensor, action:Tensor, logp:Tensor):
        if state.ndim == 1:
            state = state.unsqueeze(0); action = action.unsqueeze(0); logp = logp.unsqueeze(0)
        # Move to CPU before numpy() to avoid GPU realise cost on every call
        self.states.append(np.atleast_2d(state.detach().to("CPU").numpy()))
        self.actions.append(np.atleast_2d(action.detach().to("CPU").numpy()))
        self.logps.append(np.atleast_1d(logp.detach().to("CPU").numpy()))

    def store_reward(self, reward, term):
        # ensure 1-D tensors
        self.rewards.append(np.atleast_1d(np.asarray(reward, dtype=np.float32)))
        self.terminals.append(np.atleast_1d(np.asarray(term, dtype=np.float32)))

    # ---- rollout access ----
    def slice(self):
        states_np  = np.concatenate(self.states, axis=0)
        actions_np = np.concatenate(self.actions, axis=0)
        logps_np   = np.concatenate(self.logps, axis=0)
        rewards_np = np.concatenate(self.rewards, axis=0)
        terms_np   = np.concatenate(self.terminals, axis=0)

        # Flatten step/env dimensions
        if states_np.ndim == 3:
            states_np = states_np.reshape(-1, states_np.shape[-1])
        if actions_np.ndim == 3:
            actions_np = actions_np.reshape(-1, actions_np.shape[-1])
        logps_np = logps_np.reshape(-1)
        rewards_np = rewards_np.reshape(-1)
        terms_np = terms_np.reshape(-1)

        # ---- pad to fixed capacity to keep tensor shapes constant ----
        pad = self.capacity - states_np.shape[0]
        if pad > 0:
            states_np  = np.pad(states_np,  ((0,pad),(0,0)), 'constant')
            actions_np = np.pad(actions_np, ((0,pad),(0,0)), 'constant')
            logps_np   = np.pad(logps_np,   (0,pad), 'constant')
            rewards_np = np.pad(rewards_np, (0,pad), 'constant')
            terms_np   = np.pad(terms_np,   (0,pad), 'constant')

        states  = as_tensor(states_np.astype(np.float16))
        actions = as_tensor(actions_np)
        logps   = as_tensor(logps_np)
        rewards = as_tensor(rewards_np)
        terms   = as_tensor(terms_np)
        return states, actions, logps, rewards, terms

    def clear_memory(self):
        self.states.clear(); self.actions.clear(); self.logps.clear();
        self.rewards.clear(); self.terminals.clear()

    # --- new: store data already in NumPy to bypass Tensor conversions ---
    def store_np(self, state_np: np.ndarray, action_np: np.ndarray, logp_np: np.ndarray):
        """Fast path for when rollout collection works entirely in NumPy."""
        if state_np.ndim == 1:
            state_np  = np.expand_dims(state_np, 0)
            action_np = np.expand_dims(action_np, 0)
            logp_np   = np.expand_dims(logp_np, 0)
        self.states.append(np.atleast_2d(state_np.astype(np.float16)))
        self.actions.append(np.atleast_2d(action_np))
        self.logps.append(np.atleast_1d(logp_np))

class RunningNorm:
    """Track running mean & variance to normalise rewards."""

    def __init__(self, eps: float = 1e-6):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        import numpy as _np
        x_arr = _np.asarray(x)
        batch_mean = float(x_arr.mean())
        batch_var = float(x_arr.var())
        batch_count = x_arr.size

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta * delta * self.count * batch_count / tot_count) / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x):
        import numpy as _np
        return (x - self.mean) / (_np.sqrt(self.var) + 1e-8)

class PPO:
    def __init__(self, actor_critic, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, action_std, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size, entropy_coef_final=0.005, entropy_decay_steps=1_000_000):
        self.actor_critic = actor_critic

        # Single fused optimizer over all trainable parameters
        self.optimizer = optim.Adam([p for _, p in actor_critic.named_parameters()], lr=lr_actor)
        self.gamma = gamma
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.action_std_init = action_std  # initial std
        self.min_action_std = 0.005        # very low final exploration for deterministic control
        self.action_std_decay_steps = 1_500_000  # gradual annealing over longer horizon
        # Target final actor LR when exploration is minimal
        self.actor_lr_final = 5e-6
        self.action_std = action_std       # current std (mutable)
        self.ppo_loss_coef = ppo_loss_coef
        self.critic_loss_coef = critic_loss_coef
        # Entropy coefficient schedule
        self.entropy_start = entropy_coef
        self.entropy_final = entropy_coef_final
        self.entropy_decay_steps = entropy_decay_steps
        self.current_entropy_coef = entropy_coef  # will be updated over time
        self.batch_size = batch_size
        self.total_steps = 0
        # Store initial LR for adaptive schedule
        self.lr_init = lr_actor
        self.param_version = 0  # incremented after each update so forward_np knows when to refresh weights

        # ---- JIT compile the update step to fuse backwards/optimizer kernels ----
        self.update_jit = TinyJit(self.update)

        # Pre-sampled Gaussian noise pool to avoid per-step RNG kernel launches
        self._rng_pool = Tensor.randn(8192, action_dim)
        self._rng_pos = 0

    def select_action(self, state, memory, deterministic_mask=None):
        """Select an action for a *batch* of states.

        The input ``state`` is assumed to follow the Gymnasium vector API:
        - For a single environment it is a 1-D array of shape ``(state_dim,)``.
        - For *N* parallel envs it is a 2-D array of shape ``(N, state_dim)``.
        The function returns a numpy array of the same batch size with shape
        ``(batch, action_dim)``.
        """

        # ---- convert incoming state to numpy array (fast path) ----
        state_np = np.asarray(state, dtype=np.float32)
        if state_np.ndim == 1:
            state_np = np.expand_dims(state_np, 0)
            single_env = True
        else:
            single_env = False
        batch_size = state_np.shape[0]

        if deterministic_mask is None:
            det_mask_np = np.zeros(batch_size, dtype=bool)
        else:
            det_mask_np = np.asarray(deterministic_mask, dtype=bool)

        # Increment global step counter and linearly decay exploration noise
        self.total_steps += batch_size
        if self.total_steps < self.action_std_decay_steps:
            frac = self.total_steps / self.action_std_decay_steps
            self.action_std = self.action_std_init - (self.action_std_init - self.min_action_std) * frac
        else:
            self.action_std = self.min_action_std

        # Dynamically scale actor learning-rate as exploration noise decays
        frac_lr = min(1.0, self.total_steps / self.action_std_decay_steps)
        new_lr = self.lr_init - (self.lr_init - self.actor_lr_final) * frac_lr
        for g in self.optimizer.param_groups:
            # Skip LR decay once sprint phase triggered
            if not hasattr(self, "sprint_phase"):
                g["lr"] = new_lr

        frac_ent = min(1.0, self.total_steps / self.entropy_decay_steps)
        self.current_entropy_coef = self.entropy_start - (self.entropy_start - self.entropy_final) * frac_ent

        # --- forward pass to get action_mean via cached NumPy weights ---
        action_mean_np, _ = self.actor_critic.forward_np(state_np, version=self.param_version)

        # NumPy noise sampling (fast) using instance RNG pool not required here
        noise_np = np.random.randn(*action_mean_np.shape).astype(np.float32)
        action_np = action_mean_np + noise_np * self.action_std

        # deterministic override
        action_np = np.where(det_mask_np[:, None], action_mean_np, action_np)

        var = self.action_std ** 2
        log_coeff = -0.5 * math.log(2 * math.pi)
        logp_np = ((-0.5 * noise_np**2 / var) - 0.5 * math.log(var) + log_coeff).sum(-1)

        memory.store_np(state_np.astype(np.float16), action_np, logp_np)

        if single_env:
            return action_np[0]
        return action_np

    def compute_advantages(self, rewards, state_values, is_terminals):
        """GAE computed fully inside tinygrad to avoid CPU⇄GPU transfers.

        Args:
            rewards: (T,N) or (T,) tensor of rewards.
            state_values: matching tensor of value estimates.
            is_terminals: binary mask indicating episode termination.
        Returns:
            advantages_flat, returns_flat (both 1-D)
        """

        # Ensure 2-D shape (T,N)
        if rewards.dim() == 1:
            rewards      = rewards.unsqueeze(1)
            state_values = state_values.unsqueeze(1)
            is_terminals = is_terminals.unsqueeze(1)

        T, N = rewards.shape
        # --- CPU path: fast NumPy loop avoids GPU compile overhead ---
        import numpy as _np

        r     = rewards.detach().numpy()
        sv    = state_values.detach().numpy()
        terms = is_terminals.detach().numpy()

        adv = _np.zeros_like(r, dtype=_np.float32)
        last_gae = _np.zeros((N,), dtype=_np.float32)
        for t in range(T - 1, -1, -1):
            nonterm = 1.0 - terms[t]
            next_val = sv[t+1] if t+1 < T else 0.0
            delta = r[t] + self.gamma * next_val * nonterm - sv[t]
            last_gae = delta + self.gamma * self.lamda * nonterm * last_gae
            adv[t] = last_gae

        ret = adv + sv

        # Normalise advantages per-batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        advantages = as_tensor(adv.astype(_np.float32))
        returns     = as_tensor(ret.astype(_np.float32))
        return advantages.reshape(-1), returns.reshape(-1)
    
    def update(self, memory):
        with no_grad():
            old_states, old_actions, old_logprobs, rewards, is_terms = memory.slice()

            _, old_state_values = self.actor_critic(old_states)
            old_state_values = old_state_values.squeeze()  # (T*N,)

            # Reshape to (T, N) for advantage computation if needed
            if rewards.dim() > 1:
                N = rewards.size(1)
                old_state_values_reshaped = old_state_values.view(-1, N)
            else:
                old_state_values_reshaped = old_state_values

            advantages, returns = self.compute_advantages(rewards, old_state_values_reshaped, is_terms)
        
        total = old_states.shape[0]
        for _ in range(self.K_epochs):
            # Forward pass on *entire* rollout to avoid per-mini-batch kernel launches
            action_means, state_values = self.actor_critic(old_states)  # (T*N, action_dim) and (T*N,1)
            std_tensor = ones_like(action_means) * self.action_std
            dist = MultivariateNormal(action_means, std_tensor)
            action_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            # Critic loss
            diff = state_values.squeeze() - returns.detach()
            critic_loss = diff.square().mean()

            # Actor loss (clipped surrogate)
            ratios = (action_logprobs - old_logprobs.detach()).exp()
            surr1 = ratios * advantages
            surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -((surr1 < surr2).where(surr1, surr2)).mean()
            entropy_loss = dist_entropy.mean()

            # Joint optimisation
            self.optimizer.zero_grad()

            total_loss = (self.ppo_loss_coef * actor_loss +
                          self.critic_loss_coef * critic_loss -
                          self.current_entropy_coef * entropy_loss)

            total_loss.backward()
            clip_grad_norm_(self.optimizer, 0.25)
            self.optimizer.step()

            # KL-based adaptive LR and early stop
            approx_kl = (old_logprobs - action_logprobs).abs().mean()
            kl_val = float(approx_kl.item())
            if kl_val > 0.03:
                for g in self.optimizer.param_groups: g["lr"] = max(g["lr"] * 0.9, 1e-5)
            elif kl_val < 0.015:
                for g in self.optimizer.param_groups: g["lr"] = min(g["lr"] * 1.1, self.lr_init)

            if self.total_steps > 300_000: self.eps_clip = 0.15
            if kl_val > 0.04: break

            # notify inference cache that parameters changed
            self.param_version += 1

    # -------------------------------------------------
    # Utility to freeze actor once task is solved
    # -------------------------------------------------
    def freeze_actor(self, lr: float = 1e-6):
        """Freeze actor by dropping its learning-rate and turning off exploration noise."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        # Make subsequent actions deterministic
        self.action_std = 0.0
        self.min_action_std = 0.0

    # -------------------------------------------------
    # Helper: efficient Gaussian noise sampler
    # -------------------------------------------------
    def _sample_noise(self, batch:int, action_dim:int):
        """Return (batch, action_dim) Tensor of N(0,1) noise using a pre-filled pool."""
        if self._rng_pos + batch > self._rng_pool.shape[0]:
            # replenish pool
            with no_grad():
                self._rng_pool = Tensor.randn(8192, action_dim)
            self._rng_pos = 0
        noise = self._rng_pool[self._rng_pos:self._rng_pos+batch]
        self._rng_pos += batch
        return noise

# -----------------------------------------------------------------------------
# Lightweight profiler to track where time is spent during training
# -----------------------------------------------------------------------------
class _SimpleProfiler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.env_step = 0.0      # time spent inside env.step / env.reset
        self.actor_fwd = 0.0     # time spent in policy forward pass & sampling
        self.ppo_update = 0.0    # time spent inside PPO.update

    def summary(self):
        total = self.env_step + self.actor_fwd + self.ppo_update + 1e-12  # avoid div-by-zero
        return (
            f"time% env {self.env_step/total*100:5.1f} | "
            f"actor {self.actor_fwd/total*100:5.1f} | "
            f"update {self.ppo_update/total*100:5.1f}  (wall {total:0.2f}s)"
        )

_prof = _SimpleProfiler()

def train(env_name, max_episodes, max_timesteps, _update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size, num_envs=1):
    """Train with single or parallel environments (vectorised when num_envs>1)."""
    timestep = 0
    # Seeding for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    # ensure training flag is on inside worker processes too
    Tensor.training = True

    # Dynamic update_timestep proportional to number of envs
    update_timestep = num_envs * 2048  # larger batch per PPO update (keeps shapes stable)

    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)

    ppo = PPO(actor_critic, lr_actor, lr_critic, gamma, gae_lambda, K_epochs, eps_clip, action_std, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size, entropy_coef_final, entropy_decay_steps)

    # Running reward normaliser (optional)
    if NORMALIZE_REWARDS:
        reward_norm = RunningNorm()

    memory = Memory(update_timestep, state_dim, action_dim)
    
    episode_returns = []
    running_avg_returns = []
    eval_scores = []  # deterministic evaluation scores after solving
    
    if PLOT:
        plt.ion()
        fig, ax = plt.subplots()
    
    # ------------------------------------------------------------------
    # Environment setup (vectorised if num_envs > 1)
    # ------------------------------------------------------------------
    try:
        from gymnasium import make_vec  # gymnasium>=1.0
        env = make_vec(env_name, num_envs=num_envs)
    except ImportError:
        from gymnasium.vector import make as _vec_make  # older gymnasium
        env = _vec_make(env_name, num_envs=num_envs, asynchronous=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    if num_envs == 1:
        # Retain the original (simpler) single-environment logic for clarity
        for episode in range(1, max_episodes + 1):
            state, _ = env.reset()
            total_reward = 0.0
            for _ in range(max_timesteps):
                timestep += 1

                # ---- policy forward & sampling ----
                _t0 = time.perf_counter()
                action = ppo.select_action(state, memory)
                _prof.actor_fwd += time.perf_counter() - _t0

                # ---- environment step ----
                _t1 = time.perf_counter()
                next_state, reward, done, trunc, _ = env.step(action)
                _prof.env_step += time.perf_counter() - _t1

                # Store reward (normalised if enabled)
                if NORMALIZE_REWARDS:
                    reward_norm.update(reward)
                    rew_to_store = reward_norm.normalize(reward)
                else:
                    rew_to_store = reward
                memory.store_reward(rew_to_store, float(done or trunc))
                total_reward += reward

                if timestep % update_timestep == 0:
                    _t2 = time.perf_counter()
                    ppo.update_jit(memory)
                    _prof.ppo_update += time.perf_counter() - _t2
                    memory.clear_memory(); timestep = 0

                state = next_state
                if done or trunc:
                    break

            episode_returns.append(total_reward)
            running_avg_returns.append(np.mean(episode_returns[-log_interval:]))

            if episode % log_interval == 0:
                print(f'ep {episode:6} return {total_reward:.2f}')
                print('        ', _prof.summary())
                _prof.reset()
                if PLOT:
                    ax.clear(); ax.plot(episode_returns, label='Returns');
                    ax.plot(running_avg_returns, label='Running Avg');
                    if eval_scores:
                        ax.plot(range(len(eval_scores)), eval_scores, label='Eval', color='green')
                    ax.legend(); ax.set_xlabel('Episode'); ax.set_ylabel('Return'); plt.pause(0.01)
                if RENDER and episode % eval_interval == 0:
                    eval_ret = evaluate_policy(env_name, actor_critic, max_timesteps, episodes=1, render=True)
                    print(f'            eval return {eval_ret:.2f}')
                    eval_scores.append(eval_ret)

            # Freeze actor once solved (>200 average over last 100 episodes)
            if len(episode_returns) >= 100 and not hasattr(ppo, "actor_frozen"):
                if np.mean(episode_returns[-100:]) > 180:  # quick precheck before expensive eval
                    det_eval = evaluate_policy(env_name, actor_critic, max_timesteps, episodes=5, render=False)
                    print(f"deterministic eval avg {det_eval:.1f}")
                    if det_eval >= 200:
                        ppo.freeze_actor()
                        ppo.actor_frozen = True
                        print("Actor frozen – task SOLVED; running one human-rendered evaluation...")
                        evaluate_policy(env_name, actor_critic, max_timesteps, episodes=1, render=True)
                        return

        if PLOT:
            plt.ioff(); plt.show(); return

    # ---------- Rollout collection loop (vectorised) -----------
    states, _ = env.reset()
    per_env_returns = np.zeros(num_envs)
    completed_episodes = 0

    while completed_episodes < max_episodes:
        timestep += num_envs

        # Use deterministic behaviour for the first half of envs
        det_mask = np.arange(num_envs) < (num_envs // 2)
        _t0 = time.perf_counter()
        actions = ppo.select_action(states, memory, deterministic_mask=det_mask)  # (num_envs, action_dim)
        _prof.actor_fwd += time.perf_counter() - _t0

        _t1 = time.perf_counter()
        next_states, rewards, terminated, truncated, _ = env.step(actions)
        _prof.env_step += time.perf_counter() - _t1

        # Store rewards (normalised if enabled)
        if NORMALIZE_REWARDS:
            reward_norm.update(rewards)
            rew_to_store = reward_norm.normalize(rewards)
        else:
            rew_to_store = rewards
        memory.store_reward(rew_to_store, np.logical_or(terminated, truncated).astype(float))

        per_env_returns += rewards

        # When any env finishes an episode
        done_mask = np.logical_or(terminated, truncated)
        if np.any(done_mask):
            # Log and bookkeeping per finished env
            for idx in np.where(done_mask)[0]:
                episode_returns.append(per_env_returns[idx])
                per_env_returns[idx] = 0.0
                completed_episodes += 1

                if completed_episodes % log_interval == 0:
                    print(f'ep {completed_episodes:6} return {episode_returns[-1]:.2f}')
                    print('        ', _prof.summary())
                    _prof.reset()

                    if PLOT:
                        ax.clear()
                        ax.plot(episode_returns, label='Returns')
                        running_avg = np.convolve(episode_returns, np.ones(log_interval)/log_interval, mode='valid')
                        ax.plot(range(log_interval - 1, len(episode_returns)), running_avg, label='Running Avg')
                        if eval_scores:
                            ax.plot(range(len(eval_scores)), eval_scores, label='Eval', color='green')
                        ax.legend(); ax.set_xlabel('Episode'); ax.set_ylabel('Return'); plt.pause(0.01)
                    if RENDER and completed_episodes % eval_interval == 0:
                        eval_ret = evaluate_policy(env_name, actor_critic, max_timesteps, episodes=1, render=True)
                        print(f'            eval return {eval_ret:.2f}')
                        eval_scores.append(eval_ret)

                    # Check solved condition in vectorised path
                    if len(episode_returns) >= 100 and not hasattr(ppo, "actor_frozen"):
                        if np.mean(episode_returns[-100:]) > 180:
                            det_eval = evaluate_policy(env_name, actor_critic, max_timesteps, episodes=5, render=False)
                            print(f"deterministic eval avg {det_eval:.1f}")
                            if det_eval >= 200:
                                ppo.freeze_actor()
                                ppo.actor_frozen = True
                                print("Actor frozen – task SOLVED; running one human-rendered evaluation...")
                                evaluate_policy(env_name, actor_critic, max_timesteps, episodes=1, render=True)
                                return

            # Reset finished environments
            if hasattr(env, 'reset_done'):
                # Gymnasium ≥0.28 provides this convenience method
                states_reset, _ = env.reset_done()
                next_states[done_mask] = states_reset
            else:
                # Fallback – reset *all* envs (simpler, but slightly wasteful)
                next_states, _ = env.reset()

        # Time to update PPO?
        if timestep % update_timestep == 0:
            _t2 = time.perf_counter()
            ppo.update_jit(memory)
            _prof.ppo_update += time.perf_counter() - _t2
            memory.clear_memory()
            timestep = 0

        states = next_states

    if PLOT:
        plt.ioff()
        plt.show()


# -----------------------------------------------------------------------------
# Utility for occasional evaluation with human-rendering
# -----------------------------------------------------------------------------

def _eval_worker(conn, state_dict, env_name, max_timesteps, render):
    """Runs *one* deterministic evaluation episode in a subprocess."""
    import gymnasium as gym
    from math import sqrt

    # Reconstruct model
    state_dim = 8
    action_dim = 2
    hidden_dim = 256
    model = ActorCritic(state_dim, action_dim, hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()

    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    state, _ = env.reset()
    total_reward = 0.0

    with no_grad():
        for _ in range(max_timesteps):
            action_mean, _ = model(as_tensor(state, dtype=np.float16).unsqueeze(0))
            state, reward, done, trunc, _ = env.step(action_mean.squeeze(0).numpy())
            total_reward += reward
            if done or trunc:
                break

    env.close()
    conn.send(total_reward)
    conn.close()


def evaluate_policy(env_name, actor_critic, max_timesteps, episodes: int = 5, render: bool = False):
    """Average return over a few deterministic episodes.

    Args:
        env_name: Gymnasium environment id.
        actor_critic: Trained model.
        max_timesteps: Per-episode time limit.
        episodes: Number of evaluation episodes.
        render: If True, human-render; otherwise off-screen.
    """

    state_dict = {k: v.cpu() for k, v in actor_critic.state_dict().items()}
    returns = []
    for _ in range(episodes):
        parent_conn, child_conn = _mp.Pipe()
        p = _mp.Process(target=_eval_worker, args=(child_conn, state_dict, env_name, max_timesteps, render))
        p.start()
        ret = parent_conn.recv(); p.join()
        returns.append(ret)
    return float(np.mean(returns))

# -----------------------------------------------------------------------------
# Minimal grad-norm clipping helper (tinygrad.nn.optim doesn't expose one yet)
# -----------------------------------------------------------------------------
def clip_grad_norm_(param_source, max_norm: float):
    """Scale gradients so that global L2 norm ≤ max_norm.

    Args:
        param_source: Either an Optimizer instance (with `.params` attr) or an
            iterable of `Tensor` parameters.
        max_norm: Maximum allowed global norm (float).
    """
    params = param_source.params if hasattr(param_source, "params") else list(param_source)
    if not params:
        return  # nothing to do

    total = Tensor.zeros(1)
    for p in params:
        if p.grad is not None:
            total += (p.grad * p.grad).sum()
    total = total.sqrt()

    coef = (max_norm / (total + 1e-6)).minimum(1.0)
    for p in params:
        if p.grad is not None:
            p.grad.assign(p.grad * coef)

if __name__ == '__main__':
    train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size, num_envs=num_envs)
