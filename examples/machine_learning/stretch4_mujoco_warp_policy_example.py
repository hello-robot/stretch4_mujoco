import os
import subprocess
import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
import optax
import warp as wp
import mujoco
import mujoco.viewer
import mediapy as media
from typing import Dict, Any, Tuple

try:
    import mujoco_warp as mjw
except ImportError:
    pass  # Allow import to fail if mujoco_warp is not installed, it will error later if needed.

def get_default_config() -> Dict[str, Any]:
    config = {
        "NWORLD": 16,
        "num_iterations": 20,
        "rollout_length": 48,
        "minibatch_size": 4096,
        "NJMAX": 513,
        "NCONMAX": 1024,
        "render_during_training": True,
        "render_freq": 2
    }
    # config = {
    #     "NWORLD": 1024,
    #     "num_iterations": 2000,
    #     "rollout_length": 128,
    #     "minibatch_size": 4096,
    #     "NJMAX": 513,
    #     "NCONMAX": 1024,
    #     "render_during_training": True,
    #     "render_freq": 2
    # }
    config["batch_size"] = config["NWORLD"] * config["rollout_length"]
    return config

def patch_robocasa():
    """
    Patches Robocasa to remove strict mujoco and numpy version checks
    so it can be used with newer versions compatible with mujoco_warp.
    """
    import os
    
    robocasa_init_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        "../../third_party/robocasa/robocasa/__init__.py"
    ))
    
    if not os.path.exists(robocasa_init_path):
        try:
            import importlib.util
            spec = importlib.util.find_spec("robocasa")
            if spec is not None and spec.origin is not None:
                robocasa_init_path = spec.origin
            else:
                return
        except Exception:
            return

    with open(robocasa_init_path, 'r') as f:
        lines = f.readlines()

    modified = False
    in_mujoco_assert = False
    in_numpy_assert = False

    for i in range(len(lines)):
        line = lines[i]
        
        # Check for un-commented asserts
        if line.strip() == 'assert (' and i + 1 < len(lines) and 'mujoco.__version__' in lines[i+1] and not line.startswith('#'):
            in_mujoco_assert = True
            
        if line.startswith('assert numpy.__version__ in ['):
            in_numpy_assert = True
            
        if in_mujoco_assert:
            lines[i] = '# ' + line
            modified = True
            if 'MuJoCo version must be' in line:
                in_mujoco_assert = False
                
        if in_numpy_assert:
            lines[i] = '# ' + line
            modified = True
            if 'numpy version must be' in line:
                in_numpy_assert = False

    if modified:
        try:
            with open(robocasa_init_path, 'w') as f:
                f.writelines(lines)
            print("Successfully patched robocasa version checks.")
        except Exception as e:
            print(f"Failed to write patch for robocasa: {e}")

# Run the patch before importing robocasa
patch_robocasa()

from stretch_mujoco.robocasa_gen import model_generation_wizard
from stretch_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

def setup_environment():
    """Sets up XLA flags and configures MuJoCo for GPU rendering."""
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5'

    # Set up GPU rendering.
    if subprocess.run('nvidia-smi', shell=True, capture_output=True).returncode != 0:
        print('Warning: Cannot communicate with GPU. nvidia-smi failed.')

    NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
        try:
            os.makedirs(os.path.dirname(NVIDIA_ICD_CONFIG_PATH), exist_ok=True)
            with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
                f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")
        except PermissionError:
            print("Warning: Could not write NVIDIA_ICD_CONFIG_PATH due to permissions. EGL might not work.")

    print('Configuring MuJoCo for GPU rendering, setting MUJOCO_GL=egl')
    os.environ['MUJOCO_GL'] = 'egl'
    
    wp.config.quiet = True
    wp.init()


def generate_robocasa_model(target_category: str = "apple") -> Tuple[mujoco.MjModel, str]:
    robot_xml_path = Stretch4MujocoSimulator.get_robot_xml_path()

    model, xml, objects_info = model_generation_wizard(
        stretch_xml_absolute=robot_xml_path,
        layout=1,
        style=1,
        task="PickPlaceCounterToCabinet",
        objects_list=[target_category],
    )

    print("Available objects:", objects_info)
    print(f"Available objects matching '{target_category}':")
    target_object_body_name = None
    for body_name, info in objects_info.items():
        if target_category in info["cat"]:
            print(f" - {body_name} (Category: {info['cat']})")
            if target_object_body_name is None:
                target_object_body_name = body_name

    if target_object_body_name is None:
        target_object_body_name = "obj_main"  # Fallback
    print(f"Selected target object for training: {target_object_body_name}")
    
    return model, target_object_body_name

def load_mujoco_warp_model(model: mujoco.MjModel, config: Dict[str, Any]):
    model.opt.noslip_iterations = 0
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

    warp_model = mjw.put_model(model)
    print("Model loaded successfully. Nu:", model.nu, "Nq:", model.nq, "Nv:", model.nv)
    print(f"Initializing {config['NWORLD']} environments for training.")
    
    data = mjw.make_data(model, nworld=config['NWORLD'], nconmax=config['NCONMAX'], njmax=config['NJMAX'])
    mjw.forward(warp_model, data)
    
    return warp_model, data

def sample_normal(rng, mean, log_std):
    std = jnp.exp(log_std)
    return mean + std * jax.random.normal(rng, mean.shape)

def log_prob_normal(x, mean, log_std):
    std = jnp.exp(log_std)
    var = std ** 2
    log_scale = log_std + 0.5 * jnp.log(2.0 * jnp.pi)
    return -((x - mean) ** 2) / (2.0 * var) - log_scale

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Actor
        a = nn.Dense(256)(x)
        a = nn.relu(a)
        a = nn.Dense(256)(a)
        a = nn.relu(a)
        actor_mean = nn.Dense(self.action_dim)(a)
        
        actor_log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        
        # Critic
        c = nn.Dense(256)(x)
        c = nn.relu(c)
        c = nn.Dense(256)(c)
        c = nn.relu(c)
        critic = nn.Dense(1)(c)
        
        return actor_mean, actor_log_std, jnp.squeeze(critic, -1)

class PPOTrainer:
    def __init__(self, mj_model, warp_model, warp_data, target_object_body_name, config):
        self.mj_model = mj_model
        self.warp_model = warp_model
        self.warp_data = warp_data
        self.target_object_body_name = target_object_body_name
        self.config = config
        
        self.initial_qpos = wp.to_jax(self.warp_data.qpos)
        self.initial_qvel = wp.to_jax(self.warp_data.qvel)

        # Precompute IDs
        self.robot_gripper_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "link_grasp_center")
        self.object_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.target_object_body_name)

        self._setup_jitted_functions()

    def _setup_jitted_functions(self):
        nu = self.mj_model.nu
        nq = self.mj_model.nq
        nv = self.mj_model.nv

        @jax.jit
        def get_action_and_value(params, x, rng):
            mean, log_std, value = ActorCritic(nu).apply(params, x)
            action = sample_normal(rng, mean, log_std)
            log_prob = log_prob_normal(action, mean, log_std).sum(-1)
            return action, log_prob, value

        @jax.jit
        def get_value(params, x):
            _, _, value = ActorCritic(nu).apply(params, x)
            return value

        @jax.jit
        def compute_gae(rewards, values, next_value, dones, gamma=0.99, gae_lambda=0.95):
            advantages = jnp.zeros_like(rewards)
            lastgaelam = jnp.zeros_like(rewards[0])
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                advantages = advantages.at[t].set(lastgaelam)
            returns = advantages + values
            return advantages, returns

        import functools
        @functools.partial(jax.jit, static_argnames=['tx'])
        def update_ppo(params, opt_state, tx, obs, actions, log_probs_old, returns, advantages):
            def loss_fn(p):
                mean, log_std, value = ActorCritic(nu).apply(p, obs)
                log_probs = log_prob_normal(actions, mean, log_std).sum(-1)
                
                ratio = jnp.exp(log_probs - log_probs_old)
                surr1 = ratio * advantages
                surr2 = jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
                
                actor_loss = -jnp.minimum(surr1, surr2).mean()
                critic_loss = jnp.mean((returns - value) ** 2)
                entropy_loss = jnp.mean(log_std + 0.5 + 0.5 * jnp.log(2 * jnp.pi))
                
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
                return total_loss, (actor_loss, critic_loss, entropy_loss)
            
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (al, cl, el)), grads = grad_fn(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state, loss

        def compute_rewards(obs):
            alive_bonus = 1.0
            vel_penalty = -0.01 * jnp.sum(jnp.square(obs[:, nq:nq+nv]), axis=-1)
            
            gripper_pos = obs[:, -6:-3]
            object_pos = obs[:, -3:]
            
            dist = jnp.linalg.norm(gripper_pos - object_pos, axis=-1)
            dist_reward = -10.0 * dist
            
            return alive_bonus + vel_penalty + dist_reward

        self.get_action_and_value = get_action_and_value
        self.get_value = get_value
        self.compute_gae = compute_gae
        self.update_ppo = update_ppo
        self.compute_rewards = compute_rewards

    def get_obs_from_warp(self):
        qpos = wp.to_jax(self.warp_data.qpos)
        qvel = wp.to_jax(self.warp_data.qvel)
        xpos = wp.to_jax(self.warp_data.xpos)
        
        gripper_pos = xpos[:, self.robot_gripper_id, :]
        object_pos = xpos[:, self.object_id, :]
        
        return jnp.concatenate([qpos, qvel, gripper_pos, object_pos], axis=-1)

    def env_step(self, jax_actions, rng):
        wp_actions = wp.from_jax(jax_actions, dtype=wp.float32)
        self.warp_data.ctrl = wp_actions
        for _ in range(4):
            mjw.step(self.warp_model, self.warp_data)
            
        rng, reset_rng = jax.random.split(rng)
        resets = jax.random.bernoulli(reset_rng, p=0.01, shape=(self.config["NWORLD"],))
        
        current_qpos = wp.to_jax(self.warp_data.qpos)
        nan_mask = jnp.isnan(current_qpos).any(axis=-1)
        final_resets = jnp.logical_or(resets, nan_mask)
        
        if np.any(np.asarray(final_resets)):
            current_qvel = wp.to_jax(self.warp_data.qvel)
            reset_mask_exp = jnp.expand_dims(final_resets, axis=-1)
            
            new_qpos = jnp.where(reset_mask_exp, self.initial_qpos, current_qpos)
            new_qvel = jnp.where(reset_mask_exp, self.initial_qvel, current_qvel)
            
            self.warp_data.qpos = wp.from_jax(new_qpos, dtype=wp.float32)
            self.warp_data.qvel = wp.from_jax(new_qvel, dtype=wp.float32)
            
            mjw.forward(self.warp_model, self.warp_data)
            
        obs = self.get_obs_from_warp()
        rewards = self.compute_rewards(obs)
        
        return obs, rewards, final_resets, rng

def run_training(trainer: PPOTrainer):
    import IPython.display
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((trainer.config["NWORLD"], trainer.mj_model.nq + trainer.mj_model.nv + 6))
    params = ActorCritic(trainer.mj_model.nu).init(init_rng, dummy_obs)

    tx = optax.adam(learning_rate=3e-4)
    opt_state = tx.init(params)

    print("Starting training...")
    obs = trainer.get_obs_from_warp()

    if trainer.config["render_during_training"]:
        rc = mjw.create_render_context(
            trainer.mj_model,
            nworld=trainer.config["NWORLD"],
            cam_res=(160, 120),
            render_rgb=True,
            render_depth=False,
            render_seg=False,
        )
        cam_id = mujoco.mj_name2id(trainer.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
        if cam_id == -1: cam_id = -1
        
        num_render_worlds = min(trainer.config["NWORLD"], 16)
        grid_w = int(np.ceil(np.sqrt(num_render_worlds)))
        grid_h = int(np.ceil(num_render_worlds / grid_w))

    training_frames = []
    
    for i in range(trainer.config["num_iterations"]):
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        
        for step in range(trainer.config["rollout_length"]):
            rng, action_rng = jax.random.split(rng)
            actions, log_probs, values = trainer.get_action_and_value(params, obs, action_rng)
            
            next_obs, rewards, dones, rng = trainer.env_step(actions, rng)
            
            all_obs.append(obs)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_dones.append(dones)
            all_log_probs.append(log_probs)
            all_values.append(values)
            
            obs = next_obs
            if trainer.config["render_during_training"] and i % trainer.config["render_freq"] == 0 and step % 2 == 0:
                mjw.refit_bvh(trainer.warp_model, trainer.warp_data, rc)
                mjw.render(trainer.warp_model, trainer.warp_data, rc)
                rgb_data = wp.zeros((trainer.config["NWORLD"], 120, 160), dtype=wp.vec3)
                mjw.get_rgb(rc, camera_index=cam_id, rgb_out=rgb_data)
                
                rgb_np = rgb_data.numpy()[:num_render_worlds]
                if num_render_worlds < grid_w * grid_h:
                    padding = np.zeros((grid_w * grid_h - num_render_worlds, 120, 160, 3), dtype=np.float32)
                    rgb_np = np.concatenate([rgb_np, padding], axis=0)
                    
                rgb_grid = rgb_np.reshape(grid_h, grid_w, 120, 160, 3)
                rgb_grid = rgb_grid.transpose(0, 2, 1, 3, 4)
                rgb_grid = rgb_grid.reshape(grid_h * 120, grid_w * 160, 3)
                
                if rgb_grid.dtype != np.uint8:
                    if rgb_grid.max() <= 1.0:
                        rgb_grid = (rgb_grid * 255).astype(np.uint8)
                    else:
                        rgb_grid = rgb_grid.astype(np.uint8)
                training_frames.append(rgb_grid)
            
        all_obs = jnp.stack(all_obs)
        all_actions = jnp.stack(all_actions)
        all_rewards = jnp.stack(all_rewards)
        all_dones = jnp.stack(all_dones)
        all_log_probs = jnp.stack(all_log_probs)
        all_values = jnp.stack(all_values)
        
        next_value = trainer.get_value(params, obs)
        advantages, returns = trainer.compute_gae(all_rewards, all_values, next_value, all_dones)
        
        b_obs = all_obs.reshape(-1, all_obs.shape[-1])
        b_actions = all_actions.reshape(-1, all_actions.shape[-1])
        b_log_probs = all_log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        params, opt_state, loss = trainer.update_ppo(
            params, opt_state, tx, b_obs, b_actions, b_log_probs, b_returns, b_advantages
        )
        
        if trainer.config["render_during_training"] and i % trainer.config["render_freq"] == 0:
            IPython.display.clear_output(wait=True)
            print(f"Iteration {i}, Mean Reward: {all_rewards.mean():.4f}, Loss: {loss:.4f}")
            media.show_video(training_frames, fps=15)
        else:
            print(f"Iteration {i}, Mean Reward: {all_rewards.mean():.4f}, Loss: {loss:.4f}")
            
    print("Training finished!")
    return params, training_frames

def save_params(params, filename_prefix="trained_ppo_params"):
    bytes_output = serialization.to_bytes(params)
    filename = f"{filename_prefix}_{int(time.time())}.msgpack"
    with open(filename, "wb") as f:
        f.write(bytes_output)
    print(f"Saved trained model parameters to {filename}")
    return filename

def load_params_from_file(filename, dummy_params):
    with open(filename, "rb") as f:
        bytes_input = f.read()
    loaded_params = serialization.from_bytes(dummy_params, bytes_input)
    return loaded_params

def run_viewer(mj_model, target_object_body_name, loaded_params):
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)

    print("Launching MuJoCo Viewer...")
    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            for _ in range(10000):
                if not viewer.is_running():
                    break
                    
                qpos = jnp.array(mj_data.qpos)
                qvel = jnp.array(mj_data.qvel)
                robot_gripper_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link_grasp_center")
                object_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, target_object_body_name)
                gripper_pos = jnp.array(mj_data.xpos[robot_gripper_id])
                object_pos = jnp.array(mj_data.xpos[object_id])
                obs = jnp.concatenate([qpos, qvel, gripper_pos, object_pos], axis=-1)
                
                action_mean, _, _ = ActorCritic(mj_model.nu).apply(loaded_params, obs)
                
                mj_data.ctrl[:] = np.array(action_mean)
                mujoco.mj_step(mj_model, mj_data)
                
                viewer.sync()
                time.sleep(mj_model.opt.timestep)
    except Exception as e:
        print("Viewer closed or error:", e)

def render_video(mj_model, target_object_body_name, loaded_params, num_frames=500):
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)

    renderer = mujoco.Renderer(mj_model, 480, 640)

    print("Simulating and rendering...")
    frames = []
    for step in range(num_frames):
        qpos = jnp.array(mj_data.qpos)
        qvel = jnp.array(mj_data.qvel)
        robot_gripper_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link_grasp_center")
        object_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, target_object_body_name)
        gripper_pos = jnp.array(mj_data.xpos[robot_gripper_id])
        object_pos = jnp.array(mj_data.xpos[object_id])
        obs = jnp.concatenate([qpos, qvel, gripper_pos, object_pos], axis=-1)
        
        action_mean, _, _ = ActorCritic(mj_model.nu).apply(loaded_params, obs)
        
        mj_data.ctrl[:] = np.array(action_mean)
        mujoco.mj_step(mj_model, mj_data)
        
        if step % 10 == 0:
            renderer.update_scene(mj_data, camera="front_camera")
            frames.append(renderer.render())

    media.show_video(frames, fps=30)

if __name__ == "__main__":
    print("Running Stretch 4 MuJoCo Warp Policy Training Example...")
    setup_environment()
    config = get_default_config()
    mj_model, target_object_body_name = generate_robocasa_model()

    # sim = Stretch4MujocoSimulator(
    #     model=mj_model,
    #     scene_xml_path=None,
    #     cameras_to_use=[],
    #     camera_hz=10.00,
    # )

    # sim.start(headless=False)

    # while sim.is_running():
    #     time.sleep(0.1)

    warp_model, warp_data = load_mujoco_warp_model(mj_model, config)
    
    trainer = PPOTrainer(mj_model, warp_model, warp_data, target_object_body_name, config)
    
    # # Run training
    params, training_frames = run_training(trainer)
    
    # # Save the trained parameters
    filename = save_params(params)
    print(f"Training complete. Parameters saved to {filename}")

    if training_frames:
        video_filename = f"training_video_{int(time.time())}.mp4"
        media.write_video(video_filename, training_frames, fps=15)
        print(f"Saved training video to {video_filename}")

