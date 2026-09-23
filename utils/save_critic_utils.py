"""Save and restore the critic part of an ACFQL-style Flax agent."""

import os
import pickle

import flax
import jax
from flax import serialization


_CRITIC_KEYS = ("critic", "target_critic")


def _detect_format(filepath, save_format=None):
    if save_format is not None:
        if save_format not in ("flax", "pickle"):
            raise ValueError("save_format must be 'flax' or 'pickle'")
        return save_format
    if filepath.endswith(".flax"):
        return "flax"
    if filepath.endswith((".pkl", ".pickle")):
        return "pickle"
    raise ValueError(f"Cannot infer checkpoint format from: {filepath}")


def save_critic_params(agent, save_dir, step, phase="", save_format="flax"):
    """Save online critic and target critic without saving actor parameters.

    The checkpoint keys are ``critic`` and ``target_critic``; in the agent they
    correspond to ``modules_critic`` and ``modules_target_critic``.
    """
    if save_format not in ("flax", "pickle"):
        raise ValueError("save_format must be 'flax' or 'pickle'")

    network_params = agent.network.params
    missing = [f"modules_{name}" for name in _CRITIC_KEYS
               if f"modules_{name}" not in network_params]
    if missing:
        raise KeyError(f"Critic parameters not found in agent: {missing}")

    critic_data = {
        "step": int(step),
        "phase": phase,
        "config": dict(agent.config),
        "agent_type": agent.config.get("agent_name", "unknown"),
        "critic": network_params["modules_critic"],
        "target_critic": network_params["modules_target_critic"],
    }

    critic_save_dir = os.path.join(save_dir, "critic_checkpoints")
    os.makedirs(critic_save_dir, exist_ok=True)

    phase_prefix = f"{phase}_" if phase else ""
    extension = "flax" if save_format == "flax" else "pkl"
    filename = f"critic_{phase_prefix}step_{step}.{extension}"
    filepath = os.path.join(critic_save_dir, filename)

    if save_format == "flax":
        with open(filepath, "wb") as f:
            f.write(serialization.to_bytes(critic_data))
    else:
        with open(filepath, "wb") as f:
            pickle.dump(critic_data, f)

    # One latest pointer per phase, plus a pointer to the newest critic overall.
    pointer_names = ["latest_critic.txt"]
    if phase:
        pointer_names.append(f"latest_{phase}_critic.txt")
    for pointer_name in pointer_names:
        with open(os.path.join(critic_save_dir, pointer_name), "w") as f:
            f.write(filename)

    print(f"[{phase or 'critic'}] Critic parameters saved to: {filepath}")
    return filepath


def load_critic_params(filepath, save_format=None):
    """Read a critic checkpoint and return its data dictionary."""
    save_format = _detect_format(filepath, save_format)
    if save_format == "flax":
        with open(filepath, "rb") as f:
            critic_data = serialization.from_bytes(None, f.read())
    else:
        with open(filepath, "rb") as f:
            critic_data = pickle.load(f)

    if "critic" not in critic_data:
        raise KeyError(f"Checkpoint does not contain 'critic': {filepath}")
    return critic_data


def _assert_same_shapes(saved, current, name):
    # Flax msgpack may deserialize a FrozenDict as a plain dict. Normalize both
    # sides before comparing so equivalent parameter trees are accepted.
    saved = flax.core.unfreeze(saved)
    current = flax.core.unfreeze(current)
    saved_leaves = jax.tree_util.tree_leaves(saved)
    current_leaves = jax.tree_util.tree_leaves(current)
    if jax.tree_util.tree_structure(saved) != jax.tree_util.tree_structure(current):
        raise ValueError(f"{name} parameter tree is incompatible with this agent")
    saved_shapes = [getattr(x, "shape", None) for x in saved_leaves]
    current_shapes = [getattr(x, "shape", None) for x in current_leaves]
    if saved_shapes != current_shapes:
        raise ValueError(
            f"{name} parameter shapes are incompatible:\n"
            f"checkpoint={saved_shapes}\nagent={current_shapes}"
        )


def load_critic_into_agent(agent, filepath, save_format=None, strict=True):
    """Replace only critic parameters in an already-created compatible agent.

    Actor parameters and all other modules remain unchanged. If an old
    checkpoint lacks ``target_critic``, the restored critic is copied to it.
    For resumed training, call this on a newly created agent so that optimizer
    state is fresh and consistent with the restored parameters.
    """
    critic_data = load_critic_params(filepath, save_format)
    saved_critic = critic_data["critic"]
    saved_target = critic_data.get("target_critic", saved_critic)

    original_params = agent.network.params
    network_params = flax.core.unfreeze(original_params)

    if strict:
        _assert_same_shapes(
            saved_critic, network_params["modules_critic"], "critic"
        )
        _assert_same_shapes(
            saved_target,
            network_params["modules_target_critic"],
            "target_critic",
        )

    network_params["modules_critic"] = saved_critic
    network_params["modules_target_critic"] = saved_target
    if isinstance(original_params, flax.core.FrozenDict):
        network_params = flax.core.freeze(network_params)

    agent = agent.replace(network=agent.network.replace(params=network_params))
    print(
        "Critic restored from "
        f"{filepath} (phase={critic_data.get('phase', 'unknown')}, "
        f"step={critic_data.get('step', 'unknown')})"
    )
    return agent
