import os
import pickle
import jax.numpy as jnp
from flax import serialization
import json

def save_actor_params(agent, save_dir, step, save_format='flax'):

    actor_save_dir = os.path.join(save_dir, 'actor_checkpoints')
    os.makedirs(actor_save_dir, exist_ok=True)
    
    actor_data = {
        'step': step,
        'config': dict(agent.config), 
        'agent_type': agent.config.get('agent_name', 'unknown') 
    }

    network_params = agent.network.params
    actor_networks = []
    
    for key in network_params.keys():
        if key.startswith('modules_actor'):
            actor_networks.append(key)

            save_key = key.replace('modules_', '')
            actor_data[save_key] = network_params[key]
    
    print(f"Found actor networks: {actor_networks}")
    

    if not actor_networks:
        common_actor_keys = [
            'modules_actor_onestep_flow',
            'modules_actor_bc_flow',
            'modules_actor_transformer', 
            'modules_actor',
            'modules_policy'
        ]
        for key in common_actor_keys:
            if key in network_params:
                actor_networks.append(key)
                save_key = key.replace('modules_', '')
                actor_data[save_key] = network_params[key]
                break
    
    if save_format == 'flax':
        filename = f'actor_step_{step}.flax'
        filepath = os.path.join(actor_save_dir, filename)
        
        with open(filepath, 'wb') as f:
            serialized_data = serialization.to_bytes(actor_data)
            f.write(serialized_data)
            
    elif save_format == 'pickle':

        filename = f'actor_step_{step}.pkl'
        filepath = os.path.join(actor_save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(actor_data, f)

    latest_link = os.path.join(actor_save_dir, 'latest_actor.txt')
    with open(latest_link, 'w') as f:
        f.write(filename)
    
    print(f"Actor parameters saved to: {filepath}")
    return filepath

def load_actor_params(filepath, save_format='flax'):

    if save_format == 'flax':
        with open(filepath, 'rb') as f:
            serialized_data = f.read()
            actor_data = serialization.from_bytes(target=None, encoded_bytes=serialized_data)
    elif save_format == 'pickle':
        with open(filepath, 'rb') as f:
            actor_data = pickle.load(f)
    
    return actor_data

def save_actor_params_lightweight(agent, save_dir, step):

    actor_save_dir = os.path.join(save_dir, 'actor_checkpoints')
    os.makedirs(actor_save_dir, exist_ok=True)
    
    filename = f'actor_params_step_{step}.npz'
    filepath = os.path.join(actor_save_dir, filename)

    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    network_params = agent.network.params
    all_flat_params = {}
    
    for key in network_params.keys():
        if key.startswith('modules_actor'):
            actor_params = network_params[key]

            save_key = key.replace('modules_', '')
            flat_params = flatten_dict(actor_params, parent_key=save_key)
            all_flat_params.update(flat_params)
    
    if not all_flat_params:
        common_keys = [
            'modules_actor_onestep_flow', 
            'modules_actor_bc_flow', 
            'modules_actor_transformer',
            'modules_actor', 
            'modules_policy'
        ]
        for key in common_keys:
            if key in network_params:
                actor_params = network_params[key]
                save_key = key.replace('modules_', '')
                flat_params = flatten_dict(actor_params, parent_key=save_key)
                all_flat_params.update(flat_params)
                break

    numpy_params = {k: jnp.asarray(v) for k, v in all_flat_params.items()}
    numpy_params['step'] = step
    numpy_params['agent_type'] = agent.config.get('agent_name', 'unknown')
    
    jnp.savez(filepath, **numpy_params)
    
    print(f"Lightweight actor parameters saved to: {filepath}")
    return filepath