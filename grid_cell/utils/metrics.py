# bio_navigation/utils/metrics.py

import torch
import numpy as np

def compute_navigation_metrics(outputs, targets):

    metrics = {}
    

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()
    
    pred_positions = to_numpy(outputs['position'])
    true_positions = to_numpy(targets['positions'])
    pred_rotations = to_numpy(outputs['rotation'])
    true_rotations = to_numpy(targets['angles'])
    

    position_errors = np.sqrt(np.sum(
        (pred_positions - true_positions) ** 2,
        axis=-1
    ))
    metrics['position_error'] = float(np.mean(position_errors))
    metrics['max_position_error'] = float(np.max(position_errors))
    

    rotation_errors = np.abs(pred_rotations - true_rotations)
    rotation_errors = np.minimum(rotation_errors, 2 * np.pi - rotation_errors)
    metrics['rotation_error_deg'] = float(np.mean(np.degrees(rotation_errors)))
    

    if 'memory_activation' in outputs:
        memory_activation = to_numpy(outputs['memory_activation'])
        

        sparsity = np.mean(np.sum(memory_activation > 0.01, axis=-1) / memory_activation.shape[-1])
        metrics['memory_sparsity'] = float(sparsity)
        

        top_k = 3
        top_k_activations = np.mean(np.sort(memory_activation, axis=-1)[..., -top_k:])
        metrics['top_k_activation'] = float(top_k_activations)
        

    if 'grid_code' in outputs:
        grid_code = to_numpy(outputs['grid_code'])


        def compute_spatial_autocorr(activations):
            B, S, D = activations.shape
            activations = activations.reshape(B * S, D)
            autocorr = np.dot(activations, activations.T) / D
            return autocorr

        autocorr = compute_spatial_autocorr(grid_code)

        spatial_selectivity = np.mean(np.max(grid_code, axis=2) - np.mean(grid_code, axis=2))
        metrics['spatial_selectivity'] = float(spatial_selectivity)


        if grid_code.shape[1] > 1:
            temporal_stability = 1.0 - np.mean(np.abs(grid_code[:, 1:] - grid_code[:, :-1]))
            metrics['grid_stability'] = float(temporal_stability)
    else:
        print("\nWarning: 'grid_code' not found in outputs")

    return metrics


def print_metrics_summary(metrics, split='train'):

    print(f"\n{split.capitalize()} Metrics:")

    print(f"Position Error (grid units): {metrics['position_error']:.3f}")
    print(f"Max Position Error (grid units): {metrics['max_position_error']:.3f}")
    print(f"Rotation Error: {metrics['rotation_error_deg']:.2f}°")


    if 'memory_sparsity' in metrics:
        print(f"Memory Sparsity: {metrics['memory_sparsity']:.4f}")
        print(f"Top-k Activation: {metrics['top_k_activation']:.4f}")


    if 'spatial_selectivity' in metrics:
        print("\nGrid Cell Metrics:")
        print(f"Spatial Selectivity: {metrics['spatial_selectivity']:.4f}")
        if 'grid_stability' in metrics:
            print(f"Grid Stability: {metrics['grid_stability']:.4f}")

        if 'grid_spacing' in metrics:
            print(f"Grid Spacing: {metrics['grid_spacing']:.4f}")

def evaluate_navigation_batch(model, batch, device, memory_bank=None):

    model.eval()
    with torch.no_grad():

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        

        outputs = model(batch)
        

        metrics = compute_navigation_metrics(outputs, batch)
        

        if memory_bank is not None:
            memory_matches = find_best_matches(
                outputs['place_code'],
                memory_bank['features'],
                memory_bank['metadata']
            )
            

            if 'true_matches' in batch:
                match_accuracy = compute_memory_match_accuracy(
                    memory_matches,
                    batch['true_matches']
                )
                metrics['memory_match_accuracy'] = match_accuracy
        else:
            memory_matches = None
    
    return metrics, outputs, memory_matches

def compute_memory_match_accuracy(pred_matches, true_matches):









    total_correct = 0
    total_samples = len(true_matches)
    
    for pred, true in zip(pred_matches, true_matches):

        pred_best = (pred[0]['dataset'], pred[0]['frame_id'])
        if pred_best in true:
            total_correct += 1
    
    return total_correct / total_samples if total_samples > 0 else 0.0
