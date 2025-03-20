import os

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import entropy
import open_clip
import types


# Replace wilds function that requires torch_scatter
def _avg_over_groups(v, g, n_groups):
    """
    Args:
        v (Tensor): Vector containing the quantity to average over.
        g (Tensor): Vector of the same length as v, containing group information.
    Returns:
        group_avgs (Tensor): Vector of length num_groups
        group_counts (Tensor)
    """
    assert v.device == g.device
    assert v.numel() == g.numel()
    group_count = wilds.common.utils.get_counts(g, n_groups)
    # group_avgs = torch_scatter.scatter(src=v, index=g, dim_size=n_groups, reduce='mean')
    group_avgs = torch.zeros(n_groups, dtype=torch.float, device=v.device).scatter_(
        0, index=g, src=v, reduce="add"
    )
    group_avgs /= group_count
    return group_avgs, group_count


import wilds.common.utils

wilds.common.utils.avg_over_groups = _avg_over_groups
#

from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import F1, Accuracy, Recall
from wilds.datasets.wilds_dataset import WILDSDataset

from .wds_eval import create_webdataset, evaluate_webdataset
from clip_benchmark.metrics import zeroshot_classification as zsc


def create_metadata_loader(
    task, data_root=None, dataset_len=None, batch_size=64, num_workers=4
):
    dataset, _ = create_webdataset(
        task, None, data_root, dataset_len, batch_size, num_workers
    )
    # Load metadata (npy) and no images
    dataset.pipeline = dataset.pipeline[:5]  # This will break if webdataset changes
    metadataset = dataset.to_tuple("cls", "npy")
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)

    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataloader


def evaluate_wilds_dataset(
    task,
    model_arch,
    model_path,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=4,
    run_additional_tests=False,
):
    """Evaluate CLIP model on WILDS classification task."""

    # Evaluate
    metrics, y_pred, y_target = evaluate_webdataset(
        task,
        model_arch,
        model_path,
        data_root,
        dataset_len,
        batch_size,
        num_workers,
        return_preds=True,
    )

    # Load additional metadata
    print("Reading additional metadata")
    metadata_loader = create_metadata_loader(
        task, data_root, dataset_len, batch_size, num_workers
    )
    # Check metadata
    y_array = []
    metadata_array = []
    for label, metadata in metadata_loader:
        y_array.append(label)
        metadata_array.append(metadata)
    # assert (y_target == np.array(y_array)).all(), "Labels do not match"
    metadata = torch.cat(metadata_array)

    # Compute additional metrics
    wilds_evaluator = EVALUATORS[task](metadata)
    metrics.update(wilds_evaluator.eval(y_pred, y_target, metadata)[0])

    # clear gpu cache
    torch.cuda.empty_cache()
    
    # Run additional tests if requested
    if run_additional_tests:
        # Load model for additional tests
        model, transform, device = load_model(model_arch, model_path)
        
        # Get features and predictions
        # features, logits, attention_maps = extract_features_and_attention(
        #     task, model , model_arch, transform, data_root, dataset_len, batch_size, num_workers
        # )

        features, logits, attention_maps = extract_features_logits_and_attention(
            task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers
        )
        
        # 1. Entropy Analysis
        entropy_metrics = compute_entropy_metrics(logits)
        # metrics.update(entropy_metrics)
        
        # 2. Redundancy Reduction
        redundancy_metrics = analyze_redundancy(features)
        # metrics.update(redundancy_metrics)
        
        # 3. t-SNE Visualization
        visualize_tsne(features, y_target, metadata, task)
        
        # 4. Attention Visualization
        visualize_attention(attention_maps, task, n_samples=5)
        
        # 5. Weights/Gradient Visualization
        visualize_weights(task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers)

    # return metrics


def load_model(model_arch, model_path):
    """Load model for additional tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_path = str(model_path)
    model, _, transform = open_clip.create_model_and_transforms(
        model_arch, pretrained=model_path
    )
    model.eval()
    # Enable mixed precision for better memory efficiency
    model = model.to(device)
    # Optional: Use half precision to significantly reduce memory usage
    # model = model.half()

    return model, transform, device

def extract_features_logits_and_attention(
    task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers
):
    """Extract features, logits, and attention maps from the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers
    )

    # Get zeroshot classifier setup
    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (
        zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification"

    # Create the zero-shot classifier and move to GPU
    classifier = zsc.zero_shot_classifier(
        model,
        open_clip.get_tokenizer(model_arch),
        classnames,
        zeroshot_templates,
        device,
    )
    
    # Find all attention modules
    attn_modules = []
    for name, module in model.visual.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            attn_modules.append((name, module))
            # print(f"Found MultiheadAttention module: {name}")
    
    # Store original forward methods
    original_forwards = {}
    
    # Define the patched forward method with memory optimization
    def patched_forward(self, query, key, value, key_padding_mask=None, need_weights=True, 
                       attn_mask=None, average_attn_weights=False):
        # Call the original implementation but force need_weights=True
        output, attn_weights = self._original_forward(query, key, value, 
                                                    key_padding_mask=key_padding_mask,
                                                    need_weights=True,
                                                    attn_mask=attn_mask,
                                                    average_attn_weights=False)
        
        # Store the attention weights - use clone() to avoid modifying the original tensor
        if attn_weights is not None:
            # Move to CPU immediately to free GPU memory
            self.last_attn_weights = attn_weights.detach().cpu()
        else:
            # print("Warning: attn_weights is None even with need_weights=True")
            self.last_attn_weights = None
            
        return output, attn_weights
    
    # Apply the patch to all attention modules
    for name, module in attn_modules:
        original_forwards[name] = module.forward
        module._original_forward = module.forward
        module.forward = types.MethodType(patched_forward, module)
        module.last_attn_weights = None
    
    features_list = []
    logits_list = []
    attention_maps_list = []
    
    # Use torch.cuda.amp for mixed precision (significantly reduces memory usage)
    scaler = torch.cuda.amp.GradScaler()
    
    # Processing in smaller chunks to save memory
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features and attention")):
        # Clear CUDA cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        # for testing after 5 batches
        if batch_idx > 2:
            break
            
        images = batch[0].to(device)
        batch_attention_maps = {}
        
        # Reset the stored attention weights
        for _, module in attn_modules:
            module.last_attn_weights = None
        
        # Use mixed precision for forward pass
        # with torch.cuda.amp.autocast(enabled=True):
        with torch.no_grad():
                # Forward pass to get features and implicitly collect attention maps
                features = model.encode_image(images)
                image_features = F.normalize(features, dim=-1)
                logits = 100. * image_features @ classifier
        
        # Move results to CPU immediately to free GPU memory
        features_cpu = features.detach().cpu()
        logits_cpu = logits.detach().cpu()
        
        # Collect attention weights from all modules (already on CPU from patched_forward)
        for name, module in attn_modules:
            if hasattr(module, 'last_attn_weights') and module.last_attn_weights is not None:
                batch_attention_maps[name] = module.last_attn_weights
        
        features_list.append(features_cpu)
        logits_list.append(logits_cpu)
        attention_maps_list.append(batch_attention_maps)
        
        # Explicitly delete tensors to free memory
        del images, features, image_features, logits, features_cpu, logits_cpu
        torch.cuda.empty_cache()
    
    # Restore original forward methods
    for name, module in attn_modules:
        module.forward = original_forwards[name]
        if hasattr(module, '_original_forward'):
            delattr(module, '_original_forward')
        if hasattr(module, 'last_attn_weights'):
            delattr(module, 'last_attn_weights')
    
    # Concatenate features and logits
    features = torch.cat(features_list)
    logits = torch.cat(logits_list)
    
    return features, logits, attention_maps_list

# def extract_features_logits_and_attention(
#     task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers
# ):
#     """Extract features, logits, and attention maps from the model."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dataset, dataloader = create_webdataset(
#         task, transform, data_root, dataset_len, batch_size, num_workers
#     )

#     # Get zeroshot classifier setup
#     zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
#     classnames = dataset.classes if hasattr(dataset, "classes") else None
#     assert (
#         zeroshot_templates is not None and classnames is not None
#     ), "Dataset does not support classification"

#     # Create the zero-shot classifier
#     classifier = zsc.zero_shot_classifier(
#         model,
#         open_clip.get_tokenizer(model_arch),
#         classnames,
#         zeroshot_templates,
#         device,
#     )
    
#     # Patch the MultiheadAttention modules to extract attention maps
#     attention_maps_by_batch = []
    
#     # Find all attention modules
#     attn_modules = []
#     for name, module in model.visual.named_modules():
#         if isinstance(module, torch.nn.MultiheadAttention):
#             attn_modules.append((name, module))
#             print(f"Found MultiheadAttention module: {name}")
    
#     # Store original forward methods
#     original_forwards = {}
    
#     # Define the patched forward method
#     def patched_forward(self, query, key, value, key_padding_mask=None, need_weights=True, 
#                        attn_mask=None, average_attn_weights=False):
#         # Call the original implementation but force need_weights=True
#         output, attn_weights = self._original_forward(query, key, value, 
#                                                     key_padding_mask=key_padding_mask,
#                                                     need_weights=True,  # Force to True
#                                                     attn_mask=attn_mask,
#                                                     average_attn_weights=False)  # Force to False
        
#         # Store the attention weights
#         if attn_weights is not None:
#             self.last_attn_weights = attn_weights.detach()
#         else:
#             print("Warning: attn_weights is None even with need_weights=True")
#             self.last_attn_weights = None
            
#         return output, attn_weights
    
#     # Apply the patch to all attention modules
#     for name, module in attn_modules:
#         # Store the original forward method
#         original_forwards[name] = module.forward
        
#         # Create a new forward method that wraps the original
#         module._original_forward = module.forward
#         module.forward = types.MethodType(patched_forward, module)
#         module.last_attn_weights = None
        
#         print(f"Patched forward method for {name}")
    
#     features_list = []
#     logits_list = []
#     attention_maps_list = []
    
#     # Evaluate
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Extracting features and attention"):
#             images = batch[0].to(device)
#             batch_attention_maps = {}
            
#             # Reset the stored attention weights
#             for _, module in attn_modules:
#                 module.last_attn_weights = None
            
#             # Forward pass to get features and implicitly collect attention maps
#             features = model.encode_image(images)
#             image_features = F.normalize(features, dim=-1)
#             logits = 100. * image_features @ classifier
            
#             # Collect attention weights from all modules
#             for name, module in attn_modules:
#                 if hasattr(module, 'last_attn_weights') and module.last_attn_weights is not None:
#                     batch_attention_maps[name] = module.last_attn_weights.cpu()
            
#             features_list.append(features.cpu())
#             logits_list.append(logits.cpu())
#             attention_maps_list.append(batch_attention_maps)
    
#     # Restore original forward methods
#     for name, module in attn_modules:
#         module.forward = original_forwards[name]
#         if hasattr(module, '_original_forward'):
#             delattr(module, '_original_forward')
#         if hasattr(module, 'last_attn_weights'):
#             delattr(module, 'last_attn_weights')
    
#     # Concatenate features and logits
#     features = torch.cat(features_list)
#     logits = torch.cat(logits_list)
    
#     # Process the collected attention maps
#     # This will be a list of dictionaries, where each dictionary contains attention maps for one batch
    
#     return features, logits, attention_maps_list


# def extract_features_and_attention(
#     task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers
# ):
#     """Extract features, logits, and attention maps from the model."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dataset, dataloader = create_webdataset(
#         task, transform, data_root, dataset_len, batch_size, num_workers
#     )
    
#     features_list = []
#     logits_list = []
#     attention_maps_list = []
    
#     zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
#     classnames = dataset.classes if hasattr(dataset, "classes") else None
#     assert (
#         zeroshot_templates is not None and classnames is not None
#     ), "Dataset does not support classification"

#     # Evaluate
#     classifier = zsc.zero_shot_classifier(
#         model,
#         open_clip.get_tokenizer(model_arch),
#         classnames,
#         zeroshot_templates,
#         device,
#     )
    
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Extracting features"):
#             images = batch[0].to(device)
            
#             # Modify this based on your model architecture
#             # For example, for CLIP model:
#             features = model.encode_image(images)
#             image_features = F.normalize(features, dim=-1)
#             logits = 100. * image_features @ classifier
            
#             # If model has attention mechanism
#             attention_maps = None
#             if hasattr(model, "get_attention_maps"):
#                 attention_maps = model.get_attention_maps(images)
            
#             features_list.append(features.cpu())
#             logits_list.append(logits.cpu())
#             if attention_maps is not None:
#                 attention_maps_list.append(attention_maps.cpu())
    
#     features = torch.cat(features_list)
#     logits = torch.cat(logits_list)
#     attention_maps = torch.cat(attention_maps_list) if attention_maps_list else None
    
#     return features, logits, attention_maps


def compute_entropy_metrics(logits):
    """Compute entropy metrics for model predictions."""
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1).numpy()
    
    # Calculate entropy for each prediction
    entropies = np.array([entropy(p) for p in probs])
    
    # Calculate metrics
    metrics = {
        "mean_entropy": float(np.mean(entropies)),
        "median_entropy": float(np.median(entropies)),
        "max_entropy": float(np.max(entropies)),
        "min_entropy": float(np.min(entropies)),
    }
    
    # Plot entropy distribution
    plt.figure(figsize=(10, 6))
    plt.hist(entropies, bins=30)
    plt.title("Prediction Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.savefig("entropy_distribution.png")
    plt.close()
    
    return metrics


def analyze_redundancy(features):
    """Analyze feature redundancy using correlation and other metrics."""
    # Compute feature correlation matrix
    features_np = features.numpy()
    corr_matrix = np.corrcoef(features_np.T)
    
    # Calculate redundancy metrics
    abs_corr = np.abs(corr_matrix)
    np.fill_diagonal(abs_corr, 0)  # Exclude self-correlations
    
    metrics = {
        "mean_feature_correlation": float(np.mean(abs_corr)),
        "max_feature_correlation": float(np.max(abs_corr)),
        "feature_redundancy_score": float(np.sum(abs_corr > 0.8) / (abs_corr.size - features_np.shape[1])),
    }
    
    # Plot correlation heatmap using matplotlib instead of seaborn
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with matplotlib
    im = plt.imshow(corr_matrix[:50, :50], cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlation")
    plt.title("Feature Correlation Matrix (First 50 dimensions)")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.tight_layout()
    plt.savefig("feature_correlation.png")
    plt.close()
    
    return metrics


def visualize_tsne(features, y_target, metadata, task, perplexity=30):
    """Create t-SNE visualization of features colored by different factors."""
    # Reduce dimensionality to 2D using t-SNE
    print("Computing t-SNE projection...")
    # Use a subset if features is too large
    max_samples = 5000
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features_subset = features[indices]
        y_subset = y_target[indices]
        metadata_subset = metadata[indices]
    else:
        features_subset = features
        y_subset = y_target
        metadata_subset = metadata
    
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(features_subset.numpy())
    
    # Plot t-SNE colored by class
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_subset, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title(f't-SNE Visualization of Features by Class ({task})')
    plt.savefig(f"tsne_by_class_{task.replace('/', '_')}.png")
    plt.close()
    
    # Plot t-SNE colored by domain (e.g., hospital for Camelyon17, region for FMoW)
    if task == "wilds/camelyon17" and "hospital" in metadata_subset[0]:
        domain_value = torch.tensor([m[0] for m in metadata_subset])  # hospital is first column
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=domain_value, cmap='Set3', alpha=0.7)
        plt.colorbar(scatter, label='Hospital')
        plt.title('t-SNE Visualization of Features by Hospital (Camelyon17)')
        plt.savefig("tsne_by_hospital.png")
        plt.close()
    
    elif task == "wilds/fmow" and "region" in metadata_subset[0]:
        domain_value = torch.tensor([m[0] for m in metadata_subset])  # region is first column
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=domain_value, cmap='Set3', alpha=0.7)
        plt.colorbar(scatter, label='Region')
        plt.title('t-SNE Visualization of Features by Region (FMoW)')
        plt.savefig("tsne_by_region.png")
        plt.close()
    
    elif task == "wilds/iwildcam" and "location" in metadata_subset[0]:
        # For iWildCam, we might have many locations, so we can pick the top N
        locations = [m[0] for m in metadata_subset]  # location is first column
        top_locations = np.bincount(locations).argsort()[-10:]  # Top 10 locations
        
        # Only plot samples from top locations
        mask = np.isin(locations, top_locations)
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            tsne_result[mask, 0], 
            tsne_result[mask, 1], 
            c=[locations[i] for i in range(len(locations)) if mask[i]], 
            cmap='tab10', 
            alpha=0.7
        )
        plt.colorbar(scatter, label='Location (Top 10)')
        plt.title('t-SNE Visualization of Features by Location (iWildCam)')
        plt.savefig("tsne_by_location.png")
        plt.close()


def visualize_attention(attention_maps, task, n_samples=5):
    """Visualize attention maps for a few samples."""
    if attention_maps is None:
        print("No attention maps available for visualization")
        return
    
    # Select a few random samples
    indices = np.random.choice(len(attention_maps), min(n_samples, len(attention_maps)), replace=False)
    
    for i, idx in enumerate(indices):
        for k in attention_maps[idx]:
            attn_map = attention_maps[idx][k]
            
            # Handle different attention map formats
            if len(attn_map.shape) == 3:  # [heads, h, w]
                n_heads = attn_map.shape[0]
                fig, axes = plt.subplots(1, n_heads, figsize=(n_heads * 4, 4))
                for h in range(n_heads):
                    if n_heads == 1:
                        ax = axes
                    else:
                        ax = axes[h]
                    im = ax.imshow(attn_map[h].numpy(), cmap='viridis')
                    ax.set_title(f'Head {h+1}')
                    ax.axis('off')
                fig.colorbar(im, ax=axes.ravel().tolist())
            
            elif len(attn_map.shape) == 4:  # [layers, heads, h, w]
                n_layers = attn_map.shape[0]
                n_heads = attn_map.shape[1]
                fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 3, n_layers * 3))
                for l in range(n_layers):
                    for h in range(n_heads):
                        if n_layers == 1 and n_heads == 1:
                            ax = axes
                        elif n_layers == 1:
                            ax = axes[h]
                        elif n_heads == 1:
                            ax = axes[l]
                        else:
                            ax = axes[l, h]
                        im = ax.imshow(attn_map[l, h].numpy(), cmap='viridis')
                        ax.set_title(f'L{l+1}H{h+1}')
                        ax.axis('off')
                fig.colorbar(im, ax=axes.ravel().tolist())
            
            plt.tight_layout()
            plt.suptitle(f'Attention Maps - Sample {i+1}')
            plt.savefig(f"attention_sample_{i+1}_{k}_{task.replace('/', '_')}.png")
            plt.close()


def visualize_weights(task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers):
    """Visualize model weights and gradients."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers
    )

    # Get zeroshot classifier setup
    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (
        zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification"

    # Create the zero-shot classifier and move to GPU
    classifier = zsc.zero_shot_classifier(
        model,
        open_clip.get_tokenizer(model_arch),
        classnames,
        zeroshot_templates,
        device,
    )

    # 1. Visualize first layer filters (for CNN models)
    if hasattr(model, 'visual') and hasattr(model.visual, 'conv1'):
        # For CLIP-like models
        filters = model.visual.conv1.weight.data.cpu()
        visualize_conv_filters(filters, f"conv1_filters_{task.replace('/', '_')}.png")
    
    elif hasattr(model, 'conv1'):
        # For standard CNN models
        filters = model.conv1.weight.data.cpu()
        visualize_conv_filters(filters, f"conv1_filters_{task.replace('/', '_')}.png")
    
    # 2. Visualize final layer weights
    if classifier is not None:
        visualize_classifier_weights(classifier, task)
    
    # 3. Visualize layer activations histogram
    if hasattr(model, 'layer_activations'):
        visualize_activation_distribution(model.layer_activations, task)


def visualize_conv_filters(filters, filename, max_filters=64):
    """Visualize convolutional filters."""
    # Take only the first max_filters filters
    n_filters = min(filters.shape[0], max_filters)
    filters = filters[:n_filters]
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(n_filters)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # Plot each filter
    for i in range(grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        if i < n_filters:
            # Get the filter
            f = filters[i]
            # If RGB, convert to displayable format
            if f.shape[0] == 3:
                # Normalize each filter to [0, 1]
                f = f - f.min()
                f = f / (f.max() + 1e-8)
                # Transpose from (C, H, W) to (H, W, C)
                f = f.permute(1, 2, 0)
            else:
                # For non-RGB filters, just take the first channel
                f = f[0]
            
            if grid_size == 1:
                ax = axes
            else:
                ax = axes[row, col]
            ax.imshow(f.numpy())
            ax.axis('off')
        else:
            if grid_size == 1:
                ax = axes
            else:
                ax = axes[row, col]
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def visualize_classifier_weights(classifier, task):
    """Visualize classifier weights."""
    # Get the final layer weights
    if hasattr(classifier, 'weight'):
        weights = classifier.weight.data.cpu()
        
        # Create heatmap using matplotlib instead of seaborn
        plt.figure(figsize=(12, 8))
        im = plt.imshow(weights.numpy(), cmap='coolwarm', aspect='auto')
        plt.colorbar(im, label='Weight Value')
        plt.title(f'Classifier Weights Heatmap ({task})')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Class')
        plt.tight_layout()
        plt.savefig(f"classifier_weights_{task.replace('/', '_')}.png")
        plt.close()
        
        # Create weight norm distribution
        weight_norms = torch.norm(weights, dim=1).numpy()
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(weight_norms)), weight_norms)
        plt.title(f'Classifier Weight Norms by Class ({task})')
        plt.xlabel('Class')
        plt.ylabel('Weight Norm')
        plt.tight_layout()
        plt.savefig(f"classifier_weight_norms_{task.replace('/', '_')}.png")
        plt.close()

    if isinstance(classifier, torch.Tensor):
        print(classifier.shape)
        weights = classifier.cpu()
        
        # Create heatmap using matplotlib instead of seaborn
        plt.figure(figsize=(12, 8))
        im = plt.imshow(weights.numpy(), cmap='coolwarm', aspect='auto')
        plt.colorbar(im, label='Weight Value')
        plt.title(f'Classifier Weights Heatmap ({task})')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Class')
        plt.tight_layout()
        plt.savefig(f"classifier_weights_{task.replace('/', '_')}.png")
        plt.close()
        
        # Create weight norm distribution
        weight_norms = torch.norm(weights, dim=1).numpy()
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(weight_norms)), weight_norms)
        plt.title(f'Classifier Weight Norms by Class ({task})')
        plt.xlabel('Class')
        plt.ylabel('Weight Norm')
        plt.tight_layout()
        plt.savefig(f"classifier_weight_norms_{task.replace('/', '_')}.png")
        plt.close()


def visualize_activation_distribution(activations, task):
    """Visualize the distribution of activations across layers."""
    n_layers = len(activations)
    
    plt.figure(figsize=(12, 8))
    for i, (layer_name, activation) in enumerate(activations.items()):
        activation_flat = activation.view(-1).cpu().numpy()
        # Use matplotlib's hist function with density=True instead of seaborn's kdeplot
        plt.hist(
            activation_flat, 
            bins=50, 
            alpha=0.5, 
            density=True, 
            label=layer_name,
            histtype='step',
            linewidth=2
        )
    
    plt.title(f'Activation Distributions Across Layers ({task})')
    plt.xlabel('Activation Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"activation_distributions_{task.replace('/', '_')}.png")
    plt.close()


# WILDS


class WILDSEvaluator(WILDSDataset):
    def __init__(self, metadata):
        self._metadata_array = metadata


# iWildCam


class IWildCamEvaluator(WILDSEvaluator):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._metadata_fields = [
            "location",
            "sequence",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "y",
        ]
        self._eval_grouper = CombinatorialGrouper(
            dataset=self, groupby_fields=(["location"])
        )

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
            Recall(prediction_fn=prediction_fn, average="macro"),
            F1(prediction_fn=prediction_fn, average="macro"),
        ]
        results = {}
        for metric in metrics:
            results.update(
                {
                    **metric.compute(y_pred, y_true),
                }
            )
        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
            f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n"
        )
        return results, results_str


# Camelyon17


class Camelyon17Evaluator(WILDSEvaluator):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._metadata_fields = ["hospital", "slide", "y"]
        self._eval_grouper = CombinatorialGrouper(
            dataset=self, groupby_fields=["slide"]
        )

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric, self._eval_grouper, y_pred, y_true, metadata
        )


# FMoW


class FMoWEvaluator(WILDSEvaluator):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._metadata_fields = ["region", "year", "y"]
        self._eval_groupers = {
            "year": CombinatorialGrouper(dataset=self, groupby_fields=["year"]),
            "region": CombinatorialGrouper(dataset=self, groupby_fields=["region"]),
        }

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)
        # Overall evaluation + evaluate by year
        all_results, all_results_str = self.standard_group_eval(
            metric, self._eval_groupers["year"], y_pred, y_true, metadata
        )
        # Evaluate by region and ignore the "Other" region
        region_grouper = self._eval_groupers["region"]
        region_results = metric.compute_group_wise(
            y_pred,
            y_true,
            region_grouper.metadata_to_group(metadata),
            region_grouper.n_groups,
        )
        all_results[f"{metric.name}_worst_year"] = all_results.pop(
            metric.worst_group_metric_field
        )
        region_metric_list = []
        for group_idx in range(region_grouper.n_groups):
            group_str = region_grouper.group_field_str(group_idx)
            group_metric = region_results[metric.group_metric_field(group_idx)]
            group_counts = region_results[metric.group_count_field(group_idx)]
            all_results[f"{metric.name}_{group_str}"] = group_metric
            all_results[f"count_{group_str}"] = group_counts
            if (
                region_results[metric.group_count_field(group_idx)] == 0
                or "Other" in group_str
            ):
                continue
            all_results_str += (
                f"  {region_grouper.group_str(group_idx)}  "
                f"[n = {region_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
                f"{metric.name} = {region_results[metric.group_metric_field(group_idx)]:5.3f}\n"
            )
            region_metric_list.append(
                region_results[metric.group_metric_field(group_idx)]
            )
        all_results[f"{metric.name}_worst_region"] = metric.worst(region_metric_list)
        all_results_str += f"Worst-group {metric.name}: {all_results[f'{metric.name}_worst_region']:.3f}\n"

        return all_results, all_results_str


EVALUATORS = {
    "wilds/iwildcam": IWildCamEvaluator,
    "wilds/camelyon17": Camelyon17Evaluator,
    "wilds/fmow": FMoWEvaluator,
}