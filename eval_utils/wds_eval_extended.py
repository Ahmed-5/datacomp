"""Evaluate on standard classification webdatasets with comprehensive analysis."""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.manifold import TSNE
from scipy.stats import entropy
import open_clip
import types

def create_model(model_arch, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_path = str(model_path)
    model, _, transform = open_clip.create_model_and_transforms(
        model_arch, pretrained=model_path
    )
    model.eval()
    model = model.to(device)

    return model, transform, device


def create_webdataset(
    task, transform, data_root=None, dataset_len=None, batch_size=64, num_workers=4
):
    from clip_benchmark.datasets.builder import build_dataset
    
    data_folder = f"wds_{task.replace('/','-')}_test"
    if data_root is None:
        data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    else:
        data_root = os.path.join(data_root, data_folder)
    dataset = build_dataset(
        dataset_name=f"wds/{task}",
        root=data_root,
        transform=transform,
        split="test",
        download=False,
    )
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, dataloader


def create_metadata_loader(
    task, data_root=None, dataset_len=None, batch_size=64, num_workers=4
):
    from clip_benchmark.datasets.builder import build_dataset
    
    data_folder = f"wds_{task.replace('/','-')}_test"
    if data_root is None:
        data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    else:
        data_root = os.path.join(data_root, data_folder)
    
    dataset = build_dataset(
        dataset_name=f"wds/{task}",
        root=data_root,
        transform=None,
        split="test",
        download=False,
    )
    
    # If the dataset has metadata, extract it
    if hasattr(dataset, "metadata_field"):
        metadataset = dataset.to_tuple("cls", dataset.metadata_field)
    else:
        # If no specific metadata field, just return labels
        metadataset = dataset.to_tuple("cls")
    
    if dataset_len:
        metadataset = metadataset.with_length((dataset_len + batch_size - 1) // batch_size)

    dataloader = torch.utils.data.DataLoader(
        metadataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataloader


def evaluate_webdataset(
    task,
    model_arch,
    model_path,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=4,
    return_preds=False,
    return_topk=False,
):
    """Evaluate CLIP model on classification task."""
    from clip_benchmark.metrics import zeroshot_classification as zsc

    # Create model
    model, transform, device = create_model(model_arch, model_path)

    # Load data
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers
    )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (
        zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification"

    # Evaluate
    classifier = zsc.zero_shot_classifier(
        model,
        open_clip.get_tokenizer(model_arch),
        classnames,
        zeroshot_templates,
        device,
    )
    logits, target = zsc.run_classification(
        model, classifier, dataloader, device, amp=False
    )
    with torch.no_grad():
        pred = logits.argmax(axis=1).cpu()
        target = target.cpu()

    # Compute metrics
    if len(dataset.classes) >= 5:
        acc1, acc5 = zsc.accuracy(logits, target, topk=(1, 5))
    else:
        (acc1,) = zsc.accuracy(logits, target, topk=(1,))
        acc5 = None
    mean_per_class_recall = balanced_accuracy_score(target, pred)
    metrics = {
        "acc1": acc1,
        "acc5": acc5,
        "mean_per_class_recall": mean_per_class_recall,
    }

    if return_preds:
        if return_topk:
            with torch.no_grad():
                _, topk_pred = torch.topk(logits, int(return_topk), dim=1)
                topk_pred = topk_pred.cpu()
            return metrics, topk_pred, target
        return metrics, pred, target
    return metrics


def evaluate_webdataset_with_analysis(
    task,
    model_arch,
    model_path,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=4,
    run_additional_tests=False,
):
    """Evaluate CLIP model on WebDataset classification task with additional analysis."""

    # Basic evaluation
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

    # Try to load additional metadata if available
    try:
        print("Reading additional metadata")
        metadata_loader = create_metadata_loader(
            task, data_root, dataset_len, batch_size, num_workers
        )
        
        # Collect metadata
        y_array = []
        metadata_array = []
        for batch in metadata_loader:
            if len(batch) == 2:  # Label and metadata
                label, metadata = batch
                y_array.append(label)
                metadata_array.append(metadata)
            else:  # Just label
                y_array.append(batch[0])
                
        if metadata_array:
            metadata = torch.cat(metadata_array)
        else:
            metadata = None
            
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        metadata = None

    # Run additional tests if requested
    if run_additional_tests:
        # Load model for additional tests
        model, transform, device = create_model(model_arch, model_path)
        
        # Get features, logits and attention maps
        features, logits, attention_maps = extract_features_logits_and_attention(
            task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers
        )
        
        # 1. Entropy Analysis
        entropy_metrics = compute_entropy_metrics(logits)
        metrics.update(entropy_metrics)
        
        # 2. Redundancy Analysis
        redundancy_metrics = analyze_redundancy(features)
        metrics.update(redundancy_metrics)
        
        # 3. t-SNE Visualization
        if metadata is not None:
            visualize_tsne(features, y_target, metadata, task)
        else:
            visualize_tsne(features, y_target, None, task)
        
        # 4. Attention Visualization
        visualize_attention(attention_maps, task, n_samples=5)
        
        # 5. Weights Visualization
        visualize_weights(task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers)

    return metrics


def extract_features_logits_and_attention(
    task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers
):
    """Extract features, logits, and attention maps from the model."""
    from clip_benchmark.metrics import zeroshot_classification as zsc
    
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

    # Create the zero-shot classifier
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
    
    # Use torch.cuda.amp for mixed precision
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features and attention")):
            # Clear CUDA cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # Limit the number of batches for testing
            if dataset_len is not None and batch_idx * batch_size >= dataset_len:
                break
                
            images = batch[0].to(device)
            batch_attention_maps = {}
            
            # Reset the stored attention weights
            for _, module in attn_modules:
                module.last_attn_weights = None
            
            # Forward pass to get features and attention maps
            features = model.encode_image(images)
            image_features = F.normalize(features, dim=-1)
            logits = 100. * image_features @ classifier
            
            # Move results to CPU immediately to free GPU memory
            features_cpu = features.detach().cpu()
            logits_cpu = logits.detach().cpu()
            
            # Collect attention weights from all modules
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
    features = torch.cat(features_list) if features_list else None
    logits = torch.cat(logits_list) if logits_list else None
    
    return features, logits, attention_maps_list


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
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr_matrix[:50, :50], cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlation")
    plt.title("Feature Correlation Matrix (First 50 dimensions)")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.tight_layout()
    plt.savefig("feature_correlation.png")
    plt.close()
    
    return metrics


def visualize_tsne(features, y_target, metadata=None, task=None, perplexity=30):
    """Create t-SNE visualization of features colored by different factors."""
    # Reduce dimensionality to 2D using t-SNE
    print("Computing t-SNE projection...")
    # Use a subset if features is too large
    max_samples = 5000
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features_subset = features[indices]
        y_subset = y_target[indices]
        metadata_subset = metadata[indices] if metadata is not None else None
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
    title = f't-SNE Visualization of Features by Class'
    if task:
        title += f' ({task})'
    plt.title(title)
    plt.savefig(f"tsne_by_class_{task.replace('/', '_') if task else 'dataset'}.png")
    plt.close()
    
    # Plot t-SNE by metadata if available
    if metadata_subset is not None:
        # Try to find interesting metadata columns to visualize
        if len(metadata_subset.shape) > 1 and metadata_subset.shape[1] > 0:
            # For each metadata column, create a visualization
            for col_idx in range(metadata_subset.shape[1]):
                try:
                    domain_value = metadata_subset[:, col_idx]
                    plt.figure(figsize=(12, 10))
                    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                         c=domain_value, cmap='Set3', alpha=0.7)
                    plt.colorbar(scatter, label=f'Metadata Column {col_idx}')
                    plt.title(f't-SNE Visualization by Metadata Column {col_idx}')
                    plt.savefig(f"tsne_by_metadata_col{col_idx}_{task.replace('/', '_') if task else 'dataset'}.png")
                    plt.close()
                except Exception as e:
                    print(f"Failed to visualize metadata column {col_idx}: {e}")
        else:
            # If metadata is a single column, visualize it
            try:
                plt.figure(figsize=(12, 10))
                scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                     c=metadata_subset, cmap='Set3', alpha=0.7)
                plt.colorbar(scatter, label='Metadata')
                plt.title('t-SNE Visualization by Metadata')
                plt.savefig(f"tsne_by_metadata_{task.replace('/', '_') if task else 'dataset'}.png")
                plt.close()
            except Exception as e:
                print(f"Failed to visualize metadata: {e}")


def visualize_attention(attention_maps, task=None, n_samples=5):
    """Visualize attention maps for a few samples."""
    if not attention_maps:
        print("No attention maps available for visualization")
        return
    
    task_str = task.replace('/', '_') if task else 'dataset'
    
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
                if n_heads > 1:
                    fig.colorbar(im, ax=axes.ravel().tolist())
                else:
                    fig.colorbar(im, ax=axes)
            
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
                fig.colorbar(im, ax=axes.ravel().tolist() if n_layers > 1 or n_heads > 1 else axes)
            
            plt.tight_layout()
            plt.suptitle(f'Attention Maps - Sample {i+1}')
            plt.savefig(f"attention_sample_{i+1}_{k}_{task_str}.png")
            plt.close()


def visualize_weights(task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers):
    """Visualize model weights and gradients."""
    from clip_benchmark.metrics import zeroshot_classification as zsc
    
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

    # Create the zero-shot classifier
    classifier = zsc.zero_shot_classifier(
        model,
        open_clip.get_tokenizer(model_arch),
        classnames,
        zeroshot_templates,
        device,
    )

    task_str = task.replace('/', '_') if task else 'dataset'

    # 1. Visualize first layer filters (for CNN models)
    if hasattr(model, 'visual') and hasattr(model.visual, 'conv1'):
        # For CLIP-like models
        filters = model.visual.conv1.weight.data.cpu()
        visualize_conv_filters(filters, f"conv1_filters_{task_str}.png")
    
    elif hasattr(model, 'conv1'):
        # For standard CNN models
        filters = model.conv1.weight.data.cpu()
        visualize_conv_filters(filters, f"conv1_filters_{task_str}.png")
    
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


def visualize_classifier_weights(classifier, task=None):
    """Visualize classifier weights."""
    task_str = task.replace('/', '_') if task else 'dataset'
    
    # Get the final layer weights
    if hasattr(classifier, 'weight'):
        weights = classifier.weight.data.cpu()
        
        # Create heatmap using matplotlib
        plt.figure(figsize=(12, 8))
        im = plt.imshow(weights.numpy(), cmap='coolwarm', aspect='auto')
        plt.colorbar(im, label='Weight Value')
        plt.title(f'Classifier Weights Heatmap ({task_str})')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Class')
        plt.tight_layout()
        plt.savefig(f"classifier_weights_{task_str}.png")
        plt.close()
        
        # Create weight norm distribution
        weight_norms = torch.norm(weights, dim=1).numpy()
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(weight_norms)), weight_norms)
        plt.title(f'Classifier Weight Norms by Class ({task_str})')
        plt.xlabel('Class')
        plt.ylabel('Weight Norm')
        plt.tight_layout()
        plt.savefig(f"classifier_weight_norms_{task_str}.png")
        plt.close()

    elif isinstance(classifier, torch.Tensor):
        weights = classifier.cpu()
        
        # Create heatmap using matplotlib
        plt.figure(figsize=(12, 8))
        im = plt.imshow(weights.numpy(), cmap='coolwarm', aspect='auto')
        plt.colorbar(im, label='Weight Value')
        plt.title(f'Classifier Weights Heatmap ({task_str})')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Class')
        plt.tight_layout()
        plt.savefig(f"classifier_weights_{task_str}.png")
        plt.close()
        
        # Create weight norm distribution
        weight_norms = torch.norm(weights, dim=1).numpy()
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(weight_norms)), weight_norms)
        plt.title(f'Classifier Weight Norms by Class ({task_str})')
        plt.xlabel('Class')
        plt.ylabel('Weight Norm')
        plt.tight_layout()
        plt.savefig(f"classifier_weight_norms_{task_str}.png")
        plt.close()


def visualize_activation_distribution(activations, task=None):
    """Visualize the distribution of activations across layers."""
    task_str = task.replace('/', '_') if task else 'dataset'
    n_layers = len(activations)
    
    plt.figure(figsize=(12, 8))
    for i, (layer_name, activation) in enumerate(activations.items()):
        activation_flat = activation.view(-1).cpu().numpy()
        plt.hist(
            activation_flat, 
            bins=50, 
            alpha=0.5, 
            density=True, 
            label=layer_name,
            histtype='step',
            linewidth=2
        )
    
    plt.title(f'Activation Distributions Across Layers ({task_str})')
    plt.xlabel('Activation Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"activation_distributions_{task_str}.png")
    plt.close()