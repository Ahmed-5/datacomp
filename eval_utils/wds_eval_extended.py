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
from clip_benchmark.metrics import zeroshot_classification as zsc
import glob
import json

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
    save_dir=None
):
    """Evaluate CLIP model on WILDS classification task."""

    task_dir = f"{task.replace('/', '-')}_results"
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        task_dir = os.path.join(save_dir, task_dir)
    
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)


    y_pred_path = os.path.join(task_dir, "y_pred.npy")
    y_target_path = os.path.join(task_dir, "y_target.npy")
    metrics_path = os.path.join(task_dir, "metrics.json")
    if os.path.exists(y_pred_path):
        y_pred = np.load(y_pred_path)
        y_target = np.load(y_target_path)
        metrics = json.load(open(metrics_path, "r"))
    else:
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

        # Save results
        np.save(y_pred_path, y_pred)
        np.save(y_target_path, y_target)
        json.dump(metrics, open(metrics_path, "w"))

    # clear gpu cache
    torch.cuda.empty_cache()
    
    # Run additional tests if requested
    if run_additional_tests:
        # Load model for additional tests
        model, transform, device = load_model(model_arch, model_path)
        
        # Get features and predictions
        features_paths, logits_paths, attention_maps_paths = extract_features_logits_and_attention(
            task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers, save_dir=task_dir
        )
        
        # 1. Entropy Analysis
        entropy_metrics = compute_entropy_metrics(logits_paths, save_dir=task_dir)
        metrics.update(entropy_metrics)
        
        # 2. Redundancy Reduction
        redundancy_metrics = analyze_redundancy(features_paths, save_dir=task_dir, y_target=y_target, y_pred=y_pred)
        metrics.update(redundancy_metrics)
        
        # 3. t-SNE Visualization
        visualize_tsne(features_paths, y_target, task, save_dir=task_dir)
        
        # 4. Attention Visualization
        visualize_attention(attention_maps_paths, task, n_samples=5, save_dir=task_dir)
        
        # 5. Weights/Gradient Visualization
        visualize_weights(task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers, save_dir=task_dir)

    # save metrics
    json.dump(metrics, open(metrics_path, "w"))

    return metrics


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
    task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers, save_dir=None, num_samples=5
):
    """
    Extract features, logits from the model, and save attention maps to disk per batch.

    Args:
        task (str): Identifier for the task/dataset.
        model (nn.Module): The model to use.
        model_arch (str): Name of the model architecture (for tokenizer).
        transform (callable): Preprocessing transform for images.
        data_root (str): Path to the dataset.
        dataset_len (int): Number of samples in the dataset.
        batch_size (int): Processing batch size.
        num_workers (int): Number of data loading workers.
        save_dir (str): Directory to save attention map files and plots.

    Returns:
        tuple: (features, logits, attention_maps_dir)
               features (torch.Tensor): Concatenated features for all samples (on CPU).
               logits (torch.Tensor): Concatenated logits for all samples (on CPU).
               attention_maps_dir (str): Path to the directory containing saved attention maps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Ensure save directory exists ---
    attention_save_path = os.path.join(save_dir, f"attention_maps_{task.replace('/', '_')}")
    os.makedirs(attention_save_path, exist_ok=True)
    print(f"Attention maps will be saved in: {attention_save_path}")
    features_save_path = os.path.join(save_dir, f"features_{task.replace('/', '_')}")
    os.makedirs(features_save_path, exist_ok=True)
    print(f"Features will be saved in: {features_save_path}")
    logits_save_path = os.path.join(save_dir, f"logits_{task.replace('/', '_')}")
    os.makedirs(logits_save_path, exist_ok=True)
    print(f"Logits will be saved in: {logits_save_path}")

    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers
    )

    # check if the entropy image exists
    entropy_img_path = os.path.join(save_dir, "entropy_distribution.png")

    if os.path.exists(entropy_img_path):
        print("Attention maps and logits already exist. Skipping feature extraction.")
        return features_save_path, logits_save_path, attention_save_path

    # Get zeroshot classifier setup
    zeroshot_templates = getattr(dataset, "templates", None)
    classnames = getattr(dataset, "classes", None)
    assert (
        zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification (missing 'templates' or 'classes')"

    # Ensure model is on the correct device
    model.to(device)
    model.eval() # Set model to evaluation mode

    # Create the zero-shot classifier and move to GPU
    # Make sure zsc and get_tokenizer are correctly imported/defined
    classifier = zsc.zero_shot_classifier(
        model,
        open_clip.get_tokenizer(model_arch),
        classnames,
        zeroshot_templates,
        device,
    )

    # Find all attention modules
    attn_modules = []
    # Ensure we are looking inside the visual transformer part
    visual_encoder = model.visual if hasattr(model, 'visual') else model # Adapt if model structure differs
    for name, module in visual_encoder.named_modules():
        # Adjust the check if your model uses a different Attention class
        if isinstance(module, torch.nn.MultiheadAttention):
            attn_modules.append((name, module))
            # print(f"Found MultiheadAttention module: {name}")
        # Add checks for other potential attention types if needed
        # elif isinstance(module, SomeOtherAttentionClass):
        #     attn_modules.append((name, module))

    if not attn_modules:
        print("Warning: No MultiheadAttention modules found in model.visual. Attention maps will not be extracted.")

    # Store original forward methods
    original_forwards = {}

    # Define the patched forward method with memory optimization
    def patched_forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                       attn_mask=None, average_attn_weights=False):
        # Call the original implementation but force need_weights=True and average_attn_weights=False
        # The original forward method is stored in self._original_forward
        output, attn_weights = self._original_forward(query, key, value,
                                                    key_padding_mask=key_padding_mask,
                                                    need_weights=True, # Force calculation
                                                    attn_mask=attn_mask,
                                                    average_attn_weights=False) # Ensure we get per-head weights

        # Store the attention weights - use clone() and move to CPU immediately
        if attn_weights is not None:
            # Detach and move to CPU to free GPU memory quickly
            self.last_attn_weights = attn_weights.detach().cpu()
        else:
            # print("Warning: attn_weights is None even with need_weights=True")
            self.last_attn_weights = None

        # Return the original expected output format (respecting original need_weights)
        # Note: The original caller might not expect attn_weights if need_weights was False.
        #       However, for feature extraction, we usually don't need the attn_weights returned here.
        #       If a downstream part *of the original model* requires attn_weights conditionally,
        #       this patch might need adjustment. For just *extracting* them, this is fine.
        return output, attn_weights if need_weights else None


    # Apply the patch to all attention modules
    for name, module in attn_modules:
        if not hasattr(module, '_original_forward'): # Avoid double patching if function called multiple times
            original_forwards[name] = module.forward
            module._original_forward = module.forward # Store original
            module.forward = types.MethodType(patched_forward, module)
        module.last_attn_weights = None # Initialize storage attribute
    # No longer need attention_maps_list in RAM

    # Use torch.cuda.amp for mixed precision if GPU supports it
    use_amp = device.type == 'cuda'
    print(f"Using mixed precision (AMP): {use_amp}")

    # Processing loop
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features and attention")):
        # Clear CUDA cache periodically (optional, can sometimes help or hurt)
        # if batch_idx > 0 and batch_idx % 10 == 0:
        #    torch.cuda.empty_cache()

        # Stop early for testing if needed
        # if batch_idx > 20:
        #    print("Stopping early for testing.")
        #    break

        # Ensure batch structure is as expected (e.g., [images, labels] or just images)
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
            # labels = batch[1].to(device) # If labels are needed
        else:
            images = batch.to(device) # Assuming batch is just images

        batch_attention_maps = {} # Collect maps for this batch

        # Reset the stored attention weights before the forward pass
        for _, module in attn_modules:
            if hasattr(module, 'last_attn_weights'):
                module.last_attn_weights = None

        with torch.no_grad(): # Ensure no gradients are computed
            with torch.cuda.amp.autocast(enabled=use_amp): # Use mixed precision
                # Forward pass to get features
                # This implicitly calls the patched forward methods of attention layers
                features = model.encode_image(images)

                # Check if features require normalization (common in CLIP)
                if features.ndim == 2: # Expected shape (batch_size, embedding_dim)
                     image_features = F.normalize(features, dim=-1)
                     logits = 100. * image_features @ classifier # Calculate logits
                else:
                    # Handle cases where encode_image might return intermediate features
                    print(f"Warning: Unexpected feature shape {features.shape}. Assuming final projection.")
                    # Add logic here if needed, e.g., pooling or flattening
                    image_features = F.normalize(features.view(features.size(0), -1), dim=-1) # Example flatten
                    logits = 100. * image_features @ classifier

        # Move results to CPU immediately to free GPU memory
        features_cpu = features.detach().cpu()
        logits_cpu = logits.detach().cpu()

        # --- Save attention maps for this batch ---
        if num_samples is not None and batch_idx < num_samples:
            # Collect attention weights from all modules (already on CPU from patched_forward)
            for name, module in attn_modules:
                if hasattr(module, 'last_attn_weights') and module.last_attn_weights is not None:
                    # Store the CPU tensor
                    batch_attention_maps[name] = module.last_attn_weights
                    module.last_attn_weights = None # Clear reference in module
            
            # --- Save attention maps for this batch ---
            attn_file_path = os.path.join(attention_save_path, f"attention_batch_{batch_idx:05d}.pt")
            torch.save(batch_attention_maps, attn_file_path)
            # print(f"Saved attention maps for batch {batch_idx} to {attn_file_path}")

        
        # --- Save features and logits for this batch ---
        features_file_path = os.path.join(features_save_path, f"features_batch_{batch_idx:05d}.pt")
        torch.save(features_cpu, features_file_path)
        # print(f"Saved features for batch {batch_idx} to {features_file_path}")

        logits_file_path = os.path.join(logits_save_path, f"logits_batch_{batch_idx:05d}.pt")
        torch.save(logits_cpu, logits_file_path)
        # print(f"Saved logits for batch {batch_idx} to {logits_file_path}")

        # Explicitly delete tensors to potentially free memory sooner
        del images, features, image_features, logits, features_cpu, logits_cpu, batch_attention_maps
        if device.type == 'cuda':
             torch.cuda.empty_cache() # Clear cache after deleting


    # --- Restore original forward methods ---
    print("Restoring original forward methods...")
    for name, module in attn_modules:
        if name in original_forwards:
            module.forward = original_forwards[name]
            if hasattr(module, '_original_forward'):
                delattr(module, '_original_forward')
            if hasattr(module, 'last_attn_weights'):
                delattr(module, 'last_attn_weights')
        # else:
            # print(f"Warning: Original forward method not found for {name}. Skipping restore.")

    print(f"Extraction complete.")

    return features_save_path, logits_save_path, attention_save_path


def compute_entropy_metrics(logits_paths, save_dir):
    logits_files = sorted(glob.glob(os.path.join(logits_paths, "logits_batch_*.pt")))

    # Collect logits from all batches
    all_logits = []
    for logits_file in logits_files:
        logits = torch.load(logits_file)
        all_logits.append(logits)
    logits = torch.cat(all_logits, dim=0)

    """Compute entropy metrics for model predictions and save plot."""
    if logits.numel() == 0:
        print("Skipping entropy metrics: No logits provided.")
        return {}
        
    print("Computing entropy metrics...")
    # Convert logits to probabilities (move to CPU if necessary)
    probs = torch.softmax(logits.cpu().float(), dim=1).numpy() # Use float32 for stability

    # Calculate entropy for each prediction
    # Add small epsilon to prevent log(0)
    entropies = np.array([entropy(p + 1e-9) for p in probs])

    # Calculate metrics
    metrics = {
        "mean_entropy": float(np.mean(entropies)),
        "median_entropy": float(np.median(entropies)),
        "max_entropy": float(np.max(entropies)),
        "min_entropy": float(np.min(entropies)),
    }

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, "entropy_distribution.png")

    # Plot entropy distribution
    plt.figure(figsize=(10, 6))
    plt.hist(entropies, bins=30, color='skyblue', edgecolor='black')
    plt.title("Prediction Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    try:
        plt.savefig(img_path)
        print(f"Saved entropy distribution plot to: {img_path}")
    except Exception as e:
        print(f"Error saving entropy plot: {e}")
    plt.close() # Close the plot to free memory

    return metrics


def analyze_redundancy(features_paths, save_dir, min_feaures=None, y_target=None, y_pred=None):
    features_files = sorted(glob.glob(os.path.join(features_paths, "features_batch_*.pt")))

    # Collect features from all batches
    all_features = []
    for features_file in features_files:
        features = torch.load(features_file)
        all_features.append(features)
    features = torch.cat(all_features, dim=0)
    """Analyze feature redundancy using correlation and save plot."""
    if features.numel() == 0:
        print("Skipping feature redundancy analysis: No features provided.")
        return {}
        
    print("Analyzing feature redundancy...")
    # Compute feature correlation matrix (move to CPU if necessary)
    features_np = features.cpu().float().numpy() # Use float32

    # Handle cases with very few features or samples
    if features_np.shape[0] < 2 or features_np.shape[1] < 2:
        print(f"Skipping correlation: Not enough samples or features ({features_np.shape})")
        return {"mean_feature_correlation": 0.0, "max_feature_correlation": 0.0, "feature_redundancy_score": 0.0}

    corr_matrix = np.corrcoef(features_np.T) # Transpose to correlate features

    # Handle potential NaNs if features have zero variance
    corr_matrix = np.nan_to_num(corr_matrix)

    # Calculate redundancy metrics
    abs_corr = np.abs(corr_matrix)
    np.fill_diagonal(abs_corr, 0)  # Exclude self-correlations

    # Avoid division by zero if only one feature
    num_elements = abs_corr.size
    num_features = features_np.shape[1]
    denominator = num_elements - num_features if num_elements > num_features else 1

    metrics = {
        "mean_feature_correlation": float(np.mean(abs_corr)),
        "max_feature_correlation": float(np.max(abs_corr)),
        # Score: proportion of highly correlated pairs (excluding diagonal)
        "feature_redundancy_score": float(np.sum(abs_corr > 0.8) / denominator),
    }

    # Plot correlation heatmap (subset for visibility)
    plot_dim = corr_matrix.shape[0]
    if min_feaures is not None:
        plot_dim = min(min_feaures, plot_dim)
    plt.figure(figsize=(10, 8)) # Adjusted size

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, "feature_correlation.png")

    im = plt.imshow(corr_matrix[:plot_dim, :plot_dim], cmap="coolwarm", vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, label="Correlation", shrink=0.8) # Adjust colorbar size
    plt.title(f"Feature Correlation Matrix (First {plot_dim} dims)")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.tight_layout()
    try:
        plt.savefig(img_path)
        print(f"Saved feature correlation plot to: {img_path}")
    except Exception as e:
        print(f"Error saving correlation plot: {e}")
    plt.close() # Close the plot

    if y_pred is None or y_target is None:
        return metrics
    

    unique_targets = np.unique(y_target)
    num_classes = len(unique_targets)
    metrics["num_classes"] = num_classes

    save_dir_classified = os.path.join(save_dir, "classified")
    metrics["classified"] = {}
    os.makedirs(save_dir_classified, exist_ok=True)
    for i, target in enumerate(unique_targets):
        features_by_class = features_np[y_pred == target]
        corr_matrix = np.corrcoef(features_by_class.T) # Transpose to correlate features
        corr_matrix = np.nan_to_num(corr_matrix)
        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, 0)  # Exclude self-correlations
        num_elements = abs_corr.size
        num_features = features_by_class.shape[1]
        denominator = num_elements - num_features if num_elements > num_features else 1
        target_str = str(target)
        metrics["classified"][target_str] = {
            "mean_feature_correlation": float(np.mean(np.abs(abs_corr))),
            "max_feature_correlation": float(np.max(np.abs(abs_corr))),
            # Score: proportion of highly correlated pairs (excluding diagonal)
            "feature_redundancy_score": float(np.sum(np.abs(abs_corr) > 0.8) / denominator),
        }

        # Plot correlation heatmap (subset for visibility)
        plot_dim = corr_matrix.shape[0]
        if min_feaures is not None:
            plot_dim = min(min_feaures, plot_dim)

        img_path = os.path.join(save_dir_classified, f"feature_correlation_class_{i}.png")
        plt.figure(figsize=(10, 8)) # Adjusted size

        im = plt.imshow(corr_matrix[:plot_dim, :plot_dim], cmap="coolwarm", vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, label="Correlation", shrink=0.8) # Adjust colorbar size
        plt.title(f"Feature Correlation Matrix (First {plot_dim} dims)")
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Index")
        plt.tight_layout()
        try:
            plt.savefig(img_path)
            print(f"Saved feature correlation plot to: {img_path}")
        except Exception as e:
            print(f"Error saving correlation plot: {e}")
        plt.close() # Close the plot

    save_dir_target = os.path.join(save_dir, "target")
    os.makedirs(save_dir_target, exist_ok=True)
    metrics["target"] = {}

    for i, target in enumerate(unique_targets):
        features_by_class = features_np[y_target == target]
        corr_matrix = np.corrcoef(features_by_class.T) # Transpose to correlate features
        corr_matrix = np.nan_to_num(corr_matrix)
        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, 0)  # Exclude self-correlations
        num_elements = abs_corr.size
        num_features = features_by_class.shape[1]
        denominator = num_elements - num_features if num_elements > num_features else 1
        target_str = str(target)
        metrics["target"][target_str] = {
            "mean_feature_correlation": float(np.mean(np.abs(abs_corr))),
            "max_feature_correlation": float(np.max(np.abs(abs_corr))),
            # Score: proportion of highly correlated pairs (excluding diagonal)
            "feature_redundancy_score": float(np.sum(np.abs(abs_corr) > 0.8) / denominator),
        }

        # Plot correlation heatmap (subset for visibility)
        plot_dim = corr_matrix.shape[0]
        if min_feaures is not None:
            plot_dim = min(min_feaures, plot_dim)

        img_path = os.path.join(save_dir_target, f"feature_correlation_target_{i}.png")
        plt.figure(figsize=(10, 8)) # Adjusted size

        im = plt.imshow(corr_matrix[:plot_dim, :plot_dim], cmap="coolwarm", vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, label="Correlation", shrink=0.8) # Adjust colorbar size
        plt.title(f"Feature Correlation Matrix (First {plot_dim} dims)")
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Index")
        plt.tight_layout()
        try:
            plt.savefig(img_path)
            print(f"Saved feature correlation plot to: {img_path}")
        except Exception as e:
            print(f"Error saving correlation plot: {e}")
        plt.close() # Close the plot

    return metrics


def visualize_tsne(features_paths, y_target, task, perplexity=30, save_dir=None):
    features_files = sorted(glob.glob(os.path.join(features_paths, "features_batch_*.pt")))

    # Collect features from all batches
    all_features = []
    for features_file in features_files:
        features = torch.load(features_file)
        all_features.append(features)
    features = torch.cat(all_features, dim=0)
    """Create t-SNE visualization of features colored by class and save plot."""
    if features.numel() == 0 or y_target is None or len(y_target) == 0:
        print("Skipping t-SNE visualization: No features or targets provided.")
        return
        
    print("Computing t-SNE projection...")
    if not save_dir:
        print("Warning: save_dir not specified for t-SNE plot. Plot will not be saved.")
        
    # Use a subset if features is too large for efficiency
    max_samples = min(len(features), 5000) # Limit sample size for t-SNE
    if len(features) > max_samples:
         print(f"Using a subset of {max_samples} samples for t-SNE.")
         indices = np.random.choice(len(features), max_samples, replace=False)
         features_subset = features[indices]
         # Ensure y_target is indexable (convert to numpy array if it's a list)
         y_target_np = np.array(y_target)
         y_subset = y_target_np[indices]
    else:
         features_subset = features
         y_subset = np.array(y_target)

    # Ensure features are on CPU and numpy
    features_np = features_subset.cpu().float().numpy()

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, max_samples - 1), # Perplexity must be < n_samples
                n_iter=1000, learning_rate='auto', init='pca', random_state=42, n_jobs=-1) # Use multiple cores
    try:
        tsne_result = tsne.fit_transform(features_np)
    except ValueError as e:
        print(f"t-SNE failed: {e}. Skipping visualization.")
        return

    # Plot t-SNE colored by class
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_subset, cmap='tab20', alpha=0.7)

    # Create a legend if number of classes is manageable
    num_classes = len(np.unique(y_subset))
    if num_classes <= 20: # 'tab20' has 20 distinct colors
        handles, labels = scatter.legend_elements(prop='colors')
        plt.legend(handles, labels, title="Classes", loc="upper right")
    else:
         plt.colorbar(scatter, label='Class ID') # Use colorbar for many classes

    plt.title(f't-SNE Visualization of Features by Class ({task})')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()

    if save_dir:
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, f"tsne_by_class_{task.replace('/', '_')}.png")
        try:
            plt.savefig(img_path)
            print(f"Saved t-SNE plot to: {img_path}")
        except Exception as e:
            print(f"Error saving t-SNE plot: {e}")
    plt.close() # Close the plot


def visualize_attention(attention_maps_dir, task, n_samples=5, save_dir=None):
    """Load saved attention maps from files and visualize for a few samples."""
    print(f"Visualizing attention maps from: {attention_maps_dir}")
    if not save_dir:
        print("Warning: save_dir not specified for attention plots. Plots will not be saved.")
    else:
        save_dir = os.path.join(save_dir, "attention_maps")
        os.makedirs(save_dir, exist_ok=True)

    # Find saved attention map files
    attn_files = sorted(glob.glob(os.path.join(attention_maps_dir, "attention_batch_*.pt")))

    if not attn_files:
        print("No attention map files found in the specified directory.")
        return

    # Select a few random batch files to visualize samples from
    n_files_to_load = min(n_samples, len(attn_files))
    selected_files = np.random.choice(attn_files, n_files_to_load, replace=False)

    loaded_samples_count = 0
    for i, file_path in enumerate(selected_files):
        if loaded_samples_count >= n_samples:
            break
        try:
            # Load attention maps for one batch
            batch_attention_maps = torch.load(file_path, map_location='cpu') # Load directly to CPU
        except Exception as e:
            print(f"Error loading attention file {file_path}: {e}")
            continue

        # Get the number of samples in this batch (check shape of one map)
        if not batch_attention_maps:
            # print(f"Batch {file_path} contained no attention maps.")
            continue

        # Infer batch size from the first attention map found
        first_map_key = next(iter(batch_attention_maps))
        batch_size_in_file = batch_attention_maps[first_map_key].shape[0] # Usually (Batch, Heads, Seq, Seq) or (Batch, Seq, Seq)

        # Choose a random sample index within this batch
        sample_idx_in_batch = np.random.randint(0, batch_size_in_file)
        
        print(f"Visualizing sample {loaded_samples_count+1} (from batch file {os.path.basename(file_path)}, index {sample_idx_in_batch})")

        # Iterate through attention layers/blocks saved for this sample
        for layer_name, attn_map_batch in batch_attention_maps.items():
            # Get the attention map for the specific sample index
            attn_map_sample = attn_map_batch[sample_idx_in_batch] # Shape: (Heads, Seq, Seq) or (Seq, Seq)

            # Determine shape: Could be [Heads, TargetSeq, SourceSeq] or [TargetSeq, SourceSeq]
            # For ViTs, often [Heads, NumPatches+CLS, NumPatches+CLS]
            
            if attn_map_sample.ndim == 3:  # [Heads, SeqLen, SeqLen]
                n_heads = attn_map_sample.shape[0]
                seq_len = attn_map_sample.shape[1]
                
                # Often square, calculate grid size (e.g., 14x14=196 for 224px image / 16px patch)
                # +1 for CLS token if present
                grid_size_approx = int(np.sqrt(seq_len -1)) if seq_len > 1 else 1 
                has_cls_token = (grid_size_approx * grid_size_approx) != (seq_len -1) if seq_len > 1 else False


                fig, axes = plt.subplots(1, n_heads, figsize=(n_heads * 3, 3.5)) # Adjust size
                if n_heads == 1: axes = [axes] # Make iterable

                for h in range(n_heads):
                    ax = axes[h]
                    # Often visualize attention from CLS token to patches, or mean attention
                    # Example: Mean attention weights across all source tokens for each target token
                    head_map = attn_map_sample[h].float().numpy() # Use float32 for imshow
                    
                    # Simple visualization: just show the raw head map
                    im = ax.imshow(head_map, cmap='viridis')
                    ax.set_title(f'Head {h+1}')
                    ax.axis('off')
                    
                fig.colorbar(im, ax=axes, shrink=0.8) # Add a single colorbar

            elif attn_map_sample.ndim == 2: # [SeqLen, SeqLen] - Averaged over heads already?
                seq_len = attn_map_sample.shape[0]
                grid_size_approx = int(np.sqrt(seq_len - 1)) if seq_len > 1 else 1
                has_cls_token = (grid_size_approx * grid_size_approx) != (seq_len - 1) if seq_len > 1 else False
                
                fig, ax = plt.subplots(1, 1, figsize=(5, 4))
                im = ax.imshow(attn_map_sample.float().numpy(), cmap='viridis')
                ax.set_title('Attention Map (Avg?)')
                ax.axis('off')
                fig.colorbar(im, ax=ax, shrink=0.8)

            else: # Handle other potential shapes if necessary
                print(f"Skipping visualization for layer {layer_name}: Unexpected attention map shape {attn_map_sample.shape}")
                continue

            plt.suptitle(f'Attention - Sample {loaded_samples_count+1}, Layer: {layer_name}', fontsize=10)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                # Sanitize layer name for filename
                safe_layer_name = "".join(c if c.isalnum() else "_" for c in layer_name)
                img_path = os.path.join(save_dir, f"attention_sample_{loaded_samples_count+1}_{safe_layer_name}_{task.replace('/', '_')}.png")
                try:
                    plt.savefig(img_path)
                    # print(f"Saved attention plot: {img_path}")
                except Exception as e:
                    print(f"Error saving attention plot {img_path}: {e}")
            plt.close() # Close plot

        loaded_samples_count += 1



def visualize_weights(task, model, model_arch, transform, data_root, dataset_len, batch_size, num_workers, save_dir=None):
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

    # save the classnames
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        classnames_path = os.path.join(save_dir, f"classnames_{task.replace('/', '_')}.txt")
        with open(classnames_path, "w") as f:
            for classname in classnames:
                f.write(f"{classname}\n")

    # 1. Visualize first layer filters (for CNN models)
    if hasattr(model, 'visual') and hasattr(model.visual, 'conv1'):
        # For CLIP-like models
        filters = model.visual.conv1.weight.data.cpu()
        img_path = os.path.join(save_dir, f"conv1_filters_{task.replace('/', '_')}.png")
        visualize_conv_filters(filters, img_path)
    
    elif hasattr(model, 'conv1'):
        # For standard CNN models
        filters = model.conv1.weight.data.cpu()
        img_path = os.path.join(save_dir, f"conv1_filters_{task.replace('/', '_')}.png")
        visualize_conv_filters(filters, img_path)
    
    # 2. Visualize final layer weights
    if classifier is not None:
        visualize_classifier_weights(classifier, task, save_dir)
    
    # 3. Visualize layer activations histogram
    if hasattr(model, 'layer_activations'):
        visualize_activation_distribution(model.layer_activations, task, save_dir)


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


def visualize_classifier_weights(classifier, task, save_dir=None):
    """Visualize classifier weights."""
    # Get the final layer weights
    if hasattr(classifier, 'weight'):
        weights = classifier.weight.data.cpu()
        # save weights in .npy format
        weights_path = os.path.join(save_dir, f"classifier_weights_{task.replace('/', '_')}.npy")
        np.save(weights_path, weights.numpy())

        img_path = os.path.join(save_dir, f"classifier_weights_{task.replace('/', '_')}.png")
        
        # Create heatmap using matplotlib instead of seaborn
        plt.figure(figsize=(12, 8))
        im = plt.imshow(weights.numpy(), cmap='coolwarm', aspect='auto')
        plt.colorbar(im, label='Weight Value')
        plt.title(f'Classifier Weights Heatmap ({task})')
        plt.xlabel('Class')
        plt.ylabel('Feature Dimension')
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
        
        # Create weight norm distribution
        img_path = os.path.join(save_dir, f"classifier_weight_norms_{task.replace('/', '_')}.png")
        weight_norms = torch.norm(weights, dim=1).numpy()
        # save weight norms in .npy format
        weight_norms_path = os.path.join(save_dir, f"classifier_weight_norms_{task.replace('/', '_')}.npy")
        np.save(weight_norms_path, weight_norms)

        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(weight_norms)), weight_norms)
        plt.title(f'Classifier Weight Norms by Class ({task})')
        plt.xlabel('Class')
        plt.ylabel('Weight Norm')
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

    if isinstance(classifier, torch.Tensor):
        print(classifier.shape)
        weights = classifier.cpu()
        # save weights in .npy format
        weights_path = os.path.join(save_dir, f"classifier_weights_{task.replace('/', '_')}.npy")
        np.save(weights_path, weights.numpy())
        
        # Create heatmap using matplotlib instead of seaborn
        img_path = os.path.join(save_dir, f"classifier_weights_{task.replace('/', '_')}.png")
        plt.figure(figsize=(12, 8))
        im = plt.imshow(weights.numpy(), cmap='coolwarm', aspect='auto')
        plt.colorbar(im, label='Weight Value')
        plt.title(f'Classifier Weights Heatmap ({task})')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Class')
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
        
        # Create weight norm distribution
        img_path = os.path.join(save_dir, f"classifier_weight_norms_{task.replace('/', '_')}.png")
        weight_norms = torch.norm(weights, dim=1).numpy()
        # save weight norms in .npy format
        weight_norms_path = os.path.join(save_dir, f"classifier_weight_norms_{task.replace('/', '_')}.npy")
        np.save(weight_norms_path, weight_norms)

        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(weight_norms)), weight_norms)
        plt.title(f'Classifier Weight Norms by Class ({task})')
        plt.xlabel('Class')
        plt.ylabel('Weight Norm')
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()


def visualize_activation_distribution(activations, task, save_dir=None):
    """Visualize the distribution of activations across layers."""
    n_layers = len(activations)

    img_path = os.path.join(save_dir, f"activation_distributions_{task.replace('/', '_')}.png")
    
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
    plt.savefig(img_path)
    plt.close()
