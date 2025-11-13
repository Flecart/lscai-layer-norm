import os

# Enable Ray Train V2 for the latest train APIs
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import logging
import tempfile
import uuid

import ray
import ray.train
import ray.train.torch
import torch
import torch.profiler
import torch.distributed.checkpoint as dcp
from ray.train import Checkpoint
from ray.train import ScalingConfig, RunConfig, FailureConfig
from ray.train.torch import TorchTrainer
from torchvision.datasets import FashionMNIST
from torchvision.models import VisionTransformer
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

# Profiling and utilities
logger = logging.getLogger(__name__)


def init_model() -> torch.nn.Module:
    """Initialize a Vision Transformer model for FashionMNIST classification.

    Returns:
        torch.nn.Module: Configured ViT model
    """
    logger.info("Initializing Vision Transformer model...")

    # Create a ViT model with architecture suitable for 28x28 images
    model = VisionTransformer(
        image_size=28,        # FashionMNIST image size
        patch_size=7,         # Divide 28x28 into 4x4 patches of 7x7 pixels each
        num_layers=10,        # Number of transformer encoder layers
        num_heads=2,          # Number of attention heads per layer
        hidden_dim=128,       # Hidden dimension size
        mlp_dim=128,          # MLP dimension in transformer blocks
        num_classes=10,       # FashionMNIST has 10 classes
    )

    # Modify the patch embedding layer for grayscale images (1 channel instead of 3)
    model.conv_proj = torch.nn.Conv2d(
        in_channels=1,        # FashionMNIST is grayscale (1 channel)
        out_channels=128,     # Match the hidden_dim
        kernel_size=7,        # Match patch_size
        stride=7,             # Non-overlapping patches
    )

    return model


def train_func(config):
    """Main training function that integrates FSDP2 with Ray Train.

    Args:
        config: Training configuration dictionary containing hyperparameters
    """

    # Initialize the model
    model = init_model()

    # Configure device and move model to GPU
    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    model.to(device)

    # Apply FSDP2 sharding to the model
    shard_model(model)

    # Initialize loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.get('learning_rate', 0.001))

    # Load from checkpoint if available (for resuming training)
    start_epoch = 0
    loaded_checkpoint = ray.train.get_checkpoint()
    if loaded_checkpoint:
        latest_epoch = load_fsdp_checkpoint(model, optimizer, loaded_checkpoint)
        start_epoch = latest_epoch + 1 if latest_epoch is not None else 0
        logger.info(f"Resuming training from epoch {start_epoch}")

    # Prepare training data
    transform = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])
    data_dir = os.path.join(tempfile.gettempdir(), "data")
    train_data = FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config.get('batch_size', 64),
        shuffle=True
    )
    # Prepare data loader for distributed training
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    world_rank = ray.train.get_context().get_world_rank()

    # Set up PyTorch Profiler for memory monitoring
    with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
       record_shapes=True,
       profile_memory=True,
       with_stack=True,
   ) as prof:

        # Main training loop
        running_loss = 0.0
        num_batches = 0
        epochs = config.get('epochs', 5)

        for epoch in range(start_epoch, epochs):
            # Set epoch for distributed sampler to ensure proper shuffling
            if ray.train.get_context().get_world_size() > 1:
                train_loader.sampler.set_epoch(epoch)

            for images, labels in train_loader:
                # Note: prepare_data_loader automatically moves data to the correct device
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Standard training step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update profiler
                prof.step()

                # Track metrics
                running_loss += loss.item()
                num_batches += 1

            # Report metrics and save checkpoint after each epoch
            avg_loss = running_loss / num_batches
            metrics = {"loss": avg_loss}
            report_metrics_and_save_fsdp_checkpoint(model, optimizer, metrics, epoch)

            # Log metrics from rank 0 only to avoid duplicate outputs
            if world_rank == 0:
                logger.info(metrics)

    # Export memory profiling results to cluster storage
    run_name = ray.train.get_context().get_experiment_name()
    prof.export_memory_timeline(
        f"/mnt/cluster_storage/{run_name}/rank{world_rank}_memory_profile.html"
    )

    # Save the final model for inference
    save_model_for_inference(model, world_rank)


def shard_model(model: torch.nn.Module):
    """Apply FSDP2 sharding to the model with optimized configuration.

    Args:
        model: The PyTorch model to shard
    """
    logger.info("Applying FSDP2 sharding to model...")

    # Step 1: Create 1D device mesh for data parallel sharding
    world_size = ray.train.get_context().get_world_size()
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("data_parallel",)
    )

    # Step 2: Configure CPU offloading policy (optional)
    offload_policy = CPUOffloadPolicy()

    # Step 3: Configure mixed precision policy (optional)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float16,    # Store parameters in half precision
        reduce_dtype=torch.float16,   # Use half precision for gradient reduction
    )

    # Step 4: Apply sharding to each transformer encoder block
    for encoder_block in model.encoder.layers.children():
        fully_shard(
            encoder_block,
            mesh=mesh,
            reshard_after_forward=True,   # Free memory after forward pass
            offload_policy=offload_policy,
            mp_policy=mp_policy
        )

    # Step 5: Apply sharding to the root model
    # This wraps the entire model and enables top-level FSDP2 functionality
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=True,   # Free memory after forward pass
        offload_policy=offload_policy,
        mp_policy=mp_policy
    )


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Because this object is compliant
    with the Stateful protocol, PyTorch DCP automatically calls state_dict/load_state_dict as needed in the
    dcp.save/load APIs.

    Note: This wrapper is used to handle calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None, epoch=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch

    def state_dict(self):
        # this line automatically manages FSDP2 FQN's (Fully Qualified Name), as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "epoch": self.epoch
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that loading is complete
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        # Load epoch information if available
        if "epoch" in state_dict:
            self.epoch = state_dict["epoch"]


def load_fsdp_checkpoint(model: FSDPModule, optimizer: torch.optim.Optimizer, ckpt: Checkpoint) -> int | None:
    """Load an FSDP checkpoint into the model and optimizer.

    This function handles distributed checkpoint loading with automatic resharding
    support. It can restore checkpoints even when the number of workers differs
    from the original training run.

    Args:
        model: The FSDP-wrapped model to load state into
        optimizer: The optimizer to load state into
        ckpt: Ray Train checkpoint containing the saved state

    Returns:
        int: The epoch number saved within the checkpoint.
    """
    logger.info("Loading distributed checkpoint for resuming training...")

    try:
        with ckpt.as_directory() as checkpoint_dir:
            # Create state wrapper for DCP loading
            app_state = AppState(model, optimizer)
            state_dict = {"app": app_state}

            # Load the distributed checkpoint
            dcp.load(
                state_dict=state_dict,
                checkpoint_id=checkpoint_dir
            )

        logger.info(f"Successfully loaded distributed checkpoint from epoch {app_state.epoch}")
        return app_state.epoch
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e


def report_metrics_and_save_fsdp_checkpoint(
    model: FSDPModule, optimizer: torch.optim.Optimizer, metrics: dict, epoch: int = 0
) -> None:
    """Report training metrics and save an FSDP checkpoint.

    This function performs two critical operations:
    1. Saves the current model and optimizer state using distributed checkpointing
    2. Reports metrics to Ray Train for tracking

    Args:
        model: The FSDP-wrapped model to checkpoint
        optimizer: The optimizer to checkpoint
        metrics: Dictionary of metrics to report (e.g., loss, accuracy)
        epoch: The current epoch to be saved
    """
    logger.info("Saving checkpoint and reporting metrics...")

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        # Perform a distributed checkpoint with DCP
        state_dict = {"app": AppState(model, optimizer, epoch)}
        dcp.save(state_dict=state_dict, checkpoint_id=temp_checkpoint_dir)

        # Report each checkpoint shard from all workers
        # This saves the checkpoint to shared cluster storage for persistence
        checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
        ray.train.report(metrics, checkpoint=checkpoint)

    logger.info(f"Checkpoint saved successfully. Metrics: {metrics}")


def save_model_for_inference(model: FSDPModule, world_rank: int) -> None:
    """Save the complete unsharded model for inference.

    This function consolidates the distributed model weights into a single
    checkpoint file that can be used for inference without FSDP.

    Args:
        model: The FSDP2-wrapped model to save
        world_rank: The rank of the current worker
    """
    logger.info("Preparing model for inference...")

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        save_file = os.path.join(temp_checkpoint_dir, "full-model.pt")

        # Step 1: All-gather the model state across all ranks
        # This reconstructs the complete model from distributed shards
        model_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,    # Reconstruct full model
                cpu_offload=True,        # Move to CPU to save GPU memory
            )
        )

        logger.info("Successfully retrieved complete model state dict")
        checkpoint = None

        # Step 2: Save the complete model (rank 0 only)
        if world_rank == 0:
            torch.save(model_state_dict, save_file)
            logger.info(f"Saved complete model to {save_file}")

            # Create checkpoint for shared storage
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)

        # Step 3: Report the final checkpoint to Ray Train
        ray.train.report(
            {},
            checkpoint=checkpoint,
            checkpoint_dir_name="full_model"
        )


if __name__ == "__main__":
    # Configure distributed training resources
    scaling_config = ScalingConfig(
        num_workers=2,      # Number of distributed workers
        use_gpu=True        # Enable GPU training
    )

    # Configure training parameters
    train_loop_config = {
        "epochs": 5,
        "learning_rate": 0.001,
        "batch_size": 64,
    }

    # Create experiment name
    experiment_name = f"fsdp_mnist_{uuid.uuid4().hex[:8]}"

    # Configure run settings and storage
    run_config = RunConfig(
        # Persistent storage path accessible across all worker nodes
        storage_path="/mnt/cluster_storage/",
        # Unique experiment name (use consistent name to resume from checkpoints)
        name=experiment_name,
        # Fault tolerance configuration
        failure_config=FailureConfig(max_failures=1),
    )

    # Initialize and launch the distributed training job
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    )

    print("Starting FSDP2 training job...")
    result = trainer.fit()
    print("Training completed successfully!")

