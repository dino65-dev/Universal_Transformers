# train_ut_act_complete.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, Dict
import logging
from tqdm import tqdm
import os
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Include the DummyDataset class here
class DummyDataset(Dataset):
    """Dummy dataset for language modeling"""
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        return input_ids, target_ids

class UTACTTrainer:
    """
    Complete trainer for Universal Transformer with Adaptive Computation Time
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        ponder_cost_weight: float = 0.01,  # τ parameter from ACT math
        device: str = "auto",
        save_dir: str = "./checkpoints_act",
        log_interval: int = 100,
        eval_interval: int = 500,
        save_interval: int = 1000,
        warmup_steps: int = 1000,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.ponder_cost_weight = ponder_cost_weight
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.warmup_steps = warmup_steps

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if self.device.type == "cuda" else False
            )

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        # Learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_loader) * 10  # 10 epochs
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Save directory
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

    def train_step(self, batch) -> Dict[str, float]:
        """Single training step with ACT ponder cost"""
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device, non_blocking=True)
        target_ids = target_ids.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        # Forward pass with ACT
        logits, ponder_costs, act_info = self.model(input_ids, return_act_info=True)

        # Task loss (standard language modeling loss)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)
        task_loss = self.criterion(logits_flat, targets_flat)

        # Ponder cost: Ω = τ * mean(pondering_time_per_position)
        ponder_loss = self.ponder_cost_weight * ponder_costs.mean()

        # Total loss: L = L_task + Ω
        total_loss = task_loss + ponder_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()

        # Learning rate schedule with warmup
        if self.step < self.warmup_steps:
            lr = self.optimizer.param_groups[0]['lr'] * (self.step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()

        # Calculate metrics
        perplexity = torch.exp(task_loss).item()
        avg_ponder = ponder_costs.mean().item()

        # ACT-specific metrics
        avg_updates = act_info['n_updates'].mean().item()
        avg_effective_steps = act_info['effective_steps'].mean().item()

        metrics = {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'ponder_loss': ponder_loss.item(),
            'perplexity': perplexity,
            'avg_ponder_cost': avg_ponder,
            'avg_updates': avg_updates,
            'avg_effective_steps': avg_effective_steps,
            'grad_norm': grad_norm.item(),
            'lr': self.get_lr()
        }

        return metrics

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        if not self.val_dataset:
            return {}

        self.model.eval()
        total_task_loss = 0
        total_ponder_cost = 0
        total_samples = 0

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            input_ids, target_ids = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)

            # Forward pass
            logits, ponder_costs, _ = self.model(input_ids)

            # Calculate losses
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)
            task_loss = self.criterion(logits_flat, targets_flat)

            # Accumulate metrics
            batch_size = input_ids.size(0)
            total_task_loss += task_loss.item() * batch_size
            total_ponder_cost += ponder_costs.mean().item() * batch_size
            total_samples += batch_size

        avg_task_loss = total_task_loss / total_samples
        avg_ponder = total_ponder_cost / total_samples
        val_perplexity = math.exp(avg_task_loss)

        self.model.train()

        return {
            'val_task_loss': avg_task_loss,
            'val_perplexity': val_perplexity,
            'val_avg_ponder': avg_ponder,
            'val_total_loss': avg_task_loss + self.ponder_cost_weight * avg_ponder
        }

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'ponder_cost_weight': self.ponder_cost_weight,
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_step_{self.step}.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

    def train(self, num_epochs: int):
        """Main training loop"""
        logger.info("Starting UT+ACT training...")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Ponder cost weight (τ): {self.ponder_cost_weight}")

        if self.val_dataset:
            logger.info(f"Validation samples: {len(self.val_dataset)}")

        self.model.train()

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = []

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=True
            )

            for batch in progress_bar:
                # Training step
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['total_loss'])

                self.step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'total': f"{metrics['total_loss']:.4f}",
                    'task': f"{metrics['task_loss']:.4f}",
                    'ponder': f"{metrics['ponder_loss']:.4f}",
                    'steps': f"{metrics['avg_effective_steps']:.2f}",
                    'lr': f"{metrics['lr']:.2e}"
                })

                # Logging
                if self.step % self.log_interval == 0:
                    logger.info(
                        f"Step {self.step} | "
                        f"Total: {metrics['total_loss']:.4f} | "
                        f"Task: {metrics['task_loss']:.4f} | "
                        f"Ponder: {metrics['ponder_loss']:.4f} | "
                        f"PPL: {metrics['perplexity']:.2f} | "
                        f"Avg Steps: {metrics['avg_effective_steps']:.2f} | "
                        f"LR: {metrics['lr']:.2e}"
                    )

                # Evaluation
                if self.step % self.eval_interval == 0 and self.val_dataset:
                    val_metrics = self.evaluate()

                    logger.info(
                        f"Validation | "
                        f"Task Loss: {val_metrics['val_task_loss']:.4f} | "
                        f"PPL: {val_metrics['val_perplexity']:.2f} | "
                        f"Avg Ponder: {val_metrics['val_avg_ponder']:.2f}"
                    )

                    # Save best model
                    is_best = val_metrics['val_total_loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['val_total_loss']

                    self.save_checkpoint(is_best=is_best)

                # Regular checkpoint saving
                elif self.step % self.save_interval == 0:
                    self.save_checkpoint()

            # End of epoch summary
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch+1} completed | Average loss: {avg_epoch_loss:.4f}")

        logger.info("Training completed!")

def main():
    """Complete example with UT+ACT"""
    # Import your UT+ACT model here
    # from ut_decoder_with_act import UniversalDecoderWithACT

    # For demonstration, assuming the model class exists
    print("Setting up UT+ACT training...")

    # Model configuration
    model_config = {
        'vocab_size': 50257,
        'dim': 512,
        'max_seq_len': 512,  # Shorter for ACT demonstration
        'num_heads': 8,
        'num_kv_heads': 4,
        'max_steps': 8,  # Maximum UT recurrence steps
        'dropout': 0.1,
        'act_threshold': 0.99,  # Halt when cumulative prob >= 0.99
        'ponder_cost_weight': 0.01
    }

    # Training configuration
    train_config = {
        'batch_size': 16,  # Smaller batch for ACT
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'ponder_cost_weight': 0.01,  # τ parameter
        'num_epochs': 3,
        'warmup_steps': 500,
        'log_interval': 50,
        'eval_interval': 200,
        'save_interval': 500,
    }

    print(f"Model config: {model_config}")
    print(f"Training config: {train_config}")

    # Create datasets
    train_dataset = DummyDataset(
        vocab_size=model_config['vocab_size'],
        seq_len=256,  # Shorter sequences for ACT
        num_samples=5000
    )

    val_dataset = DummyDataset(
        vocab_size=model_config['vocab_size'],
        seq_len=256,
        num_samples=500
    )

    print(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Note: You would uncomment these lines when you have the actual model
    # model = UniversalDecoderWithACT(**model_config)
    #
    # trainer = UTACTTrainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     **train_config
    # )
    #
    # # Start training
    # trainer.train(num_epochs=train_config['num_epochs'])
    #
    # # Test generation with ACT statistics
    # model.eval()
    # test_input = torch.randint(0, model_config['vocab_size'], (1, 10))
    # generated, act_stats = model.generate(
    #     test_input,
    #     max_new_tokens=20,
    #     verbose_act=True
    # )
    #
    # print(f"Generated sequence length: {generated.shape[1]}")
    # if act_stats:
    #     print("ACT Statistics for generation:")
    #     for i, stat in enumerate(act_stats[:5]):
    #         print(f"  Token {i}: {stat['avg_computation_steps']:.2f} avg UT steps")

if __name__ == "__main__":
    main()
