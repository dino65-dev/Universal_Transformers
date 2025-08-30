import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import os
import json
from tqdm import tqdm
from typing import Dict, Optional, List
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.
    Expects tokenized text as input.
    """
    def __init__(self, tokenized_text: List[int], seq_len: int):
        self.data = tokenized_text
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        # Return input and target (shifted by 1)
        chunk = self.data[idx:idx + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, target_ids

class DummyDataset(Dataset):
    """Dummy dataset for testing"""
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random sequence
        sequence = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        return input_ids, target_ids

class UTTrainer:
    """
    Trainer class for Universal Transformer decoder-only model.
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
        device: str = "auto",
        save_dir: str = "./checkpoints",
        log_interval: int = 100,
        eval_interval: int = 500,
        save_interval: int = 1000,
        warmup_steps: int = 1000,
        max_steps: Optional[int] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

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

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        # Learning rate scheduler with warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps if max_steps else len(self.train_loader) * 10
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

        # Training metrics
        self.train_losses = []
        self.val_losses = []

    def get_lr(self) -> float:
        """Get current learning rate with warmup"""
        if self.step < self.warmup_steps:
            # Linear warmup
            return self.optimizer.param_groups[0]['lr'] * (self.step / self.warmup_steps)
        else:
            return self.optimizer.param_groups[0]['lr']

    def train_step(self, batch) -> Dict[str, float]:
        """Single training step"""
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device, non_blocking=True)
        target_ids = target_ids.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        # Forward pass
        logits, _ = self.model(input_ids)

        # Calculate loss
        # Reshape for loss calculation: [batch * seq_len, vocab_size]
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)

        loss = self.criterion(logits_flat, targets_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )

        # Update parameters
        self.optimizer.step()

        # Update learning rate with warmup
        if self.step < self.warmup_steps:
            lr = self.optimizer.param_groups[0]['lr'] * (self.step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()

        # Calculate perplexity
        perplexity = torch.exp(loss).item()

        metrics = {
            'loss': loss.item(),
            'perplexity': perplexity,
            'grad_norm': grad_norm.item(),
            'lr': self.get_lr()
        }

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        if not self.val_dataset:
            return {}

        self.model.eval()
        total_loss = 0
        total_tokens = 0

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            input_ids, target_ids = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)

            # Forward pass
            logits, _ = self.model(input_ids)

            # Calculate loss
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)

            loss = self.criterion(logits_flat, targets_flat)

            total_loss += loss.item() * targets_flat.size(0)
            total_tokens += targets_flat.size(0)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        self.model.train()

        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
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
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
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

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        logger.info(f"Loaded checkpoint from step {self.step}")

    def train(self, num_epochs: int):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Training on {len(self.train_dataset)} samples")
        if self.val_dataset:
            logger.info(f"Validation on {len(self.val_dataset)} samples")

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
                epoch_losses.append(metrics['loss'])
                self.train_losses.append(metrics['loss'])

                self.step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'ppl': f"{metrics['perplexity']:.2f}",
                    'lr': f"{metrics['lr']:.2e}"
                })

                # Logging
                if self.step % self.log_interval == 0:
                    logger.info(
                        f"Step {self.step} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"PPL: {metrics['perplexity']:.2f} | "
                        f"LR: {metrics['lr']:.2e} | "
                        f"Grad Norm: {metrics['grad_norm']:.3f}"
                    )

                # Evaluation
                if self.step % self.eval_interval == 0 and self.val_dataset:
                    val_metrics = self.evaluate()
                    self.val_losses.append(val_metrics['val_loss'])

                    logger.info(
                        f"Validation | "
                        f"Loss: {val_metrics['val_loss']:.4f} | "
                        f"PPL: {val_metrics['val_perplexity']:.2f}"
                    )

                    # Save best model
                    is_best = val_metrics['val_loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['val_loss']

                    self.save_checkpoint(is_best=is_best)

                # Save checkpoint
                elif self.step % self.save_interval == 0:
                    self.save_checkpoint()

                # Early stopping check
                if self.max_steps and self.step >= self.max_steps:
                    logger.info(f"Reached maximum steps ({self.max_steps})")
                    return

            # End of epoch logging
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch+1} completed | Average loss: {avg_epoch_loss:.4f}")

        logger.info("Training completed!")

def main():
    """Example training script"""
    # Model configuration
    model_config = {
        'vocab_size': 257,  # GPT-2 vocab size
        'dim': 512,
        'max_seq_len': 1024,
        'num_heads': 8,
        'num_kv_heads': 4,
        'T': 6,  # Number of UT recurrent steps
        'dropout': 0.1
    }

    # Training configuration
    train_config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'num_epochs': 10,
        'warmup_steps': 1000,
        'log_interval': 50,
        'eval_interval': 500,
        'save_interval': 1000,
    }

    # Create model
    from Universal_Transformers_.ut_decoder import UniversalTransformerDecoder
    model = UniversalTransformerDecoder(**model_config)

    # Create datasets (using dummy data for this example)
    train_dataset = DummyDataset(
        vocab_size=model_config['vocab_size'],
        seq_len=128,  # Shorter sequences for faster training
        num_samples=10000
    )

    val_dataset = DummyDataset(
        vocab_size=model_config['vocab_size'],
        seq_len=128,
        num_samples=1000
    )

    # Create trainer (exclude 'num_epochs' since it's not a parameter of UTTrainer.__init__)
    train_config_for_trainer = {k: v for k, v in train_config.items() if k != 'num_epochs'}
    trainer = UTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **train_config_for_trainer
    )

    # Start training
    trainer.train(num_epochs=train_config['num_epochs'])

    # Save final model
    final_checkpoint_path = os.path.join(trainer.save_dir, 'final_model.pt')
    trainer.save_checkpoint()

    print(f"Training completed! Final model saved to: {final_checkpoint_path}")

if __name__ == "__main__":
    main()
