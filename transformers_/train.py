import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
from init_transformer import build_transformer

# Load the dataset
dataset = load_dataset("lmsys/lmsys-chat-1m")

# Create or load a tokenizer
# For this example, we'll use an existing tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# Add special tokens including pad token
special_tokens = {
    'pad_token': '[PAD]',
    'additional_special_tokens': ["<user>", "<assistant>"]
}
num_added = tokenizer.add_special_tokens(special_tokens)
print(f"Pad token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
print(f"Added {num_added} special tokens")

# Dataset class definition
class ConversationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get conversation
        conversation = self.dataset[idx]['conversation']

        # Format conversation
        formatted_text = ""
        for turn in conversation:
            if turn["role"] == "user":
                formatted_text += f"<user> {turn['content']} "
            elif turn["role"] == "assistant":
                formatted_text += f"<assistant> {turn['content']} "

        # Tokenize
        encodings = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Mask generation function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Debug function for NaN values
def debug_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

# Modified train function
def train(model, dataset, tokenizer, d_model, device="cuda", epochs=3, batch_size=8, lr=1e-5):
    # Create DataLoader with a small subset for testing
    print(f"Preparing {batch_size*10} samples for training...")
    train_dataset = ConversationDataset(dataset["train"].select(range(batch_size*10)), tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set up optimizer
    initial_lr = 1e-7
    max_lr = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Create masks
            src_mask = None  # Not needed for decoder-only
            tgt_mask = generate_square_subsequent_mask(input_ids.size(1)).to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Instead of zeros, use the embedded input as both query and key/value
            tgt_embeddings = model.tgt_embed(input_ids)
            tgt_embeddings = model.tgt_pos(tgt_embeddings)

            # Debug NaN values
            debug_nan(tgt_embeddings, "tgt_embeddings after embedding layer")
            debug_nan(tgt_embeddings, "tgt_embeddings after position encoding")

            # Use these as encoder outputs for cross-attention
            output, _ = model.decode(
                tgt_embeddings,  # Use embedded input instead of zeros
                tgt_embeddings,  # Use embedded input instead of zeros
                src_mask,
                input_ids,
                tgt_mask
            )

            # Apply projection layer
            logits = model.projection_layer(output)

            # Calculate loss
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            total_loss += loss.item()

            if i % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

        # Linear warmup over first 3 epochs
        if epoch < 3:
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr + (max_lr - initial_lr) * (epoch / 3)

        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(train_loader):.4f}")

    return model

# Main execution

# Model parameters
print("Building model...")
d_model = 576  # Define d_model here to use in both model building and training
vocab_size = len(tokenizer)
seq_len = 512
N = 8
h = 9
kv_h = 3
dropout = 0.1
d_ff = 2048

# Build model
model = build_transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    src_seq_len=seq_len,
    tgt_seq_len=seq_len,
    d_model=d_model,
    N=N,
    h=h,
    kv_h=kv_h,
    dropout=dropout,
    d_ff=d_ff
)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Resize token embeddings
print("Resizing token embeddings...")
# REMOVE THESE LINES - they're causing the NaNs
# model.tgt_embed.weight = torch.nn.Parameter(torch.randn(len(tokenizer), d_model).to(device))
# model.projection_layer.proj.weight = torch.nn.Parameter(torch.randn(len(tokenizer), d_model).to(device))
# model.projection_layer.proj.bias = torch.nn.Parameter(torch.zeros(len(tokenizer)).to(device))

# INSTEAD, use these more stable initialization methods directly
# Use conservative initialization
torch.nn.init.normal_(model.tgt_embed.weight, mean=0.0, std=0.02)
torch.nn.init.normal_(model.projection_layer.proj.weight, mean=0.0, std=0.02)
torch.nn.init.zeros_(model.projection_layer.proj.bias)

# Train - passing d_model explicitly
print("Starting training...")
trained_model = train(model, dataset, tokenizer, d_model, device, epochs=9, batch_size=2)

# Save
print("Saving model...")
torch.save(trained_model.state_dict(), "trained_transformer.pt")
print("Training complete!")
