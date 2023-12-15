import torch
import torch.nn.functional as F


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_length, num_heads, context_length, dropout):
        super().__init__()
        self.H = num_heads

        self.queries = torch.nn.Linear(embed_length, embed_length, bias=False)
        self.keys = torch.nn.Linear(embed_length, embed_length, bias=False)
        self.values = torch.nn.Linear(embed_length, embed_length, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_length, context_length))
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.proj = torch.nn.Linear(embed_length, embed_length)

    def forward(self, input_data):
        B, T, K = input_data.size()  # (batch_size, context_length, embed_length)

        q = self.queries(input_data)  # (B, T, K)
        k = self.keys(input_data)  # (B, T, K)
        v = self.values(input_data)  # (B, T, K)

        S = K // self.H
        q = q.view(B, T, self.H, S)  # (B, T, H, S)
        k = k.view(B, T, self.H, S)  # (B, T, H, S)
        v = v.view(B, T, self.H, S)  # (B, T, H, S)

        q = q.transpose(1, 2).reshape(B * self.H, T, S)  # (B * H, T, S)
        k = k.transpose(1, 2).reshape(B * self.H, T, S)  # (B * H, T, S)
        v = v.transpose(1, 2).reshape(B * self.H, T, S)  # (B * H, T, S)

        wei = (
            q @ k.transpose(1, 2) * self.H ** (-0.5)
        )  # (B * H, T, S) @ (B * H, S, T) --> (B * H, T, T)

        # Slice :T, :T in case context is larger when generating
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B * H, T, T)
        wei = F.softmax(wei, dim=2)  # (B * H, T, T)
        wei = self.dropout(wei)

        out = wei @ v  # (B * H, T, T) @ (B * H, T, S) --> (B * H, T, S)
        out = (
            out.view(B, self.H, T, S).transpose(1, 2).reshape(B, T, S * self.H)
        )  # (B, T, K)

        # Apply dropout before projecting back into residual pathway
        return self.proj(self.dropout(out))


class FeedForward(torch.nn.Module):
    def __init__(self, embed_length, dropout):
        super().__init__()
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(embed_length, 4 * embed_length),
            torch.nn.ReLU(),
            # Project back into residual pathway
            torch.nn.Linear(4 * embed_length, embed_length),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input_data):
        return self.ff(input_data)


class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_length, num_heads, context_length, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(embed_length, num_heads, context_length, dropout)
        self.ffwd = FeedForward(embed_length, dropout)
        self.ln1 = torch.nn.LayerNorm(embed_length)
        self.ln2 = torch.nn.LayerNorm(embed_length)

    def forward(self, input_data):
        # Residual connections: lay computation off to the side and back
        # Apply layer norm on inputs
        input_data = input_data + self.sa(self.ln1(input_data))
        input_data = input_data + self.ffwd(self.ln1(input_data))
        return input_data


class Megatron(torch.nn.Module):
    def __init__(
        self, vocab_size, embed_length, context_length, dropout, num_heads, num_blocks
    ):
        super().__init__()
        assert embed_length % num_heads == 0
        self.vocab_size = vocab_size
        self.embed_length = embed_length
        self.context_length = context_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.token_embedding_table = torch.nn.Embedding(
            vocab_size, embed_length, max_norm=1000
        )
        self.position_embedding_table = torch.nn.Embedding(
            context_length, embed_length, max_norm=1000
        )

        self.blocks = torch.nn.Sequential(
            *[
                TransformerBlock(embed_length, num_heads, context_length, dropout)
                for _ in range(num_blocks)
            ],
            torch.nn.LayerNorm(embed_length),
        )
        self.lm_head = torch.nn.Linear(embed_length, vocab_size)

    def forward(self, input_data, targets=None):
        # B is the batch_size, T is the context_length (time)
        B, T = input_data.shape

        # Encode the tokens
        token_embed = self.token_embedding_table(input_data)  # (B, T, embed_length)

        # Encode the positions of the tokens
        pos_embed = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, embed_length)

        # Complete encoding of input
        total_embed = token_embed + pos_embed  # (B, T, embed_length)

        # Apply attention block & MLP (let tokens "think" about learned attention)
        transformer_embed = self.blocks(total_embed)

        logits = self.lm_head(transformer_embed)  # (B, T, vocab_size)

        # When generating or evaluating, loss calculation is overhead
        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape  # C is the vocab_size (channel/#classes)

            # Calculate loss and fit inputs to required dimensions "(N, C)"
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, start_data, max_tokens=100):
        # start_data is (B, T) tensor containing current context

        for i in range(max_tokens):
            # Predict using starting symbol(s)
            pred_logits, _ = self.forward(
                start_data[:, -self.context_length :]
            )  # (B, T, C)

            # Get final character logits
            final_logits = pred_logits[
                :,
                -1,
                :,
            ]  # (B, C)

            # Apply softmax to get distribution
            pred_distribution = F.softmax(final_logits, dim=1)

            # Sample from distribution
            prediction = torch.multinomial(pred_distribution, num_samples=1)  # (B, 1)

            # Append to output
            start_data = torch.cat((start_data, prediction), dim=1)  # (B, T+1)

        return start_data  # (B, T+max_tokens, C)
