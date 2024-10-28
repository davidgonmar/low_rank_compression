import torch
import torch.nn as nn
import datasets
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
import tqdm



class LowRankLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, bias: bool = True
    ):
        super(LowRankLinear, self).__init__()
        self.w0 = nn.Parameter(torch.randn(in_features, rank))
        self.w1 = nn.Parameter(torch.randn(rank, out_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None

    @staticmethod
    def from_linear(linear: nn.Linear, eps):
        # Original linear -> O = X @ W
        # Low rank linear -> W = U @ S @ V_T -> O = X @ U @ S @ V_T
        W, b = linear.weight.T, linear.bias
        U, S, V_T = torch.linalg.svd(W, full_matrices=True)  # complete SVD

        rank = max((S > eps).sum(), 16)
        original_n_params = linear.weight.shape[0] * linear.weight.shape[1]
        new_n_params = (
            linear.weight.shape[0] * rank + rank * linear.weight.shape[1]
        )
        if new_n_params * 1.2 > original_n_params:
            print(
                f"Rank {rank} is too high, not using low rank approximation. "
            )
            return linear
        
        S = torch.diag(S[:rank])  # in R^{MIN(IN, OUT) x MIN(IN, OUT)}
        # pad S to be {IN x OUT}
        in_f, out_f = W.shape
        assert S.shape == (rank, rank)
        assert U.shape == (in_f, in_f)
        assert V_T.shape == (out_f, out_f)
        W0 = U[:, :rank] @ S  # in R^{IN x RANK}
        W1 = V_T[:rank, :]  # in R^{RANK x OUT}
        print(
            f"shape gone from {tuple(linear.weight.shape)} to rank {rank} approximation with shapes {tuple(W0.shape)} and {tuple(W1.shape)}"
        )
        # measure distance between W and W0 @ W1
        print(f"Distance between W and W0 @ W1: {torch.norm(W - W0 @ W1)}")

        low_rank_linear = LowRankLinear(
            linear.weight.shape[1],
            linear.weight.shape[0],
            rank,
            bias=linear.bias is not None,
        )
        low_rank_linear.w0.data = W0
        low_rank_linear.w1.data = W1
        if b is not None:
            low_rank_linear.bias.data = b
        else:
            low_rank_linear.bias = None
        return low_rank_linear

    def forward(self, x: torch.Tensor):
        # X in R^{... x IN}
        # W0 in R^{IN x RANK} -> X @ W0 in R^{... x RANK}
        # W1 in R^{RANK x OUT} -> O = (X @ W0) @ W1 in R^{... x OUT}
        if self.bias is not None:
            return torch.matmul(torch.matmul(x, self.w0), self.w1) + self.bias
        else:
            return torch.matmul(torch.matmul(x, self.w0), self.w1)


def to_low_rank(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LowRankLinear.from_linear(module, eps=0.5))
        else:
            to_low_rank(module)
    return model

max_len = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tiny_llama = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1").to(device)

dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1", model_max_length=512)

tokenizer.pad_token_id = tokenizer.eos_token_id


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))


evaluator = Evaluator(dataset, tokenizer, device)


training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="no", num_train_epochs=5, per_device_train_batch_size=2, per_device_eval_batch_size=2)

def process_fn(examples):
    batch = tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")
    inputs = batch.input_ids[:, :-1]
    labels = batch.input_ids[:, 1:]
    return {"input_ids": inputs, "labels": labels}

train_dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train").shuffle(seed=42).select(range(200)).map(process_fn, batched=True)
eval_dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="validation").select(range(200)).map(process_fn, batched=True)

predictions = []

#llama_perplexity = evaluator.evaluate(tiny_llama)

#print("Tiny llama perplexity", llama_perplexity)

low_rank_llama = to_low_rank(tiny_llama)
del tiny_llama

low_rank_llama_perplexity_before_training = evaluator.evaluate(low_rank_llama)

print("Low rank llama perplexity before training", low_rank_llama_perplexity_before_training)
trainer = Trainer(
    model=low_rank_llama,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

low_rank_llama_perplexity_after_training = evaluator.evaluate(low_rank_llama)
print("Low rank llama perplexity after training", low_rank_llama_perplexity_after_training)

