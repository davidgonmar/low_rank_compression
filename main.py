import torch
import torch.nn as nn
import datasets
from transformers import LlamaForCausalLM, AutoTokenizer
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
    def from_linear(linear: nn.Linear, rank: int):
        # Original linear -> O = X @ W
        # Low rank linear -> W = U @ S @ V_T -> O = X @ U @ S @ V_T
        rank = min(linear.weight.shape[0], linear.weight.shape[1])
        W, b = linear.weight.T, linear.bias
        U, S, V_T = torch.linalg.svd(W, full_matrices=True)  # complete SVD

        eps = 0.5
        rank = (S > eps).sum()
        print(
            "original rank",
            min(linear.weight.shape[0], linear.weight.shape[1]),
            "new rank",
            rank,
        )
        S = torch.diag(S[:rank])  # in R^{MIN(IN, OUT) x MIN(IN, OUT)}
        # pad S to be {IN x OUT}
        in_f, out_f = W.shape
        assert S.shape == (rank, rank)
        assert U.shape == (in_f, in_f)
        assert V_T.shape == (out_f, out_f)
        W0 = U[:, :rank] @ S  # in R^{IN x RANK}
        W1 = V_T[:rank, :]  # in R^{RANK x OUT}

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


def to_low_rank(model: nn.Module, rank: int):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LowRankLinear.from_linear(module, rank))
        else:
            to_low_rank(module, rank)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tiny_llama = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1").to(device)

dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")


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


low_rank_llama = to_low_rank(tiny_llama, rank=128).to(device)


nll_low_rank_offline = evaluator.evaluate(low_rank_llama)

del tiny_llama
del low_rank_llama


print(f"LowRankLlama ppl: {nll_low_rank_offline}")
