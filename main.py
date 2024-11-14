import torch
import torch.nn as nn
import datasets
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, default_data_collator
import tqdm
from low_rank_compression.weighted import to_low_rank

max_len = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tiny_llama = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1").to(device)

dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama_v1.1", model_max_length=512
)

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
    output_dir="test_trainer",
    evaluation_strategy="no",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
)


def process_fn(examples):
    batch = tokenizer(
        examples["text"], padding="max_length", truncation=True, return_tensors="pt"
    )
    inputs = batch.input_ids[:, :-1]
    labels = batch.input_ids[:, 1:]
    return {"input_ids": inputs, "labels": labels}


train_dataset = (
    datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    .shuffle(seed=42)
    .select(range(2000))
    .map(process_fn, batched=True)
)
eval_dataset = (
    datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    .select(range(200))
    .map(process_fn, batched=True)
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset.select(300), batch_size=2, collate_fn=default_data_collator
)
predictions = []

# llama_perplexity = evaluator.evaluate(tiny_llama)

# print("Tiny llama perplexity", llama_perplexity)


low_rank_llama = to_low_rank(tiny_llama, train_dataloader, eps=1e-2)
print(low_rank_llama)

del tiny_llama

low_rank_llama_perplexity_before_training = evaluator.evaluate(low_rank_llama)

print(
    "Low rank llama perplexity before training",
    low_rank_llama_perplexity_before_training,
)
trainer = Trainer(
    model=low_rank_llama,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

low_rank_llama_perplexity_after_training = evaluator.evaluate(low_rank_llama)
print(
    "Low rank llama perplexity after training", low_rank_llama_perplexity_after_training
)
