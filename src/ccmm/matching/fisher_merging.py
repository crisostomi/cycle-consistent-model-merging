import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def _compute_exact_fisher_for_batch(batch, model, num_classes):

    model = model.to("cuda")
    model.eval()

    fishers = [_fisher_single_example(sample, model, num_classes) for sample in batch]

    del model
    torch.cuda.empty_cache()
    del batch

    return [torch.stack(f).sum(0) for f in zip(*fishers)]


def _fisher_single_example(single_example_batch, model, num_classes):

    with torch.enable_grad():
        single_example_batch = single_example_batch.to("cuda")

        logits = model(single_example_batch.unsqueeze(0), return_logits=True).squeeze(0)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        del single_example_batch

        sq_grads = []
        for i in range(num_classes):
            model.zero_grad()
            log_prob = log_probs[i]
            retain_graph = i < num_classes - 1
            log_prob.backward(retain_graph=retain_graph)
            sq_grad = [probs[i].detach() * (v.grad**2).detach() for v in model.parameters()]
            sq_grads.append(sq_grad)

        example_fisher = [torch.stack(g).sum(0).cpu() for g in zip(*sq_grads)]

    model.zero_grad()

    for p in model.parameters():
        p.grad = None

    del logits, log_probs, probs, sq_grads, model
    torch.cuda.empty_cache()

    gc.collect()

    return example_fisher


def compute_fisher_for_model(model, loader, num_classes):

    # initialize a zero matrix for each param tensor in the model
    fishers = {k: torch.zeros_like(v, requires_grad=False).cpu() for k, v in model.state_dict().items()}

    n_examples = 0
    for batch, _ in tqdm(loader):
        n_examples += len(batch)
        batch_fishers = _compute_exact_fisher_for_batch(batch, model, num_classes)

        for f, bf in zip(fishers.values(), batch_fishers):
            f.add_(bf)

    for fisher in fishers.values():
        fisher.div_(float(n_examples))

    return fishers
