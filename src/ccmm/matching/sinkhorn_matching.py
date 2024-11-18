import logging
import time
from copy import deepcopy
from typing import List, Literal

import torch
import torch.nn as nn

from ccmm.matching.permutation_spec import PermutationSpec
from ccmm.matching.weight_matching import solve_linear_assignment_problem
from ccmm.utils.perm_graph import solve_graph

pylogger = logging.getLogger(__name__)


def sinkhorn_matching(
    fixed: nn.Module,
    permutee: nn.Module,
    perm_spec: PermutationSpec,
    max_iter: int,
    example_input_shape: List[int],
    criterion: Literal["L1", "L2"] = "L2",
    lr: float = 0.1,
    device="cuda",
    verbose=False,
):
    """
    Weight matching via optimal transport
    """

    modelA = fixed.to(device)
    modelB = permutee.to(device)

    pi_modelB = RebasinNet(modelB, perm_spec, input_shape=example_input_shape).to(device)
    pi_modelB.identity_init()
    pi_modelB.train()

    pylogger.info("Check if permutation matrices are initialized to I:")
    pylogger.info(torch.allclose(pi_modelB.p[0].data, torch.eye(pi_modelB.p[0].shape[0], device=device)))

    # distance loss
    criterion = globals()["Dist{}Loss".format(criterion)](modelA)

    # optimizer for rebasin network
    optimizer = torch.optim.AdamW(pi_modelB.p.parameters(), lr=lr)

    # try to find the permutation matrices that originated modelB
    pylogger.info("\nTraining Re-Basing network")
    t1 = time.time()

    for iteration in range(max_iter):
        # training step
        pi_modelB.train()  # this uses soft permutation matrices
        rebased_model = pi_modelB()
        loss_training = criterion(rebased_model)  # this compared rebased_model with modelB

        optimizer.zero_grad()
        loss_training.backward()
        optimizer.step()  # only updates the permutation matrices

        # validation step
        pi_modelB.eval()  # this uses hard permutation matrices
        rebased_model = pi_modelB()
        loss_validation = criterion(rebased_model)
        pylogger.info(
            "Iteration {:02d}: loss training {:1.3f}, loss validation {:1.3f}".format(
                iteration, loss_training, loss_validation
            )
        )
        if loss_validation == 0:
            break

    pylogger.info("Elapsed time {:1.3f} secs".format(time.time() - t1))

    # if loss validation is 0, then we found the same permutation matrix
    pi_modelB.eval()

    perm_indices = {
        p_name: solve_linear_assignment_problem(pi_modelB.p[p_idx])
        for p_name, p_idx in pi_modelB.p_name_to_idx.items()
        if p_name is not None
    }

    return perm_indices, None


# Sinkhorn differentiation from https://github.com/marvin-eisenberger/implicit-sinkhorn
class Sinkhorn(torch.autograd.Function):
    """
    An implementation of a Sinkhorn layer with our custom backward module, based on implicit differentiation
    :param c: input cost matrix, size [*,m,n], where * are arbitrarily many batch dimensions
    :param a: first input marginal, size [*,m]
    :param b: second input marginal, size [*,n]
    :param num_sink: number of Sinkhorn iterations
    :param lambd_sink: entropy regularization weight
    :return: optimized soft permutation matrix
    """

    @staticmethod
    def forward(ctx, c, a, b, num_sink, lambd_sink):
        log_p = -c / lambd_sink
        log_a = torch.log(a).unsqueeze(dim=-1)
        log_b = torch.log(b).unsqueeze(dim=-2)
        for _ in range(num_sink):
            log_p -= torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b
            log_p -= torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a
        p = torch.exp(log_p)

        ctx.save_for_backward(p, torch.sum(p, dim=-1), torch.sum(p, dim=-2))
        ctx.lambd_sink = lambd_sink
        return p

    @staticmethod
    def backward(ctx, grad_p):
        p, a, b = ctx.saved_tensors

        device = grad_p.device

        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        grad_p *= -1 / ctx.lambd_sink * p
        K = torch.cat(
            (
                torch.cat((torch.diag_embed(a), p), dim=-1),
                torch.cat((p.transpose(-2, -1), torch.diag_embed(b)), dim=-1),
            ),
            dim=-2,
        )[..., :-1, :-1]
        t = torch.cat((grad_p.sum(dim=-1), grad_p[..., :, :-1].sum(dim=-2)), dim=-1).unsqueeze(-1)
        grad_ab = torch.linalg.solve(K, t)
        grad_a = grad_ab[..., :m, :]
        grad_b = torch.cat(
            (
                grad_ab[..., m:, :],
                torch.zeros(batch_shape + [1, 1], device=device, dtype=torch.float32),
            ),
            dim=-2,
        )
        U = grad_a + grad_b.transpose(-2, -1)
        grad_p -= p * U
        grad_a = -ctx.lambd_sink * grad_a.squeeze(dim=-1)
        grad_b = -ctx.lambd_sink * grad_b.squeeze(dim=-1)
        return grad_p, grad_a, grad_b, None, None, None


class ReparamNet(torch.nn.Module):
    def __init__(self, model, permutation_type="mat_mul"):
        super().__init__()
        _permutation_types = ["mat_mul", "broadcast"]
        assert permutation_type in _permutation_types, "Permutation type must be in {}".format(_permutation_types)
        self.permutation_type = permutation_type
        self.output = deepcopy(model)
        self.model = deepcopy(model)
        for p1, p2 in zip(self.model.parameters(), self.output.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False

    def set_model(self, model):
        self.model = deepcopy(model)
        for p1 in self.model.parameters():
            p1.requires_grad = False

    def training_rebasin(self, P):
        for (name, p1), p2 in zip(self.output.named_parameters(), self.model.parameters()):
            if name not in self.map_param_index or name not in self.map_prev_param_index:
                continue
            i = self.perm_dict[self.map_param_index[name]]
            j = self.perm_dict[self.map_prev_param_index[name]] if self.map_prev_param_index[name] is not None else None

            if "bias" in name[-4:]:
                if i is not None:
                    p1.copy_(P[i] @ p2)
                else:
                    continue

            # batchnorm
            elif len(p1.shape) == 1:
                if i is not None:
                    p1.copy_((P[i] @ p2.view(p1.shape[0], -1)).view(p2.shape))

            # mlp / cnn
            elif "weight" in name[-6:]:
                if i is not None and j is None:
                    p1.copy_((P[i] @ p2.view(P[i].shape[0], -1)).view(p2.shape))

                if i is not None and j is not None:
                    p1.copy_(
                        (
                            P[j].view(1, *P[j].shape)
                            @ (P[i] @ p2.view(P[i].shape[0], -1)).view(p2.shape[0], P[j].shape[0], -1)
                        ).view(p2.shape)
                    )

                if i is None and j is not None:
                    p1.copy_((P[j].view(1, *P[j].shape) @ p2.view(p2.shape[0], P[j].shape[0], -1)).view(p2.shape))

    def update_batchnorm(self, model):
        for m1, m2 in zip(self.model.modules(), model.modules()):
            if "BatchNorm" in str(type(m2)):
                if m2.running_mean is None:
                    m1.running_mean = None
                else:
                    m1.running_mean.copy_(m2.running_mean)
                if m2.running_var is None:
                    m1.running_var = None
                    m1.track_running_stats = False
                else:
                    m1.running_var.copy_(m2.running_var)

    def permute_batchnorm(self, P):
        for (name, m1), m2 in zip(self.output.named_modules(), self.model.modules()):
            if "BatchNorm" in str(type(m2)):
                if name + ".weight" in self.map_param_index:
                    if m2.running_mean is None and m2.running_var is None:
                        continue
                    i = self.perm_dict[self.map_param_index[name + ".weight"]]
                    index = torch.argmax(P[i], dim=1) if i is not None else torch.arange(m2.running_mean.shape[0])
                    m1.running_mean.copy_(m2.running_mean[index, ...])
                    m1.running_var.copy_(m2.running_var[index, ...])

    def eval_rebasin(self, P):
        for (name, p1), p2 in zip(self.output.named_parameters(), self.model.parameters()):
            if name not in self.map_param_index or name not in self.map_prev_param_index:
                continue
            i = self.perm_dict[self.map_param_index[name]]
            j = self.perm_dict[self.map_prev_param_index[name]] if self.map_prev_param_index[name] is not None else None

            if "bias" in name[-4:]:
                if i is not None:
                    index = torch.argmax(P[i], dim=1)
                    p1.copy_(p2.data[index, ...])
                else:
                    continue

            # batchnorm
            elif len(p1.shape) == 1:
                if i is not None:
                    index = torch.argmax(P[i], dim=1)
                    p1.copy_(p2.data[index, ...])

            # mlp / cnn
            elif "weight" in name[-6:]:
                if i is not None and j is None:
                    index = torch.argmax(P[i], dim=1)
                    p1.copy_(p2.data.view(P[i].shape[0], -1)[index, ...].view(p2.shape))

                if i is not None and j is not None:
                    index = torch.argmax(P[i], dim=1)
                    p1.copy_(p2.data[index, ...])
                    index = torch.argmax(P[j], dim=1)
                    p1.copy_(p1.data[:, index, ...])

                if i is None and j is not None:
                    index = torch.argmax(P[j], dim=1)
                    p1.copy_((p2.data.view(p2.shape[0], P[j].shape[0], -1)[:, index, ...]).view(p2.shape))

    def forward(self, P):
        for p1, p2 in zip(self.output.parameters(), self.model.parameters()):
            p1.data = p2.data.clone()

        for p1 in self.output.parameters():
            p1._grad_fn = None

        if self.training or self.permutation_type == "mat_mul":
            self.training_rebasin(P)
        else:
            self.eval_rebasin(P)

        self.permute_batchnorm(P)

        return self.output

    def to(self, device):
        self.output.to(device)
        self.model.to(device)

        return self


class RebasinNet(torch.nn.Module):
    def __init__(
        self,
        model,
        perm_spec,
        input_shape,
        remove_nodes=list(),
        l=1.0,  # NOQA
        tau=1.0,
        n_iter=20,
        operator="implicit",
        permutation_type="mat_mul",
    ):
        super().__init__()
        assert operator in [
            "implicit",
        ], "Operator must be either `implicit`"

        self.reparamnet = ReparamNet(model, permutation_type=permutation_type)
        self.param_precision = next(iter(model.parameters())).data.dtype
        input = torch.randn(input_shape, dtype=self.param_precision)
        perm_dict, n_perm, permutation_g, parameter_map = solve_graph(model, input, remove_nodes=remove_nodes)

        self.p_name_to_idx = dict()
        P_sizes = [None] * n_perm
        map_param_index = dict()
        map_prev_param_index = dict()
        nodes = list(permutation_g.nodes.keys())
        for name, p in model.named_parameters():
            if parameter_map[name] not in nodes:
                continue
            else:
                map_param_index[name] = permutation_g.naming[parameter_map[name]]
            parents = permutation_g.parents(parameter_map[name])
            map_prev_param_index[name] = None if len(parents) == 0 else permutation_g.naming[parents[0]]

            p_name = perm_spec[1][name][0]
            self.p_name_to_idx[p_name] = perm_dict[map_param_index[name]]

            if "weight" in name[-6:]:
                if len(p.shape) == 1:  # batchnorm
                    pass  # no permutation : bn is "part" for the previous one like bias
                else:
                    if map_param_index[name] is not None and perm_dict[map_param_index[name]] is not None:
                        perm_index = perm_dict[map_param_index[name]]
                        P_sizes[perm_index] = (p.shape[0], p.shape[0])

        self.reparamnet.map_param_index = map_param_index
        self.reparamnet.map_prev_param_index = map_prev_param_index
        self.reparamnet.perm_dict = perm_dict

        self.p = torch.nn.ParameterList(
            [
                (
                    torch.nn.Parameter(
                        torch.eye(ps[0], dtype=self.param_precision)
                        + torch.randn(ps, dtype=self.param_precision) * 0.1,
                        requires_grad=True,
                    )
                    if ps is not None
                    else None
                )
                for ps in P_sizes
            ]
        )

        self.l = l  # NOQA
        self.tau = tau
        self.n_iter = n_iter
        self.operator = operator

    def update_batchnorm(self, model):
        self.reparamnet.update_batchnorm(model)

    def random_init(self):
        for p in self.p:
            ci = torch.randperm(p.shape[0])
            p.data = (torch.eye(p.shape[0])[ci, :]).to(p.data.device)

    def identity_init(self):
        for p in self.p:
            p.data = torch.eye(p.shape[0]).to(p.data.device)

    def eval(self):
        self.reparamnet.eval()
        return super().eval()

    def train(self, mode: bool = True):
        self.reparamnet.train(mode)
        return super().train(mode)

    def forward(self, x=None):

        if self.training:
            gk = list()
            for i in range(len(self.p)):
                if self.operator == "implicit":
                    sk = Sinkhorn.apply(
                        -self.p[i] * self.l,
                        torch.ones((self.p[i].shape[0])).to(self.p[0].device),
                        torch.ones((self.p[i].shape[1])).to(self.p[0].device),
                        self.n_iter,
                        self.tau,
                    )

                gk.append(sk)

        else:
            gk = [
                solve_linear_assignment_problem(p, return_matrix=True).to(self.param_precision).to(self.p[0].device)
                for p in self.p
            ]

        m = self.reparamnet(gk)
        if x is not None and x.ndim == 1:
            x.unsqueeze_(0)

        if x is not None:
            return m(x)

        return m

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.reparamnet.output.zero_grad(set_to_none)
        return super().zero_grad(set_to_none)

    def parameters(self, recurse: bool = True):
        return self.p.parameters(recurse)

    def to(self, device):
        for p in self.p:
            if p is not None:
                p.data = p.data.to(device)

        return self


class DistL2Loss(nn.Module):
    """
    Suitable for neurons aligment
    """

    def __init__(self, modela=None):
        super(DistL2Loss, self).__init__()
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def set_model(self, modela):
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def forward(self, modelb):
        loss = 0
        num_params = 0
        for p1, p2 in zip(self.modela.parameters(), modelb.parameters()):
            num_params += p1.numel()
            loss += torch.pow(p1 - p2, 2).sum()

        loss /= num_params

        return loss


class DistL1Loss(nn.Module):
    """
    Suitable for neurons aligment
    """

    def __init__(self, modela=None):
        super(DistL1Loss, self).__init__()
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def set_model(self, modela):
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def forward(self, modelb):
        loss = 0
        num_params = 0
        for p1, p2 in zip(self.modela.parameters(), modelb.parameters()):
            num_params += p1.numel()
            loss += (p1 - p2).abs().sum()

        loss /= num_params

        return loss
