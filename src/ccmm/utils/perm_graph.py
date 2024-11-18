# import graphviz
import copy
import functools
import warnings
from collections import defaultdict
from typing import List

import graphviz
import numpy as np
from numpy import arange, argmax, unique
from torchviz import make_dot


class TorchVizPermutationGraph:
    def __init__(self):
        self.nodes = dict()
        self.edges = defaultdict(list)

        # maps a long node id to a progressive integer, e.g. {'138150777418896': 0, '138150777420384': 1, ..}
        self.naming = dict()

    def view(self, directory="doctest-output", filename="perm_graph"):
        node_attr = dict(
            style="filled",
            shape="box",
            align="left",
            fontsize="10",
            ranksep="0.1",
            height="0.2",
            fontname="monospace",
        )
        dot = graphviz.Digraph(
            "param_list",
            format="pdf",
            node_attr=node_attr,
            graph_attr=dict(size="12,12"),
        )

        names = self.naming
        if hasattr(self, "org_naming"):
            names = self.org_naming

        for key in self.nodes.keys():
            dot.node(key, str(names[key]))

        for key, childs in self.edges.items():
            for c in childs:
                dot.edge(key, c)

        dot.render(directory=directory, filename=filename, view=True, engine="dot")

    def add_node(self, name, value, is_output=False, is_param=False):
        self.nodes[name] = dict(type=value, is_output=is_output, is_param=is_param)
        self.naming[name] = int(len(self.naming))

    def index2name(self, index):
        for key, value in self.naming.items():
            if value == index:
                return key
        return None

    def mark_as_leaf(self, name):
        self.nodes[name]["is_output"] = True
        childs = self.edges[name]
        self.edges[name] = []
        parent = self.parents(self.parents(name)[0])[0]
        for child in childs:
            self.add_edge(parent, child)

    def add_edge(self, from_node, to_node):
        if to_node not in self.edges[from_node]:
            self.edges[from_node].append(to_node)

    def remove_node(self, name):
        self.nodes.pop(name)
        self.naming.pop(name)
        self.edges.pop(name)
        for key, value in self.edges.items():
            if name in value:
                value.remove(name)

    def paramid(self, name):
        for key, node in self.nodes.items():
            name_from_type = node["type"].split("\n")[0][1:]
            if node["is_param"] and name == name_from_type:
                return key

    def parents(self, name):
        parents = []
        for key, value in self.edges.items():
            if name in value:
                parents.append(key)
        return parents

    def closer_perm(self, key):
        if self.nodes[key]["type"] in [
            "ConvolutionBackward0",
            "AddmmBackward0",
            "MmBackward0",
            "AddBackward0",  # this line is transformer-stuff
        ]:
            return key
        if self.nodes[key]["type"] in [
            "NativeBatchNormBackward0",
            "NativeGroupNormBackward0",
            "NativeLayerNormBackward0",
        ]:
            return key

        child = self.edges[key]
        assert len(child) == 1, f"self.nodes[key]['type'] = {self.nodes[key]['type']}"

        return self.closer_perm(child[0])

    def not_output_nodes(self):
        return [key for key, value in self.nodes.items() if not value["is_output"]]

    def child_perm(self, key, perms):

        queue = [key]
        visited = [key]

        childs = []
        fused_nodes = []

        notfirst = False
        while queue:
            node = queue.pop(0)
            type_node = self.nodes[node]["type"]

            if node in perms and notfirst and type_node in ["ConvolutionBackward0", "AddmmBackward0", "MmBackward0"]:
                childs.append(node)
            else:
                if type_node in ["NativeBatchNormBackward0", "NativeGroupNormBackward0", "NativeLayerNormBackward0"]:
                    fused_nodes.append(node)
                for child in self.edges[node]:
                    if child not in visited:
                        queue.append(child)
                        visited.append(child)
            notfirst = True
        return childs, fused_nodes

    def from_dot(self, dot):

        for n in dot.body:

            n = n.strip()
            is_edge = "->" in n

            # node
            if not is_edge:
                idnode, label = n.split(" [label=")
                # remove ]
                label = label[:-1]
                # output node
                is_output = "fillcolor=darkolivegreen1" in label
                label = label.replace(" fillcolor=darkolivegreen1", "")
                # param node
                is_param = "fillcolor=lightblue" in label
                label = label.replace(" fillcolor=lightblue", "")

                self.add_node(idnode, label, is_output, is_param)

            # edge
            else:
                from_node, to_node = n.split(" -> ")
                if " [style=" in to_node:
                    to_node, _ = to_node.split(" [style=")
                if " [style=" in from_node:
                    from_node, _ = from_node.split(" [style=")
                self.add_edge(from_node, to_node)


def build_permutation_graph(
    model, input, fix_multiple=False, mark_as_leaf=list(), remove_nodes=list()
) -> tuple[TorchVizPermutationGraph, dict]:

    device = next(model.parameters()).device

    model.to("cpu")
    input = input.to("cpu")

    model_output = model(input)

    # Graphviz representation of PyTorch autograd graph
    dot = make_dot(model_output, params=dict(model.named_parameters()))

    g = TorchVizPermutationGraph()
    g.from_dot(dot)

    # map param name to id of the corresponding permutation in the graph
    param_to_node_id_map = dict()
    for name, param in model.named_parameters():
        key = g.paramid(name)
        param_to_node_id_map[name] = g.closer_perm(key)

    permutation_list = list(param_to_node_id_map.values())
    visited = set()

    # construct permutation params graph
    permutation_graph = TorchVizPermutationGraph()

    for p in permutation_list:

        if p in visited:
            continue

        # TODO: check that we are not missing some other backwards that we might want to skip
        if g.nodes[p]["type"] in ["NativeBatchNormBackward0", "NativeGroupNormBackward0", "NativeLayerNormBackward0"]:
            continue

        permutation_graph.add_node(p, g.nodes[p]["type"], g.nodes[p]["is_output"], g.nodes[p]["is_param"])
        childs, fused_nodes = g.child_perm(p, permutation_list)

        if fused_nodes:
            for k, v in param_to_node_id_map.items():
                if v in fused_nodes:
                    param_to_node_id_map[k] = p

        for c in childs:
            permutation_graph.add_edge(p, c)

        visited.add(p)

    copy_naming = permutation_graph.naming.copy()
    permutation_graph.org_naming = copy_naming

    # find leaf permutations nodes
    for k, v in permutation_graph.naming.items():
        c, _ = permutation_graph.child_perm(k, permutation_list)
        if not c:
            permutation_graph.nodes[k]["is_output"] = True

    for k in mark_as_leaf:
        permutation_graph.mark_as_leaf(permutation_graph.index2name(k))

    for k in remove_nodes:
        permutation_graph.remove_node(permutation_graph.index2name(k))

    max_index = max(list(permutation_graph.naming.values())) + 1
    for k, v in permutation_graph.nodes.items():
        if v["is_output"]:
            permutation_graph.naming[k] = max_index
            max_index += 1

    remap_index = {r: v for (v, r) in enumerate(sorted(list(permutation_graph.naming.values())))}
    for k in permutation_graph.naming.keys():
        permutation_graph.naming[k] = remap_index[permutation_graph.naming[k]]

    # fix permutation nodes with multiple parents
    if fix_multiple:

        for n in list(permutation_graph.nodes.keys()):

            if n not in permutation_graph.nodes:
                continue

            parents = permutation_graph.parents(n)

            if len(parents) > 1:
                print(f"Multiple parents for {permutation_graph.naming[n]}")

                for p in parents[1:]:
                    print(f"Removing {permutation_graph.naming[p]} of type {permutation_graph.nodes[p]['type']}")
                    permutation_graph.remove_node(p)

        list_nodes = list(permutation_graph.nodes.keys())

        for k, v in list(param_to_node_id_map.items()):
            if v not in list_nodes:
                param_to_node_id_map.pop(k)

    model.to(device)
    return permutation_graph, param_to_node_id_map


def get_connected_from(idx, permutation_g):
    """
    get the ids of the parents of the node idx
    """
    return [permutation_g.naming[k] for k, l in permutation_g.edges.items() if permutation_g.index2name(idx) in l]


def perm_graph_to_perm_dict(perm_graph: TorchVizPermutationGraph):
    """
    perm_dict maps groups of layers that are permuted (on the rows?) by the same permutation to the corresponding permutation id
    """
    perm_dict = {}

    i = -1
    for node in perm_graph.naming.values():

        node_parents = get_connected_from(node, perm_graph)
        j = i
        i += 1

        for parent in node_parents:

            if parent in perm_dict.keys():
                j = perm_dict[parent]
                i = max(perm_dict.values()) + 1

            perm_dict[parent] = j

    # add last node (output) with no perm
    perm_dict[node] = None

    return perm_dict


def remove_nodes_from_perm_dict(nodes_id, perm_dict):
    """
    removes the permutation associated with the nodes as well as other nodes using the same permutation
    """
    for node_id in nodes_id:
        if node_id not in perm_dict.keys():
            warnings.warn("Node_id {} cannot be removed, this node is not in the graph".format(node_id))
            continue
        perm_id = perm_dict[node_id]
        list_to_remove = [n_id for n_id in perm_dict.keys() if perm_dict[n_id] == perm_id]
        for node in list_to_remove:
            perm_dict[node] = None
    return perm_dict


def re_id_perm(perm_dict):
    """
    fill in the gaps in the perm_ids
    """
    list_perm_id = unique([p_id for p_id in perm_dict.values() if p_id is not None])

    if len(list_perm_id) == 0:
        # no permutation left
        return perm_dict

    first_gap = argmax((list_perm_id != arange(len(list_perm_id))).astype(int))

    if list_perm_id[first_gap] == 0:
        # no re_id needed
        return perm_dict

    for n_id, p_id in perm_dict.items():
        if p_id is not None and p_id > first_gap:
            # fill the gap
            perm_dict[n_id] = p_id - 1

    # if there is more that one gap
    return re_id_perm(perm_dict)


def solve_graph(model, input, remove_nodes=list()):

    perm_graph, param_to_node_id_map = build_permutation_graph(
        model, input, fix_multiple=False, mark_as_leaf=[], remove_nodes=[]
    )

    perm_dict = perm_graph_to_perm_dict(perm_graph)

    # remove the nodes disabled by the user
    perm_dict = remove_nodes_from_perm_dict(remove_nodes, perm_dict)

    # fill the gaps
    perm_dict = re_id_perm(perm_dict)
    num_perms = len(unique([p_id for p_id in perm_dict.values() if p_id is not None]))

    if num_perms == 0:
        warnings.warn("No permutation left in graph, you might let more nodes free")

    return perm_dict, num_perms, perm_graph, param_to_node_id_map


def graph_permutations_to_layer_and_axes_to_perm(model, perm_dict, param_to_perm_map, param_to_prev_perm_map):

    perm_dict_copy = copy.deepcopy(perm_dict)
    perm_dict_copy[None] = None

    def rgetattr(obj, attr, *args):  # Recursive getattr
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split("."))

    layer_and_axes_to_perm = {}

    for param_name, perm in param_to_perm_map.items():

        row_perm = perm_dict_copy[param_to_perm_map[param_name]]
        col_perm = perm_dict_copy[param_to_prev_perm_map[param_name]]
        param_num_dims = rgetattr(model, param_name).dim()

        # TODO: FIX THIS STUFF ONCE AND FOR ALL
        # param_name = param_name.replace("model.", "")

        if "pos_embedding" in param_name or "cls_token" in param_name:
            layer_and_axes_to_perm[param_name] = [None, None, row_perm]
            continue

        layer_and_axes_to_perm[param_name] = tuple([row_perm, col_perm] + [None] * (param_num_dims - 2))[
            :param_num_dims
        ]

    return layer_and_axes_to_perm


def get_perm_dict(
    model,
    input,
    remove_nodes: List[str] = list(),
):
    perm_dict, num_perms, perm_graph, param_to_node_id_map = solve_graph(model, input, remove_nodes=remove_nodes)

    P_sizes = [None] * num_perms

    param_to_perm_map = dict()
    param_to_prev_perm_map = dict()

    nodes = list(perm_graph.nodes.keys())

    for name, p in model.named_parameters():

        if "temperature" in name:
            continue

        param_node_id = param_to_node_id_map[name]

        if param_node_id not in nodes:
            continue
        else:
            param_to_perm_map[name] = perm_graph.naming[param_node_id]

        parents = perm_graph.parents(param_node_id)
        param_to_prev_perm_map[name] = None if len(parents) == 0 else perm_graph.naming[parents[0]]

        if "weight" in name[-6:]:

            if len(p.shape) == 1:  # batchnorm
                # no permutation : bn is "part" for the previous one like bias
                pass
            else:
                if param_to_perm_map[name] is not None and perm_dict[param_to_perm_map[name]] is not None:
                    perm_index = perm_dict[param_to_perm_map[name]]
                    P_sizes[perm_index] = (p.shape[0], p.shape[0])

    return perm_dict, param_to_perm_map, param_to_prev_perm_map
