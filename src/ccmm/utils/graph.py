# import graphviz
import warnings
import graphviz
import functools
import copy

from numpy import unique, argmax, arange
from collections import defaultdict
from torchviz import make_dot

from ccmm.matching.permutation_spec import PermutationSpec


class graph:
    def __init__(self):
        self.nodes = dict()
        self.edges = defaultdict(list)
        self.naming = dict()

    def view(self):
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

        dot.render(directory="doctest-output", view=True, engine="dot")

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
        ]:
            return key
        if self.nodes[key]["type"] in [
            "NativeBatchNormBackward0",
            "NativeGroupNormBackward0",
            "NativeLayerNormBackward0"
        ]:
            return key

        child = self.edges[key]
        assert len(child) == 1
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

            if (
                node in perms
                and notfirst
                and type_node
                in ["ConvolutionBackward0", "AddmmBackward0", "MmBackward0"]
            ):
                childs.append(node)
            else:
                if type_node in [
                    "NativeBatchNormBackward0",
                    "NativeGroupNormBackward0",
                    "NativeLayerNormBackward0"
                ]:
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


def permutation_graph(
    model, input, fix_multiple=False, mark_as_leaf=list(), remove_nodes=list()
):
    prev_dev = next(model.parameters()).device
    model.to("cpu")
    input = input.to("cpu")
    y = model(input)
    dot = make_dot(y, params=dict(model.named_parameters()))
    # dot.view()
    g = graph()
    g.from_dot(dot)

    # map param name to permutation
    permutation_param = dict()
    for name, param in model.named_parameters():
        key = g.paramid(name)
        permutation_param[name] = g.closer_perm(key)

    permutation_list = list(permutation_param.values())
    visited = set()

    # construct permutation params graph
    permutation_graph = graph()
    for p in permutation_list:
        if p in visited:
            continue
        if g.nodes[p]["type"] in [
            "NativeBatchNormBackward0",
            "NativeGroupNormBackward0",
            "NativeLayerNormBackward0"
        ]:
            continue

        permutation_graph.add_node(
            p, g.nodes[p]["type"], g.nodes[p]["is_output"], g.nodes[p]["is_param"]
        )
        childs, fused_nodes = g.child_perm(p, permutation_list)

        if fused_nodes:
            for k, v in permutation_param.items():
                if v in fused_nodes:
                    permutation_param[k] = p

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

    remap_index = {
        r: v for (v, r) in enumerate(sorted(list(permutation_graph.naming.values())))
    }
    for k in permutation_graph.naming.keys():
        permutation_graph.naming[k] = remap_index[permutation_graph.naming[k]]

    # fix permutation nodes with multiple parents
    if fix_multiple:
        for n in list(permutation_graph.nodes.keys()):
            if n not in permutation_graph.nodes:
                continue
            parents = permutation_graph.parents(n)
            if len(parents) > 1:
                for p in parents[1:]:
                    permutation_graph.remove_node(p)

        list_nodes = list(permutation_graph.nodes.keys())
        for k, v in list(permutation_param.items()):
            if v not in list_nodes:
                permutation_param.pop(k)

    model.to(prev_dev)
    return permutation_graph, permutation_param


def get_connected_from(idx, permutation_g):
    """
    get the ids of the parents of the node idx
    """
    return [
        permutation_g.naming[k]
        for k, l in permutation_g.edges.items()
        if permutation_g.index2name(idx) in l
    ]


def get_perm_dict(permutation_g):
    """
    get the permutation dict
    """
    perm_dict = {}
    i = -1
    for node in permutation_g.naming.values():
        p = get_connected_from(node, permutation_g)
        j = i
        i += 1
        for p_ in p:
            if p_ in perm_dict.keys():
                j = perm_dict[p_]
                i = max(perm_dict.values()) + 1
            perm_dict[p_] = j
    # add last node (output) with no perm
    perm_dict[node] = None
    return perm_dict


def remove_nodes_from_perm_dict(nodes_id, perm_dict):
    """
    removes the permutation associated with the nodes as well as other nodes using the same permutation
    """
    for node_id in nodes_id:
        if not node_id in perm_dict.keys():
            warnings.warn(
                "Node_id {} cannot be removed, this node is not in the graph".format(
                    node_id
                )
            )
            continue
        perm_id = perm_dict[node_id]
        list_to_remove = [
            n_id for n_id in perm_dict.keys() if perm_dict[n_id] == perm_id
        ]
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
    # if there is more that one gap :
    return re_id_perm(perm_dict)


def solve_graph(model, input, remove_nodes=list()):
    permutation_g, parameter_map = permutation_graph(model, input, False, [], [])
    perm_dict = get_perm_dict(permutation_g)
    # remove the nodes disabled by the user
    perm_dict = remove_nodes_from_perm_dict(remove_nodes, perm_dict)
    # fill the gaps
    perm_dict = re_id_perm(perm_dict)
    n_perm = len(unique([p_id for p_id in perm_dict.values() if p_id is not None]))
    if n_perm == 0:
        warnings.warn("No permutation left in graph, you might let more nodes free")
    return perm_dict, n_perm, permutation_g, parameter_map


def graph_permutations_to_perm_spec(model, perm_dict, map_param_index, map_prev_param_index):
    
    perm_dict_copy = copy.deepcopy(perm_dict)
    perm_dict_copy[None] = None

    def rgetattr(obj, attr, *args):  # Recursive getattr
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)
        return functools.reduce(_getattr, [obj] + attr.split('.'))
    
    layer_and_axes_to_perm = {
        layer: tuple(
            [perm_dict_copy[block], perm_dict_copy[map_prev_param_index[layer]]] + \
            [None] * (rgetattr(model, layer).dim() - 2)
        )[:rgetattr(model, layer).dim()]
        for layer, block in map_param_index.items()
    }
    perm_to_layers_and_axes = {
        perm_id: 
            [(layer, 0) for layer, block in map_param_index.items() if perm_dict[block] == perm_id] + \
            [(layer, 1) for layer, block in map_prev_param_index.items() if perm_dict[block] == perm_id] 
        for perm_id in set(perm_dict_copy.values()) if perm_id is not None
    }
    
    return PermutationSpec(perm_to_layers_and_axes, layer_and_axes_to_perm)