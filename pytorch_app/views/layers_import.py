from collections import OrderedDict


def extract_nodes(last_node):

    operations = OrderedDict()
    seen = set()
    redundant_ops = set(['ExpandBackward', 'TBackward', 'AccumulateGrad'])

    def add_nodes(node):

        if node not in seen:
            node_name = str(type(node).__name__)
            if node_name not in redundant_ops:
                operations[node] = node_name
            seen.add(node)
            if hasattr(node, 'next_functions'):
                for next_node in node.next_functions:
                    # May have to modify slightly for multiple input functions
                    if next_node[0] is not None:
                        add_nodes(next_node[0])

    add_nodes(last_node.grad_fn)

    return operations
