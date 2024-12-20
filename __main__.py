from enum import Enum
from typing import Any
import tkinter as tk
import numpy as np
from math import *

class DirectedMultihypergraph:
        def __init__(self) -> None:
                self.node_to_sources = {}
                self.node_to_targets = {}
                self.source_to_edges = {}
                self.target_to_edges = {}
                self.edge_to_source = {}

        def add_node(self, node: Any) -> None:
                if self.node_to_sources.get(node) != None:
                        return
                self.node_to_sources[node] = []
                self.node_to_targets[node] = []

        def remove_node(self, node: Any) -> None:
                source_list = self.node_to_sources.pop(node)
                for source in source_list:
                        edges = self.source_to_edges.pop(tuple(source))
                        for edge, target in edges:
                                self.edge_to_source.pop(edge)

                target_list = self.node_to_targets.pop(node)
                for target in target_list:
                        self.target_to_edges.pop(target)

        def add_edge(
                self,
                name: str,
                source_set: set | Any,
                target_set: set | Any
                ) -> None:
                if not isinstance(source_set, set):
                       source_set = {source_set,}
                for source in source_set:
                        if source in self.node_to_sources.keys():
                                self.node_to_sources[source].append(source_set)
                        else:
                                self.node_to_sources[source] = [source_set]

                if not isinstance(target_set, set):
                       target_set = {target_set,}
                for target in target_set:
                        if target in self.node_to_targets.keys():
                                self.node_to_targets[target].append(target_set)
                        else:
                                self.node_to_targets[target] = [target_set]

                if self.source_to_edges.get(tuple(source_set)) == None:
                        self.source_to_edges[tuple(source_set)] = [(name, target_set)]
                else:
                        self.source_to_edges[tuple(source_set)].append((name, target_set))

                if self.target_to_edges.get(tuple(target_set)) == None:
                        self.target_to_edges[tuple(target_set)] = [(name, source_set)]
                else:
                        self.target_to_edges[tuple(target_set)].append((name, source_set))

                self.edge_to_source[name] = tuple(source_set)

        def remove_edge(self, edge):
                edge_list = self.source_to_edges[self.edge_to_source[edge]]
                for i, (_name, target) in enumerate(edge_list):
                        if _name == edge:
                                edge_list.pop(i)
                                for j, (_name2, source) in enumerate(self.target_to_edges[tuple(target)]):
                                        if _name2 == edge:
                                                self.target_to_edges[tuple(target)].pop(j)
                self.edge_to_source.pop(edge)

        def __str__(self) -> str:
                ret = "nodes: " + ", ".join(
                        [str(x) for x in self.node_to_sources.keys()]) + "\n"
                for source, name_and_target in self.source_to_edges.items():
                        for (name, target_set) in name_and_target:
                                ret += str(name) + ": "
                                ret += ", ".join([str(x) for x in source])
                                ret += " -> "
                                ret += ", ".join([str(x) for x in target_set])
                                ret += "\n"
                ret = ret[:-1]
                return ret
        def edge_to_target(self, edge):
                '''
                edge -> source -> (edge, target) -> target
                '''
                source = self.edge_to_source[edge]
                for (_edge, target) in self.source_to_edges[source]:
                        if _edge == edge:
                                return target

        def nodes(self):
                return self.node_to_sources.keys()

        class FakeNode:
                def __init__(self, idx):
                        self.data = idx

        def degenerate(self):
                fakenode_index = 0

                ret_nodes = set()
                ret_edges = []

                for source_set in self.source_to_edges.keys():
                        for edgename, target_set in self.source_to_edges[source_set]:
                                if len(source_set) == 1 and len(target_set) == 1:
                                        s = list(source_set)[0]
                                        t = list(target_set)[0]
                                        ret_nodes.add(s)
                                        ret_nodes.add(t)
                                        ret_edges.append((s, t, edgename))
                                else:
                                        fakenode = self.FakeNode(fakenode_index)
                                        fakenode_index += 1
                                        ret_nodes.add(fakenode)
                                        for s in source_set:
                                                ret_nodes.add(s)
                                                ret_edges.append((s, fakenode, edgename))
                                        for t in target_set:
                                                ret_nodes.add(t)
                                                ret_edges.append((fakenode, t, edgename))

                return ret_nodes, ret_edges

        def degenerate_from_edges(self, edges: set):
                fakenode_index = 0

                ret_nodes = set()
                ret_edges = []

                for source_set in self.source_to_edges.keys():
                        for edgename, target_set in self.source_to_edges[source_set]:
                                if edgename not in edges:
                                        continue
                                if len(source_set) == 1 and len(target_set) == 1:
                                        s = list(source_set)[0]
                                        t = list(target_set)[0]
                                        ret_nodes.add(s)
                                        ret_nodes.add(t)
                                        ret_edges.append((s, t, edgename))
                                else:
                                        fakenode = self.FakeNode(fakenode_index)
                                        fakenode_index += 1
                                        ret_nodes.add(fakenode)
                                        for s in source_set:
                                                ret_nodes.add(s)
                                                ret_edges.append((s, fakenode, edgename))
                                        for t in target_set:
                                                ret_nodes.add(t)
                                                ret_edges.append((fakenode, t, edgename))

                return ret_nodes, ret_edges

def force_directed_layout(nodes, edges, width_and_height, iterations=100, k=None):

        width = width_and_height[0]
        height = width_and_height[1]

        num_nodes = len(nodes)
        positions = {node: np.random.rand(2) * [width, height] for node in nodes}  # 随机初始化节点位置
        # 理想间距
        area = width * height
        if k is None:
                k = np.sqrt(area / num_nodes) / 1.5

        # 模拟力导向算法
        for iter_n in range(iterations):
                forces = {node: np.array([0.0, 0.0]) for node in nodes}

                # 计算引力 (attractive force)
                for edge in edges:
                        node_i, node_j = edge[0], edge[1]
                        delta = positions[node_i] - positions[node_j]
                        distance = np.linalg.norm(delta)
                        if distance == 0:
                                distance = 0.01
                        attractive_force = (distance * log(distance / k)) * (delta / distance)
                        forces[node_i] -= attractive_force
                        forces[node_j] += attractive_force

                for node in nodes:
                        positions[node] += forces[node] * 0.1
                        positions[node] = np.clip(positions[node], [0, 0], [width, height])

                forces = {node: np.array([0.0, 0.0]) for node in nodes}

                # 计算斥力 (repulsive force)
                for i in range(num_nodes):
                        for j in range(i + 1, num_nodes):
                                node_i, node_j = nodes[i], nodes[j]
                                delta = positions[node_i] - positions[node_j]
                                distance = np.linalg.norm(delta)
                                if distance == 0:
                                        distance = 0.01
                                repulsive_force = (1000 * k / (distance ** 3)) * (delta / distance)
                                forces[node_i] += repulsive_force
                                forces[node_j] -= repulsive_force

                # 更新节点位置
                for node in nodes:
                        positions[node] += forces[node]
                        positions[node] = np.clip(positions[node], [0, 0], [width, height])

        #         if iter_n % 4 == 0:
        #                 print(1111111)
        #                 ret.append(positions.copy())

        # ret.append(positions.copy())

        return positions

def create_circle(canvas, x, y, r, **kwargs):
        x1, y1 = x - r, y - r
        x2, y2 = x + r, y + r
        return canvas.create_oval(x1, y1, x2, y2, **kwargs)

def add_arrow_to_line(canvas, line_id, arrow_length=7, arrow_angle=30):
        x1, y1, x2, y2 = canvas.coords(line_id)

        tip_x, tip_y = x2, y2
        base_x, base_y = (x1 + x2) / 2, (y1 + y2) / 2

        line_angle = atan2(tip_y - base_y, tip_x - base_x)

        angle1 = line_angle + radians(arrow_angle)
        angle2 = line_angle - radians(arrow_angle)

        arrow_x1 = base_x - arrow_length * cos(angle1)
        arrow_y1 = base_y - arrow_length * sin(angle1)
        arrow_x2 = base_x - arrow_length * cos(angle2)
        arrow_y2 = base_y - arrow_length * sin(angle2)

        canvas.create_line(base_x, base_y, arrow_x1, arrow_y1, fill="white", width=2)
        canvas.create_line(base_x, base_y, arrow_x2, arrow_y2, fill="white", width=2)

def hsl_to_rgb(h, s, l):
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2

        if 0 <= h < 60:
                r, g, b = c, x, 0
        elif 60 <= h < 120:
                r, g, b = x, c, 0
        elif 120 <= h < 180:
                r, g, b = 0, c, x
        elif 180 <= h < 240:
                r, g, b = 0, x, c
        elif 240 <= h < 300:
                r, g, b = x, 0, c
        else:
                r, g, b = c, 0, x

        r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255
        return int(r), int(g), int(b)

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def nbits_max(n):
        return 2 ** n - 1

COLOR_PALETTE_BITS = 32 # = 8 * n

# e.g. COLOR_PALETTE_BITS = 32 = 8 * n
# 32 - 1 = 1 + 2 + 4 + 8 + 16
# 16 = 2 ^ 4
# obviously, 4 = binary_digits(32) - 2
# consequently, if COLOR_PALETTE_BITS = 32, INDEX_MAX = 4
INDEX_MAX = COLOR_PALETTE_BITS.bit_length() - 2

def bits_field(n, start, len):
        return (n & nbits_max(start + len)) >> start

def set_bit(num, k, value):
        mask = ~(1 << k)
        num &= mask

        num |= (value << k)

        return num

frac = tuple

def frac_to_rgb(frac_color: frac, s = 0.5, l = 0.5):
        denominator = frac_color[0]
        numberator = frac_color[1]
        hue = 360 / denominator * numberator
        return hsl_to_rgb(hue, s, l)

class ColorPalette:
        def __init__(self):
                self.data = 0

        def next(self) -> frac:
                if bits_field(self.data, start = 0, len = 1) == 0b0:
                        self.data = set_bit(self.data, 0, 1)
                        return 1, 1
                start = index = 1
                while index <= INDEX_MAX:
                        field = bits_field(self.data, start, index)
                        if field != nbits_max(index):
                                for x in range(0, index):
                                        if bits_field(field, x, 1) == 0b0:
                                                self.data = set_bit(self.data, start + x, 1)
                                                return index + 1, x + 1

                        index <<= 1

                raise "数据结构已满"
                # 其实是可以引入动态长度的，很简单，但是目前不想做
                # 同一个 field 可以压缩到一个bit, 声明它已满，如果数据量极大可以这样做


        def add(self, frac_color: frac):
                denominator = frac_color[0]
                numberator = frac_color[1]

                if denominator == 1:
                        field_start = 0
                else:
                        field_start = 2 * ((1 << (denominator - 1)) - 1)

                self.data = set_bit(self.data, field_start + numberator - 1, 1)

        def remove(self, frac_color: frac):
                denominator = frac_color[0]
                numberator = frac_color[1]

                if denominator == 1:
                        field_start = 0
                else:
                        field_start = 2 * ((1 << (denominator - 1)) - 1)

                self.data = set_bit(self.data, field_start + numberator - 1, 0)

# 根据source找到所有的source-target对
# 对于一个source-target对：
# 如果source或target的长度都为1,直接解包后塞进ret
# 否则创建一个fakenode，然后把遍历source和target，将它们与fakenode的连接塞进ret

# 分布式数据结构也是未来要做的

class ContextSet:
        def __init__(
                        self,
                        name,
                        first_node_bs_edge: int,
                        sources,
                        target
                ):

                self.name = name
                self.data = [(first_node_bs_edge, sources, target)]

        def add(self, bs_edge, sources, target):
                if bs_edge in self.data:
                        raise "对 Context 的操作会引入圈"

                self.data.append((bs_edge, sources, target))

        def bsnodes_to_bs_edges_in_this(
                        self,
                        bsnodes,
                        bsnode_to_bs_edge_list
                ) -> set:

                ret = set()
                for bsnode in bsnodes:
                        bs_edge_list = bsnode_to_bs_edge_list[bsnode]
                        for bs_edge in bs_edge_list:
                                if bs_edge in self.data:
                                        ret.add(bs_edge)
                return ret

        def exist(self, sources, target):
                for (_edge, _sources, _target) in self.data:
                        if sources == _sources and target == _target:
                                return True
                return False

class LINKER(Enum):
        CONS = 0
        MERGE = 1

class Deterritorializer:
        def get_count(self) -> int:
                self.counter += 1
                return self.counter

        def __init__(self) -> None:
                self.base_space = DirectedMultihypergraph()
                self.base_space.add_node("@ORIGIN")
                self.base_space.add_edge("@ORIGIN", "@ORIGIN", "@ORIGIN")

                self.edge_to_linker = {}

                origin_cs = ContextSet("@ORIGIN", "@ORIGIN", "@ORIGIN", "@ORIGIN")
                self.context_to_cs = {"@ORIGIN" : origin_cs}

                self.counter = 0


        def add_normal_description(self, desc_str, sources, target, context: str):
                self.base_space.add_node(target)
                bs_edge = self.get_count()
                self.base_space.add_edge(
                        bs_edge,
                        sources,
                        target
                )

                self.context_to_cs[context].add(bs_edge, sources, target)

                self.edge_to_linker[bs_edge] = (
                       self.context_to_cs[context],
                       LINKER.CONS,      # linker:      apply name of s_d
                       desc_str
                )

        def add_description(self, desc_str, sources, target, contexts, new_context = None):
                if new_context == None:
                        self.add_normal_description(
                                desc_str,
                                sources,
                                target,
                                contexts
                        )
                        return

                self.base_space.add_node(target)
                bs_edge = self.get_count()
                self.base_space.add_edge(
                        bs_edge,
                        sources,
                        target
                )

                if new_context in contexts and self.context_to_cs[new_context].exist(sources, target):
                        raise "对 Context 的操作会引入重边"

                if new_context not in contexts:
                        self.context_to_cs[new_context] = ContextSet(new_context, bs_edge, sources, target)
                else:
                        self.context_to_cs[new_context].add(bs_edge, sources, target)

                self.edge_to_linker[bs_edge] = (
                       self.context_to_cs[new_context],
                       LINKER.MERGE,      # linker:      apply name of s_d
                       desc_str
                )

        # 这个算法目前非常慢，它需要遍历整个 base_space
        def context_extension(self, context):
                edges = set()
                cs = self.context_to_cs[context]
                for _edge, sources, target in cs.data:
                        if isinstance(sources, set):
                                for source in sources:
                                        for _source in self.base_space.node_to_sources[source]:
                                                if tuple(_source) not in self.base_space.target_to_edges.keys():
                                                        continue

                                                for name, _ in self.base_space.target_to_edges[tuple(_source)]:
                                                        edges.add(name)
                        else:
                                source = sources
                                for _source in self.base_space.node_to_sources[source]:
                                        if tuple(_source) not in self.base_space.target_to_edges.keys():
                                                continue

                                        for name, _ in self.base_space.target_to_edges[tuple(_source)]:
                                                edges.add(name)

                        if isinstance(sources, set):
                                for source in sources:
                                        for _source in self.base_space.node_to_targets[source]:
                                                if tuple(_source) not in self.base_space.target_to_edges.keys():
                                                        continue

                                                for name, _ in self.base_space.target_to_edges[tuple(_source)]:
                                                        edges.add(name)
                        else:
                                source = sources
                                for _source in self.base_space.node_to_targets[source]:
                                        if tuple(_source) not in self.base_space.target_to_edges.keys():
                                                continue

                                        for name, _ in self.base_space.target_to_edges[tuple(_source)]:
                                                edges.add(name)

                        if isinstance(sources, set):
                                for source in sources:
                                        for _source in self.base_space.node_to_sources[source]:
                                                if tuple(_source) not in self.base_space.source_to_edges.keys():
                                                        continue

                                                for name, _ in self.base_space.source_to_edges[tuple(_source)]:
                                                        edges.add(name)
                        else:
                                source = sources
                                for _source in self.base_space.node_to_sources[source]:
                                        if tuple(_source) not in self.base_space.source_to_edges.keys():
                                                continue

                                        for name, _ in self.base_space.source_to_edges[tuple(_source)]:
                                                edges.add(name)

                        if isinstance(sources, set):
                                for source in sources:
                                        for _source in self.base_space.node_to_targets[source]:
                                                if tuple(_source) not in self.base_space.source_to_edges.keys():
                                                        continue

                                                for name, _ in self.base_space.source_to_edges[tuple(_source)]:
                                                        edges.add(name)
                        else:
                                source = sources
                                for _source in self.base_space.node_to_targets[source]:
                                        if tuple(_source) not in self.base_space.source_to_edges.keys():
                                                continue

                                        for name, _ in self.base_space.source_to_edges[tuple(_source)]:
                                                edges.add(name)

                        # 必然是单 target

                        if (target, ) in self.base_space.source_to_edges.keys():
                                for name, _ in self.base_space.source_to_edges[(target, )]:
                                        edges.add(name)
                        if (target, ) in self.base_space.target_to_edges.keys():
                                for name, _ in self.base_space.target_to_edges[(target, )]:
                                        edges.add(name)
                return self.base_space.degenerate_from_edges(edges)


        def __render_context_list(self, canvas: tk.Canvas, cs_to_color: dict[ContextSet, frac], canvas_height: int):
                height = len(cs_to_color.keys()) * 10
                xstart = 10
                ystart = canvas_height - height
                font_style = ('Arial', 6)
                for cs, color in cs_to_color.items():
                        canvas.create_line(xstart, ystart, xstart + 20, ystart, width=6, fill=rgb_to_hex(frac_to_rgb(color, 0.5, 0.5)))
                        canvas.create_text(xstart + 25, ystart - 6, text=cs.name, fill="white", anchor='nw', font=font_style)
                        ystart += 10

        def render(self, width_and_height, context = None, iterations = 4, window: tk.Tk | None = None):
                if window == None:
                        is_window_none = True
                else:
                        is_window_none = False

                if window == None:
                        window = tk.Tk()
                        window.protocol("WM_DELETE_WINDOW", window.quit)
                window.title("Deterritorializer Alpha")

                canvas_height = 500

                canvas = tk.Canvas(window, width=500, height=canvas_height, bg="black")
                canvas.pack()

                if context == None:
                        nodes, edges = self.base_space.degenerate()
                else:
                        nodes, edges = self.context_extension(context)
                positions = force_directed_layout(list(nodes), edges, width_and_height, iterations)

                color_palette = ColorPalette()
                cs_to_color = {}

                def on_click_line_button1(edge):
                        canvas.destroy()
                        self.render(width_and_height, context=self.edge_to_linker[edge][0].name, iterations = iterations, window = window)
                def on_click_line_button3(edge):
                        print(1)
                        text_window = tk.Tk()
                        text = tk.Text(text_window, height=8)
                        text.pack()
                        text.insert('1.0', self.edge_to_linker[edge][2])
                        text_window.mainloop()

                for edge in edges:
                        node_i, node_j = edge[0], edge[1]

                        edge_name = edge[2]
                        if edge_name != '@ORIGIN':
                                cs = self.edge_to_linker[edge_name][0]
                                if cs not in cs_to_color:
                                        color = color_palette.next()
                                        cs_to_color[cs] = color
                                else:
                                        color = cs_to_color[cs]

                                line = canvas.create_line(
                                        *positions[node_i],
                                        *positions[node_j],
                                        fill=rgb_to_hex(frac_to_rgb(color, 0.5, 0.5)),
                                        width=3
                                )

                                canvas.tag_bind(
                                        line,
                                        "<Button-1>",
                                        lambda event: on_click_line_button1(edge_name)
                                )

                                canvas.tag_bind(
                                        line,
                                        "<Button-3>",
                                        lambda event: on_click_line_button3(edge_name)
                                )


                        if not isinstance(node_i, DirectedMultihypergraph.FakeNode):
                                create_circle(canvas, *positions[node_i],4, fill="red", outline="gray"),
                        if not isinstance(node_j, DirectedMultihypergraph.FakeNode):
                                create_circle(canvas, *positions[node_j],4, fill="red", outline="gray"),

                        if node_i != node_j:
                                add_arrow_to_line(canvas, line)

                for node in nodes:
                        if isinstance(node, DirectedMultihypergraph.FakeNode):
                                continue
                        x, y = positions[node]
                        canvas.create_text(x, y, text=str(node), fill="white", anchor='nw', font=('Arial', 8))


                self.__render_context_list(canvas, cs_to_color, canvas_height)

                if is_window_none:
                        window.mainloop()
                else:
                        window.update()

d15r = Deterritorializer()
d15r.add_description("Test description", "@ORIGIN", "概念1", "@ORIGIN", new_context="new context")
d15r.add_description("Test description", "概念1", "概念2", "new context")
d15r.add_description("Test description", set(["概念1", "概念2"]), "概念3", "new context")
d15r.add_description("Test description", "概念3", "概念4", "new context", new_context="new context2")
d15r.render((400, 300), context="new context")
