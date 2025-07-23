<template>
    <div class="hull-select">
        <span>“press key <v-icon icon="mdi-alpha-h-box"></v-icon>”↴</span>
        <v-switch v-model="groupHull" inset color="#FFC000" class="switch"
            :label="groupHull ? 'Hull ON' : 'Hull OFF'" />
        <span>“press key <v-icon icon="mdi-alpha-c-box"></v-icon>”↴</span>
        <v-switch v-model="checkbox" inset color="#55C000" class="switch"
            :label="checkbox ? 'Checkbox ON disable zoom' : 'Checkbox OFF enable zoom'" />
        <div class="input-group">
            <v-combobox v-model="eps" :items="eps_list || []" label="Eps（control Cluster Number）" class="input-box" />
        </div>
        <div class="input-group">
            <v-text-field v-model="min" :min="1" :max="20" step="1" label="Minimum_nodes in a Cluster" type="number"
                class="input-box" />
        </div>
        <div class="input-group">
            <v-text-field v-model="link" :min="0.0" :max="1.0" step="0.1" label="Link Density" type="number"
                class="input-box" />
        </div>
    </div>
    <div class="svg-container">
        <svg viewBox="0 0 1200 700" ref="svg"></svg>
    </div>
</template>

<script setup>
import { ref, onMounted, watch, onUnmounted } from 'vue';
import { debounce } from 'lodash';
import * as d3 from 'd3';
import { useStore } from 'vuex';
const store = useStore();

const width = 1200;
const height = 700;
const svg = ref(null);
const apiUrl = 'http://127.0.0.1:8000/community_data_mult';
const runClusteringUrl = 'http://127.0.0.1:8000/run_clustering';
const epsUrl = 'http://127.0.0.1:8000/get_eps_list';
const groupHull = ref(true);
const checkbox = ref(false);
const eps = ref(null);
const eps_list = ref(null)
const min = ref(1);
const link = ref(0.3);

let simulation;
const isSelecting = ref(false);
let selectionStart = { x: 0, y: 0 };
let selectionRect = null;

const graphData = ref({
    nodes: [],
    links: [],
    groups: []
});

const color = d3.scaleLinear().domain([-2, 4]).range(["#252525", "#cccccc"]);
const groupHullColor = "#9ecae1";

const customColorMap = {
    "circle": "#FFE119", // 鲜黄
    "rect": "#E6194B", // 猩红
    "line": "#4363D8", // 宝蓝
    "polyline": "#911EB4", // 紫色
    "polygon": "#F58231", // 橙色
    "path": "#3CB44B", // 明绿
    "text": "#46F0F0", // 青色
    "ellipse": "#F032E6", // 紫罗兰
    "image": "#BCF60C", // 酸橙
    "use": "#FFD700", // 金色
    "defs": "#FF4500", // 橙红色
    "linearGradient": "#1E90FF", // 道奇蓝
    "radialGradient": "#FF6347", // 番茄
    "stop": "#4682B4", // 钢蓝
    "symbol": "#D2691E", // 巧克力
    "clipPath": "#FABEBE", // 粉红
    "mask": "#8B008B", // 深紫罗兰红色
    "pattern": "#A52A2A", // 棕色
    "filter": "#5F9EA0", // 冰蓝
    "feGaussianBlur": "#D8BFD8", // 紫丁香
    "feOffset": "#FFDAB9", // 桃色
    "feBlend": "#32CD32", // 酸橙绿
    "feFlood": "#FFD700", // 金色
    "feImage": "#FF6347", // 番茄
    "feComposite": "#FF4500", // 橙红色
    "feColorMatrix": "#1E90FF", // 道奇蓝
    "feMerge": "#FF1493", // 深粉色
    "feMorphology": "#00FA9A", // 中春绿色
    "feTurbulence": "#8B008B", // 深紫罗兰红色
    "feDisplacementMap": "#FFD700", // 金色
    "unknown": "#696969" // 暗灰色
};

onMounted(async () => {
    count.value = 1;
    try {
        const response = await fetch(epsUrl);
        if (!response.ok) {
            throw new Error('Failed to fetch initial EPS value');
        }
        const data = await response.json();
        if (data && typeof data.max_eps === 'number') {
            eps.value = parseFloat(data.max_eps.toFixed(4));
            eps_list.value = data.epss.map(eps => Number(parseFloat(eps).toFixed(4)));
        } else {
            console.error('Unexpected data format received:', data);
        }
    } catch (error) {
        console.error('Error fetching initial EPS value:', error);
    }

    fetchData();
    window.addEventListener('keydown', handleKeyDown);
});

onUnmounted(() => {
    window.removeEventListener('keydown', handleKeyDown);
});

function handleKeyDown(event) {
    if (event.key === 'h' || event.key === 'H') {
        groupHull.value = !groupHull.value;

        updatahull();
    }

    if (event.key === 'c' || event.key === 'C') {
        checkbox.value = !checkbox.value;
        if (checkbox.value) {
            disableZoom();
        } else {
            enableZoom();
        }
        updatahull();
    }
}

const count = ref(0);
watch([eps, min, link], debounce(() => {
    if (count.value !== 1) {
        runClusteringWithParams();
    }
    count.value++;
}, 200));

watch(groupHull, (newValue) => {
    updatahull();
});

watch(checkbox, (newValue) => {
    if (newValue) {
        disableZoom(); // 禁用缩放
    } else {
        enableZoom(); // 启用缩放
    }
});

function runClusteringWithParams() {
    fetch(runClusteringUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            eps: eps.value,
            min_samples: min.value,
            distance_threshold_ratio: link.value
        })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            d3.select(svg.value).selectAll('*').remove();
            fetchData();
        })

        .catch(error => {
            console.error('Error running clustering:', error);
        });
}


async function fetchData() {
    try {
        const response = await fetch(apiUrl);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        graphData.value.nodes = data.GraphData.node;
        graphData.value.links = data.GraphData.links;
        graphData.value.groups = data.GraphData.group;
        submitAllNodes();
        initializeGraph();
    } catch (error) {
        console.error('Error fetching or parsing data:', error);
    }
}

function submitAllNodes() {
    const nodeIds = graphData.value.nodes.map(node => {
        const parts = node.id.split("/");
        return parts[parts.length - 1];
    });

    store.commit('SET_ALL_VISIBLE_NODES', nodeIds);
}


function initializeGraph() {
    const svgEl = d3.select(svg.value)
        .attr('width', width)
        .attr('height', height)
        .on('mousedown', onMouseDown)
        .on('mousemove', onMouseMove)
        .on('mouseup', onMouseUp);

    const zoom = d3.zoom()
        .on("zoom", (event) => {
            contentGroup.attr("transform", event.transform);
        });

    svgEl.call(zoom);

    const contentGroup = svgEl.append('g').attr('class', 'content');

    const hullGroup = contentGroup.append('g').attr('class', 'hulls');
    const linkGroup = contentGroup.append('g').attr('class', 'links');
    const nodeGroup = contentGroup.append('g').attr('class', 'nodes');

    // First stage: Force-directed layout for groups
    const groupCenters = computeGroupCenters();
    const groupSimulation = d3.forceSimulation(groupCenters)
        .force('charge', d3.forceManyBody().strength(-3000)) // Increased repulsion between groups
        .force('center', d3.forceCenter(width / 3, height / 3))
        .force('collision', d3.forceCollide().radius(d => d.radius * 2.0)) // Increased collision radius
        .force('x', d3.forceX(width / 2).strength(0.1))
        .force('y', d3.forceY(height / 2).strength(0.1))
        .stop();

    // Run the group simulation
    for (let i = 0; i < 10; ++i) groupSimulation.tick();

    // Assign group centers to nodes
    assignGroupCentersToNodes(groupCenters);

    // Second stage: Force-directed layout for nodes within groups
    const simulation = d3.forceSimulation(graphData.value.nodes)
        .force('link', d3.forceLink(graphData.value.links).id(d => d.id).distance(d => 350 - d.value * 500))
        .force('charge', d3.forceManyBody().strength(-1100))
        .force('center', d3.forceCenter(width / 1.7, height / 2.3))
        .force('group', forceGroup().strength(0.08));

    const link = linkGroup.selectAll('line')
        .data(graphData.value.links)
        .enter().append('line')
        .attr('class', 'link')
        .style('stroke-width', d => Math.sqrt(d.value * 60));

    const node = nodeGroup.selectAll('circle')
        .data(graphData.value.nodes)
        .enter().append('circle')
        .attr('class', 'node')
        .attr('r', 20)
        .attr("id", d => {
            const parts = d.id.split("/");
            return parts[parts.length - 1];
        })
        .attr("style", "cursor: pointer;")
        .attr("fill", d => {
            const svgTag = d.id.split('/');
            const parts = svgTag[svgTag.length - 1];
            const index = parts.split('_')[0];
            return customColorMap[index] || color(d.propertyValue * 1);
        })
        .on('click', hullClicked)
        .call(drag(simulation));

    node.append('title')
        .text(d => {
            const idParts = d.id.split('/');
            const lastPart = idParts.pop();
            return `${lastPart}`;
        });

    simulation.on('tick', () => {
        hullGroup.selectAll('path').remove();
        drawHulls(hullGroup, graphData.value.groups, groupHullColor, 'group-hull');
        updatahull();

        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
    });

    const renderedTags = new Set();
    graphData.value.nodes.forEach(node => {
        const parts = node.id.split('/');
        const tag = parts[parts.length - 1].split('_')[0];
        if (customColorMap[tag]) {
            renderedTags.add(tag);
        }
    });

    const legendGroup = svgEl.append("g")
        .attr("class", "legend-group")
        .attr("transform", `translate(${width - 57}, 7)`);

    Array.from(renderedTags).forEach((tag, index) => {
        const legendItem = legendGroup.append("g")
            .attr("class", "legend-item")
            .attr("transform", `translate(0, ${index * 25})`);

        legendItem.append("circle")
            .attr("r", 6)
            .attr("cx", -10)
            .attr("cy", -1)
            .attr("fill", customColorMap[tag]);

        legendItem.append("text")
            .attr("x", 0)
            .attr("y", 1.6)
            .text(tag)
            .attr("font-size", "14px")
            .attr("fill", "#000");
    });

    const initialZoom = d3.zoomIdentity.translate(width / 2, height / 2).scale(0.2).translate(-width / 2, -height / 2);
    svgEl.call(zoom.transform, initialZoom);

    if (checkbox.value) {
        disableZoom();
    }
}

function computeGroupCenters() {
    const groupCenters = graphData.value.groups.map(group => {
        const nodes = group.map(id => graphData.value.nodes.find(node => node.id === id));
        const x = d3.mean(nodes, d => d.x);
        const y = d3.mean(nodes, d => d.y);
        const radius = Math.sqrt(nodes.length) * 30; // Adjust this multiplier as needed
        return { x, y, radius, group };
    });
    return groupCenters;
}

function assignGroupCentersToNodes(groupCenters) {
    graphData.value.nodes.forEach(node => {
        const group = graphData.value.groups.find(g => g.includes(node.id));
        if (group) {
            const groupCenter = groupCenters.find(gc => gc.group === group);
            node.groupX = groupCenter.x;
            node.groupY = groupCenter.y;
        }
    });
}

function forceGroup() {
    let nodes;
    let strength = 0.1;

    function force(alpha) {
        for (let i = 0, n = nodes.length, node; i < n; ++i) {
            node = nodes[i];
            if (node.groupX != null && node.groupY != null) {
                node.vx += (node.groupX - node.x) * strength * alpha;
                node.vy += (node.groupY - node.y) * strength * alpha;
            }
        }
    }

    force.initialize = function (_) {
        nodes = _;
    };

    force.strength = function (_) {
        return arguments.length ? (strength = +_, force) : strength;
    };

    return force;
}

function disableZoom() {
    d3.select(svg.value).on('.zoom', null);
    d3.select(svg.value).style('cursor', 'crosshair');
}

function enableZoom() {
    const zoomHandler = d3.zoom().on("zoom", (event) => {
        window.requestAnimationFrame(() => {
            d3.select(svg.value).select('.content').attr("transform", event.transform);
        });
    });

    d3.select(svg.value).call(zoomHandler);
    d3.select(svg.value).style('cursor', 'default');
}

function onMouseDown(event) {
    if (!checkbox.value) return;

    isSelecting.value = true;
    selectionStart = d3.pointer(event);

    if (!selectionRect) {
        selectionRect = d3.select(svg.value).append('rect')
            .attr('class', 'selection')
            .attr('x', selectionStart[0])
            .attr('y', selectionStart[1])
            .attr('width', 0)
            .attr('height', 0)
            .style('stroke', '#55C000')
            .style('stroke-width', '1px')
            .style('fill', '#55C000')
            .style('fill-opacity', 0.2);
    }
}

function onMouseMove(event) {
    if (!isSelecting.value) return;

    const currentPos = d3.pointer(event);
    const x = Math.min(selectionStart[0], currentPos[0]);
    const y = Math.min(selectionStart[1], currentPos[1]);
    const width = Math.abs(selectionStart[0] - currentPos[0]);
    const height = Math.abs(selectionStart[1] - currentPos[1]);

    selectionRect
        .attr('x', x)
        .attr('y', y)
        .attr('width', width)
        .attr('height', height);
}

function onMouseUp() {
    if (!isSelecting.value) return;

    isSelecting.value = false;
    const selectionBox = selectionRect.node().getBBox();
    selectNodesInBox(selectionBox);
    selectionRect.remove();
    selectionRect = null;
}

function selectNodesInBox(selectionBox) {
    const selectedNodes = [];
    const svgElement = d3.select(svg.value);
    const transform = d3.zoomTransform(svgElement.node());
    const adjustedSelectionBox = {
        x: (selectionBox.x - transform.x) / transform.k,
        y: (selectionBox.y - transform.y) / transform.k,
        width: selectionBox.width / transform.k,
        height: selectionBox.height / transform.k,
    };


    svgElement.selectAll('.node').each(function (d) {
        const cx = d.x;
        const cy = d.y;


        if (cx >= adjustedSelectionBox.x && cx <= adjustedSelectionBox.x + adjustedSelectionBox.width &&
            cy >= adjustedSelectionBox.y && cy <= adjustedSelectionBox.y + adjustedSelectionBox.height) {
            selectedNodes.push(d);
        }
    });

    const nodeIds = selectedNodes.map(node => node.id.split('/').pop());
    store.commit('UPDATE_SELECTED_NODES', { nodeIds, group: null });
}


function updatahull() {
    const svgElement = d3.select(svg.value);
    svgElement.selectAll('.group-hull').style('display', groupHull.value ? null : 'none');
}

function hullClicked(event, d) {
    let activeHullLevel;
    let groupName;

    const isAllHullsOff = !groupHull.value;

    if (!isAllHullsOff) {
        if (groupHull.value) {
            activeHullLevel = graphData.value.groups;
        }

        for (const group of activeHullLevel) {
            if (group.includes(d.id)) {
                groupName = group[0];
                break;
            }
        }

        const groupNodes = graphData.value.nodes.filter(node => activeHullLevel.find(group => group.includes(node.id) && group[0] === groupName));
        const nodeIds = groupNodes.map(node => {
            const parts = node.id.split("/");
            return parts[parts.length - 1];
        });

        store.commit('UPDATE_SELECTED_NODES', { nodeIds, group: null });
    } else {
        const parts = d.id.split("/");
        const nodeId = parts[parts.length - 1];

        store.commit('UPDATE_SELECTED_NODES', { nodeIds: [nodeId], group: null });
    }
}

const drag = (simulation) => {
    function start(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function end(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    return d3.drag()
        .on('start', start)
        .on('drag', dragged)
        .on('end', end);
};

function drawHulls(hullGroup, groups, fillColor, className) {
    const hullsData = groups.map(group => {
        const points = group.map(member => graphData.value.nodes.find(n => n.id === member)).map(n => [n.x, n.y]);
        if (points.length === 1) {
            const [x, y] = points[0];
            return [
                [x, y],
                [x + 0.1, y + 0.1],
                [x - 0.1, y - 0.1]
            ];
        }
        if (points.length === 2) {
            const [p1, p2] = points;
            const midX = (p1[0] + p2[0]) / 2;
            const midY = (p1[1] + p2[1]) / 2;
            return [
                p1,
                p2,
                [midX + 0.1, midY + 0.1],
            ];
        }
        return points;
    }).map(points => d3.polygonHull(points));

    hullGroup.selectAll(`.${className}`)
        .data(hullsData)
        .join('path')
        .attr('class', className)
        .attr('d', d => `M${d.join('L')}Z`)
        .style('fill', fillColor)
        .style('stroke', fillColor)
        .style('stroke-width', 60)
        .style('stroke-linejoin', 'round')
        .style('opacity', 0.25);
}
</script>

<style lang="scss">
svg {
    width: 100%;

}
.links {
    stroke: #905F29;
    stroke-opacity: 0.6;
}

.nodes {
    stroke: #fff;
    stroke-width: 6px;
}

.hulls {
    fill: #905F29;
    stroke: #905F29;
}

.selection {
    fill: #905F29;
    fill-opacity: 0.2;
    stroke: #905F29;
    stroke-width: 1px;
}

.hull-select {
    position: absolute;
    display: flex;
    flex-direction: column;
    left: 20px;
    width: 210px;
}
</style>
