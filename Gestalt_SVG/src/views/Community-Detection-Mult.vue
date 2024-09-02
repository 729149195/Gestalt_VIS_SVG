<template>
    <div class="hull-select">
        <span>“<v-icon icon="mdi-alpha-h-box"></v-icon>”↴</span>
        <v-switch v-model="groupHull" inset color="#FFC000" class="switch"
            :label="groupHull ? 'Hull ON' : 'Hull OFF'" />
        <span>“<v-icon icon="mdi-alpha-c-box"></v-icon>”↴</span>
        <v-switch v-model="checkbox" inset color="#55C000" class="switch"
            :label="checkbox ? 'Checkbox ON' : 'Checkbox OFF'" />
        <div class="input-group">
            <v-combobox v-model="eps" :items="eps_list || []" label="DBSCAN Eps" class="input-box"/>
        </div>
        <div class="input-group">
            <v-text-field v-model="min" :min="1" :max="20" step="1" label="Min_Samples" type="number"
                class="input-box" />
        </div>
        <div class="input-group">
            <v-text-field v-model="link" :min="0.0" :max="1.0" step="0.1" label="link_threshold_ratio" type="number"
                class="input-box" />
        </div>
    </div>
    <svg :width="width" :height="height" ref="svg"></svg>
</template>

<script setup>
import { ref, onMounted, watch, onUnmounted } from 'vue';
import { debounce } from 'lodash';
import * as d3 from 'd3';
import { useStore } from 'vuex';
const store = useStore();

const width = 1250;
const height = 1300;
const svg = ref(null);
const apiUrl = 'http://localhost:5000/community_data_mult';
const runClusteringUrl = 'http://localhost:5000/run_clustering';
const epsUrl = 'http://127.0.0.1:5000/get_eps_list';
const groupHull = ref(true);
const checkbox = ref(false); // This checkbox will control the selection box mode
const eps = ref(null); // Initialize eps as null
const eps_list = ref(null)
const min = ref(1);
const link = ref(0.5);


let simulation;
let isSelecting = false; // Track if the user is currently selecting
let selectionStart = { x: 0, y: 0 }; // Start coordinates of selection
let selectionRect = null; // D3 element for the selection rectangle

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
            // console.log(eps_list.value)
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
        if (groupHull.value) {
            checkbox.value = false;
        }
        updatahull();
    } else if (event.key === 'c' || event.key === 'C') {
        checkbox.value = !checkbox.value;
        if (checkbox.value) {
            groupHull.value = false;
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
        console.log(count)
    }
    count.value ++;
}, 200));


// 添加监听器以确保 groupHull 和 checkbox 之间的一开一关逻辑
watch(groupHull, (newValue) => {
    if (newValue) {
        checkbox.value = false; // 自动关闭 checkbox
    }
    updatahull(); // 重新渲染 hull
});

watch(checkbox, (newValue) => {
    if (newValue) {
        groupHull.value = false; // 自动关闭 groupHull
        disableZoom(); // 禁用缩放
    } else {
        enableZoom(); // 启用缩放
        updatahull(); // 重新渲染 hull
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
        .then(() => {
            if (checkbox.value) {
                disableZoom();  // 重新禁用缩放和拖拉
            }
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
        return parts[parts.length - 1]; // 假设节点的唯一标识位于id的最后一部分
    });

    // console.log(nodeIds);
    store.commit('SET_ALL_VISIBLE_NODES', nodeIds); // 假设这是你的mutation
}

function initializeGraph() {
    const svgEl = d3.select(svg.value)
        .attr('width', width)
        .attr('height', height)
        .on('mousedown', onMouseDown) // Add mouse events for selection
        .on('mousemove', onMouseMove)
        .on('mouseup', onMouseUp);

    // Define zoom behavior
    const zoom = d3.zoom()
        .on("zoom", (event) => {
            contentGroup.attr("transform", event.transform);
        });

    svgEl.call(zoom);

    const contentGroup = svgEl.append('g').attr('class', 'content');

    const hullGroup = contentGroup.append('g').attr('class', 'hulls');
    const linkGroup = contentGroup.append('g').attr('class', 'links');
    const nodeGroup = contentGroup.append('g').attr('class', 'nodes');

    simulation = d3.forceSimulation(graphData.value.nodes)
        .force('link', d3.forceLink(graphData.value.links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-80))
        .force('center', d3.forceCenter(width / 2, height / 2));

    // Create links
    const link = linkGroup.selectAll('line')
        .data(graphData.value.links)
        .enter().append('line')
        .attr('class', 'link')
        .style('stroke-width', d => Math.sqrt(d.value * 35));

    // Create nodes
    const node = nodeGroup.selectAll('circle')
        .data(graphData.value.nodes)
        .enter().append('circle')
        .attr('class', 'node')
        .attr('r', 12)
        .attr("id", d => {
            const parts = d.id.split("/");
            return parts[parts.length - 1];
        })
        .attr("style", "cursor: pointer;")
        .attr("fill", d => {
            const svgTag = d.id.split('/');
            const parts = svgTag[svgTag.length - 1];
            const index = parts.split('_')[0];
            return customColorMap[index] || color(d.propertyValue * 1); // Use custom color or default color
        })
        .on('click', hullClicked)
        .call(drag(simulation));

    node.append('title')
        .text(d => {
            const idParts = d.id.split('/'); // Use '/' to separate id
            const lastPart = idParts.pop(); // Get the last part
            return `${lastPart}`; // Return processed text
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
        .attr("transform", `translate(${width - 57}, 7)`); // Place legend in top right corner of SVG

    Array.from(renderedTags).forEach((tag, index) => {
        const legendItem = legendGroup.append("g")
            .attr("class", "legend-item")
            .attr("transform", `translate(0, ${index * 25})`); // Offset each legend item to fit view

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

    const initialZoom = d3.zoomIdentity.translate(width / 2, height / 2).scale(0.5).translate(-width / 2, -height / 2);
    svgEl.call(zoom.transform, initialZoom);

    if (checkbox.value) {
        disableZoom();
    }
}

function disableZoom() {
    d3.select(svg.value).on('.zoom', null); // Remove zoom events
    d3.select(svg.value).style('cursor', 'crosshair'); // Change cursor to crosshair
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
    if (!checkbox.value) return; // Exit if not in selection mode

    isSelecting = true;
    selectionStart = d3.pointer(event); // Get starting coordinates
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
    if (!isSelecting) return; // Exit if not selecting

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
    if (!isSelecting) return; // Exit if not selecting

    isSelecting = false;
    const selectionBox = selectionRect.node().getBBox();
    selectNodesInBox(selectionBox);
    selectionRect.remove();
    selectionRect = null;
}

function selectNodesInBox(selectionBox) {
    const selectedNodes = [];
    const svgElement = d3.select(svg.value);
    const transform = d3.zoomTransform(svgElement.node());

    // Adjust selection box coordinates to account for zoom and pan transformations
    const adjustedSelectionBox = {
        x: (selectionBox.x - transform.x) / transform.k,
        y: (selectionBox.y - transform.y) / transform.k,
        width: selectionBox.width / transform.k,
        height: selectionBox.height / transform.k,
    };

    // Access nodes directly from the graphData
    svgElement.selectAll('.node').each(function (d) {
        const cx = d.x; // Directly access x-coordinate from the data bound to the node
        const cy = d.y; // Directly access y-coordinate from the data bound to the node

        // Check if the node's center is within the adjusted selection box
        if (cx >= adjustedSelectionBox.x && cx <= adjustedSelectionBox.x + adjustedSelectionBox.width &&
            cy >= adjustedSelectionBox.y && cy <= adjustedSelectionBox.y + adjustedSelectionBox.height) {
            selectedNodes.push(d); // Add node to selectedNodes if it falls within the selection box
        }
    });

    const nodeIds = selectedNodes.map(node => node.id.split('/').pop());
    // console.log(nodeIds);
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
        .style('stroke-width', 50)
        .style('stroke-linejoin', 'round')
        .style('opacity', 0.15);
}
</script>

<style lang="scss">
.links {
    stroke: #333;
    stroke-opacity: 0.6;
}

.nodes {
    stroke: #fff;
    stroke-width: 1.5px;
}

.hulls {
    fill: none;
    stroke: #c0c0c0;
}

.selection {
    fill: #55C000;
    fill-opacity: 0.2;
    stroke: #55C000;
    stroke-width: 1px;
}

.hull-select {
    position: absolute;
    display: flex;
    flex-direction: column;
    left: 20px;
    width: 180px;
}
</style>
