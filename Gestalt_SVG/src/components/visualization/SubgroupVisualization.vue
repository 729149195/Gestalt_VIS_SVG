<template>
    <div class="force-graph-container">
        <div class="controls">
            <el-button class="clear-button" @click="clearSelectedNodes">清空并高亮所有节点</el-button>
            <v-switch v-model="checkbox" inset color="#55C000"
                :label="checkbox ? '框选模式开启(缩放已禁用)' : '框选模式关闭(缩放已启用)'" />  
        </div>
        <div class="graph-grid">
            <div v-for="(dim, index) in dimensions" :key="dim" class="graph-item">
                <div class="graph-title">Dimension {{ dim }}</div>
                <div :ref="el => { if (el) graphContainers[index] = el }" class="graph-container"></div>
            </div>
        </div>
    </div>
</template>
<script setup>
import { ref, onMounted, nextTick, watch, onUnmounted } from 'vue';
import * as d3 from 'd3';
import { computed } from 'vue';
import { useStore } from 'vuex';
import { ElButton } from 'element-plus'

const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);
const checkbox = ref(false);
const dimensions = [0, 1, 2, 3];
const graphContainers = ref(Array(dimensions.length).fill(null));
const isSelecting = ref(false);
let selectionRect = null;
let selectionStart = { x: 0, y: 0 };

// 添加一个延迟函数
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function loadAndRenderAllGraphs() {
    try {
        // 等待DOM更新
        await nextTick();
        // 增加延迟时间以确保布局完全计算完成
        await delay(300);

        for (let i = 0; i < dimensions.length; i++) {
            try {
                const data = await d3.json(`http://localhost:5000/subgraph/${i}`);
                const container = graphContainers.value[i];

                if (!container) {
                    console.error(`Container for dimension ${i} not found`);
                    continue;
                }

                // 等待一小段时间确保每个容器都准备好
                await delay(100);

                // 检查容器尺寸并设置默认值
                const width = container.clientWidth || 600;
                const height = container.clientHeight || 400;

                if (width <= 0 || height <= 0) {
                    console.warn(`Container ${i} dimensions not ready, using default values`);
                }

                renderGraph(container, data);
            } catch (error) {
                console.error(`Error loading data for dimension ${i}:`, error);
            }
        }
    } catch (error) {
        console.error('Error in loadAndRenderAllGraphs:', error);
    }
}

function renderGraph(container, graphData) {
    // 确保容器和数据都存在
    if (!container || !graphData || !graphData.nodes || !graphData.links) {
        console.error('Invalid container or graph data');
        return;
    }

    // 确保容器有有效的尺寸
    const width = container.clientWidth || 600; // 提供默认值
    const height = container.clientHeight || 400; // 提供默认值

    if (width <= 0 || height <= 0) {
        console.error('Invalid container dimensions:', container);
        return;
    }

    // 清除可能存在的旧SVG
    d3.select(container).selectAll('svg').remove();

    // 创建新的SVG
    const svg = d3
        .select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', [-width / 2, -height / 2, width, height]);

    // 添加缩放功能
    const zoom = d3.zoom()
        .filter((event) => {
            // Allow zooming only when selection mode is off
            return !checkbox.value;
        })
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });


    // 如果不在框选模式下,启用缩放
    if (!checkbox.value) {
        svg.call(zoom);
    }

    // 添加鼠标事件监听
    svg.on('mousedown', (event) => onMouseDown(event, svg))
        .on('mousemove', (event) => onMouseMove(event, svg))
        .on('mouseup', (event) => onMouseUp(event, svg));

    const g = svg.append('g');

    // 力导引仿真器
    const simulation = d3
        .forceSimulation(graphData.nodes)
        .force('link', d3.forceLink(graphData.links).id((d) => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-50))
        .force('center', d3.forceCenter(0, 0));

    // 绘制连线
    const link = g
        .append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(graphData.links)
        .join('line')
        .attr('stroke', '#aaa')
        .attr('stroke-width', (d) => Math.sqrt(d.value));

    // 绘制节点
    const node = g
        .append('g')
        .attr('class', 'nodes')
        .selectAll('circle')
        .data(graphData.nodes)
        .join('circle')
        .attr('r', 8)
        .attr('fill', '#69b3a2')
        .attr('stroke-width', '1')
        .style('cursor', 'pointer')
        .on('click', function (event, d) {
            if (checkbox.value) return; // 在框选模式下禁用点击
            const nodeName = d.name.split('/').pop();
            if (selectedNodeIds.value.includes(nodeName)) {
                store.commit('REMOVE_SELECTED_NODE', nodeName);
                d3.select(this).attr('fill', '#69b3a2');
            } else {
                store.commit('ADD_SELECTED_NODE', nodeName);
                d3.select(this).attr('fill', '#ff6347');
            }
        })
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended)
        );

    // 节点标签
    g.append('g')
        .attr('class', 'labels')
        .selectAll('text')
        .data(graphData.nodes)
        .join('text')
        .attr('dy', -10)
        .attr('text-anchor', 'middle')
        .text((d) => d.name.split('/').pop())
        .style('font-size', '14px');

    simulation.on('tick', () => {
        link
            .attr('x1', (d) => d.source.x)
            .attr('y1', (d) => d.source.y)
            .attr('x2', (d) => d.target.x)
            .attr('y2', (d) => d.target.y);

        node.attr('cx', (d) => d.x).attr('cy', (d) => d.y);

        g.selectAll('.labels text')
            .attr('x', (d) => d.x)
            .attr('y', (d) => d.y);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        const width = event.sourceEvent.target.ownerSVGElement.clientWidth / 2;
        const height = event.sourceEvent.target.ownerSVGElement.clientHeight / 2;

        d.fx = Math.max(-width, Math.min(width, event.x));
        d.fy = Math.max(-height, Math.min(height, event.y));
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

function clearSelectedNodes() {
    store.dispatch('clearSelectedNodes');

    graphContainers.value.forEach((container) => {
        d3.select(container)
            .selectAll('circle')
            .attr('fill', '#69b3a2');
    });
}

watch(selectedNodeIds, () => {
    nextTick(() => {
        graphContainers.value.forEach((container) => {
            const svg = d3.select(container).select('svg');
            svg.selectAll('circle')
                .attr('fill', (d) => {
                    const nodeName = d.name.split('/').pop();
                    return selectedNodeIds.value.includes(nodeName) ? '#ff6347' : '#69b3a2';
                });
        });
    });
});

function onMouseDown(event, svg) {
    if (!checkbox.value) return;

    isSelecting.value = true;
    const point = d3.pointer(event);
    selectionStart = { x: point[0], y: point[1] };

    if (!selectionRect) {
        selectionRect = svg.append('rect')
            .attr('class', 'selection')
            .attr('x', selectionStart.x)
            .attr('y', selectionStart.y)
            .attr('width', 0)
            .attr('height', 0);
    }
}

function onMouseMove(event, svg) {
    if (!isSelecting.value) return;

    const point = d3.pointer(event);
    const x = Math.min(selectionStart.x, point[0]);
    const y = Math.min(selectionStart.y, point[1]);
    const width = Math.abs(selectionStart.x - point[0]);
    const height = Math.abs(selectionStart.y - point[1]);

    selectionRect
        .attr('x', x)
        .attr('y', y)
        .attr('width', width)
        .attr('height', height);
}

function onMouseUp(event, svg) {
    if (!isSelecting.value) return;

    isSelecting.value = false;
    const selectionBox = selectionRect.node().getBBox();
    selectNodesInBox(selectionBox, svg);
    selectionRect.remove();
    selectionRect = null;
}

function selectNodesInBox(selectionBox, svg) {
    const selectedNodes = [];
    const transform = d3.zoomTransform(svg.node());
    const adjustedSelectionBox = {
        x: (selectionBox.x - transform.x) / transform.k,
        y: (selectionBox.y - transform.y) / transform.k,
        width: selectionBox.width / transform.k,
        height: selectionBox.height / transform.k,
    };

    svg.selectAll('circle').each(function (d) {
        const cx = d.x;
        const cy = d.y;

        if (cx >= adjustedSelectionBox.x && cx <= adjustedSelectionBox.x + adjustedSelectionBox.width &&
            cy >= adjustedSelectionBox.y && cy <= adjustedSelectionBox.y + adjustedSelectionBox.height) {
            selectedNodes.push(d);
        }
    });

    const nodeIds = selectedNodes.map(node => node.name.split('/').pop());
    store.commit('UPDATE_SELECTED_NODES', { nodeIds, group: null });
}

function handleKeyDown(event) {
    if (event.key === 'c' || event.key === 'C') {
        checkbox.value = !checkbox.value;
        graphContainers.value.forEach((container) => {
            const svg = d3.select(container).select('svg');
            if (checkbox.value) {
                disableZoom(svg);
            } else {
                enableZoom(svg);
            }
        });
    }
}

function disableZoom(svg) {
    svg.on('.zoom', null);
    svg.style('cursor', 'crosshair');
}

function enableZoom(svg) {
    const zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            svg.select('g').attr('transform', event.transform);
        });

    svg.call(zoom);
    svg.style('cursor', 'default');
}

// 修改onMounted钩子
onMounted(async () => {
    try {
        // 等待DOM完全渲染
        await nextTick();
        // 等待一小段时间确保布局计算完成
        await delay(100);
        await loadAndRenderAllGraphs();
        window.addEventListener('keydown', handleKeyDown);
    } catch (error) {
        console.error('Error in onMounted:', error);
    }
});

onUnmounted(() => {
    window.removeEventListener('keydown', handleKeyDown);
});
</script>
<style scoped>
.graph-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    width: 100%;
}

.graph-item {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 500px;
    border: 1px solid #ccc;
    border-radius: 5px;
    background: #f9f9f9;
    overflow: hidden;
}

.graph-title {
    padding: 8px;
    font-weight: bold;
    background-color: #eee;
}

.graph-container {
    flex: 1;
    width: 100%;
    height: calc(100% - 40px);
    position: relative;
}

.clear-button {
    padding: 8px 16px;
    border: none;
    background-color: #ff6347;
    color: #fff;
    cursor: pointer;
    border-radius: 6px;
    font-size: 14px;
    transition: all 0.3s ease;
}

.selection {
    fill: #55C000;
    fill-opacity: 0.2;
    stroke: #55C000;
    stroke-width: 1px;
}

.checkbox-control span {
    color: #666;
    font-size: 14px;
    font-weight: 500;
}
</style>
