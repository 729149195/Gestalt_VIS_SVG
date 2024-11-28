<template>
    <div class="force-graph-container">
        <button class="clear-button" @click="clearSelectedNodes">清空节点</button>
        <div class="graph-grid">
            <div v-for="dim in dimensions" :key="dim" class="graph-item" ref="graphContainers">
               Dimension {{ dim }}
            </div>
        </div>
    </div>
</template>


<script setup>
import { ref, onMounted, nextTick, watch  } from 'vue';
import * as d3 from 'd3';
import { computed } from 'vue';
import { useStore } from 'vuex';

const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);

const dimensions = [0, 1, 2, 3];
const graphContainers = ref([]);

const clearSelectedNodes = () => {
    store.dispatch('clearSelectedNodes');

    // 遍历所有维度的节点，重置颜色
    graphContainers.value.forEach((container) => {
        d3.select(container)
            .selectAll('circle')
            .attr('fill', '#69b3a2'); // 重置为默认颜色
    });
};

watch(selectedNodeIds, () => {
    nextTick(() => {
        graphContainers.value.forEach((container) => {
            const svg = d3.select(container).select('svg');
            svg.selectAll('circle')
                .attr('fill', (d) => {
                    const nodeName = d.name.split('/').pop();
                    // 如果节点在 selectedNodeIds 中，则使用高亮颜色
                    return selectedNodeIds.value.includes(nodeName) ? '#ff6347' : '#69b3a2';
                });
        });
    });
});


// 处理每个维度的数据和渲染逻辑
function renderGraph(container, graphData) {
    const width = container.clientWidth;
    const height = container.clientHeight;

    // 创建 SVG 容器，设置 viewBox 使其自适应
    const svg = d3
        .select(container)
        .append('svg')
        .attr('viewBox', [-width / 2, -height / 2, width, height])
        .style('width', '100%')
        .style('height', '100%')
        .call(
            d3.zoom().on('zoom', (event) => {
                svg.attr('transform', event.transform);
            })
        )
        .append('g');

    // 力导引仿真器
    const simulation = d3
        .forceSimulation(graphData.nodes)
        .force('link', d3.forceLink(graphData.links).id((d) => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-50))
        .force('center', d3.forceCenter(0, 0));

    // 绘制连线
    const link = svg
        .append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(graphData.links)
        .join('line')
        .attr('stroke', '#aaa')
        .attr('stroke-width', (d) => Math.sqrt(d.value));

    // 绘制节点
    const node = svg
        .append('g')
        .attr('class', 'nodes')
        .selectAll('circle')
        .data(graphData.nodes)
        .join('circle')
        .attr('r', 8)
        .attr('fill', '#69b3a2')
        .attr('stroke-width', '1') // 设置较细的边框
        .style('cursor', 'pointer') // 鼠标悬停时变成小手
        .on('click', function (event, d) {
            // 获取节点的最后部分名称
            const nodeName = d.name.split('/').pop();
            // 检查节点是否在 selectedNodeIds 中
            if (selectedNodeIds.value.includes(nodeName)) {
                store.commit('REMOVE_SELECTED_NODE', nodeName);
                d3.select(this).attr('fill', '#69b3a2'); // 恢复原颜色
            } else {
                store.commit('ADD_SELECTED_NODE', nodeName);
                d3.select(this).attr('fill', '#ff6347'); // 更改颜色标识为选中
            }
        })
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended)
        );
    // 节点标签
    svg
        .append('g')
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

        svg.selectAll('.labels text')
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

async function loadAndRenderAllGraphs() {
    for (let i = 0; i < dimensions.length; i++) {
        const data = await d3.json(`http://localhost:5000/subgraph/${i}`);
        renderGraph(graphContainers.value[i], data);
    }
}

onMounted(async () => {
    await nextTick(); // 等待 DOM 渲染
    loadAndRenderAllGraphs();
});
</script>

<style scoped>
.force-graph-container {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.graph-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    width: 100%;
}

.graph-item {
    position: relative;
    width: 100%;
    height: 440px;
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 10px;
    background: #f9f9f9;
    overflow: hidden;
}

.clear-button {
    padding: 5px 10px;
    border: none;
    background-color: #ff6347;
    color: #fff;
    cursor: pointer;
    border-radius: 4px;
    font-size: 14px;
    position: absolute;
    z-index: 999;
    right: 10px;
    top: 10px;
}

.clear-button:hover {
    background-color: #ff4500;
}
</style>