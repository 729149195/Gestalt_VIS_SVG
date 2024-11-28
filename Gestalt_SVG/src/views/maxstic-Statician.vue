<template>
    <div ref="init_chart_container" class="init_chart_container"></div>
</template>

<script setup>
import { onMounted, ref, computed, watch } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';

const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);

const eleURL = "http://localhost:5000/cluster_features";
const init_chart_container = ref(null);
const margin = { top: 0, right: 10, bottom: 80, left: 40 };
let svg;
let xScale;
let resizeObserver;

onMounted(async () => {
    if (!init_chart_container.value) return;

    try {
        const response = await fetch(eleURL);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const rawData = await response.json();
        const data = processData(rawData);

        // 使用 ResizeObserver 监听宽度变化
        resizeObserver = new ResizeObserver(entries => {
            for (let entry of entries) {
                const newWidth = entry.contentRect.width;
                render(data, newWidth);
            }
        });

        resizeObserver.observe(init_chart_container.value);

        // 初始渲染
        const initialWidth = init_chart_container.value.clientWidth;
        render(data, initialWidth);

        // 监控 selectedNodeIds 变化，更新高亮显示
        watch(selectedNodeIds, (newVal) => {
            updateHighlights(newVal);
        });

    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
});

const updateHighlights = (selectedIds) => {
    // 更新 X 轴文本的颜色和加粗
    d3.selectAll('.x-axis text')
        .style('fill', d => d && selectedIds.includes(d) ? 'red' : 'black')
        .style('font-weight', d => d && selectedIds.includes(d) ? 'bold' : 'normal');

    // 移除旧的高亮矩形
    d3.selectAll('.highlight-rect-cc').remove();

    // 根据 selectedNodeIds 添加新的红框矩形
    selectedIds.forEach(id => {
        const xPos = xScale(id);
        if (xPos !== undefined) {
            svg.append('rect')
                .attr('class', 'highlight-rect-cc')
                .attr('x', xPos)
                .attr('y', 0)
                .attr('width', xScale.bandwidth())
                .attr('height', init_chart_container.value.clientHeight - margin.top - margin.bottom)
                .style('fill', 'none')
                .style('stroke', 'red')
                .style('stroke-width', '1.5px');
        }
    });
};

const processData = (rawData) => {
    let processedData = [];
    rawData.forEach((node) => {
        node.features.forEach((featureValue, featureIndex) => {
            processedData.push({
                node: node.id,
                group: featureIndex,
                featureValue: featureValue,
            });
        });
    });
    return processedData;
};

const render = (data, containerWidth) => {
    const width = containerWidth - margin.left - margin.right;
    const height = 50 + margin.top + margin.bottom;

    // 如果存在旧的 SVG 则移除
    d3.select(init_chart_container.value).selectAll('*').remove();

    // 设置 SVG 容器
    svg = d3.select(init_chart_container.value)
        .append('svg')
        .attr('width', containerWidth)
        .attr('height', height)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const ids = [...new Set(data.map(d => d.node.split('/').pop()))];
    const groups = d3.range(0, 4);
    const groupname = groups.map(String);

    // 更新 xScale
    xScale = d3.scaleBand().domain(ids).range([0, width]).padding(0.05);
    const yScale = d3.scaleBand().domain(groups).range([height - margin.top - margin.bottom, 0]).padding(0.05);

    const colorScale = d3.scaleSequential(d3.interpolateInferno)
        .domain([d3.max(data, d => d.featureValue), d3.min(data, d => d.featureValue)]);

    // 创建悬停提示框
    const tooltip = d3.select(init_chart_container.value)
        .append('div')
        .attr('class', 'tooltip')
        .style('position', 'absolute')
        .style('background', '#fff')
        .style('padding', '5px')
        .style('border', '1px solid #ccc')
        .style('border-radius', '5px')
        .style('pointer-events', 'none')
        .style('visibility', 'hidden');

    svg.selectAll('.block')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', d => xScale(d.node.split('/').pop()))
        .attr('y', d => yScale(d.group))
        .attr('width', xScale.bandwidth())
        .attr('height', yScale.bandwidth())
        .style('fill', d => colorScale(d.featureValue))
        .on('mouseover', function (event, d) {
            tooltip.style('visibility', 'visible')
                .text(`Feature Value: ${d.featureValue}`);
        })
        .on('mousemove', function (event) {
            const [mouseX, mouseY] = d3.pointer(event);
            tooltip.style('top', `${mouseY - 20}px`)
                .style('left', `${mouseX - 20}px`);
        })
        .on('mouseout', function () {
            tooltip.style('visibility', 'hidden');
        });

    const xAxis = d3.axisBottom(xScale).tickSizeOuter(0);
    const yAxis = d3.axisLeft(yScale).tickFormat((d) => groupname[d]).tickSizeOuter(0);

    svg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
        .call(xAxis)
        .selectAll('text')
        .style('text-anchor', 'end')
        .attr('dx', '-1em')
        .attr('dy', '-0.5em')
        .attr('transform', 'rotate(-90)')
        .style('fill', 'black')
        .style('font-size', '12px')
        .style('cursor', 'pointer')
        .on('click', function (event, d) {
            if (selectedNodeIds.value.includes(d)) {
                store.commit('REMOVE_SELECTED_NODE', d);
            } else {
                store.commit('ADD_SELECTED_NODE', d);
            }
        });

    svg.append('g').call(yAxis)
        .selectAll('text')
        .style('fill', 'black')
        .style('font-size', '12px');

    svg.selectAll('.domain')
        .style('stroke', 'black')
        .style('stroke-width', '1px');

    svg.selectAll('.tick line')
        .style('stroke', 'black')
        .style('stroke-width', '1px');
};
</script>

<style scoped>
.init_chart_container {
    max-width: 100%;
    height: auto;
    position: relative;
}
.tooltip {
    font-size: 12px;
    color: #333;
}
</style>
