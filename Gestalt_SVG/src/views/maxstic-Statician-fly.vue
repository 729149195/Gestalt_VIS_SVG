<template>
    <div ref="chartContainer" class="chart-container"></div>
</template>

<script setup>
import { onMounted, ref, computed, watch } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';

const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);

const eleURL = "http://localhost:5000/fourier_file_path";
const chartContainer = ref(null);
const margin = { top: 20, right: 20, bottom: 100, left: 30 };
const width = 1180 + margin.left + margin.right;
const height = 520 + margin.top + margin.bottom;

// 将这些变量定义在 setup 中，使得它们在 watch 中也可以访问
let svg;
let xScale;

onMounted(async () => {
    if (!chartContainer.value) return;

    try {
        const response = await fetch(eleURL);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const rawData = await response.json();
        const data = processData(rawData);
        render(data);

        // 监控 selectedNodeIds 变化，更新高亮显示
        watch(selectedNodeIds, (newVal) => {
            // 高亮 X 轴的文本并加粗
            d3.selectAll('.x-axis text')
                .style('fill', d => d && newVal.includes(d) ? 'red' : 'black')
                .style('font-weight', d => d && newVal.includes(d) ? 'bold' : 'normal');

            // 移除已有的红框
            d3.selectAll('.highlight-rect').remove();

            // 添加红框矩形
            newVal.forEach(id => {
                const xPos = xScale(id); // 使用xScale计算x坐标
                if (xPos !== undefined) {
                    svg.append('rect')
                        .attr('class', 'highlight-rect')
                        .attr('x', xPos)
                        .attr('y', 0) // 红框从顶部开始
                        .attr('width', xScale.bandwidth())
                        .attr('height', height - margin.top - margin.bottom)
                        .style('fill', 'none')
                        .style('stroke', 'red')
                        .style('stroke-width', '2px');
                }
            });
        });

    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
});


const processData = (rawData) => {
    let processedData = [];
    rawData.forEach((node, nodeIndex) => {
        node.fourier_features.forEach((probability, groupIndex) => {
            processedData.push({
                node: node.id,
                group: groupIndex,
                probability: probability,
            });
        });
    });
    return processedData;
};

const render = (data) => {
    const textYOffset = 10;

    // 设置 SVG 容器
    svg = d3.select(chartContainer.value)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // 获取所有唯一的 id 后缀
    const ids = [...new Set(data.map(d => d.node.split('/').pop()))];
    const groups = d3.range(1, 16);

    // 创建比例尺
    xScale = d3.scaleBand().domain(ids).range([0, width - margin.left - margin.right]).padding(0.05);
    const yScale = d3.scaleBand().domain(groups).range([height - margin.top - margin.bottom, 0]).padding(0.05);

    // 颜色比例尺
    const colorScale = d3.scaleSequential(d3.interpolateInferno)
        .domain([d3.max(data, d => d.probability), 0]);

    // 绘制方块
    svg.selectAll('.block')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', d => xScale(d.node.split('/').pop()))
        .attr('y', d => yScale(d.group))
        .attr('width', xScale.bandwidth())
        .attr('height', yScale.bandwidth())
        .style('fill', d => colorScale(d.probability));

    // 添加坐标轴
    const xAxis = d3.axisBottom(xScale).tickSizeOuter(0);
    const yAxis = d3.axisLeft(yScale).tickSizeOuter(0);

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
        .style('font-size', '12px');

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


    svg.selectAll('.x-axis text')
        .style('cursor', 'pointer')
        .on('click', function (event, d) {
            // 检查当前点击的 ID 是否已经在 selectedNodeIds 中
            if (selectedNodeIds.value.includes(d)) {
                // 如果存在，移除该 ID
                store.commit('REMOVE_SELECTED_NODE', d);
            } else {
                // 如果不存在，添加该 ID
                store.commit('ADD_SELECTED_NODE', d);
            }
            // console.log(selectedNodeIds.value);
        });



    const legendHeight = 480;
    const legendWidth = 10;
    const numSwatches = 50;
    const legendDomain = colorScale.domain();
    const legendScale = d3.scaleLinear()
        .domain([0, numSwatches - 1])
        .range([legendDomain[1], legendDomain[0]]);
    const legendData = Array.from(Array(numSwatches).keys());

    const legend = svg.append('g')
        .attr('transform', `translate(${width - 40}, 10)`);

    legend.selectAll('rect')
        .data(legendData)
        .enter()
        .append('rect')
        .attr('y', (d, i) => legendHeight - (i + 1) * (legendHeight / numSwatches))
        .attr('x', 0)
        .attr('height', legendHeight / numSwatches)
        .attr('width', legendWidth)
        .attr('fill', d => colorScale(legendScale(d)));

    legend.append('text')
        .attr('transform', `translate(${legendWidth + textYOffset}, 0) rotate(90)`)
        .style('font-size', '10px')
        .style('fill', 'black')
        .text(d3.format(".2f")(legendDomain[0]));

    legend.append('text')
        .attr('transform', `translate(${legendWidth + textYOffset}, ${legendHeight}) rotate(90)`)
        .style('font-size', '10px')
        .style('fill', 'black')
        .text(d3.format(".2f")(legendDomain[1]));

    legend.append('text')
        .attr('transform', `translate(${legendWidth + 20}, ${legendHeight / 2}) rotate(90)`)
        .style('font-size', '12px')
        .style('text-anchor', 'middle')
        .style('fill', 'black')
        .text('Probability');
};
</script>

<style scoped>
.chart-container {
    max-width: 100%;
    height: auto;
}
</style>
