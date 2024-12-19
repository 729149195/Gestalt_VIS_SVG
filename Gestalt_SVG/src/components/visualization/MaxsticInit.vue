<template>
    <div ref="inti_chartContainer" class="init_chart_container"></div>
</template>

<script setup>
import { onMounted, ref, computed, watch } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';

const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);

const eleURL = "http://localhost:5000/init_json";
const inti_chartContainer = ref(null);
const margin = { top: 20, right: 20, bottom: 80, left: 100 };
const width = 1100 + margin.left + margin.right;
const height = 350 + margin.top + margin.bottom;

// 将这些变量定义在 setup 中，使得它们在 watch 中也可以访问
let svg;
let xScale;

onMounted(async () => {
    if (!inti_chartContainer.value) return;

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
            d3.selectAll('.highlight-rect-init').remove();

            // 添加红框矩形
            newVal.forEach(id => {
                const xPos = xScale(id); // 使用xScale计算x坐标
                if (xPos !== undefined) {
                    svg.append('rect')
                        .attr('class', 'highlight-rect-init')
                        .attr('x', xPos)
                        .attr('y', 0) // 红框从顶部开始
                        .attr('width', xScale.bandwidth())
                        .attr('height', height - margin.top - margin.bottom)
                        .style('fill', 'none')
                        .style('stroke', 'red')
                        .style('stroke-width', '0.5px');
                }
            });
        });

    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
});


const processData = (rawData) => {
    let processedData = [];
    
    rawData.forEach((node) => {
        // 修改第9个元素的值（对应 groupname 中的 'layer'）为 0
        node.features[9] = 0;

        // 处理数据，将每个 feature 及其索引保存
        node.features.forEach((probability, groupIndex) => {
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

    // 创建唯一的 tooltip 类名
    const uniqueTooltipClass = `init_tooltip_${Math.random().toString(36).substr(2, 9)}`;

    // 设置 SVG 容器
    svg = d3.select(inti_chartContainer.value)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // 获取所有唯一的 id 后缀
    const ids = [...new Set(data.map(d => d.node.split('/').pop()))];
    const groups = d3.range(0, 20);
    const groupname = ['tag', 'opacity',
        'fill_h', 'fill_s', 'fill_l',
        'stroke_h', 'stroke_s', 'stroke_l', 'stroke_width',
        'layer', 'bbox_left', 'bbox_right', 'bbox_top',
        'bbox_bottom', 'bbox_center_x', 'bbox_center_y',
        'bbox_width', 'bbox_height', 'bbox_fill_area', 'bbox_stroke_area'];

    // 创建比例尺
    xScale = d3.scaleBand().domain(ids).range([0, width - margin.left - margin.right]).padding(0.05);
    const yScale = d3.scaleBand().domain(groups).range([height - margin.top - margin.bottom, 0]).padding(0.05);
    const yScalename = d3.scaleBand().domain(groupname).range([height - margin.top - margin.bottom, 0]).padding(0.05);

    // 颜色比例尺
    const colorScale = d3.scaleSequential(d3.interpolateInferno)
        .domain([d3.max(data, d => d.probability), 0]);

    // 创建 init_tooltip 元素，使用唯一的类名
    const init_tooltip = d3.select(inti_chartContainer.value)
        .append('div')
        .attr('class', uniqueTooltipClass)
        .style('position', 'absolute')
        .style('background', '#fff')
        .style('padding', '5px')
        .style('border', '1px solid #ccc')
        .style('border-radius', '5px')
        .style('pointer-events', 'none')
        .style('visibility', 'hidden');

    // 绘制方块
    svg.selectAll('.block')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', d => xScale(d.node.split('/').pop()))
        .attr('y', d => yScale(d.group))
        .attr('width', xScale.bandwidth())
        .attr('height', yScale.bandwidth())
        .style('fill', d => colorScale(d.probability))
        .on('mouseover', function(event, d) {
            init_tooltip.style('visibility', 'visible')
                .text(`Probability: ${d.probability}`);
        })
        .on('mousemove', function(event) {
            // 获取鼠标的位置，并设置 init_tooltip 在鼠标的上方显示
            const [mouseX, mouseY] = d3.pointer(event);

            init_tooltip.style('top', `${mouseY - 20}px`) 
                   .style('left', `${mouseX - 20}px`);
        })
        .on('mouseout', function() {
            init_tooltip.style('visibility', 'hidden');
        });

    // 添加坐标轴
    const xAxis = d3.axisBottom(xScale).tickSizeOuter(0);
    const yAxis = d3.axisLeft(yScalename).tickSizeOuter(0);

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
.init_chart_container {
    max-width: 100%;
    height: auto;
    position: relative;
}
</style>
