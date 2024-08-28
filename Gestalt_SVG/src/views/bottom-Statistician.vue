<template>
    <span style="color: #666">bottom-position</span>
    <div ref="chartContainer" style="width: 410px; height: 250px;"></div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
const store = useStore();

const chartContainer = ref(null);
const eleURL = "http://localhost:5000/bottom_position";

onMounted(async () => {
    if (!chartContainer.value) return;

    try {
        const response = await fetch(eleURL);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        const dataset = Object.keys(data).map((range) => ({
            range,
            tags: data[range].tags, // 确保包含tags
            totals: Object.entries(data[range].total).map(([key, value]) => ({ key, value }))
        }));
        renderChart(dataset);
    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
});

const renderChart = (dataset) => {
    const svgWidth = 650, svgHeight = 350;
    const margin = { top: 20, right: 25, bottom: 50, left: 85 };
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;

    const xScale = d3.scaleBand()
        .range([0, width])
        .padding(0.2)
        .domain(dataset.map(d => d.range));

    const yScale = d3.scaleLinear()
        .range([height, 0])
        .domain([0, d3.max(dataset, d => d3.sum(d.totals, t => t.value))]).nice();

    const svg = d3.select(chartContainer.value).append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${svgWidth} ${svgHeight}`)
        .style("position", "relative");

    const tooltip = d3.select(chartContainer.value)
        .append("div")
        .attr("class", "tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden")
        .style("background", "white")
        .style("border", "1px solid #ddd")
        .style("padding", "10px")
        .style("border-radius", "5px")
        .style("pointer-events", "none")
        .style("box-shadow", "0px 0px 10px rgba(0,0,0,0.1)")
        .style("white-space", "nowrap");

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale));

    const yAxis = g.append('g')
        .attr('class', 'y-axis')
        .call(d3.axisLeft(yScale))
        .call(g => g.selectAll('.tick line')
            .clone()
            .attr('x2', width)
            .attr('stroke', '#ddd')); // 设置颜色

    const zoom = d3.zoom()
        .scaleExtent([1, 3])
        .translateExtent([[0, 0], [width, height]])
        .extent([[0, 0], [width, height]])
        .on('zoom', (event) => {
            const transform = event.transform;
            xScale.range([0, width].map(d => transform.applyX(d)));
            g.select('.x-axis').call(d3.axisBottom(xScale));
            g.selectAll('.range').attr('transform', d => `translate(${xScale(d.range)},0)`);
            g.selectAll('.range rect').attr('width', xScale.bandwidth());
        });

    svg.call(zoom);

    const rangeGroup = g.selectAll('.range')
        .data(dataset)
        .enter().append('g')
        .attr('class', 'range')
        .attr('transform', d => `translate(${xScale(d.range)},0)`)
        .attr("style", "cursor: pointer;")
        .on('mouseover', (event, d) => {
            tooltip.style("visibility", "visible")
                .html(() => {
                    const tagsContent = d.tags.map(tag => `${tag}<br/>`).join("");
                    const content = `<strong>Range:</strong> ${d.range}<br/><strong>Tags:</strong><br/>${tagsContent}`;
                    return content;
                });
            const tooltipHeight = tooltip.node().getBoundingClientRect().height;
            tooltip.style("top", (event.pageY - tooltipHeight - 10) + "px")
                .style("left", (event.pageX + 10) + "px");
        })
        .on('mouseout', () => {
            tooltip.style("visibility", "hidden");
        }).on('click', (event, d) => {
            store.commit('UPDATE_SELECTED_NODES', { nodeIds: d.tags, group: null });
        });

    rangeGroup.each(function (d) {
        let yAccumulator = 0;
        d3.select(this).selectAll('rect')
            .data(d.totals)
            .enter().append('rect')
            .attr('x', 0)
            .attr('y', t => {
                const y = yScale(yAccumulator + t.value);
                yAccumulator += t.value;
                return y;
            })
            .attr('width', xScale.bandwidth())
            .attr('height', t => height - yScale(t.value))
            .attr('fill', t => ({
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
            }[t.key]))
            .attr('rx', 2)  // 设置x轴方向的圆角半径
            .attr('ry', 2); // 设置y轴方向的圆角半径
    });

    svg.append("text")
        .attr("transform", `translate(${width / 2},${height - 5})`)
        .style("text-anchor", "middle")
        .style("font-size", "23px")
        .attr("dx", "12.5em")
        .attr("dy", "3.5em")
        .text("Position  zones");

    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 45)
        .attr("x", 0 - (height / 2))
        .style("text-anchor", "middle")
        .style("font-size", "23px")
        .attr("dx", ".5em")
        .attr("dy", "0em")
        .text("Position Number");
};
</script>

<style scoped>
.tooltip strong {
    color: blue;
    /* 标题颜色 */
}

.tooltip span {
    color: black;
    /* 内容颜色 */
}
</style>
