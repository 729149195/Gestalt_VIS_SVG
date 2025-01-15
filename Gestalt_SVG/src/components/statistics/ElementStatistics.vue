<template>
    <div class="statistics-container">
        <span style="color: #666">EleProportions</span>
        <div ref="chartContainer" class="chart-container"></div>
    </div>
</template>

<script setup>
import { onMounted, ref, computed } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
const store = useStore();

const eleURL = "http://127.0.0.1:5000/ele_num_data"
const chartContainer = ref(null);

onMounted(async () => {
    if (!chartContainer.value) return;
    try {
        const response = await fetch(eleURL);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        store.commit('GET_ELE_NUM_DATA', data);
        render(data)
    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
});

const render = (data) => {
    if (!chartContainer.value) return;
    
    const container = chartContainer.value;
    const width = container.clientWidth;
    const height = container.clientHeight;
    const marginTop = height * 0.08;
    const marginRight = width * 0.02;
    const marginBottom = height * 0.25;
    const marginLeft = width * 0.15;

    const x = d3.scaleBand()
        .domain(data.map(d => d.tag))
        .range([marginLeft, width - marginRight])
        .padding(0.1);

    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.num)]).nice()
        .range([height - marginBottom, marginTop]);

    const svg = d3.select(chartContainer.value)
        .append('svg')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('width', width)
        .attr('height', height)
        .attr('style', 'max-width: 100%; height: auto;');

    const zoom = (svg) => {
        const extent = [[marginLeft, marginTop], [width - marginRight, height - marginBottom]];
        svg.call(d3.zoom()
            .scaleExtent([1, 8])
            .translateExtent(extent)
            .extent(extent)
            .on('zoom', (event) => {
                x.range([marginLeft, width - marginRight].map(d => event.transform.applyX(d)));
                svg.selectAll('.bars')
                    .attr('d', d => roundedRectPath(d, x, y)); // 更新路径
                svg.selectAll('.bar-text')
                    .attr('x', d => x(d.tag) + x.bandwidth() / 2); // 更新文本位置
                svg.selectAll('.x-axis').call(d3.axisBottom(x));
            }));
    };

    svg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${height - marginBottom})`)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .style("text-anchor", "end")
        .style("pointer-events", "none")
        .style("font-size", "12px")
        .attr("dx", "-.8em")
        .attr("dy", ".15em")
        .attr("transform", "rotate(-45)");

    const yAxis = svg.append('g')
        .attr('class', 'y-axis')
        .style("pointer-events", "none")
        .attr('transform', `translate(${marginLeft},0)`)
        .call(d3.axisLeft(y));

    // 添加 y 轴上的灰色横线
    yAxis.selectAll('line')
        .attr('x2', width - marginLeft - marginRight)
        .attr('stroke', '#ddd');

    svg.append('g')
        .selectAll('path')
        .data(data)
        .join('path')
        .attr('class', 'bars')
        .attr('fill', d => d.visible ? 'steelblue' : '#999')
        .attr('d', d => roundedRectPath(d, x, y));

    // 添加 x 轴图例
    svg.append("text")
        .attr("transform", `translate(${width / 2},${height - marginBottom / 10})`)
        .style("text-anchor", "middle")
        .style("font-size", "14px")
        .text("Element Tag");

    // 添加 y 轴图例
    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", marginLeft / 3)
        .attr("x", 0 - (height - marginBottom) / 2)
        .style("text-anchor", "middle")
        .style("font-size", "14px")
        .text("Element Number");

    zoom(svg);
};

const roundedRectPath = (d, x, y) => {
    const x0 = x(d.tag);
    const y0 = y(d.num);
    const x1 = x0 + x.bandwidth();
    const y1 = y(0);
    const r = Math.min(x.bandwidth(), y(0) - y(d.num)) / 8; // Radius for the rounded corners

    return `M${x0},${y0 + r}
            Q${x0},${y0} ${x0 + r},${y0}
            L${x1 - r},${y0}
            Q${x1},${y0} ${x1},${y0 + r}
            L${x1},${y1}
            L${x0},${y1}
            Z`;
};
</script>

<style scoped>
.statistics-container {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.chart-container {
    flex: 1;
    width: 100%;
    min-height: 0;
}
</style>
