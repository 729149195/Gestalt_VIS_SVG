<template>
    <div class="statistics-container">
        <span style="color: #666">Fill color Statiction</span>
        <div ref="chartContainer" class="chart-container"></div>
    </div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
const store = useStore();

// 更新了数据接口地址
const eleURL = "http://127.0.0.1:5000/fill_num";
const chartContainer = ref(null);

onMounted(async () => {
    if (!chartContainer.value) return;
    try {
        const response = await fetch(eleURL);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const rawData = await response.json();
        // 将接收到的数据转换为适合绘图的格式
        const data = Object.keys(rawData).map(key => ({
            tag: key, // 颜色值作为标签
            num: rawData[key], // 对应的数量
            visible: true // 可见性标志，根据需要调整
        }));
        store.commit('GET_ELE_NUM_DATA', data);
        render(data);
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

    // 添加横轴
    svg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${height - marginBottom})`)
        .call(d3.axisBottom(x))
        .selectAll("text").remove(); // 移除标签文本，后续添加颜色圆点

    // 添加颜色圆点于x轴
    data.forEach(d => {
        svg.select('.x-axis').append('circle')
            .attr('cx', x(d.tag) + x.bandwidth() / 2)
            .attr('cy', 15) // 轴线下方适当位置
            .attr('r', 5)
            .attr('fill', d.tag)
            .attr('stroke', '#999');
    });

    // 添加纵轴及横线
    const yAxis = svg.append('g')
        .attr('class', 'y-axis')
        .attr('transform', `translate(${marginLeft},0)`)
        .call(d3.axisLeft(y));

    yAxis.selectAll('line')
        .attr('x2', width - marginLeft - marginRight)
        .attr('stroke', '#ddd');

    // 绘制带圆角的条形图
    svg.selectAll('.bar')
        .data(data)
        .enter()
        .append('path')
        .attr('fill', d => d.tag) // 使用数据中的颜色值作为填充色
        .attr('stroke', '#666')
        .attr('d', d => roundedRectPath(d, x, y));

    // 添加 x 轴图例
    svg.append("text")
        .attr("transform", `translate(${width / 2},${height - marginBottom / 10})`)
        .style("text-anchor", "middle")
        .style("font-size", "14px")
        .text("Fill Color");

    // 添加 y 轴图例
    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", marginLeft / 3)
        .attr("x", 0 - (height - marginBottom) / 2)
        .style("text-anchor", "middle")
        .style("font-size", "14px")
        .text("Number");

}
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
