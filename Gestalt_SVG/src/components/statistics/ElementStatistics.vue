<template>
    <div class="statistics-container">
        <span class="title">Type</span>
        <div ref="chartContainer" class="chart-container"></div>
        <div v-if="!hasData" class="no-data-message">No Elements</div>
    </div>
</template>

<script setup>
import { onMounted, ref, computed, watch, nextTick } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
const store = useStore();

const eleURL = "http://127.0.0.1:5000/ele_num_data"
const chartContainer = ref(null);
const hasData = ref(false);
const rawJsonData = ref(null);
const isInitialized = ref(false);

onMounted(async () => {
    await nextTick();
    isInitialized.value = true;
    await fetchData();
});

const fetchData = async () => {
    if (!isInitialized.value) {
        return;
    }
    
    if (!chartContainer.value) {
        return;
    }
    
    try {
        const response = await fetch(eleURL);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        rawJsonData.value = data;

        const visibleData = data.filter(d => d.visible === true);
    
        if (visibleData.length === 0) {
            hasData.value = false;
            return;
        }
        
        hasData.value = true;
        store.commit('GET_ELE_NUM_DATA', data);
        
        setTimeout(() => {
            render(data);
        }, 0);
    } catch (error) {
        console.error('获取元素类型数据时出错:', error);
        hasData.value = false;
    }
};

// 监听组件容器变化
watch([() => chartContainer.value, () => isInitialized.value], ([newContainer, newInitialized]) => {
    if (newContainer && newInitialized && rawJsonData.value) {
        
        // 检查是否有可见元素
        const visibleData = rawJsonData.value.filter(d => d.visible === true);
        if (visibleData.length === 0) {
            hasData.value = false;
            return;
        }
        
        hasData.value = true;
        // 清除之前的图表内容
        if (chartContainer.value) {
            chartContainer.value.innerHTML = '';
        }
        render(rawJsonData.value);
    }
});

const render = (data) => {
    if (!chartContainer.value) return;
    
    // 过滤数据，只保留可见元素
    const visibleData = data.filter(d => d.visible === true);
    
    // 如果没有可见元素，显示提示信息
    if (visibleData.length === 0) {
        const container = chartContainer.value;
        container.innerHTML = '<div class="no-data-message">没有可见元素</div>';
        return;
    }
    
    const container = chartContainer.value;
    const width = container.clientWidth;
    const height = container.clientHeight;
    const marginTop = height * 0.08;
    const marginRight = width * 0.02;
    const marginBottom = height * 0.25;
    const marginLeft = width * 0.15;

    const x = d3.scaleBand()
        .domain(visibleData.map(d => d.tag))
        .range([marginLeft, width - marginRight])
        .padding(0.1);

    const y = d3.scaleLinear()
        .domain([0, d3.max(visibleData, d => d.num)]).nice()
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
        .call(d3.axisLeft(y)
            .ticks(5)
            .tickFormat(d => {
                if (d >= 1000) {
                    return d3.format('.1k')(d);
                }
                return d;
            }))
        .call(g => {
            g.selectAll('.tick line')
                .attr('x2', width - marginLeft - marginRight)
                .attr('stroke', '#ddd');
            g.selectAll('.tick text')
                .style('font-size', '12px');
        });

    svg.append('g')
        .selectAll('path')
        .data(visibleData)
        .join('path')
        .attr('class', 'bars')
        .attr('fill', 'steelblue')
        .attr('d', d => roundedRectPath(d, x, y));

    // 添加 x 轴图例
    // svg.append("text")
    //     .attr("transform", `translate(${width / 2},${height - marginBottom / 10})`)
    //     .style("text-anchor", "middle")
    //     .style("font-size", "14px")
    //     .text("Element Tag");

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
    position: relative;
}

.chart-container {
    flex: 1;
    width: 100%;
    min-height: 180px;
    display: block;
}

.no-data-message {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #999;
    font-size: 14px;
    pointer-events: none;
    z-index: 5;
}

.title {
  top: 12px;
  left: 16px;
  font-size: 14px;
  font-weight: bold;
  color: #000;
  margin: 0;
  padding: 0;
  z-index: 10;
  letter-spacing: -0.01em;
  opacity: 0.8;
}

/* 添加条形图样式 */
.bars {
  transition: opacity 0.3s;
}
.bars:hover {
  opacity: 0.8;
}
</style>
