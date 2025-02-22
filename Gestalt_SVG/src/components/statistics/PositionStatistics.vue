<template>
    <div class="statistics-container">
        <span class="title">{{ title }}</span>
        <div ref="chartContainer" class="chart-container"></div>
    </div>
</template>

<script setup>
import { onMounted, ref, watch } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';

const props = defineProps({
    position: {
        type: String,
        required: true,
        validator: (value) => ['top', 'bottom', 'left', 'right'].includes(value)
    },
    title: {
        type: String,
        default: 'Position Statistics'
    }
});

const store = useStore();
const chartContainer = ref(null);
const svg = ref(null);

// 计算选中比例的函数
const calculateSelectedRatio = (tags, selectedNodes) => {
    if (!selectedNodes || !selectedNodes.length) return 0;
    const intersection = tags.filter(tag => selectedNodes.includes(tag));
    return intersection.length / tags.length;
};

// 监听 selectedNodes 变化
watch(
    () => store.state.selectedNodes,
    (newSelectedNodes) => {
        if (!svg.value) return;
        
        // 更新所有柱子的颜色
        svg.value.selectAll('.range')
            .each(function(d) {
                const ratio = calculateSelectedRatio(d.tags, newSelectedNodes);
                if (ratio > 0) {
                    // 创建渐变
                    const gradientId = `gradient-${d.range.replace(/\./g, '-')}`;
                    const gradient = svg.value.select('defs')
                        .append('linearGradient')
                        .attr('id', gradientId)
                        .attr('x1', '0%')
                        .attr('x2', '0%')
                        .attr('y1', '0%')
                        .attr('y2', '100%');

                    gradient.append('stop')
                        .attr('offset', `${ratio * 100}%`)
                        .attr('stop-color', '#1E90FF');

                    gradient.append('stop')
                        .attr('offset', `${ratio * 100}%`)
                        .attr('stop-color', '#808080');

                    d3.select(this).selectAll('rect')
                        .transition()
                        .duration(300)
                        .attr('fill', `url(#${gradientId})`);
                } else {
                    d3.select(this).selectAll('rect')
                        .transition()
                        .duration(300)
                        .attr('fill', '#808080');
                }
            });
    },
    { deep: true }
);

const eleURL = `http://127.0.0.1:5000/${props.position}_position`;

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
            tags: data[range].tags,
            totals: Object.entries(data[range].total).map(([key, value]) => ({ key, value }))
        }));
        renderChart(dataset);
    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
});

const renderChart = (dataset) => {
    dataset.sort((a, b) => {
        const aStart = parseInt(a.range.split('-')[0]);
        const bStart = parseInt(b.range.split('-')[0]);
        return aStart - bStart;
    });

    const container = chartContainer.value;
    const svgWidth = container.clientWidth;
    const svgHeight = container.clientHeight;
    const margin = { 
        top: svgHeight * 0.05, 
        right: svgWidth * 0.05, 
        bottom: svgHeight * 0.45,
        left: svgWidth * 0.15 
    };
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;

    const xScale = d3.scaleBand()
        .range([0, width])
        .padding(0.2)
        .domain(dataset.map(d => d.range));

    const yScale = d3.scaleLinear()
        .range([height, 0])
        .domain([0, d3.max(dataset, d => d3.sum(d.totals, t => t.value))]).nice();

    svg.value = d3.select(chartContainer.value).append('svg')
        .attr('viewBox', `0 0 ${svgWidth} ${svgHeight}`)
        .attr('width', svgWidth)
        .attr('height', svgHeight)
        .attr('style', 'max-width: 100%; height: auto;')
        .style("position", "relative");

    // 添加 defs 元素用于存放渐变定义
    svg.value.append('defs');

    const tooltip = d3.select(chartContainer.value)
        .append("div")
        .attr("class", "tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden")
        .style("background", "white")
        .style("border", "1px solid #ddd")
        .style("padding", "10px")
        .style("border-radius", "2px")
        .style("pointer-events", "none")
        .style("box-shadow", "0px 0px 10px rgba(0,0,0,0.1)")
        .style("white-space", "nowrap");

    const g = svg.value.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale))
        .selectAll("text")
        .style("text-anchor", "end")
        .style("pointer-events", "none")
        .style("font-size", "10px")
        .attr("dx", "-.8em")
        .attr("dy", ".15em")
        .attr("transform", "rotate(-45)");

    const yAxis = g.append('g')
        .attr('class', 'y-axis')
        .call(d3.axisLeft(yScale)
            .ticks(4.5)
            .tickFormat(d => {
                if (d >= 1000) {
                    return d3.format('.1k')(d);
                }
                return d;
            }))
        .call(g => g.selectAll('.tick line')
            .clone()
            .attr('x2', width)
            .attr('stroke', '#ddd'))
        .call(g => g.selectAll('.tick text')
            .style('font-size', '12px'));

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

    svg.value.call(zoom);

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
        })
        .on('click', (event, d) => {
            store.commit('UPDATE_SELECTED_NODES', { nodeIds: d.tags, group: null });
        });

    rangeGroup.each(function (d) {
        let yAccumulator = 0;
        const selectedRatio = calculateSelectedRatio(d.tags, store.state.selectedNodes);
        
        // 如果有选中的节点，创建渐变
        if (selectedRatio > 0) {
            const gradientId = `gradient-${d.range.replace(/\./g, '-')}`;
            const gradient = svg.value.select('defs')
                .append('linearGradient')
                .attr('id', gradientId)
                .attr('x1', '0%')
                .attr('x2', '0%')
                .attr('y1', '0%')
                .attr('y2', '100%');

            gradient.append('stop')
                .attr('offset', `${selectedRatio * 100}%`)
                .attr('stop-color', '#1E90FF');

            gradient.append('stop')
                .attr('offset', `${selectedRatio * 100}%`)
                .attr('stop-color', '#808080');

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
                .attr('fill', `url(#${gradientId})`)
                .attr('rx', 2)
                .attr('ry', 2);
        } else {
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
                .attr('fill', '#808080')
                .attr('rx', 2)
                .attr('ry', 2);
        }
    });

    svg.value.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", margin.left / 3)
        .attr("x", 0 - (height + margin.top + margin.bottom) / 2)
        .style("text-anchor", "middle")
        .style("font-size", "12px")
        .text("Position Number");
};
</script>

<style scoped>
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

.tooltip strong {
    color: blue;
}

.tooltip span {
    color: black;
}
</style> 