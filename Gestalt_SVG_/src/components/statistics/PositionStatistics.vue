<template>
    <div class="statistics-container">
        <div class="title-container">
            <span class="title">{{ title }}</span>
        </div>
        <div ref="chartContainer" class="chart-container"></div>
    </div>
</template>

<script setup>
import { onMounted, ref, watch, computed } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';

const props = defineProps({
    position: {
        type: String,
        required: true,
        validator: (value) => ['top', 'bottom', 'left', 'right', 'width', 'height'].includes(value)
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
    if (!selectedNodes || !tags || tags.length === 0) return 0;
    
    // 标准化标签格式
    const normalizedTags = tags.map(tag => tag.split('/').pop());
    const normalizedSelectedNodes = selectedNodes.map(node => node.split('/').pop());
    
    // 计算交集
    const intersection = normalizedTags.filter(tag => 
        normalizedSelectedNodes.includes(tag)
    );
    
    return intersection.length / tags.length;
};

// 计算API端点URL
const eleURL = computed(() => {
    // 根据不同的position属性返回不同的URL
    const positionMap = {
        'top': 'top_position',
        'bottom': 'bottom_position',
        'left': 'left_position',
        'right': 'right_position',
        'width': 'width_position',
        'height': 'height_position'
    };
    return `http://127.0.0.1:8000/${positionMap[props.position]}`;
});

onMounted(async () => {
    if (!chartContainer.value) return;

    try {
        const response = await fetch(eleURL.value);
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
        bottom: svgHeight * 0.3,
        left: svgWidth * 0.2
    };
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;

    const xScale = d3.scaleBand()
        .range([0, width])
        .padding(0.2)
        .domain(dataset.map(d => d.range));

    // 设置条形的最大宽度
    const maxBarWidth = 50; // 设置条形的最大宽度为50px
    const actualBandwidth = Math.min(xScale.bandwidth(), maxBarWidth);

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
            
            // 更新条形宽度，确保不超过最大宽度
            const newBandwidth = Math.min(xScale.bandwidth(), maxBarWidth);
            g.selectAll('.range rect').each(function() {
                const rect = d3.select(this);
                const barX = (xScale.bandwidth() - newBandwidth) / 2;
                rect.attr('x', barX)
                    .attr('width', newBandwidth);
            });
        });

    svg.value.call(zoom);

    // 创建一个函数来计算矩形的高度和位置
    const calculateRectDimensions = (value, yAccumulator) => {
        const y = yScale(yAccumulator + value);
        const rectHeight = height - yScale(value);
        return { y, height: rectHeight };
    };

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
            // 标准化标签格式
            const normalizedTags = d.tags.map(tag => 
                tag.startsWith('svg/') ? tag : `svg/${tag}`
            );
            store.commit('UPDATE_SELECTED_NODES', { 
                nodeIds: normalizedTags,
                group: null 
            });
        });

    // 渲染基础条形图
    rangeGroup.each(function(d) {
        let yAccumulator = 0;
        const group = d3.select(this);
        
        d.totals.forEach(t => {
            const { y, height: rectHeight } = calculateRectDimensions(t.value, yAccumulator);
            
            // 计算条形的中心位置，使其居中显示
            const barX = (xScale.bandwidth() - actualBandwidth) / 2;
            
            // 背景灰色条形
            group.append('rect')
                .attr('class', 'background-bar')
                .attr('x', barX)
                .attr('y', y)
                .attr('width', actualBandwidth)
                .attr('height', rectHeight)
                .attr('fill', '#E0E0E0') // 使用更浅的灰色
            
            // 前景蓝色条形
            group.append('rect')
                .attr('class', 'highlight-bar')
                .attr('x', barX)
                .attr('y', y + rectHeight) // 初始位置在底部
                .attr('width', actualBandwidth)
                .attr('height', 0) // 初始高度为0
                .attr('fill', '#905F29')
                .style('opacity', 0.7);
                
            yAccumulator += t.value;
        });
    });

    // 更新选中状态的函数
    const updateSelection = (selectedNodes) => {
        if (!selectedNodes) return;

        rangeGroup.each(function(d) {
            const ratio = calculateSelectedRatio(d.tags, selectedNodes);
            let yAccumulator = 0;
            
            // 获取该组的所有条形
            const bars = d3.select(this).selectAll('rect.highlight-bar');
            
            d.totals.forEach((t, i) => {
                const { y, height: rectHeight } = calculateRectDimensions(t.value, yAccumulator);
                
                // 更新高亮条形的高度和位置
                d3.select(this).select(`.highlight-bar:nth-of-type(${i * 2 + 2})`)
                    .transition()
                    .duration(300)
                    .attr('height', rectHeight * ratio)
                    .attr('y', y + rectHeight * (1 - ratio));
                
                yAccumulator += t.value;
            });
        });
    };

    // 监听选中节点变化
    watch(
        () => store.state.selectedNodes.nodeIds, // 直接监听 nodeIds 数组
        (newSelectedNodes) => {
            if (!svg.value) return;
            updateSelection(newSelectedNodes || []); // 确保传入空数组而不是 undefined
        },
        { deep: true, immediate: true } // 添加 immediate: true 确保初始化时也执行
    );

    svg.value.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", margin.left / 2.5)
        .attr("x", 0 - (height + margin.top) / 1.8)
        .style("text-anchor", "middle")
        .style("font-size", "12px")
        .text("Count");
};
</script>

<style scoped>
.statistics-container {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.title-container {
    display: flex;
    justify-content: center;
    width: 100%;
    padding-top: 8px;
}

.title {
    font-size: 14px;
    color: #000;
    margin: 0;
    padding: 0;
    z-index: 10;
    letter-spacing: -0.01em;
    opacity: 0.8;
}

.chart-container {
    flex: 1;
    width: 100%;
    min-height: 0;
}

.tooltip strong {
    color: #905F29;
}

.tooltip span {
    color: black;
}
</style> 