<template>
    <div class="statistics-container">
        <span class="title">Elements Type</span>
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
const svgURL = "http://127.0.0.1:5000/get_svg"
const chartContainer = ref(null);
const hasData = ref(false);
const rawJsonData = ref(null);
const isInitialized = ref(false);
const svgRef = ref(null);
const filteredElementIds = ref([]); // 存储过滤后的SVG元素ID

// 从SVG内容中提取所有元素ID
const extractElementIds = (svgContent) => {
    const parser = new DOMParser();
    const svgDoc = parser.parseFromString(svgContent, "image/svg+xml");
    const allElements = svgDoc.querySelectorAll('[id]');
    
    return Array.from(allElements).map(el => el.id);
};

// 获取过滤后的SVG元素ID
const fetchFilteredSvgIds = async () => {
    try {
        const response = await fetch(svgURL);
        if (!response.ok) {
            console.error('获取SVG内容失败');
            return [];
        }
        
        const svgContent = await response.text();
        return extractElementIds(svgContent);
    } catch (error) {
        console.error('获取过滤后SVG元素ID时出错:', error);
        return [];
    }
};

// 计算选中元素的类型比例
const calculateSelectedRatio = (tag, selectedNodes) => {
    if (!selectedNodes || selectedNodes.length === 0) return 0;
    
    // 从 selectedNodes 中提取元素类型和ID
    const selectedElements = selectedNodes.map(nodeId => {
        // 通常 nodeId 格式为 "svg/type_number" 或 "type_number"
        const parts = nodeId.split('/');
        const lastPart = parts[parts.length - 1];
        // 返回类型和编号
        const [type, id] = lastPart.split('_');
        return { type, id, fullId: nodeId };
    });
    
    // 只保留在过滤后SVG中存在的元素
    const validSelectedElements = selectedElements.filter(el => {
        // 检查元素ID是否在过滤后的SVG中
        // 需要考虑多种可能的ID格式: fullId, type_id, 或者纯id
        return filteredElementIds.value.length === 0 || 
               filteredElementIds.value.includes(el.fullId) || 
               filteredElementIds.value.includes(`${el.type}_${el.id}`) ||
               filteredElementIds.value.includes(el.id);
    });
    
    // 找出该类型的所有有效选中元素
    const selectedOfThisType = validSelectedElements.filter(el => el.type === tag);
    
    // 获取该类型的总数量 (从 rawJsonData 中)
    const typeData = rawJsonData.value.find(d => d.tag === tag);
    if (!typeData) return 0;
    
    const totalOfThisType = typeData.num;
    
    // 如果没有选中该类型，返回 0
    if (selectedOfThisType.length === 0) return 0;
    
    // 计算选中比例 (选中的该类型元素 / 该类型的总元素)
    return selectedOfThisType.length / totalOfThisType;
};

onMounted(async () => {
    await nextTick();
    isInitialized.value = true;
    // 先获取过滤后的SVG元素ID
    filteredElementIds.value = await fetchFilteredSvgIds();
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
        // 获取最新的过滤后SVG元素ID
        filteredElementIds.value = await fetchFilteredSvgIds();
        
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

// 监听选中节点变化
watch(
    () => store.state.selectedNodes.nodeIds,
    async (newSelectedNodes) => {
        if (!svgRef.value || !rawJsonData.value) return;
        
        // 确保每次节点变化时都更新过滤后的SVG元素ID
        if (filteredElementIds.value.length === 0) {
            filteredElementIds.value = await fetchFilteredSvgIds();
        }
        
        updateSelection(newSelectedNodes || []);
    },
    { deep: true, immediate: true }
);

// 更新选中状态的函数
const updateSelection = (selectedNodes) => {
    if (!selectedNodes || !svgRef.value) return;
    
    const visibleData = rawJsonData.value.filter(d => d.visible === true);
    
    visibleData.forEach(d => {
        const ratio = calculateSelectedRatio(d.tag, selectedNodes);
        
        // 更新对应标签的背景条和高亮条
        const barGroup = svgRef.value.select(`.bar-group-${d.tag}`);
        if (!barGroup.empty()) {
            // 更新高亮条的高度和位置
            barGroup.select('.highlight-bar')
                .transition()
                .duration(300)
                .attr('height', rect => rect.height * ratio)
                .attr('y', rect => rect.y + rect.height * (1 - ratio));
        }
    });
};

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
    const marginBottom = height * 0.2;
    const marginLeft = width * 0.15;

    const x = d3.scaleBand()
        .domain(visibleData.map(d => d.tag))
        .range([marginLeft, width - marginRight])
        .padding(0.1);

    // 设置条形的最大宽度
    const maxBarWidth = 50; // 设置条形的最大宽度为50px
    const actualBandwidth = Math.min(x.bandwidth(), maxBarWidth);

    const y = d3.scaleLinear()
        .domain([0, d3.max(visibleData, d => d.num)]).nice()
        .range([height - marginBottom, marginTop]);

    const svg = d3.select(chartContainer.value)
        .append('svg')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('width', width)
        .attr('height', height)
        .attr('style', 'max-width: 100%; height: auto;');
    
    svgRef.value = svg;

    const zoom = (svg) => {
        const extent = [[marginLeft, marginTop], [width - marginRight, height - marginBottom]];
        svg.call(d3.zoom()
            .scaleExtent([1, 8])
            .translateExtent(extent)
            .extent(extent)
            .on('zoom', (event) => {
                x.range([marginLeft, width - marginRight].map(d => event.transform.applyX(d)));
                
                // 更新条形宽度，确保不超过最大宽度
                const newBandwidth = Math.min(x.bandwidth(), maxBarWidth);
                
                svg.selectAll('.background-bar, .highlight-bar')
                    .attr('x', d => x(d.tag) + (x.bandwidth() - newBandwidth) / 2)
                    .attr('width', newBandwidth);
                    
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

    // 为每个元素类型创建条形图组
    const barGroups = svg.selectAll('.bar-group')
        .data(visibleData)
        .enter()
        .append('g')
        .attr('class', d => `bar-group bar-group-${d.tag}`);
    
    // 添加背景灰色条形
    barGroups.append('rect')
        .attr('class', 'background-bar')
        .attr('x', d => x(d.tag) + (x.bandwidth() - actualBandwidth) / 2) // 居中显示
        .attr('y', d => y(d.num))
        .attr('width', actualBandwidth)
        .attr('height', d => height - marginBottom - y(d.num))
        .attr('fill', '#E0E0E0')
        .each(function(d) {
            // 存储矩形的位置和尺寸信息
            d.x = x(d.tag) + (x.bandwidth() - actualBandwidth) / 2;
            d.y = y(d.num);
            d.width = actualBandwidth;
            d.height = height - marginBottom - y(d.num);
        });
    
    // 添加前景蓝色条形（初始高度为0）
    barGroups.append('rect')
        .attr('class', 'highlight-bar')
        .attr('x', d => x(d.tag) + (x.bandwidth() - actualBandwidth) / 2) // 居中显示
        .attr('y', d => height - marginBottom) // 初始位置在底部
        .attr('width', actualBandwidth)
        .attr('height', 0) // 初始高度为0
        .attr('fill', '#905F29')
        .style('opacity', 0.7);

    // 添加 y 轴图例
    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", marginLeft / 3)
        .attr("x", 0 - (height - marginBottom) / 1.7)
        .style("text-anchor", "middle")
        .style("font-size", "12px")
        .text("Count");

    zoom(svg);
    
    // 初始更新选中状态
    const selectedNodes = store.state.selectedNodes.nodeIds || [];
    updateSelection(selectedNodes);
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
  top: 1.1em;
  left: 16px;
  font-size: 14px;
  color: #000;
  margin: 0;
  padding: 0;
  z-index: 10;
  letter-spacing: -0.01em;
  opacity: 0.8;
}

/* 添加条形图样式 */
.background-bar, .highlight-bar {
  transition: opacity 0.3s;
}
.background-bar:hover, .highlight-bar:hover {
  opacity: 0.8;
}
</style>
