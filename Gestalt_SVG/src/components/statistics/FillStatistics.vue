<template>
    <div class="statistics-container">
        <span class="title">Fill color</span>
        <div ref="chartContainer" class="chart-container"></div>
        <div v-if="!hasData" class="no-data-message">No Fill</div>
    </div>
</template>

<script setup>
import { onMounted, ref, watch, nextTick } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
const store = useStore();

// 更新了数据接口地址
const eleURL = "http://127.0.0.1:5000/fill_num";
const elementColorsURL = "http://127.0.0.1:5000/element_colors"; // 新增元素颜色接口
const chartContainer = ref(null);
const hasData = ref(false);
const rawJsonData = ref(null);
const isInitialized = ref(false);
const svg = ref(null);
const elementColors = ref({}); // 存储元素颜色映射

// 计算选中比例的函数
const calculateSelectedRatio = (color, selectedNodes) => {
    if (!selectedNodes || selectedNodes.length === 0 || !elementColors.value) return 0;
    
    // 找出所有使用该颜色的元素
    const elementsWithColor = Object.keys(elementColors.value).filter(
        element => elementColors.value[element] === color
    );
    
    if (elementsWithColor.length === 0) return 0;
    
    // 标准化选中节点格式
    const normalizedSelectedNodes = selectedNodes.map(node => node.split('/').pop());
    
    // 计算交集
    const intersection = elementsWithColor.filter(element => 
        normalizedSelectedNodes.includes(element)
    );
    
    return intersection.length / elementsWithColor.length;
};

onMounted(async () => {
    // 延迟执行数据获取，确保在父组件准备好后执行
    await nextTick();
    isInitialized.value = true;
    await fetchElementColors(); // 先获取元素颜色映射
    await fetchData();
});

// 获取元素颜色映射
const fetchElementColors = async () => {
    try {
        const response = await fetch(elementColorsURL);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        elementColors.value = data;
    } catch (error) {
        console.error('获取元素颜色映射时出错:', error);
    }
};

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
        
        const rawData = await response.json();
        rawJsonData.value = rawData;
        
        // 检查数据是否为空
        if (!rawData || Object.keys(rawData).length === 0) {
            hasData.value = false;
            return;
        }
        
        // 将接收到的数据转换为适合绘图的格式
        const data = Object.keys(rawData).map(key => ({
            tag: key, // 颜色值作为标签
            num: rawData[key], // 对应的数量
            visible: true // 可见性标志，根据需要调整
        }));

        if (data.length > 0) {
            // 先清除之前的图表内容
            if (chartContainer.value) {
                chartContainer.value.innerHTML = '';
            }
            
            // 先设置hasData为true，再渲染
            hasData.value = true;
            store.commit('GET_ELE_NUM_DATA', data);
            
            // 强制下一个渲染周期再渲染图表
            setTimeout(() => {
                render(data);
            }, 0);
        } else {
            hasData.value = false;
        }
    } catch (error) {
        console.error('获取fill_num数据时出错:', error);
        hasData.value = false;
    }
};

// 监听组件的容器变化
watch([() => chartContainer.value, () => isInitialized.value], ([newContainer, newInitialized]) => {
    if (newContainer && newInitialized && rawJsonData.value) {
        const data = Object.keys(rawJsonData.value).map(key => ({
            tag: key,
            num: rawJsonData.value[key],
            visible: true
        }));
        
        if (data.length > 0) {
            hasData.value = true;
            // 清除之前的图表内容
            if (chartContainer.value) {
                chartContainer.value.innerHTML = '';
            }
            render(data);
        }
    }
});

const render = (data) => {
    if (!chartContainer.value) {
        console.error('渲染失败：chartContainer不存在');
        return;
    }
    
    if (!data || !Array.isArray(data) || data.length === 0) {
        console.error('渲染失败：数据为空或格式不正确', data);
        hasData.value = false;
        return;
    }

    try {
        // 为确保DOM已更新，测量容器大小
        const container = chartContainer.value;
        
        const width = container.clientWidth || 300;
        const height = container.clientHeight || 200;
        
        if (width <= 0 || height <= 0) {
            console.warn('容器尺寸异常，使用默认值');
        }
        
        const marginTop = height * 0.08;
        const marginRight = width * 0.02;
        const marginBottom = height * 0.2;
        const marginLeft = width * 0.15;

        // 清空容器
        container.innerHTML = '';
        
        const x = d3.scaleBand()
            .domain(data.map(d => d.tag))
            .range([marginLeft, width - marginRight])
            .padding(0.1);

        // 设置条形的最大宽度
        const maxBarWidth = 50; // 设置条形的最大宽度为50px
        const actualBandwidth = Math.min(x.bandwidth(), maxBarWidth);
        
        const y = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.num)]).nice()
            .range([height - marginBottom, marginTop]);

        // 创建SVG元素
        svg.value = d3.select(container)
            .append('svg')
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('width', width)
            .attr('height', height)
            .attr('style', 'max-width: 100%; height: auto;');
        
        // 添加横轴
        svg.value.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${height - marginBottom})`)
            .call(d3.axisBottom(x))
            .selectAll("text").remove(); // 移除标签文本，后续添加颜色圆点

        // 添加颜色圆点于x轴
        data.forEach(d => {
            svg.value.select('.x-axis').append('circle')
                .attr('cx', x(d.tag) + x.bandwidth() / 2)
                .attr('cy', 15) // 轴线下方适当位置
                .attr('r', 8)
                .attr('fill', d.tag)
                .attr('stroke', '#999');
        });
        
        // 添加纵轴及横线
        const yAxis = svg.value.append('g')
            .attr('class', 'y-axis')
            .attr('transform', `translate(${marginLeft},0)`)
            .call(d3.axisLeft(y)
                .ticks(5)  // 减少刻度数量
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

        // 创建颜色组
        const colorGroups = svg.value.selectAll('.color-group')
            .data(data)
            .enter()
            .append('g')
            .attr('class', 'color-group')
            .attr('transform', d => `translate(${x(d.tag) + x.bandwidth() / 2},0)`)
            .attr('style', 'cursor: pointer;')
            .on('click', (event, d) => {
                // 找出所有使用该颜色的元素
                const elementsWithColor = Object.keys(elementColors.value).filter(
                    element => elementColors.value[element] === d.tag
                );
                
                // 标准化元素格式
                const normalizedElements = elementsWithColor.map(element => 
                    element.startsWith('svg/') ? element : `svg/${element}`
                );
                
                // 更新选中节点
                store.commit('UPDATE_SELECTED_NODES', { 
                    nodeIds: normalizedElements,
                    group: null 
                });
            });

        // 绘制背景条形图（灰色）
        colorGroups.append('rect')
            .attr('class', 'background-bar')
            .attr('x', -actualBandwidth / 2)
            .attr('y', d => y(d.num))
            .attr('width', actualBandwidth)
            .attr('height', d => height - marginBottom - y(d.num))
            .attr('fill', '#E0E0E0'); // 使用灰色作为背景

        // 绘制高亮条形图（初始高度为0）
        colorGroups.append('rect')
            .attr('class', 'highlight-bar')
            .attr('x', -actualBandwidth / 2)
            .attr('y', height - marginBottom) // 初始位置在底部
            .attr('width', actualBandwidth)
            .attr('height', 0) // 初始高度为0
            .attr('fill', '#905F29') // 使用与PositionStatistics相同的高亮颜色
            .style('opacity', 0.7);

        // 添加 y 轴图例
        svg.value.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", marginLeft / 3)
            .attr("x", 0 - (height - marginTop) / 2)
            .style("text-anchor", "middle")
            .style("font-size", "12px")
            .text("Count");
        
        // 添加提示框
        const tooltip = d3.select(container)
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
        
        // 添加鼠标悬停事件
        colorGroups.on('mouseover', (event, d) => {
            // 找出所有使用该颜色的元素
            const elementsWithColor = Object.keys(elementColors.value).filter(
                element => elementColors.value[element] === d.tag
            );
            
            tooltip.style("visibility", "visible")
                .html(() => {
                    const elementsContent = elementsWithColor.slice(0, 10).map(element => `${element}<br/>`).join("");
                    const moreElements = elementsWithColor.length > 10 ? `<br/>...and ${elementsWithColor.length - 10} more` : "";
                    const content = `<strong>Color:</strong> ${d.tag}<br/><strong>Count:</strong> ${d.num}<br/><strong>Elements:</strong><br/>${elementsContent}${moreElements}`;
                    return content;
                });
            
            const tooltipHeight = tooltip.node().getBoundingClientRect().height;
            tooltip.style("top", (event.pageY - tooltipHeight - 10) + "px")
                .style("left", (event.pageX + 10) + "px");
        })
        .on('mouseout', () => {
            tooltip.style("visibility", "hidden");
        });
        
        // 更新选中状态的函数
        const updateSelection = (selectedNodes) => {
            if (!selectedNodes) return;
            
            colorGroups.each(function(d) {
                const ratio = calculateSelectedRatio(d.tag, selectedNodes);
                const barHeight = height - marginBottom - y(d.num);
                
                // 更新高亮条形的高度和位置
                d3.select(this).select('.highlight-bar')
                    .transition()
                    .duration(300)
                    .attr('height', barHeight * ratio)
                    .attr('y', y(d.num) + barHeight * (1 - ratio));
            });
        };
        
        // 监听选中节点变化
        watch(
            () => store.state.selectedNodes.nodeIds,
            (newSelectedNodes) => {
                if (!svg.value) return;
                updateSelection(newSelectedNodes || []);
            },
            { deep: true, immediate: true }
        );
        
    } catch (error) {
        console.error('渲染FillStatistics图表时出错:', error);
        hasData.value = false;
    }
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
    min-height: 180px; /* 确保有足够的高度 */
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

.color-group:hover .background-bar {
  opacity: 0.8;
}

.tooltip strong {
    color: #905F29;
}

.tooltip span {
    color: black;
}
</style>
