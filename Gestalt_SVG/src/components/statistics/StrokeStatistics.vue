<template>
    <div class="statistics-container">
        <span class="title">Stroke color</span>
        <div ref="chartContainer" class="chart-container"></div>
        <div v-if="!hasData" class="no-data-message">No Stroke</div>
    </div>
</template>

<script setup>
import { onMounted, ref, watch, nextTick } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
const store = useStore();

// 更新了数据接口地址
const eleURL = "http://127.0.0.1:5000/stroke_num";
const chartContainer = ref(null);
const hasData = ref(false);
const rawJsonData = ref(null);
const isInitialized = ref(false); // 添加初始化标志

onMounted(async () => {
    // 延迟执行数据获取，确保在父组件准备好后执行
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
        console.error('获取stroke_num数据时出错:', error);
        hasData.value = false;
    }
};

// 监听组件的key变化和容器变化
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
        const marginBottom = height * 0.25;
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
        const svg = d3.select(container)
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

        // 绘制带圆角的条形图
        svg.selectAll('.bar')
            .data(data)
            .enter()
            .append('path')
            .attr('class', 'bar')
            .attr('fill', d => d.tag) // 使用数据中的颜色值作为填充色
            .attr('stroke', '#666')
            .attr('d', d => roundedRectPath(d, x, y, actualBandwidth));

        // 添加 y 轴图例
        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", marginLeft / 3)
            .attr("x", 0 - (height - marginBottom) / 2)
            .style("text-anchor", "middle")
            .style("font-size", "14px")
            .text("Number");
        
    } catch (error) {
        console.error('渲染StrokeStatistics图表时出错:', error);
        hasData.value = false;
    }
}

const roundedRectPath = (d, x, y, maxWidth) => {
    const bandWidth = x.bandwidth();
    const barWidth = maxWidth || bandWidth;
    // 计算条形的中心位置
    const barCenter = x(d.tag) + bandWidth / 2;
    // 根据最大宽度计算条形的起始和结束位置
    const x0 = barCenter - barWidth / 2;
    const y0 = y(d.num);
    const x1 = barCenter + barWidth / 2;
    const y1 = y(0);
    
    // 使用直角矩形路径
    return `M${x0},${y0}
            L${x1},${y0}
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
.bar {
  transition: opacity 0.3s;
}
.bar:hover {
  opacity: 0.8;
}
</style>
