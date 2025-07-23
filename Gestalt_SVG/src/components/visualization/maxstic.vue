<template>
    <div ref="parentContainer" class="combined_chart_container">
        <div ref="linesContainer" class="lines_container"></div>
        <div ref="normalChartContainer" class="normal_chart_container"></div>
        <div ref="initChartContainer" class="init_chart_container"></div>
        <div ref="axisContainer" class="axis_container"></div>
        <div ref="mdsContainer" class="mds_container"></div>
        <div class="tooltip" ref="lineTooltip"></div>
    </div>
</template>

<script setup>
import { onMounted, ref, computed, watch, onUnmounted, nextTick } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';


// 获取Vuex store
const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);

// 添加 emit
const emit = defineEmits(['update-analysis']);

// 数据源URL
const NORMAL_DATA_URL = "http://127.0.0.1:8000/normalized_init_json";
const INIT_DATA_URL = "http://127.0.0.1:8000/cluster_features";
const MAPPING_DATA_URL = "http://127.0.0.1:8000/average_equivalent_mapping";
const EQUIVALENT_WEIGHTS_URL = "http://127.0.0.1:8000/equivalent_weights_by_tag"; // 新的数据源

// 引用
const parentContainer = ref(null);
const normalChartContainer = ref(null);
const initChartContainer = ref(null);
const linesContainer = ref(null); // 连线容器引用
const lineTooltip = ref(null); // 全局连线工具提示引用
const axisContainer = ref(null);
const mdsContainer = ref(null);

// 统一的边距设置，确保左右对齐
const marginNormal = { top: 10, right: 10, bottom: 60, left: 280 };
const marginInit = { top: 0, right: 10, bottom:80, left: 280 };

// SVG和缩放变量
let svgNormal, svgInit;
let xScaleNormal, xScaleInit;
let resizeObserver;

// 圆点半径
const circleRadius = 5;

// 圆角半径
const cornerRadius = 8;

// 在顶层声明数据变量
let dataEquivalentWeights; // 声明 dataEquivalentWeights
let dataMapping; // 如果其他函数需要

// 全局鼠标释放事件监听器
const handleGlobalMouseUp = () => {
    // 恢复所有连线的显示
    d3.selectAll('.lines_container path')
        .style('display', null); // 显示所有连线
};

// 生成分析文字的函数
const generateAnalysis = (dataMapping, dataEquivalentWeights) => {
    if (!dataMapping || !dataEquivalentWeights) return '';

    let analysis = '';
    const inputDimensions = dataMapping.input_dimensions;
    const outputDimensions = dataMapping.output_dimensions;
    const weights = dataMapping.weights;

    // 分析每个输出维度的主要特征
    outputDimensions.forEach((outDim, j) => {
        analysis += `维度 Z_${j + 1}：\n`;
        
        // 获取该维度的权重
        const dimensionWeights = weights[j];
        
        // 找出最重要的输入特征（权重绝对值最大的两个）
        const weightEntries = dimensionWeights.map((w, i) => ({ weight: w, index: i }));
        weightEntries.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
        
        // 取前2个最重要的特征
        const topFeatures = weightEntries.slice(0, 2);
        
        analysis += `主要由 `;
        topFeatures.forEach(({ weight, index }, i) => {
            const featureName = inputDimensions[index];
            const influence = weight > 0 ? '正向' : '负向';
            analysis += `${featureName}(${influence})${i < topFeatures.length - 1 ? ' 和 ' : ' '}`;
        });
        analysis += '特征组成\n\n';
    });

    return analysis;
};

// 添加相关性矩阵的渲染函数
const renderCorrelationMatrix = (correlationData, svg, width) => {
    const matrixWidth = 200;
    const matrixHeight = 200;
    const cellSize = matrixWidth / 4;

    // 移除旧的相关性矩阵
    svg.selectAll('.correlation-matrix').remove();
    svg.selectAll('.dimension-stats').remove();

    const correlationGroup = svg.append('g')
        .attr('class', 'correlation-matrix')
        .attr('transform', `translate(${width + 50}, 20)`);

    // 添加标题
    correlationGroup.append('text')
        .attr('x', matrixWidth / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', 'bold')
        .text('维度相关性矩阵');

    // 创建颜色比例尺
    const colorScale = d3.scaleSequential(d3.interpolateRdBu)
        .domain([1, -1]);

    // 绘制相关性矩阵
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            const cell = correlationGroup.append('g')
                .attr('transform', `translate(${j * cellSize}, ${i * cellSize})`);

            // 添加背景矩形
            cell.append('rect')
                .attr('width', cellSize)
                .attr('height', cellSize)
                .style('fill', colorScale(correlationData.correlations[i][j]))
                .style('stroke', 'white');

            // 添加相关系数文本
            cell.append('text')
                .attr('x', cellSize / 2)
                .attr('y', cellSize / 2)
                .attr('dy', '0.35em')
                .attr('text-anchor', 'middle')
                .style('font-size', '12px')
                .style('fill', Math.abs(correlationData.correlations[i][j]) > 0.5 ? 'white' : 'black')
                .text(correlationData.correlations[i][j].toFixed(2));
        }
    }

    // 添加维度标签
    for (let i = 0; i < 4; i++) {
        // Y轴标签
        correlationGroup.append('text')
            .attr('x', -10)
            .attr('y', i * cellSize + cellSize / 2)
            .attr('dy', '0.35em')
            .attr('text-anchor', 'end')
            .style('font-size', '12px')
            .text(`Z_${i + 1}`);

        // X轴标签
        correlationGroup.append('text')
            .attr('x', i * cellSize + cellSize / 2)
            .attr('y', matrixHeight + 20)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .text(`Z_${i + 1}`);
    }

    // 添加维度统计信息
    const statsGroup = svg.append('g')
        .attr('class', 'dimension-stats')
        .attr('transform', `translate(${width + 50}, ${matrixHeight + 100})`);

    statsGroup.append('text')
        .attr('x', 0)
        .attr('y', -20)
        .style('font-size', '14px')
        .style('font-weight', 'bold')
        .text('维度统计特征');

    correlationData.dimensionStats.forEach((stats, i) => {
        const statText = statsGroup.append('g')
            .attr('transform', `translate(0, ${i * 60})`);

        statText.append('text')
            .attr('x', 0)
            .attr('y', 0)
            .style('font-weight', 'bold')
            .text(`Z_${i + 1}:`);

        statText.append('text')
            .attr('x', 0)
            .attr('y', 20)
            .text(`偏度: ${stats.skewness.toFixed(2)}, 峰度: ${stats.kurtosis.toFixed(2)}`);
    });
};

// 更新相关性矩阵的位置
const updateCorrelationMatrix = () => {
    const correlationMatrix = d3.select(axisContainer.value).select('.correlation-matrix');
    if (!correlationMatrix.empty()) {
        correlationMatrix.attr('transform', `translate(${width + 50}, 20)`);
    }
};

// 添加防抖函数
const debounce = (fn, delay) => {
    let timeoutId;
    return (...args) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
};

// 在 setup 函数顶部声明清理函数
let cleanup = () => {};

// 在组件挂载后执行
onMounted(async () => {
    if (!parentContainer.value) return;

    try {
        // 并行获取四个数据源
        const [responseNormal, responseInit, responseMapping, responseEquivalentWeights] = await Promise.all([
            fetch(NORMAL_DATA_URL),
            fetch(INIT_DATA_URL),
            fetch(MAPPING_DATA_URL),
            fetch(EQUIVALENT_WEIGHTS_URL)
        ]);

        if (!responseNormal.ok || !responseInit.ok || !responseMapping.ok || !responseEquivalentWeights.ok) {
            throw new Error('网络响应有问题');
        }

        const rawDataNormal = await responseNormal.json();
        const rawDataInit = await responseInit.json();
        const rawDataMapping = await responseMapping.json();
        const rawDataEquivalentWeights = await responseEquivalentWeights.json();

        // 生成分析文字
        const analysis = generateAnalysis(rawDataMapping, rawDataEquivalentWeights);
        emit('update-analysis', analysis);

        const dataNormal = processDataNormal(rawDataNormal);
        const dataInit = processDataInit(rawDataInit);
        dataMapping = rawDataMapping;
        dataEquivalentWeights = rawDataEquivalentWeights;

        const groupNames = dataMapping.input_dimensions;

        // 初始渲染
        const initialWidth = parentContainer.value.clientWidth;
        
        // 按顺序渲染各个组件
        await renderNormal(dataNormal, initialWidth, groupNames);
        await renderInit(dataInit, initialWidth, dataMapping.output_dimensions);
        await nextTick();
        
        // 确保比例尺已经初始化
        if (xScaleNormal && xScaleInit) {
            updateHighlights(selectedNodeIds.value);
        }
        
        // 渲染其他组件
        drawLines(dataMapping, initialWidth, groupNames);
        renderDimensionAxes(rawDataInit, initialWidth);

        // 设置 ResizeObserver 使用防抖
        const handleResize = debounce(async (entries) => {
            try {
                for (let entry of entries) {
                    const newWidth = entry.contentRect.width;
                    await renderNormal(dataNormal, newWidth, groupNames);
                    await renderInit(dataInit, newWidth, dataMapping.output_dimensions);
                    await nextTick();
                    if (xScaleNormal && xScaleInit) {
                        updateHighlights(selectedNodeIds.value);
                    }
                    drawLines(dataMapping, newWidth, groupNames);
                    renderDimensionAxes(rawDataInit, newWidth);
                    renderMDSScatterplot(rawDataNormal, newWidth);
                }
            } catch (error) {
                console.error('处理resize事件时出错:', error);
            }
        }, 150);

        resizeObserver = new ResizeObserver((entries) => {
            window.requestAnimationFrame(() => {
                handleResize(entries);
            });
        });

        resizeObserver.observe(parentContainer.value);

        // 添加全局事件监听器
        window.addEventListener('mouseup', handleGlobalMouseUp);

        // 更新清理函数
        cleanup = () => {
            window.removeEventListener('mouseup', handleGlobalMouseUp);
            if (resizeObserver) {
                resizeObserver.disconnect();
            }
            if (parentContainer.value) {
                resizeObserver.unobserve(parentContainer.value);
            }
        };

        // 计算并渲染相关性矩阵
        const correlationData = calculateDimensionCorrelations(rawDataInit);
        const axisSvg = d3.select(axisContainer.value).select('svg');
        renderCorrelationMatrix(correlationData, axisSvg, initialWidth);

        // 监听 selectedNodeIds 的变化
        watch(selectedNodeIds, (newVal) => {
            if (xScaleNormal && xScaleInit) {
                updateHighlights(newVal);
            }
        }, { immediate: true });

        // 添加点击空白处的事件监听
        d3.select(initChartContainer.value).on('click', function(event) {
            if (event.target.tagName !== 'circle') {
                // 取消所有圆点的高亮
                d3.selectAll('.y-axis circle')
                    .classed('highlighted', false)
                    .style('fill', '#999');
                // 显示所有连线
                d3.selectAll('.lines_container path')
                    .style('display', null);
            }
        });

    } catch (error) {
        console.error('数据获取或处理时出错:', error);
    }
});

// 在 setup 函数中直接调用 onUnmounted
onUnmounted(() => {
    cleanup();
});

// 更新高亮显示
const updateHighlights = (selectedIds) => {
    // 更新第一张热力图的高亮
    d3.selectAll('.normal_chart_container .x-axis text')
        .style('fill', d => selectedIds.includes(d) ? 'red' : 'black')
        .style('font-weight', d => selectedIds.includes(d) ? 'bold' : 'normal');

    // 移除旧的高亮矩形
    d3.selectAll('.normal_chart_container .highlight-rect-normal').remove();

    // 添加新的高亮矩形
    selectedIds.forEach(id => {
        const xPos = xScaleNormal(id);
        if (xPos !== undefined) {
            svgNormal.append('rect')
                .attr('class', 'highlight-rect-normal')
                .attr('x', xPos)
                .attr('y', 0)
                .attr('width', xScaleNormal.bandwidth())
                .attr('height', normalChartContainer.value.clientHeight - marginNormal.top - marginNormal.bottom)
                .style('fill', 'none')
                .style('stroke', 'red')
                .style('stroke-width', '1.5px');
        }
    });

    // 更新第二张热力图的高亮
    d3.selectAll('.init_chart_container .x-axis text')
        .style('fill', d => selectedIds.includes(d) ? 'red' : 'black')
        .style('font-weight', d => selectedIds.includes(d) ? 'bold' : 'normal');

    // 移除旧的高亮矩形
    d3.selectAll('.init_chart_container .highlight-rect-cc').remove();

    // 添加新的高亮矩形
    selectedIds.forEach(id => {
        const xPos = xScaleInit(id);
        if (xPos !== undefined) {
            svgInit.append('rect')
                .attr('class', 'highlight-rect-cc')
                .attr('x', xPos)
                .attr('y', 0)
                .attr('width', xScaleInit.bandwidth())
                .attr('height', initChartContainer.value.clientHeight - marginInit.top - marginInit.bottom)
                .style('fill', 'none')
                .style('stroke', 'red')
                .style('stroke-width', '1.5px');
        }
    });
};

// 处理第一张热力图的数据
const processDataNormal = (rawData) => {
    let processedData = [];
    rawData.forEach((node) => {
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

// 处理第二张热力图的数据
const processDataInit = (rawData) => {
    let processedData = [];
    rawData.forEach((node) => {
        node.features.forEach((featureValue, featureIndex) => {
            processedData.push({
                node: node.id,
                group: featureIndex,
                featureValue: featureValue,
            });
        });
    });
    return processedData;
};

// 渲染第一张热力图
const renderNormal = async (data, containerWidth, groupNames) => {
    return new Promise((resolve) => {
        const width = containerWidth - marginNormal.left - marginNormal.right;
        const height = 350 + marginNormal.top + marginNormal.bottom;

        // 清空旧的SVG
        d3.select(normalChartContainer.value).selectAll('*').remove();

        // 创建SVG
        svgNormal = d3.select(normalChartContainer.value)
            .append('svg')
            .attr('width', containerWidth)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${marginNormal.left},${marginNormal.top})`);

        // 获取唯一的节点ID
        const ids = [...new Set(data.map(d => d.node.split('/').pop()))];
        const groups = d3.range(0, groupNames.length);

        // 创建比例尺
        xScaleNormal = d3.scaleBand().domain(ids).range([0, width]).padding(0.05);
        const yScale = d3.scaleBand().domain(groups).range([height - marginNormal.top - marginNormal.bottom, 0]).padding(0.05);
        const yScaleName = d3.scaleBand().domain(groupNames).range([height - marginNormal.top - marginNormal.bottom, 0]).padding(0.05);

        // 修改颜色比例尺为分歧颜色比例尺，定 domain 为 [-1, 0, 1]
        const colorScale = d3.scaleDiverging(d3.interpolateRdBu)
            .domain([1, 0, -1]);

        // 创建悬停提示框
        const tooltip = d3.select(normalChartContainer.value)
            .append('div')
            .attr('class', 'tooltip')
            .style('position', 'absolute')
            .style('background', '#fff')
            .style('padding', '5px')
            .style('border', '1px solid #ccc')
            .style('border-radius', '5px')
            .style('pointer-events', 'none')
            .style('visibility', 'hidden');

        // 绘制方块
        svgNormal.selectAll('.block')
            .data(data)
            .enter()
            .append('rect')
            .attr('x', d => xScaleNormal(d.node.split('/').pop()))
            .attr('y', d => yScale(d.group))
            .attr('width', xScaleNormal.bandwidth())
            .attr('height', yScale.bandwidth())
            .style('fill', d => colorScale(d.probability))
            .on('mouseover', function (event, d) {
                tooltip.style('visibility', 'visible')
                    .text(`Probability: ${d.probability}`);
            })
            .on('mousemove', function (event) {
                const [mouseX, mouseY] = d3.pointer(event);
                tooltip.style('top', `${mouseY - 30}px`)
                    .style('left', `${mouseX + 10}px`);
            })
            .on('mouseout', function () {
                tooltip.style('visibility', 'hidden');
            });

        // 添加X轴
        const xAxis = d3.axisBottom(xScaleNormal).tickSizeOuter(0);

        svgNormal.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${height - marginNormal.top - marginNormal.bottom})`)
            .call(xAxis)
            .selectAll('text')
            .style('text-anchor', 'end')
            .attr('dx', '-1em')
            .attr('dy', '-0.5em')
            .attr('transform', 'rotate(-90)')
            .style('fill', 'black')
            .style('font-size', '12px')
            .style('cursor', 'pointer')
            .on('click', function (event, d) {
                // 点击事件更新Vuex中的 selectedNodeIds
                if (selectedNodeIds.value.includes(d)) {
                    store.commit('REMOVE_SELECTED_NODE', d);
                } else {
                    store.commit('ADD_SELECTED_NODE', d);
                }
            });

        // 创建Y轴
        const yAxis = d3.axisLeft(yScaleName).tickSizeOuter(0);

        // 添加Y轴圆点并绑定事件监听器
        svgNormal.append('g')
            .attr('class', 'y-axis')
            .call(yAxis)
            .selectAll('.tick')
            .each(function (d) {
                const tick = d3.select(this);
                tick.insert('circle', 'text')
                    .attr('cx', -120)
                    .attr('cy', 0)
                    .attr('r', circleRadius)
                    .style('fill', '#666');
            })
            .selectAll('text')
            .style('fill', 'black')
            .style('font-size', '12px');

        // 美化坐标轴
        svgNormal.selectAll('.domain')
            .style('stroke', 'black')
            .style('stroke-width', '1px');

        svgNormal.selectAll('.tick line')
            .style('stroke', 'black')
            .style('stroke-width', '1px');

        // 在渲染完成后解析 Promise
        resolve();
    });
};

// 渲染第二张热力图
const renderInit = async (data, containerWidth, outputDimensions) => {
    return new Promise((resolve) => {
        const width = containerWidth - marginInit.left - marginInit.right;
        const height = 60 + marginInit.top + marginInit.bottom;

        // 清空旧的SVG
        d3.select(initChartContainer.value).selectAll('*').remove();

        // 创建SVG
        svgInit = d3.select(initChartContainer.value)
            .append('svg')
            .attr('width', containerWidth)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${marginInit.left},${marginInit.top})`);

        // 获取唯一的节点ID
        const ids = [...new Set(data.map(d => d.node.split('/').pop()))];
        const groups = d3.range(0, outputDimensions.length);
        const groupNames = outputDimensions; // 使用 output_dimensions 作为 groupNames

        // 创建比例尺
        xScaleInit = d3.scaleBand().domain(ids).range([0, width]).padding(0.05);
        const yScale = d3.scaleBand().domain(groups).range([height - marginInit.top - marginInit.bottom, 0]).padding(0.05);

        // 修改颜色比例尺为分歧颜色比例尺，固定 domain 为 [-1, 0, 1]
        const colorScale = d3.scaleDiverging(d3.interpolateRdBu)
            .domain([1, 0, -1]);

        // 创建悬停提示框
        const tooltip = d3.select(initChartContainer.value)
            .append('div')
            .attr('class', 'tooltip')
            .style('position', 'absolute')
            .style('background', '#fff')
            .style('padding', '5px')
            .style('border', '1px solid #ccc')
            .style('border-radius', '5px')
            .style('pointer-events', 'none')
            .style('visibility', 'hidden');

        // 绘制方块
        svgInit.selectAll('.block')
            .data(data)
            .enter()
            .append('rect')
            .attr('x', d => xScaleInit(d.node.split('/').pop()))
            .attr('y', d => yScale(d.group))
            .attr('width', xScaleInit.bandwidth())
            .attr('height', yScale.bandwidth())
            .style('fill', d => colorScale(d.featureValue))
            .on('mouseover', function (event, d) {
                tooltip.style('visibility', 'visible')
                    .text(`Value: ${d.featureValue}`);
            })
            .on('mousemove', function (event) {
                const [mouseX, mouseY] = d3.pointer(event);
                tooltip.style('top', `${mouseY - 40}px`)
                    .style('left', `${mouseX + 10}px`);
            })
            .on('mouseout', function () {
                tooltip.style('visibility', 'hidden');
            })
            .on('click', function (event, d) {
                event.stopPropagation(); // 防止立即清除

                const nodeId = d.node;

                const perSampleWeights = dataEquivalentWeights[nodeId];
                if (!perSampleWeights) {
                    console.warn(`未找到节点 ${nodeId} 的等效权重`);
                    return;
                }

                // 获取点击的输出维度的权重
                const weightsForDimension = perSampleWeights[d.group]; // 长度为 20 的权重数组

                // 调用函数绘制按样本的连线
                drawPerSampleLines(weightsForDimension, d, this); // 传递 'this' 以获取被点击的矩形节点
            });

        // 添加X轴
        const xAxis = d3.axisBottom(xScaleInit).tickSizeOuter(0);

        svgInit.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${height - marginInit.top - marginInit.bottom})`)
            .call(xAxis)
            .selectAll('text')
            .style('text-anchor', 'end')
            .attr('dx', '-1em')
            .attr('dy', '-0.5em')
            .attr('transform', 'rotate(-90)')
            .style('fill', 'black')
            .style('font-size', '12px')
            .style('cursor', 'pointer')
            .on('click', function (event, d) {
                // 点击事件更新Vuex中的 selectedNodeIds
                if (selectedNodeIds.value.includes(d)) {
                    store.commit('REMOVE_SELECTED_NODE', d);
                } else {
                    store.commit('ADD_SELECTED_NODE', d);
                }
            });

        // 创建Y轴
        const yAxis = d3.axisLeft(yScale).tickSizeOuter(0)
            .tickFormat(d => groupNames[d]);

        // 添加Y轴圆点并绑定事件监听器
        svgInit.append('g')
            .attr('class', 'y-axis')
            .call(yAxis)
            .selectAll('.tick')
            .each(function (d) {
                const tick = d3.select(this);
                tick.insert('circle', 'text')
                    .attr('cx', -40)
                    .attr('cy', 0)
                    .attr('r', circleRadius)
                    .style('fill', '#999')
                    .style('z-index', 999)
                    .attr('data-init-index', d) // 添加数据属性
                    .style('cursor', 'pointer') // 指示可交互
                    .on('click', function (event, d) {
                        event.stopPropagation(); // 阻止事件冒泡
                        const circle = d3.select(this);
                        const isHighlighted = circle.classed('highlighted');
                        
                        // 如果当前圆点已经高亮，则取消高亮并显示所有连线
                        if (isHighlighted) {
                            circle.classed('highlighted', false)
                                .style('fill', '#999');
                            d3.selectAll('.lines_container path')
                                .style('display', null);
                        } else {
                            // 取消其他圆点的高亮
                            d3.selectAll('.y-axis circle')
                                .classed('highlighted', false)
                                .style('fill', '#999');
                            
                            // 高亮当前圆点并只显示相关连线
                            circle.classed('highlighted', true)
                                .style('fill', '#ff6347');
                            d3.selectAll('.lines_container path')
                                .style('display', function () {
                                    return this.getAttribute('data-init-index') == d ? null : 'none';
                                });
                        }
                    });
            })
            .selectAll('text')
            .style('fill', 'black')
            .style('font-size', '12px');

        // 美化坐标轴
        svgInit.selectAll('.domain')
            .style('stroke', 'black')
            .style('stroke-width', '1px');

        svgInit.selectAll('.tick line')
            .style('stroke', 'black')
            .style('stroke-width', '1px');

        // 在渲染完成后解析 Promise
        resolve();
    });
};

// 当第二张热力图中的矩形被点击时，绘制按样本的连线
const drawPerSampleLines = (weightsForDimension, clickedRectData, clickedRectNode) => {
    // 清除现有的按样本连线
    d3.selectAll('.per-sample-lines').remove();

    // 降低其他元素的透明度
    d3.selectAll('.lines_container path') // 现有的平连线
        .style('opacity', 0.1);

    d3.selectAll('.normal_chart_container rect') // 第一张热力图中的矩形
        .style('opacity', 0.1);

    d3.selectAll('.init_chart_container rect') // 第二张热力图中的矩形
        .style('opacity', 0.1);

    // 将被点击的矩形的透明度恢复为 1
    d3.select(clickedRectNode)
        .style('opacity', 1);

    // 创建一个新的组用于按样本连线
    const perSampleLinesGroup = d3.select(linesContainer.value).select('svg')
        .append('g')
        .attr('class', 'per-sample-lines');

    // 获取输入特征（第一张热力图的 Y 轴上的圆点）
    const normalCircles = d3.selectAll('.normal_chart_container .y-axis circle');
    const normalData = normalCircles.nodes(); // 20 个输入节点

    // 获取被点击矩形的位置
    const clickedRectPos = clickedRectNode.getBoundingClientRect();
    const parentRect = parentContainer.value.getBoundingClientRect();

    const x2_center = (clickedRectPos.left + clickedRectPos.right) / 2 - parentRect.left;
    const y2_center = (clickedRectPos.top + clickedRectPos.bottom) / 2 - parentRect.top;

    const x2 = x2_center;
    const y2 = y2_center;

    // 定义线条粗细和颜色的比例尺
    const maxLineWidth = 4;
    const weightMax = d3.max(weightsForDimension);
    const weightMin = d3.min(weightsForDimension);

    const lineWidthScale = d3.scaleLinear()
        .domain([d3.min([Math.abs(weightMin), Math.abs(weightMax)]), d3.max([Math.abs(weightMin), Math.abs(weightMax)])])
        .range([0.5, maxLineWidth]);

    const lineColorScale = d3.scaleDiverging(d3.interpolateRdBu)
        .domain([weightMax, 0, weightMin]); // 动态 domain

    // 对于每个输入特征，绘制一条连线到被点击的矩形
    weightsForDimension.forEach((w_j_i, i) => {
        const inputCircle = normalData[i];
        const inputPos = inputCircle.getBoundingClientRect();

        const x1_center = (inputPos.left + inputPos.right) / 2 - parentRect.left;
        const y1_center = (inputPos.top + inputPos.bottom) / 2 - parentRect.top;

        const x1 = x1_center; // 从输入圆点边缘开始
        const y1 = y1_center;

        // 定义路径
        const pathData = `
            M${x1},${y1}
            L${x2},${y2}
        `;

        perSampleLinesGroup.append('path')
            .attr('d', pathData)
            .attr('stroke', lineColorScale(w_j_i))
            .attr('stroke-width', lineWidthScale(Math.abs(w_j_i)))
            .attr('fill', 'none')
            .attr('class', 'per-sample-line')
            .style('opacity', 0.8)
            .on('mouseover', function(event) {
                d3.select(this)
                    .style('stroke', 'orange')
                    .style('opacity', 1);
                // 显示权重值的提示框
                d3.select(lineTooltip.value)
                    .style('visibility', 'visible')
                    .text(`w_${clickedRectData.group+1},${i+1}: ${w_j_i.toFixed(4)}`);
            })
            .on('mousemove', function(event) {
                const [mouseX, mouseY] = d3.pointer(event);
                d3.select(lineTooltip.value)
                    .style('top', `${mouseY - 30}px`)
                    .style('left', `${mouseX + 10}px`);
            })
            .on('mouseout', function() {
                d3.select(this)
                    .attr('stroke', lineColorScale(w_j_i))
                    .style('opacity', 0.8);
                d3.select(lineTooltip.value).style('visibility', 'hidden');
            });
    });

    // 添加一个点击监听器到背景，用于清除按样本连线
    d3.select('body').on('click.perSample', function() {
        // 恢复其他元素的透明度
        d3.selectAll('.lines_container path')
            .style('opacity', 0.8);

        d3.selectAll('.normal_chart_container rect')
            .style('opacity', 1);

        d3.selectAll('.init_chart_container rect')
            .style('opacity', 1);

        // 清除按样本连线
        d3.select('.per-sample-lines').remove();

        // 移除此事件监听器
        d3.select('body').on('click.perSample', null);
    });
};

// 画连接线的函数，使用平均等效权重
const drawLines = (dataMapping, containerWidth, groupNames) => {
    if (!linesContainer.value) return;

    // 清空旧的线条
    d3.select(linesContainer.value).selectAll('*').remove();

    // 创建SVG用于绘制线条
    const svgLines = d3.select(linesContainer.value)
        .append('svg')
        .attr('width', containerWidth)
        .attr('height', parentContainer.value.clientHeight)
        .style('position', 'absolute')
        .style('top', '0')
        .style('left', '0');

    // 加载等效映射数据
    const outputDimensions = dataMapping.output_dimensions;
    const inputDimensions = dataMapping.input_dimensions;
    const weights = dataMapping.weights;

    // 获取节点对应的圆点
    const normalCircles = d3.selectAll('.normal_chart_container .y-axis circle');
    const initCircles = d3.selectAll('.init_chart_container .y-axis circle');

    const normalData = normalCircles.nodes();
    const initData = initCircles.nodes();

    // 确保数量匹配
    if (normalData.length !== inputDimensions.length || initData.length !== outputDimensions.length) {
        console.error('热力图节点数量与映射数据不匹配');
        return;
    }

    // 定义线条的最大粗细
    const maxLineWidth = 4;
    const minLineWidth = 0.5;

    // 为每个输出维度创建单独的线宽比例尺
    const lineWidthScales = weights.map(outputWeight => {
        const absWeights = outputWeight.map(Math.abs);
        return d3.scaleLinear()
            .domain([d3.min(absWeights), d3.max(absWeights)])
            .range([minLineWidth, maxLineWidth]);
    });

    // 计算所有权重的范围，用于颜色映射
    const allWeights = weights.flat();
    const weightMax = d3.max(allWeights);
    const weightMin = d3.min(allWeights);

    // 定义颜色比例尺
    const lineColorScale = d3.scaleDiverging(d3.interpolateRdBu)
        .domain([weightMax, 0, weightMin]);

    // 遍历每个输出维度
    weights.forEach((outputWeight, j) => {
        outputWeight.forEach((w_j_i, i) => {
            const inputCircle = normalData[i];
            const outputCircle = initData[j];

            // 获取 bounding rectangles
            const inputPos = inputCircle.getBoundingClientRect();
            const outputPos = outputCircle.getBoundingClientRect();

            // 获取父容器的位以计算相对坐标
            const parentRect = parentContainer.value.getBoundingClientRect();

            // 计算中心位置
            const x1_center = (inputPos.left + inputPos.right) / 2 - parentRect.left;
            const y1_center = (inputPos.top + inputPos.bottom) / 2 - parentRect.top;
            const x2_center = (outputPos.left + outputPos.right) / 2 - parentRect.left;
            const y2_center = (outputPos.top + outputPos.bottom) / 2 - parentRect.top;

            // 计算边缘位置
            const x1 = x1_center;
            const y1 = y1_center;
            const x2 = x2_center;
            const y2 = y2_center;

            // 计算水平偏移
            const horizontalOffset = 100;
            const outputOffset = j * -15;

            const verticalX = x1 - horizontalOffset + outputOffset;

            // 定义路径，应用水平偏移并添加圆角
            const pathData = `
                M${x1},${y1}
                L${verticalX + cornerRadius},${y1}
                Q${verticalX},${y1},${verticalX},${y1 + cornerRadius}
                L${verticalX},${y2 - cornerRadius}
                Q${verticalX},${y2},${verticalX + cornerRadius},${y2}
                L${x2},${y2}
            `;

            // 使用对应维度的线宽比例尺
            const lineWidth = lineWidthScales[j](Math.abs(w_j_i));

            // 追加路径到连线SVG，并设置线条粗细和颜色
            svgLines.append('path')
                .attr('d', pathData)
                .attr('stroke', lineColorScale(w_j_i))
                .attr('stroke-width', lineWidth)
                .attr('fill', 'none')
                .attr('data-init-index', j)
                .style('opacity', 0.8)
                .on('mouseover', function(event) {
                    d3.select(this)
                        .style('stroke', 'orange')
                        .style('opacity', 1);
                    d3.select(lineTooltip.value)
                        .style('visibility', 'visible')
                        .text(`w_${j+1},${i+1}: ${w_j_i.toFixed(4)}`);
                })
                .on('mousemove', function(event) {
                    const [mouseX, mouseY] = d3.pointer(event);
                    d3.select(lineTooltip.value)
                        .style('top', `${mouseY - 30}px`)
                        .style('left', `${mouseX + 10}px`);
                })
                .on('mouseout', function() {
                    d3.select(this)
                        .attr('stroke', lineColorScale(w_j_i))
                        .style('opacity', 0.8);
                    d3.select(lineTooltip.value).style('visibility', 'hidden');
                });
        });
    });
};

// 在renderDimensionAxes函数之前添加新的辅助函数
// 生成所有可能的维度组合
const generateDimensionCombinations = () => {
    const dimensions = [0, 1, 2, 3];
    const combinations = [];
    
    // 生成1到4维度的所有组合
    for (let len = 1; len <= dimensions.length; len++) {
        const getCombinations = (arr, len) => {
            const result = [];
            
            if (len === 1) return arr.map(x => [x]);
            
            arr.forEach((first, index) => {
                const rest = arr.slice(index + 1);
                const subCombinations = getCombinations(rest, len - 1);
                subCombinations.forEach(sub => {
                    result.push([first, ...sub]);
                });
            });
            
            return result;
        };
        
        combinations.push(...getCombinations(dimensions, len));
    }
    
    return combinations;
};

// 计算组合维度的值
const calculateCombinedDimensionValue = (features, combination) => {
    if (combination.length === 1) {
        return features[combination[0]];
    }
    
    // 对于多维组合，计算欧几里得距离
    return Math.sqrt(combination.reduce((sum, dim) => {
        return sum + Math.pow(features[dim], 2);
    }, 0));
};

// 修改renderDimensionAxes函数
const renderDimensionAxes = async (data, containerWidth) => {
    try {
        // 清空旧的轴
        d3.select(axisContainer.value).selectAll('*').remove();

        // 创建或更新 tooltip
        let tooltip;
        if (lineTooltip.value) {
            tooltip = d3.select(lineTooltip.value)
                .style('position', 'absolute')
                .style('background', '#fff')
                .style('padding', '5px')
                .style('border', '1px solid #ccc')
                .style('border-radius', '5px')
                .style('pointer-events', 'none')
                .style('visibility', 'hidden')
                .style('z-index', 9999);
        }

        // 修改使用 tooltip 的地方
        const showTooltip = (text) => {
            if (tooltip && lineTooltip.value) {
                tooltip.style('visibility', 'visible')
                    .text(text);
            }
        };

        const moveTooltip = (event) => {
            if (tooltip && lineTooltip.value) {
                const [mouseX, mouseY] = d3.pointer(event);
                tooltip.style('left', `${mouseX + margin.left + 10}px`)
                    .style('top', `${mouseY + margin.top - 10}px`);
            }
        };

        const hideTooltip = () => {
            if (tooltip && lineTooltip.value) {
                tooltip.style('visibility', 'hidden');
            }
        };

        const margin = { top: 20, right: 15, bottom: 20, left: 420 };
        const width = containerWidth - margin.left - margin.right;
        const height = 60; // 每个轴的高度
        const gmmHeight = 30; // GMM曲线的高度

        // 获取所有维度组合
        const dimensionCombinations = generateDimensionCombinations();
        const totalHeight = height * dimensionCombinations.length;

        // 创建SVG
        const svg = d3.select(axisContainer.value)
            .append('svg')
            .attr('width', containerWidth)
            .attr('height', totalHeight + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // 为每个维度组合创建数据
        const combinedData = dimensionCombinations.map(combination => {
            return data.map(d => ({
                id: d.id,
                value: calculateCombinedDimensionValue(d.features, combination),
                combination: combination
            }));
        });

        // 创建GMM曲线生成器
        const gmmCurve = d3.line()
            .x(d => d[0])
            .y(d => d[1])
            .curve(d3.curveBasis);

        // 为每个维度组合创建统计信息
        const dimensionStats = await Promise.all(combinedData.map(async dimData => {
            const values = dimData.map(d => d.value);
            try {
                const gmm = await fitGMM(values);
                return {
                    variance: d3.variance(values) || 0,
                    range: (d3.max(values) - d3.min(values)) || 0,
                    dispersionDegree: calculateDispersionDegree(values),
                    localClustering: calculateLocalClustering(gmm.components),
                    gmm: gmm
                };
            } catch (error) {
                console.error('处理GMM时出错:', error);
                return {
                    variance: d3.variance(values) || 0,
                    range: (d3.max(values) - d3.min(values)) || 0,
                    dispersionDegree: calculateDispersionDegree(values),
                    localClustering: 0,
                    gmm: { components: [{ mean: 0, variance: 0.0001, weight: 1.0 }] }
                };
            }
        }));

        // 定义表格列
        const columns = ['方差', '数据范围', '分散程度', '局部聚集性'];
        const columnWidths = [100, 100, 100, 100];

        // 为每个维度组合创建表头
        const header = svg.append('g')
            .attr('class', 'stats-header')
            .attr('transform', `translate(-400,0)`);

        header.selectAll('text')
            .data(columns)
            .enter()
            .append('text')
            .attr('x', (d, i) => {
                let x = 0;
                for (let j = 0; j < i; j++) {
                    x += columnWidths[j];
                }
                return x;
            })
            .attr('y', 0)
            .text(d => d)
            .style('font-weight', 'bold')
            .style('font-size', '12px');

        // 创建全局缩放行为
        const zoom = d3.zoom()
            .scaleExtent([0.5, 10])
            .extent([[0, 0], [width, height]])
            .on('zoom', function(event) {
                svg.selectAll('.dimension-group').each(function(d, i) {
                    const group = d3.select(this);
                    const transform = event.transform;
                    
                    // 创建固定domain的比例尺
                    const xScale = d3.scaleLinear()
                        .domain([-1, 1])
                        .range([0, width]);
                    
                    // 使用transform来调整比例尺
                    const newXScale = transform.rescaleX(xScale);
                    
                    // 更新轴
                    group.select('.x-axis').call(d3.axisBottom(newXScale));
                    
                    // 更新节点
                    group.selectAll('.dimension-nodes')
                        .attr('cx', d => newXScale(d.value));

                    // 更新GMM曲线和参数标签
                    const stats = dimensionStats[i];
                    if (stats && stats.gmm && stats.gmm.components) {
                        stats.gmm.components.forEach((component, idx) => {
                            const { points, containedNodes } = updateGMMCurve(component, newXScale, gmmHeight, combinedData[i]);
                            
                            // 更新曲线路径 - 修改选择器
                            const componentGroup = group.select(`.gmm-curve .gmm-component-group:nth-child(${idx + 1})`);
                            if (componentGroup.node()) {
                                // 更新路径
                                componentGroup.select('path')
                                    .datum(points)
                                    .attr('d', gmmCurve)
                                    .attr('data-contained-nodes', JSON.stringify(containedNodes));

                                // 更新文字标签位置
                                componentGroup.select('text')
                                    .attr('x', newXScale(component.mean));
                            }
                        });
                    }
                });
            });

        // 添加缩放矩形
        const zoomRect = svg.append('rect')
            .attr('class', 'zoom-rect')
            .attr('width', width + margin.left + margin.right)
            .attr('height', totalHeight)
            .attr('x', -margin.left)
            .style('fill', 'none')
            .style('pointer-events', 'all');

        // 将缩放行为应用到整个SVG
        svg.call(zoom);

        // 添加双击重置功能
        svg.on('dblclick.zoom', function() {
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        });

        // 创建一个新的组用于放置所有可交互元素
        const interactiveGroup = svg.append('g')
            .attr('class', 'interactive-layer')
            .style('pointer-events', 'all');

        // 在新的组中渲染维度组合
        dimensionCombinations.forEach((combination, index) => {
            const yPos = index * height;
            const dimensionGroup = interactiveGroup.append('g')
                .attr('class', `dimension-group dimension-group-${combination.join('-')}`)
                .attr('transform', `translate(0,${yPos})`);

            // 创建比例尺
            const xScale = d3.scaleLinear()
                .domain([-1, 1])
                .range([0, width]);

            // 添加GMM曲线
            const gmmGroup = dimensionGroup.append('g')
                .attr('class', 'gmm-curve')
                .attr('transform', `translate(0,${height/2 - gmmHeight})`);

            const stats = dimensionStats[index];
            if (stats && stats.gmm && stats.gmm.components) {
                stats.gmm.components.forEach((component, idx) => {
                    const { points, containedNodes } = updateGMMCurve(component, xScale, gmmHeight, combinedData[index]);
                    
                    // 创建一个组来包含路径和文字
                    const componentGroup = gmmGroup.append('g')
                        .attr('class', 'gmm-component-group');

                    const gmmPath = componentGroup.append('path')
                        .datum(points)
                        .attr('d', gmmCurve)
                        .attr('class', 'gmm-component')
                        .style('fill', d3.schemeCategory10[idx % 10])
                        .style('fill-opacity', 0.2)
                        .style('stroke', d3.schemeCategory10[idx % 10])
                        .style('stroke-width', 1.5)
                        .style('opacity', 0.6)
                        .style('cursor', 'pointer')
                        .attr('data-contained-nodes', JSON.stringify(containedNodes));

                    // 添加文字标签
                    const label = componentGroup.append('text')
                        .attr('x', xScale(component.mean))
                        .attr('y', -5)
                        .attr('text-anchor', 'middle')
                        .style('font-size', '10px')
                        .style('fill', d3.schemeCategory10[idx % 10])
                        .style('cursor', 'pointer')
                        .text(`μ=${component.mean.toFixed(2)}, σ²=${component.variance.toFixed(2)}`);

                    // 为路径和文字添加相同的点击事件
                    const handleClick = function(event) {
                        event.stopPropagation();
                        const nodes = JSON.parse(gmmPath.attr('data-contained-nodes'));
                        
                        // 高亮显示当前曲线
                        d3.selectAll('.gmm-component')
                            .style('fill-opacity', 0.2)
                            .style('stroke-width', 1.5);
                        gmmPath
                            .style('fill-opacity', 0.4)
                            .style('stroke-width', 2.5);

                        // 先清除所有已选中的节点
                        store.state.selectedNodes.nodeIds.forEach(id => {
                            store.commit('REMOVE_SELECTED_NODE', id);
                        });
                        
                        // 添加新选中的节点
                        nodes.forEach(nodeId => {
                            store.commit('ADD_SELECTED_NODE', nodeId);
                        });
                    };

                    // 为路径和文字添加相同的悬停事件
                    const handleMouseOver = function() {
                        gmmPath
                            .style('fill-opacity', 0.4)
                            .style('stroke-width', 2.5);
                    };

                    const handleMouseOut = function() {
                        if (!gmmPath.classed('selected')) {
                            gmmPath
                                .style('fill-opacity', 0.2)
                                .style('stroke-width', 1.5);
                        }
                    };

                    // 绑定事件到路径和文字
                    gmmPath
                        .on('click', handleClick)
                        .on('mouseover', handleMouseOver)
                        .on('mouseout', handleMouseOut);

                    label
                        .on('click', handleClick)
                        .on('mouseover', handleMouseOver)
                        .on('mouseout', handleMouseOut);
                });
            }

            // 添加X轴
            const xAxis = d3.axisBottom(xScale);
            dimensionGroup.append('g')
                .attr('class', 'x-axis')
                .attr('transform', `translate(0,${height/2})`)
                .call(xAxis);

            // 添加维度标签
            const tagGroup = dimensionGroup.append('g')
                .attr('class', 'dimension-tag')
                .attr('transform', `translate(-10, ${height/2})`);

            // 添加tag背景
            const tagBg = tagGroup.append('rect')
                .attr('x', -30)
                .attr('y', -12)
                .attr('width', 40)
                .attr('height', 24)
                .attr('rx', 8)
                .attr('ry', 4)
                .style('fill', '#f0f2f5')
                .style('stroke', '#e4e7ed')
                .style('stroke-width', 1);

            // 添加文本标签
            const tagText = tagGroup.append('text')
                .attr('x', -10)
                .attr('y', 4)
                .style('text-anchor', 'middle')
                .style('font-size', '14px')
                .style('fill', '#909399')
                .text(`Z_${combination.map(d => d + 1).join(',')}`);

            // 检查是否被过滤
            if (combinedData[index] && combinedData[index].length > 0) {
                const stats = dimensionStats[index];
                if (stats) {
                    const dispersion = stats.dispersionDegree;
                    const clustering = stats.localClustering;
                    let filterStatus = '';

                    // 根据评估矩阵进行判断
                    if (clustering > 0.7) {  // 高聚集
                        if (dispersion > 0.7 || dispersion > 0.3) {
                            filterStatus = '保留(多类别特征)';
                            tagBg.style('fill', '#d4edda');  // 浅绿色背景
                            tagText.style('fill', '#28a745');
                        } else {
                            filterStatus = '需要检查(区分不明显)';
                            tagBg.style('fill', '#fff3cd');  // 浅黄色背景
                            tagText.style('fill', '#856404');
                        }
                    } else if (clustering > 0.3) {  // 中等聚集
                        if (dispersion > 0.7) {
                            filterStatus = '可能保留(需要检查)';
                            tagBg.style('fill', '#fff3cd');  // 浅黄色背景
                            tagText.style('fill', '#856404');
                        } else if (dispersion > 0.3) {
                            filterStatus = '需要检查(需要进一步分析)';
                            tagBg.style('fill', '#fff3cd');  // 浅黄色背景
                            tagText.style('fill', '#856404');
                        } else {
                            filterStatus = '可能过滤(区分不明显)';
                            tagBg.style('fill', '#ffd6d6');  // 浅红色背景
                            tagText.style('fill', '#d63031');
                        }
                    } else {  // 低聚集
                        if (dispersion > 0.7) {
                            filterStatus = '过滤(纯噪声)';
                            tagBg.style('fill', '#ffb8b8');  // 红色背景
                            tagText.style('fill', '#c0392b');
                        } else if (dispersion > 0.3) {
                            filterStatus = '过滤(杂乱无规律)';
                            tagBg.style('fill', '#ffb8b8');  // 红色背景
                            tagText.style('fill', '#c0392b');
                        } else {
                            filterStatus = '过滤(无区分度)';
                            tagBg.style('fill', '#ffb8b8');  // 红色背景
                            tagText.style('fill', '#c0392b');
                        }
                    }

                    // 添加过滤状态标签
                    tagGroup.append('text')
                        .attr('x', 20)
                        .attr('y', -10)
                        .style('text-anchor', 'start')
                        .style('font-size', '12px')
                        .style('fill', tagText.style('fill'))
                        .text(filterStatus);

                    // 如果被过滤，添加半透明遮罩
                    if (filterStatus.includes('过滤')) {
                        dimensionGroup.style('opacity', 0.5);
                    }
                }
            }

            // 添加节点
            if (combinedData[index]) {
                const nodes = dimensionGroup.selectAll('.dimension-nodes')
                    .data(combinedData[index])
                    .enter()
                    .append('circle')
                    .attr('class', d => `dimension-nodes node-${d.id.split('/').pop()}`)
                    .attr('cx', d => xScale(d.value))
                    .attr('cy', height/2)
                    .attr('r', 4)
                    .style('fill', d => selectedNodeIds.value.includes(d.id.split('/').pop()) ? '#ff6347' : '#333')
                    .style('opacity', d => selectedNodeIds.value.includes(d.id.split('/').pop()) ? 1 : 0.6)
                    .style('cursor', 'pointer')
                    .on('mouseover', function(event, d) {
                        d3.select(this)
                            .attr('r', 6)
                            .style('fill', '#ff6347')
                            .style('opacity', 1);
                        showTooltip(`${d.id.split('/').pop()}: ${d.value.toFixed(4)}`);
                    })
                    .on('mousemove', moveTooltip)
                    .on('mouseout', function(event, d) {
                        d3.select(this)
                            .attr('r', 4)
                            .style('fill', selectedNodeIds.value.includes(d.id.split('/').pop()) ? '#ff6347' : '#333')
                            .style('opacity', selectedNodeIds.value.includes(d.id.split('/').pop()) ? 1 : 0.6);
                        hideTooltip();
                    })
                    .on('click', function(event, d) {
                        event.stopPropagation();
                        const nodeId = d.id.split('/').pop();
                        if (selectedNodeIds.value.includes(nodeId)) {
                            store.commit('REMOVE_SELECTED_NODE', nodeId);
                        } else {
                            store.commit('ADD_SELECTED_NODE', nodeId);
                        }
                    });
            }

            // 添加统计信息
            if (dimensionStats[index]) {
                const statsRow = svg.append('g')
                    .attr('class', `stats-row-${index}`)
                    .attr('transform', `translate(-400,${yPos + height/2})`);

                const stats = dimensionStats[index];
                const values = [
                    stats.variance.toFixed(4),
                    stats.range.toFixed(4),
                    stats.dispersionDegree.toFixed(4),
                    stats.localClustering.toFixed(4)
                ];

                values.forEach((value, i) => {
                    let x = 0;
                    for (let j = 0; j < i; j++) {
                        x += columnWidths[j];
                    }
                    statsRow.append('text')
                        .attr('x', x)
                        .attr('y', 8)
                        .text(value)
                        .style('font-size', '12px');
                });
            }
        });

        // 修改 zoomRect 的事件处理
        zoomRect.style('pointer-events', 'painted')
            .on('mouseover', hideTooltip);
    } catch (error) {
        console.error('渲染维度轴时出错:', error);
    }
};

// 修改 updateGMMCurve 函数
const updateGMMCurve = (component, xScale, height, dimensionData) => {
    const points = [];
    const numPoints = 100;
    const xMin = xScale.domain()[0];
    const xMax = xScale.domain()[1];
    const step = (xMax - xMin) / numPoints;

    // 计算最大高度用于归一化
    const maxHeight = component.weight / Math.sqrt(2 * Math.PI * component.variance);

    for (let i = 0; i <= numPoints; i++) {
        const x = xMin + i * step;
        const gaussian = component.weight * Math.exp(-Math.pow(x - component.mean, 2) / (2 * component.variance)) 
            / Math.sqrt(2 * Math.PI * component.variance);
        points.push([
            xScale(x), 
            height * (1 - gaussian / maxHeight * 0.8) // 使用相对高度，保留20%空间
        ]);
    }

    // 添加闭合路径的点，使其成为一个可填充的区域
    points.push([xScale(xMax), height]); // 右下角
    points.push([xScale(xMin), height]); // 左下角

    return {
        points,
        // 添加属性用于确定哪些节点属于这个组件
        containedNodes: dimensionData.filter(d => {
            // 计算节点值属于该高斯成分的概率
            const prob = Math.exp(-Math.pow(d.value - component.mean, 2) / (2 * component.variance)) 
                / Math.sqrt(2 * Math.PI * component.variance);
            // 如果概率大于阈值，认为节点属于该组件
            return prob > 0.1; // 可以调整这个阈值
        }).map(d => d.id.split('/').pop())
    };
};

// 添加计算分散程度的函数
const calculateDispersionDegree = (values) => {
    const sorted = values.sort(d3.ascending);
    const q1 = d3.quantile(sorted, 0.25);
    const q3 = d3.quantile(sorted, 0.75);
    const min = d3.min(values);
    const max = d3.max(values);
    return (q3 - q1) / (max - min);
};

// 添加监听 selectedNodes 变化的 watch
watch(selectedNodeIds, (newSelectedIds) => {
    // 更新所有维度组合中的节点状态
    d3.select(axisContainer.value)
        .selectAll('.dimension-nodes')
        .style('fill', d => newSelectedIds.includes(d.id.split('/').pop()) ? '#ff6347' : '#333')
        .style('opacity', d => newSelectedIds.includes(d.id.split('/').pop()) ? 1 : 0.6);

    // 更新第一张热力图的高亮
    d3.selectAll('.normal_chart_container .x-axis text')
        .style('fill', d => newSelectedIds.includes(d) ? 'red' : 'black')
        .style('font-weight', d => newSelectedIds.includes(d) ? 'bold' : 'normal');

    // 移除旧的高亮矩形
    d3.selectAll('.normal_chart_container .highlight-rect-normal').remove();

    // 添加新的高亮矩形
    selectedNodeIds.value.forEach(id => {
        const xPos = xScaleNormal(id);
        if (xPos !== undefined) {
            svgNormal.append('rect')
                .attr('class', 'highlight-rect-normal')
                .attr('x', xPos)
                .attr('y', 0)
                .attr('width', xScaleNormal.bandwidth())
                .attr('height', normalChartContainer.value.clientHeight - marginNormal.top - marginNormal.bottom)
                .style('fill', 'none')
                .style('stroke', 'red')
                .style('stroke-width', '1.5px');
        }
    });

    // 更新第二张热力图的高亮
    d3.selectAll('.init_chart_container .x-axis text')
        .style('fill', d => newSelectedIds.includes(d) ? 'red' : 'black')
        .style('font-weight', d => newSelectedIds.includes(d) ? 'bold' : 'normal');

    // 移除旧的高亮矩形
    d3.selectAll('.init_chart_container .highlight-rect-cc').remove();

    // 添加新的高亮矩形
    selectedNodeIds.value.forEach(id => {
        const xPos = xScaleInit(id);
        if (xPos !== undefined) {
            svgInit.append('rect')
                .attr('class', 'highlight-rect-cc')
                .attr('x', xPos)
                .attr('y', 0)
                .attr('width', xScaleInit.bandwidth())
                .attr('height', initChartContainer.value.clientHeight - marginInit.top - marginInit.bottom)
                .style('fill', 'none')
                .style('stroke', 'red')
                .style('stroke-width', '1.5px');
        }
    });

    // 更新MDS散点图中的点
    d3.selectAll('.mds-point')
        .style('fill', d => newSelectedIds.includes(d.id.split('/').pop()) ? '#ff6347' : '#333')
        .style('opacity', d => newSelectedIds.includes(d.id.split('/').pop()) ? 1 : 0.6);
}, { deep: true });

// 添加GMM拟合函数
async function fitGMM(data) {
    try {
        // 如果数据无效，返回默认单组件
        if (!data || !Array.isArray(data) || data.length === 0) {
            console.warn('无效的输入数据，返回默认GMM组件');
            return {
                components: [{
                    mean: 0,
                    variance: 0.0001,
                    weight: 1.0
                }]
            };
        }

        // 调用后端API计算GMM
        const response = await fetch('http://127.0.0.1:8000/calculate_gmm', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ values: data })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        if (!result.success || !result.components || !Array.isArray(result.components) || result.components.length === 0) {
            throw new Error(result.error || '无效的GMM结果');
        }

        // 验证每个组件都有必要的属性
        const validComponents = result.components.map(comp => ({
            mean: comp.mean || 0,
            variance: comp.variance || 0.0001,
            weight: comp.weight || 1.0/result.components.length
        }));

        return {
            components: validComponents
        };
    } catch (error) {
        console.error('Error calculating GMM:', error);
        // 如果API调用失败，返回一个简单的单组件结果
        const mean = data ? d3.mean(data) || 0 : 0;
        const variance = data ? d3.variance(data) || 0.0001 : 0.0001;
        return {
            components: [{
                mean: mean,
                variance: variance,
                weight: 1.0
            }]
        };
    }
}

// 添加计算维度相关性的函数
function calculateDimensionCorrelations(data) {
    const dimensions = 4;
    const correlations = [];
    const dimensionStats = [];

    // 计算每个维度的基本统计信息
    for (let i = 0; i < dimensions; i++) {
        const values = data.map(d => d.features[i]);
        dimensionStats.push({
            mean: d3.mean(values),
            std: Math.sqrt(d3.variance(values)),
            skewness: calculateSkewness(values),
            kurtosis: calculateKurtosis(values)
        });
    }

    // 计算维度间的相关系数
    for (let i = 0; i < dimensions; i++) {
        correlations[i] = [];
        for (let j = 0; j < dimensions; j++) {
            if (i === j) {
                correlations[i][j] = 1;
            } else {
                const values1 = data.map(d => d.features[i]);
                const values2 = data.map(d => d.features[j]);
                correlations[i][j] = calculatePearsonCorrelation(values1, values2);
            }
        }
    }

    return {
        correlations,
        dimensionStats
    };
}

// 计算偏度
function calculateSkewness(values) {
    const n = values.length;
    const mean = d3.mean(values);
    const std = Math.sqrt(d3.variance(values));
    const skewness = values.reduce((acc, val) => 
        acc + Math.pow((val - mean) / std, 3), 0) / n;
    return skewness;
}

// 计算峰度
function calculateKurtosis(values) {
    const n = values.length;
    const mean = d3.mean(values);
    const std = Math.sqrt(d3.variance(values));
    const kurtosis = values.reduce((acc, val) => 
        acc + Math.pow((val - mean) / std, 4), 0) / n - 3;
    return kurtosis;
}

// 计算皮尔逊相关系数
function calculatePearsonCorrelation(x, y) {
    const n = x.length;
    const meanX = d3.mean(x);
    const meanY = d3.mean(y);
    let numerator = 0;
    let denominatorX = 0;
    let denominatorY = 0;

    for (let i = 0; i < n; i++) {
        const xDiff = x[i] - meanX;
        const yDiff = y[i] - meanY;
        numerator += xDiff * yDiff;
        denominatorX += xDiff * xDiff;
        denominatorY += yDiff * yDiff;
    }

    return numerator / Math.sqrt(denominatorX * denominatorY);
}

// 添加局部聚集性计算函数
function calculateLocalClustering(components) {
    // 添加参数检查
    if (!components || !Array.isArray(components) || components.length === 0) {
        console.warn('无效的GMM组件数据，返回默认值0');
        return 0;
    }
    
    try {
        // 使用组件方差的倒数作为权重
        const totalWeight = components.reduce((sum, comp) => {
            // 确保方差不为0，如果为0则使用一个很小的值
            const variance = comp.variance || 0.0001;
            return sum + 1/variance;
        }, 0);

        const weightedClusteringScore = components.reduce((sum, comp) => {
            const variance = comp.variance || 0.0001;
            const weight = 1/variance / totalWeight;
            return sum + weight * (1 - Math.sqrt(variance));
        }, 0);
        
        return weightedClusteringScore;
    } catch (error) {
        console.error('计算局部聚集性时出错:', error);
        return 0;
    }
}

// 添加 MDS 散点图渲染函数
const renderMDSScatterplot = (data, containerWidth) => {
    // 清空旧的图表
    d3.select(mdsContainer.value).selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = containerWidth - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    // 创建 SVG
    const svg = d3.select(mdsContainer.value)
        .append('svg')
        .attr('width', containerWidth)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // 提取 MDS 坐标
    const mdsPoints = data.map(d => ({
        id: d.id,
        x: d.features[15], // bbox_mds_1
        y: d.features[16], // bbox_mds_2
    }));

    // 创建比例尺
    const xScale = d3.scaleLinear()
        .domain(d3.extent(mdsPoints, d => d.x))
        .range([0, width])
        .nice();

    const yScale = d3.scaleLinear()
        .domain(d3.extent(mdsPoints, d => d.y))
        .range([height, 0])
        .nice();

    // 添加 X 轴
    svg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale))
        .append('text')
        .attr('x', width / 2)
        .attr('y', 35)
        .attr('fill', 'black')
        .style('text-anchor', 'middle')
        .text('bbox_mds_1');

    // 添加 Y 轴
    svg.append('g')
        .attr('class', 'y-axis')
        .call(d3.axisLeft(yScale))
        .append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', -40)
        .attr('x', -height / 2)
        .attr('fill', 'black')
        .style('text-anchor', 'middle')
        .text('bbox_mds_2');

    // 添加标题
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', 'bold')
        .text('MDS 二维投影');

    // 创建缩放行为
    const zoom = d3.zoom()
        .scaleExtent([0.5, 20])
        .on('zoom', (event) => {
            const newXScale = event.transform.rescaleX(xScale);
            const newYScale = event.transform.rescaleY(yScale);
            
            // 更新轴
            svg.select('.x-axis').call(d3.axisBottom(newXScale));
            svg.select('.y-axis').call(d3.axisLeft(newYScale));
            
            // 更新点的位置
            svg.selectAll('.mds-point')
                .attr('cx', d => newXScale(d.x))
                .attr('cy', d => newYScale(d.y));
        });

    // 添加透明的缩放矩形
    svg.append('rect')
        .attr('class', 'zoom-rect')
        .attr('width', width)
        .attr('height', height)
        .style('fill', 'none')
        .style('pointer-events', 'all')
        .call(zoom);

    // 绘制散点
    const points = svg.selectAll('.mds-point')
        .data(mdsPoints)
        .enter()
        .append('circle')
        .attr('class', d => `mds-point node-${d.id.split('/').pop()}`)
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .attr('r', 4)
        .style('fill', d => selectedNodeIds.value.includes(d.id.split('/').pop()) ? '#ff6347' : '#333')
        .style('opacity', d => selectedNodeIds.value.includes(d.id.split('/').pop()) ? 1 : 0.6)
        .style('cursor', 'pointer');

    // 添加交互
    points.on('mouseover', function(event, d) {
        d3.select(this)
            .attr('r', 6)
            .style('fill', '#ff6347')
            .style('opacity', 1);

        d3.select(lineTooltip.value)
            .style('visibility', 'visible')
            .text(`${d.id.split('/').pop()}: (${d.x.toFixed(4)}, ${d.y.toFixed(4)})`);
    })
    .on('mousemove', function(event) {
        const [mouseX, mouseY] = d3.pointer(event);
        d3.select(lineTooltip.value)
            .style('left', `${mouseX + margin.left + 10}px`)
            .style('top', `${mouseY + margin.top - 10}px`);
    })
    .on('mouseout', function(event, d) {
        d3.select(this)
            .attr('r', 4)
            .style('fill', selectedNodeIds.value.includes(d.id.split('/').pop()) ? '#ff6347' : '#333')
            .style('opacity', selectedNodeIds.value.includes(d.id.split('/').pop()) ? 1 : 0.6);

        d3.select(lineTooltip.value)
            .style('visibility', 'hidden');
    })
    .on('click', function(event, d) {
        event.stopPropagation();
        const nodeId = d.id.split('/').pop();
        if (selectedNodeIds.value.includes(nodeId)) {
            store.commit('REMOVE_SELECTED_NODE', nodeId);
        } else {
            store.commit('ADD_SELECTED_NODE', nodeId);
        }
    });

    // 添加双击重置功能
    svg.select('.zoom-rect').on('dblclick.zoom', function() {
        svg.transition()
            .duration(750)
            .call(zoom.transform, d3.zoomIdentity);
    });
};

// 更新 watch 函数以包含 MDS 散点图的更新
watch(selectedNodeIds, (newSelectedIds) => {
    // ... 现有代码 ...

    // 更新 MDS 散点图中的点
    d3.selectAll('.mds-point')
        .style('fill', d => newSelectedIds.includes(d.id.split('/').pop()) ? '#ff6347' : '#333')
        .style('opacity', d => newSelectedIds.includes(d.id.split('/').pop()) ? 1 : 0.6);
}, { deep: true });

</script>

<style scoped>
.combined_chart_container {
    display: flex;
    flex-direction: column;
    width: 100%;
    position: relative;
}

.normal_chart_container,
.init_chart_container,
.axis_container {
    max-width: 100%;
    height: auto;
    position: relative;
    z-index: 2;
}

.lines_container {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
}

.tooltip {
    font-size: 12px;
    color: #333;
    position: fixed;
    pointer-events: none;
    padding: 5px;
    border: 1px solid #ccc;
    border-radius: 5px;
    background-color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    z-index: 9999;
}

/* 添加新的样式 */
.axis_container {
    margin-top: 20px;
}

.axis_container .domain {
    stroke: #000;
}

.axis_container .tick line {
    stroke: #000;
}

.axis_container .tick text {
    fill: #000;
}

/* 统计表格样式 */
.stats-header text {
    fill: #333;
    font-weight: bold;
}

.stats-row-0 text,
.stats-row-1 text,
.stats-row-2 text,
.stats-row-3 text {
    fill: #666;
}

.stats-header text,
[class^="stats-row-"] text {
    font-family: Arial, sans-serif;
}

.mds_container {
    max-width: 100%;
    height: auto;
    position: relative;
    z-index: 2;
    margin-top: 20px;
    background-color: white;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.mds_container .x-axis path,
.mds_container .y-axis path,
.mds_container .x-axis line,
.mds_container .y-axis line {
    stroke: #000;
}

.mds_container .x-axis text,
.mds_container .y-axis text {
    fill: #000;
    font-size: 12px;
}
</style>
