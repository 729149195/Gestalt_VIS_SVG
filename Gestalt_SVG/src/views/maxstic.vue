<template>
    <div ref="parentContainer" class="combined_chart_container">
        <div ref="linesContainer" class="lines_container"></div> <!-- 连线容器放在最下方 -->
        <div ref="normalChartContainer" class="normal_chart_container"></div>
        <div ref="initChartContainer" class="init_chart_container"></div>
        <div class="tooltip" ref="lineTooltip"></div> <!-- 全局连线工具提示 -->
    </div>
</template>

<script setup>
import { onMounted, ref, computed, watch, onUnmounted, nextTick } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';


// 获取Vuex store
const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);

// 数据源URL
const NORMAL_DATA_URL = "http://localhost:5000/normalized_init_json";
const INIT_DATA_URL = "http://localhost:5000/cluster_features";
const MAPPING_DATA_URL = "http://localhost:5000/average_equivalent_mapping";
const EQUIVALENT_WEIGHTS_URL = "http://localhost:5000/equivalent_weights_by_tag"; // 新的数据源

// 引用
const parentContainer = ref(null);
const normalChartContainer = ref(null);
const initChartContainer = ref(null);
const linesContainer = ref(null); // 连线容器引用
const lineTooltip = ref(null); // 全局连线工具提示引用

// 统一的边距设置，确保左右对齐
const marginNormal = { top: 10, right: 10, bottom: 60, left: 280 };
const marginInit = { top: 0, right: 10, bottom: 80, left: 280 };

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

// 在组件挂载后执行
onMounted(async () => {
    if (!parentContainer.value) return;

    try {
        // 并行获取四个数据源
        const [responseNormal, responseInit, responseMapping, responseEquivalentWeights] = await Promise.all([
            fetch(NORMAL_DATA_URL),
            fetch(INIT_DATA_URL),
            fetch(MAPPING_DATA_URL),
            fetch(EQUIVALENT_WEIGHTS_URL) // 获取按标签的等效权重
        ]);

        if (!responseNormal.ok || !responseInit.ok || !responseMapping.ok || !responseEquivalentWeights.ok) {
            throw new Error('网络响应有问题');
        }

        const rawDataNormal = await responseNormal.json();
        const rawDataInit = await responseInit.json();
        const rawDataMapping = await responseMapping.json(); // 映射数据
        const rawDataEquivalentWeights = await responseEquivalentWeights.json(); // 等效权重数据

        const dataNormal = processDataNormal(rawDataNormal);
        const dataInit = processDataInit(rawDataInit);
        dataMapping = rawDataMapping; // 赋值给顶层变量
        dataEquivalentWeights = rawDataEquivalentWeights; // 赋值给顶层变量

        // 从映射数据中获取 groupNames
        const groupNames = dataMapping.input_dimensions;

        // 使用 ResizeObserver 监听父容器宽度变化
        resizeObserver = new ResizeObserver(entries => {
            for (let entry of entries) {
                const newWidth = entry.contentRect.width;
                renderNormal(dataNormal, newWidth, groupNames);
                renderInit(dataInit, newWidth, dataMapping.output_dimensions);
                drawLines(dataMapping, newWidth, groupNames);
            }
        });

        resizeObserver.observe(parentContainer.value);

        // 初始渲染
        const initialWidth = parentContainer.value.clientWidth;
        renderNormal(dataNormal, initialWidth, groupNames);
        renderInit(dataInit, initialWidth, dataMapping.output_dimensions);

        await nextTick(); // 确保 DOM 已更新

        drawLines(dataMapping, initialWidth, groupNames); // 初始绘制连线

        // 添加全局鼠标释放事件监听器
        window.addEventListener('mouseup', handleGlobalMouseUp);

        // 监控 selectedNodeIds 变化，更新高亮
        watch(selectedNodeIds, (newVal) => {
            updateHighlights(newVal);
        });

    } catch (error) {
        console.error('数据获取或处理时出错:', error);
    }
});

// 在组件��载时移除全局事件监听器
onUnmounted(() => {
    window.removeEventListener('mouseup', handleGlobalMouseUp);
});

// 更新高亮显示
const updateHighlights = (selectedIds) => {
    // 更新第一张热力图的高亮
    d3.selectAll('.normal_chart_container .x-axis text')
        .style('fill', d => d && selectedIds.includes(d) ? 'red' : 'black')
        .style('font-weight', d => d && selectedIds.includes(d) ? 'bold' : 'normal');

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
        .style('fill', d => d && selectedNodeIds.value.includes(d) ? 'red' : 'black')
        .style('font-weight', d => d && selectedNodeIds.value.includes(d) ? 'bold' : 'normal');

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
        // 确保 node.features 长度为 21
        if (node.features.length !== 20) {
            console.warn(`节点 ${node.id} 的特征数量为 ${node.features.length}，预期为 21`);
        }
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
        // 确保 node.features 长度为 4
        if (node.features.length !== 4) {
            console.warn(`节点 ${node.id} 的特征数量为 ${node.features.length}，预期为 4`);
        }
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
const renderNormal = (data, containerWidth, groupNames) => {
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

    // 修改颜色比例尺为分歧颜色比例尺，固定 domain 为 [-1, 0, 1]
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
};

// 渲染第二张热力图
const renderInit = (data, containerWidth, outputDimensions) => {
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

    // 修��颜色比例尺为分歧颜色比例尺，固定 domain 为 [-1, 0, 1]
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
            const weightsForDimension = perSampleWeights[d.group]; // 长度为 21 的权重数组

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
                .on('mousedown', function (event, d) {
                    // 按下时仅显示与此 initIndex 连接的连线
                    d3.selectAll('.lines_container path')
                        .style('display', function () {
                            return this.getAttribute('data-init-index') == d ? null : 'none';
                        });
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
};

// 当第二张热力图中的矩形被点击时，绘制按样本的连线
const drawPerSampleLines = (weightsForDimension, clickedRectData, clickedRectNode) => {
    // 清除现有的按样本连线
    d3.selectAll('.per-sample-lines').remove();

    // 降低其他元素的透明度
    d3.selectAll('.lines_container path') // 现有的平均连线
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

    // 获取输入特征（第一张热力图的 Y 轴上的圆���）
    const normalCircles = d3.selectAll('.normal_chart_container .y-axis circle');
    const normalData = normalCircles.nodes(); // 21 个输入节点

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

        // 移除按样本连线
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

            // 获取父容器的位置以计算相对坐标
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

</script>

<style scoped>
.combined_chart_container {
    display: flex;
    flex-direction: column;
    width: 100%;
    position: relative; /* 允许 lines_container 绝对定位 */
}

.normal_chart_container,
.init_chart_container {
    max-width: 100%;
    height: auto;
    position: relative;
    z-index: 2; /* 设置较高的 z-index，使其在连线上方 */
}

.lines_container {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1; /* 设置较低的 z-index，使其在热力图下方 */
}

.tooltip {
    font-size: 12px;
    color: #333;
    position: absolute;
    pointer-events: none;
    background: rgba(255, 255, 255, 0.8);
    padding: 5px;
    border: 1px solid #ccc;
    border-radius: 5px;
}
</style>
