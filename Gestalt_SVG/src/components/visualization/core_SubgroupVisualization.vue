<template>
    <div class="core-graph-container">
        <!-- 修改总览条 -->
        <div class="clusters-overview">
            <!-- 添加左侧标题 -->
            <div class="overview-title">List overview</div>
            <!-- 使用wrapper-container包裹实际内容 -->
            <div class="overview-wrapper-container">
                <div class="overview-wrapper">
                    <div class="overview-content">
                        <svg class="overview-svg" ref="overviewSvg">
                            <!-- 将连接线组放在最后，确保它绘制在最上层 -->
                            <g class="clusters-group">
                                <g v-for="(node, index) in flattenedNodes" :key="`overview-${node.id}`" class="cluster-item" :class="{
                                    'cluster-core': node.type === 'core',
                                    'cluster-extension': node.type === 'extension'
                                    /* 移除选中状态的类，保持矩形样式不变 */
                                    /* 'cluster-selected': isNodeSelected(node) */
                                }" :data-node-id="node.id" :data-cluster-id="node.clusterId" :data-index="index">
                                    <rect :width="clusterItemSize" :height="clusterItemSize" :x="index * (clusterItemSize + clusterItemGap)" :y="20" rx="3" ry="3" :style="{
                                        fill: getSalienceColor(node),
                                        stroke: 'none' /* 移除边框 */
                                    }" @click="handleOverviewClick(node.id, $event)" />
                                    <!-- 移除小圆点标记 -->
                                </g>
                            </g>
                            <g class="connection-lines"></g>
                        </svg>
                        <!-- 添加自定义滚动条到总览条内部 -->
                        <div class="overview-scrollbar-container">
                            <div class="custom-scrollbar-track" @mousedown="startScrollbarDrag">
                                <div class="custom-scrollbar-thumb" ref="scrollbarThumb" :style="{ width: thumbWidth + 'px', left: thumbPosition + 'px' }"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div v-if="loading" class="loading-overlay">
            <div class="loading-spinner"></div>
        </div>
        <div class="cards-container" ref="cardsContainer">
            <div class="cards-wrapper" ref="cardsWrapper" @mousedown="startDrag" @mousemove="onDrag" @mouseup="endDrag" @mouseleave="endDrag" @scroll="handleScroll">
                <div v-for="node in flattenedNodes" :key="node.id" class="card" :class="{
                    'card-selected': isNodeSelected(node),
                    'card-core': node.type === 'core',
                    'card-extension': node.type === 'extension',
                    'has-extension': hasExtension(node)
                }" :data-type="node.type" :data-node-id="node.id" :data-cluster-id="node.clusterId" @click.stop="handleCardClick(node, $event)">

                    <!-- 移除扩展指示器和导航控制按钮 -->

                    <div class="card-svg-container" ref="graphContainer"></div>
                    <div class="card-info">
                        <div class="highlight-stats">
                            <template v-if="Object.keys(getStats(node)).length > 0">
                                Included elements: <span v-for="(count, type) in getStats(node)" :key="type">
                                    {{ count }} <{{ type }}>
                                </span>
                            </template>
                        </div>
                        <div class="encodings-wrapper">
                            <div class="visual-encodings">
                                Used visual encodings:
                            </div>
                            <div class="analysis-content" v-html="generateAnalysis(node)"></div>
                        </div>
                        <div class="attention-probability">
                            <span class="attention-probability-label">Visual salience</span>
                            <span class="attention-probability-value">{{ (calculateAttentionProbability(node) * 100).toFixed(3) }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch, onUnmounted, computed } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
import axios from 'axios';

const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);
const graphContainer = ref(null);
const cardsContainer = ref(null);
const cardsWrapper = ref(null);
const scrollbarThumb = ref(null);
const graphData = ref(null);
const nodes = ref([]);
const originalSvgContent = ref('');
const loading = ref(true);
const currentPages = ref(new Map());
const elementStats = ref(new Map());

// 修改总览条相关的变量
const overviewSvg = ref(null);
const clusterItemSize = 32; // 小方块高度和宽度相同，形成正方形
const clusterItemGap = 8;  // 小方块之间的间隔为3px
const overviewHeight = 50; // 总览条高度
// 不再需要单独的itemWidth变量，使用clusterItemSize代替

// 添加滚动条相关的状态
const thumbWidth = ref(100);
const thumbPosition = ref(0);
const isScrollbarDragging = ref(false);
const scrollbarStartX = ref(0);
const scrollbarInitialLeft = ref(0);

// 添加颜色缓存映射
const nodeColorCache = ref(new Map());

// 添加缩略图缓存
const thumbnailCache = new Map();

// 添加拖动相关的状态
const isDragging = ref(false);
const wasRecentlyDragging = ref(false);  // 新增：用于跟踪最近是否有拖动行为
const startX = ref(0);
const scrollLeft = ref(0);
const lastX = ref(0);
const velocity = ref(0);
const animationFrame = ref(null);
const dragEndTimeout = ref(null);  // 新增：用于拖动结束后的延时器

// 首先添加一个标志变量来控制是否需要重新渲染
const isInitialRender = ref(true);
const isDataUpdated = ref(false);

// 添加滚动阴影状态变量
const showLeftShadow = ref(false);
const showRightShadow = ref(true); // 默认显示右侧阴影，提示有更多内容

// 添加新的计算属性，将所有节点（核心+外延）扁平化展示
const flattenedNodes = computed(() => {
    if (!nodes.value) return [];

    // 创建所有节点的扁平化数组（核心节点和独立的外延节点）
    let allNodes = [];

    nodes.value.forEach(coreNode => {
        // 添加核心节点
        allNodes.push({
            ...coreNode,
            clusterId: coreNode.id.split('_')[1] // 记录聚类ID
        });

        // 添加关联的外延节点
        if (coreNode.extensions && coreNode.extensions.length > 0) {
            coreNode.extensions.forEach((ext, extIndex) => {
                allNodes.push({
                    ...ext,
                    clusterId: coreNode.id.split('_')[1], // 记录聚类ID
                    extIndex: extIndex // 记录在扩展数组中的索引
                });
            });
        }
    });

    // 计算每个节点的显著性分数和总面积
    const nodeScores = allNodes.map(node => {
        // 获取原始分数用于排序
        const rawScore = calculateAttentionProbability(node, true);
        const totalArea = calculateTotalArea(node);
        return { node, rawScore, totalArea };
    });

    // 按照修改后的排序逻辑排序
    nodeScores.sort((a, b) => {
        // 如果显著性分数差值不超过0.03 (3%)，则按面积排序
        if (Math.abs(a.rawScore - b.rawScore) <= 0.0) {
            return b.totalArea - a.totalArea; // 面积大的排前面
        }
        // 否则按显著性分数排序
        return b.rawScore - a.rawScore;
    });

    // 返回排序后的节点数组
    return nodeScores.map(item => item.node);
});

// 添加计算节点总面积的函数
function calculateTotalArea(nodeData) {
    if (!normalizedData.value || normalizedData.value.length === 0) return 0;

    try {
        let nodesToAnalyze = [];

        if (nodeData.type === 'extension') {
            // 对于外延节点，同时分析核心节点的内容
            const coreIndex = parseInt(nodeData.clusterId);
            const coreNode = nodes.value.find(n => n.id === `core_${coreIndex}`);
            if (coreNode) {
                nodesToAnalyze.push(...coreNode.originalNodes);
            }
            // 再添加外延节点内容
            nodesToAnalyze.push(...nodeData.originalNodes);
        } else {
            // 核心节点只分析自身
            nodesToAnalyze = [...nodeData.originalNodes];
        }

        if (nodesToAnalyze.length === 0) return 0;

        // 提取高亮元素的特征
        const highlightedFeatures = [];

        // 遍历normalized数据
        normalizedData.value.forEach(item => {
            // 标准化ID格式以便比较
            const normalizedItemId = item.id;

            // 检查当前元素是否是高亮元素
            const isHighlighted = nodesToAnalyze.some(analyzeNode => {
                // 移除开头的 'svg/' 并标准化分析节点的路径
                const normalizedAnalyzeNode = analyzeNode.replace(/^svg\//, '');
                return normalizedItemId === `svg/${normalizedAnalyzeNode}` ||
                    normalizedItemId === normalizedAnalyzeNode;
            });

            if (isHighlighted) {
                highlightedFeatures.push(item.features);
            }
        });

        if (highlightedFeatures.length === 0) return 0;

        // 计算总面积 - bbox_fill_area 在特征向量中的索引是19
        const AREA_INDEX = 19;
        const totalArea = highlightedFeatures.reduce((sum, features) =>
            sum + features[AREA_INDEX], 0);

        return totalArea;
    } catch (error) {
        console.error('Error calculating total area:', error);
        return 0;
    }
}

// 添加扩展相关的方法
const hasExtension = (node) => {
    if (node.type === 'core') {
        return node.extensions && node.extensions.length > 0;
    }
    return false;
};

const getExtensionCount = (node) => {
    return node.extensions ? node.extensions.length : 0;
};

// 导航相关的函数
const hasRelatedCards = (node) => {
    if (node.type === 'core') {
        return hasExtension(node);
    } else if (node.type === 'extension') {
        // 外延节点总是有关联卡片（至少有核心节点）
        return true;
    }
    return false;
};

const getRelatedCardIds = (node) => {
    const clusterId = node.clusterId;
    if (!clusterId) return [];

    // 找出同一聚类中的所有节点ID
    const relatedIds = flattenedNodes.value
        .filter(n => n.clusterId === clusterId)
        .map(n => n.id);

    return relatedIds;
};

const getCurrentCardIndex = (node) => {
    const relatedIds = getRelatedCardIds(node);
    const index = relatedIds.indexOf(node.id);
    return index !== -1 ? index + 1 : 1;
};

const getTotalRelatedCards = (node) => {
    return getRelatedCardIds(node).length;
};

const hasPrevCard = (node) => {
    return getCurrentCardIndex(node) > 1;
};

const hasNextCard = (node) => {
    return getCurrentCardIndex(node) < getTotalRelatedCards(node);
};

const navigateToPrevCard = (node) => {
    // 无需检查拖动状态，导航按钮应始终可用
    if (!hasPrevCard(node)) return;

    const relatedIds = getRelatedCardIds(node);
    const currentIndex = relatedIds.indexOf(node.id);
    if (currentIndex > 0) {
        scrollToNodeId(relatedIds[currentIndex - 1]);
    }
};

const navigateToNextCard = (node) => {
    // 无需检查拖动状态，导航按钮应始终可用
    if (!hasNextCard(node)) return;

    const relatedIds = getRelatedCardIds(node);
    const currentIndex = relatedIds.indexOf(node.id);
    if (currentIndex < relatedIds.length - 1) {
        scrollToNodeId(relatedIds[currentIndex + 1]);
    }
};

// 修改scrollToNodeId方法，确保点击后能正确滚动到对应卡片，但不影响原有样式
const scrollToNodeId = (nodeId) => {
    const cardElement = document.querySelector(`[data-node-id="${nodeId}"]`);
    if (!cardElement || !cardsWrapper.value) {
        console.error('Card element or wrapper not found');
        return;
    }

    try {
        // 标记正在进行跳转以防止其他拖动事件干扰
        wasRecentlyDragging.value = true;

        // 获取卡片在滚动容器中的位置
        const container = cardsWrapper.value;
        const cardLeft = cardElement.offsetLeft;
        const containerWidth = container.clientWidth;

        // 计算目标滚动位置（居中显示）
        const targetScrollLeft = cardLeft - (containerWidth / 2) + (cardElement.offsetWidth / 2);

        // 使用平滑滚动
        container.scrollTo({
            left: Math.max(0, targetScrollLeft),
            behavior: 'smooth'
        });

        // 使用临时样式创建闪烁效果，而不是修改卡片本身的类
        const tempHighlight = document.createElement('div');
        tempHighlight.className = 'temp-card-highlight';
        tempHighlight.style.position = 'absolute';
        tempHighlight.style.top = `${cardElement.offsetTop}px`;
        tempHighlight.style.left = `${cardElement.offsetLeft}px`;
        tempHighlight.style.width = `${cardElement.offsetWidth}px`;
        tempHighlight.style.height = `${cardElement.offsetHeight}px`;
        tempHighlight.style.borderRadius = '12px';
        tempHighlight.style.pointerEvents = 'none';
        tempHighlight.style.zIndex = '10';
        tempHighlight.style.animation = 'pulse-highlight-subtle 1.5s ease-in-out forwards';

        container.appendChild(tempHighlight);

        // 移除临时高亮元素
        setTimeout(() => {
            if (tempHighlight.parentNode) {
                tempHighlight.parentNode.removeChild(tempHighlight);
            }
        }, 1500);

        // 滚动完成后重置状态
        setTimeout(() => {
            wasRecentlyDragging.value = false;
        }, 600);
    } catch (error) {
        console.error('Error scrolling to card:', error);
        wasRecentlyDragging.value = false;
    }
};

// 修改updateOverview方法，动态计算方块宽度
const updateOverview = () => {
    if (!overviewSvg.value || !flattenedNodes.value.length) return;

    try {
        // 获取容器宽度
        const overviewWrapper = overviewSvg.value.closest('.overview-wrapper');
        if (!overviewWrapper) return;

        // 计算总宽度 - 使用正方形尺寸
        const nodeCount = flattenedNodes.value.length;
        const totalGapWidth = (nodeCount - 1) * clusterItemGap;
        const totalWidth = nodeCount * clusterItemSize + totalGapWidth;

        // 设置SVG的宽度
        overviewSvg.value.setAttribute('width', `${totalWidth}px`);
        overviewSvg.value.setAttribute('height', `${overviewHeight}px`);
        
        // 设置滚动条容器的宽度与SVG宽度相同
        const overviewContent = overviewSvg.value.closest('.overview-content');
        if (overviewContent) {
            overviewContent.style.width = `${totalWidth}px`;
        }

        // 绘制连接线
        drawConnectionLines();

        // 标记当前可见的卡片
        updateVisibleCards();
    } catch (error) {
        console.error('Error updating overview:', error);
    }
};

// 添加绘制连接线的方法
const drawConnectionLines = () => {
    if (!overviewSvg.value || !flattenedNodes.value.length) return;

    try {
        // 清除现有的连接线
        const linesGroup = d3.select(overviewSvg.value).select('.connection-lines');
        linesGroup.selectAll('*').remove();

        // 创建一个映射，存储每个核心节点的索引
        const coreNodeIndices = new Map();
        flattenedNodes.value.forEach((node, index) => {
            if (node.type === 'core') {
                coreNodeIndices.set(node.id, index);
            }
        });

        // 获取SVG容器的高度
        const svgHeight = overviewSvg.value.clientHeight || 50;
        
        // 计算最大可用高度（留出一些边距）
        const maxAvailableHeight = svgHeight - 5; // 5px的安全边距
        
        // 为每个外延节点绘制连接线
        flattenedNodes.value.forEach((node, index) => {
            if (node.type === 'extension') {
                // 找到对应的核心节点
                const coreNodeId = `core_${node.clusterId}`;
                const coreIndex = coreNodeIndices.get(coreNodeId);

                if (coreIndex !== undefined) {
                    // 计算外延节点矩形的顶部中心点
                    const extX = index * (clusterItemSize + clusterItemGap) + clusterItemSize / 2;
                    const extY = 20; // 矩形的顶部y坐标

                    // 计算核心节点矩形的顶部中心点
                    const coreX = coreIndex * (clusterItemSize + clusterItemGap) + clusterItemSize / 2;
                    const coreY = 20; // 矩形的顶部y坐标

                    // 计算两点之间的距离
                    const distance = Math.abs(extX - coreX);
                    
                    // 计算垂直线段的高度，确保不会超出容器
                    const minOffset = 8;
                    const maxOffset = Math.min(25, maxAvailableHeight - 5); // 确保不超出容器
                    const normalizedDistance = Math.min(distance / 200, 1);
                    const verticalOffset = Math.min(
                        minOffset + Math.round(normalizedDistance * normalizedDistance * (maxOffset - minOffset)),
                        maxAvailableHeight - 5
                    );
                    
                    // 创建一个圆角方形路径
                    const pathData = createArcPath(coreX, coreY, extX, extY, verticalOffset);

                    // 添加路径到SVG
                    linesGroup.append('path')
                        .attr('d', pathData)
                        .attr('fill', 'none')
                        .attr('stroke', '#905F29')
                        .attr('stroke-width', 1.5)
                        .attr('stroke-opacity', 0.7) // 降低不透明度
                        .attr('data-ext-id', node.id)
                        .attr('data-core-id', coreNodeId)
                        .style('pointer-events', 'none') // 禁用鼠标事件，防止干扰
                        .style('overflow', 'visible'); // 确保内容可以超出边界
                    
                    // 计算箭头位置 - 在水平线段的中点
                    const midPoint = calculateArcMidPoint(coreX, coreY, extX, extY, verticalOffset);
                    
                    // 计算箭头方向 - 从核心指向外延
                    const arrowAngle = calculateArrowAngle(coreX, coreY, extX, extY);
                    
                    // 添加箭头 - 始终显示箭头
                    linesGroup.append('polygon')
                        .attr('points', '0,-3 6,0 0,3') // 调整箭头大小
                        .attr('fill', '#905F29')
                        .attr('transform', `translate(${midPoint.x}, ${midPoint.y}) rotate(${arrowAngle})`)
                        .style('pointer-events', 'none');
                }
            }
        });

        // 添加RevelioGood聚类之间的连线
        drawRevelioGoodClusterConnections(linesGroup, coreNodeIndices, maxAvailableHeight);

        // 简化节点的鼠标事件处理
        d3.selectAll('.cluster-item').on('mouseover', null).on('mouseout', null);
    } catch (error) {
        console.error('Error drawing connection lines:', error);
    }
};

// 添加新函数: 绘制RevelioGood聚类之间的连线
const drawRevelioGoodClusterConnections = (linesGroup, coreNodeIndices, maxAvailableHeight) => {
    try {
        // 找出所有RevelioGood相关节点，包括显式标记的和standalone聚类
        const revelioGoodNodes = flattenedNodes.value.filter(node => 
            node.type === 'core' && (
                node.isRevelioGood === true || 
                (node.id && node.id.includes('standalone')) ||
                (node.groupKey && (
                    node.groupKey.includes('revelioGood') || 
                    node.groupKey.includes('reveliogood')
                ))
            )
        );
        
        if (revelioGoodNodes.length <= 1) {
            // 如果只有一个或没有revelioGood节点，则无需绘制连线
            return;
        }
        
        console.log(`Found ${revelioGoodNodes.length} revelioGood nodes for overlap analysis (including standalone)`);
        
        // 所有节点的列表
        const allNodes = revelioGoodNodes;
        
        // 计算所有节点对之间的重叠关系
        const overlaps = [];
        for (let i = 0; i < allNodes.length; i++) {
            for (let j = i + 1; j < allNodes.length; j++) {
                const nodeA = allNodes[i];
                const nodeB = allNodes[j];
                const overlapResult = calculateNodeOverlap(nodeA, nodeB);
                
                // 如果重叠度达到阈值
                if (overlapResult.overlapPercentage >= 0.8) {
                    overlaps.push({
                        nodeAId: nodeA.id,
                        nodeBId: nodeB.id,
                        overlapPercentage: overlapResult.overlapPercentage
                    });
                }
            }
        }
        
        console.log(`Found ${overlaps.length} node pairs with >80% overlap`);
        
        // 构建节点之间的连接图
        const graph = {};
        allNodes.forEach(node => {
            graph[node.id] = {
                node: node,
                connections: []
            };
        });
        
        // 填充连接信息
        overlaps.forEach(overlap => {
            graph[overlap.nodeAId].connections.push(overlap.nodeBId);
            graph[overlap.nodeBId].connections.push(overlap.nodeAId);
        });
        
        // 使用BFS找出所有连通分量（重叠组）
        const visited = new Set();
        const overlapGroups = [];
        
        for (const nodeId in graph) {
            if (visited.has(nodeId)) continue;
            
            // 新的连通分量
            const group = [];
            const queue = [nodeId];
            visited.add(nodeId);
            
            // BFS遍历
            while (queue.length > 0) {
                const currentId = queue.shift();
                group.push(graph[currentId].node);
                
                // 访问所有未访问的邻居
                for (const neighborId of graph[currentId].connections) {
                    if (!visited.has(neighborId)) {
                        visited.add(neighborId);
                        queue.push(neighborId);
                    }
                }
            }
            
            // 只有当组内至少有两个节点时才添加
            if (group.length >= 2) {
                overlapGroups.push(group);
            }
        }
        
        console.log(`Identified ${overlapGroups.length} overlap groups`);
        
        // 为每个组确定唯一的核心
        const relationsToDraw = [];
        
        overlapGroups.forEach(group => {
            // 按原始节点数量排序，找出节点数量最少的作为核心
            group.sort((a, b) => {
                const aSize = a.originalNodes?.length || 0;
                const bSize = b.originalNodes?.length || 0;
                return aSize - bSize;
            });
            
            // 组中第一个节点（节点数量最少的）作为核心
            const coreNode = group[0];
            
            // 其余节点作为外延
            for (let i = 1; i < group.length; i++) {
                relationsToDraw.push({
                    coreNode: coreNode,
                    extNode: group[i]
                });
            }
        });
        
        console.log(`Generated ${relationsToDraw.length} core-extension relations`);
        
        // 为每个关系绘制连线
        relationsToDraw.forEach(relation => {
            const coreNodeId = relation.coreNode.id;
            const extNodeId = relation.extNode.id;
            
            // 获取节点在flattenedNodes中的索引
            const coreIndex = flattenedNodes.value.findIndex(node => node.id === coreNodeId);
            const extIndex = flattenedNodes.value.findIndex(node => node.id === extNodeId);
            
            if (coreIndex !== -1 && extIndex !== -1) {
                // 计算外延节点矩形的顶部中心点
                const extX = extIndex * (clusterItemSize + clusterItemGap) + clusterItemSize / 2;
                const extY = 20; // 矩形的顶部y坐标

                // 计算核心节点矩形的顶部中心点
                const coreX = coreIndex * (clusterItemSize + clusterItemGap) + clusterItemSize / 2;
                const coreY = 20; // 矩形的顶部y坐标

                // 计算两点之间的距离
                const distance = Math.abs(extX - coreX);
                
                // 计算垂直线段的高度，确保不会超出容器
                const minOffset = 8;
                const maxOffset = Math.min(25, maxAvailableHeight - 5); // 确保不超出容器
                const normalizedDistance = Math.min(distance / 200, 1);
                const verticalOffset = Math.min(
                    minOffset + Math.round(normalizedDistance * normalizedDistance * (maxOffset - minOffset)),
                    maxAvailableHeight - 5
                );
                
                // 创建一个圆角方形路径，与API聚类连线样式完全一致
                const pathData = createArcPath(coreX, coreY, extX, extY, verticalOffset);

                // 添加路径到SVG，样式与API聚类连线相同
                linesGroup.append('path')
                    .attr('d', pathData)
                    .attr('fill', 'none')
                    .attr('stroke', '#905F29') // 相同颜色
                    .attr('stroke-width', 1.5)
                    .attr('stroke-opacity', 0.7) // 相同透明度
                    .attr('data-ext-id', extNodeId)
                    .attr('data-core-id', coreNodeId)
                    .attr('data-reveliogood', 'true')
                    .style('pointer-events', 'none')
                    .style('overflow', 'visible');
                
                // 计算箭头位置 - 在水平线段的中点
                const midPoint = calculateArcMidPoint(coreX, coreY, extX, extY, verticalOffset);
                
                // 计算箭头方向 - 从核心指向外延
                const arrowAngle = calculateArrowAngle(coreX, coreY, extX, extY);
                
                // 添加箭头 - 与API聚类完全相同的样式
                linesGroup.append('polygon')
                    .attr('points', '0,-3 6,0 0,3') // 相同的箭头大小
                    .attr('fill', '#905F29') // 相同颜色
                    .attr('transform', `translate(${midPoint.x}, ${midPoint.y}) rotate(${arrowAngle})`)
                    .attr('data-reveliogood', 'true')
                    .style('pointer-events', 'none');
            }
        });
        
    } catch (error) {
        console.error('Error drawing revelioGood cluster connections:', error);
    }
};

// 添加新函数: 计算两个节点之间的重叠度
const calculateNodeOverlap = (nodeA, nodeB) => {
    // 节点A和节点B的原始节点数组
    const nodesA = nodeA.originalNodes || [];
    const nodesB = nodeB.originalNodes || [];
    
    // 确保两个节点数组都有元素
    if (nodesA.length === 0 || nodesB.length === 0) {
        return { 
            overlapPercentage: 0, 
            smallerNode: nodesA.length <= nodesB.length ? nodeA : nodeB,
            largerNode: nodesA.length <= nodesB.length ? nodeB : nodeA
        };
    }
    
    // 为了进行比较，提取每个id的最后部分（去掉路径部分）
    const extractLastPart = (id) => {
        // 确保id是字符串
        if (typeof id !== 'string') {
            return '';
        }
        return id.split('/').pop();
    };
    
    // 处理nodeA的ID，去掉路径部分
    const processedNodesA = nodesA.map(extractLastPart).filter(id => id);
    
    // 处理nodeB的ID，去掉路径部分
    const processedNodesB = nodesB.map(extractLastPart).filter(id => id);
    
    // 计算交集
    const intersection = processedNodesA.filter(id => processedNodesB.includes(id));
    
    // 确定节点中元素较少的作为参考
    const referenceNodes = processedNodesA.length <= processedNodesB.length ? processedNodesA : processedNodesB;
    
    // 计算重叠百分比 (交集大小 / 较小节点的大小)
    const overlapPercentage = referenceNodes.length > 0 ? intersection.length / referenceNodes.length : 0;
    
    return {
        overlapPercentage,
        // 确定哪个是核心节点（节点较少的）和外延节点（节点较多的）
        smallerNode: processedNodesA.length <= processedNodesB.length ? nodeA : nodeB,
        largerNode: processedNodesA.length <= processedNodesB.length ? nodeB : nodeA
    };
};

// 创建节点的缩略图
function createThumbnail(nodeData) {
    const cacheKey = JSON.stringify(nodeData.originalNodes);
    if (thumbnailCache.has(cacheKey)) {
        return thumbnailCache.get(cacheKey);
    }

    try {
        const parser = new DOMParser();
        const svgDoc = parser.parseFromString(originalSvgContent.value, 'image/svg+xml');
        const svgElement = svgDoc.querySelector('svg');

        if (!svgElement) {
            console.error('SVG element not found');
            return '';
        }

        const clonedSvg = svgElement.cloneNode(true);

        clonedSvg.querySelectorAll('*').forEach(el => {
            if (el.tagName !== 'svg' && el.tagName !== 'g') {
                el.style.opacity = '0.05';
                if (el.hasAttribute('fill')) {
                    el.style.fill = el.getAttribute('fill');
                }
                if (el.hasAttribute('stroke')) {
                    el.style.stroke = el.getAttribute('stroke');
                }
            }
        });

        let nodesToHighlight = [];

        if (nodeData.type === 'extension') {
            // 恢复同时显示核心节点和外延节点的元素
            const coreIndex = parseInt(nodeData.clusterId);
            const coreNode = nodes.value.find(n => n.id === `core_${coreIndex}`);
            if (coreNode) {
                // 先添加核心节点
                nodesToHighlight.push(...coreNode.originalNodes);
            }
            // 再添加扩展节点
            nodesToHighlight.push(...nodeData.originalNodes);
        } else {
            nodesToHighlight = [...nodeData.originalNodes];
        }

        let highlightedCount = 0;
        nodesToHighlight.forEach(nodeId => {
            const element = clonedSvg.getElementById(nodeId.split('/').pop());
            if (element) {
                element.style.opacity = '1';
                highlightedCount++;
                if (element.hasAttribute('fill')) {
                    element.style.fill = element.getAttribute('fill');
                }
                if (element.hasAttribute('stroke')) {
                    element.style.stroke = element.getAttribute('stroke');
                }
            }
        });

        if (highlightedCount === 0) {
            clonedSvg.querySelectorAll('*').forEach(el => {
                if (el.tagName !== 'svg' && el.tagName !== 'g') {
                    el.style.opacity = '1';
                }
            });
        }

        clonedSvg.setAttribute('width', '100%');
        clonedSvg.setAttribute('height', '100%');
        clonedSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

        const thumbnail = clonedSvg.outerHTML;
        thumbnailCache.set(cacheKey, thumbnail);
        return thumbnail;
    } catch (error) {
        console.error('Error creating thumbnail:', error);
        return '';
    }
}

// 处理核心聚类数据
function processGraphData(coreData, revelioGoodGroups = new Map(), revelioBadGroups = new Map()) {
    const processedNodes = [];
    
    // 首先收集所有API聚类的元素ID集合
    const apiClusters = [];
    
    if (coreData && coreData.core_clusters) {
        coreData.core_clusters.forEach((cluster, clusterIndex) => {
            // 收集核心聚类的元素
            if (cluster.core_nodes && cluster.core_nodes.length > 0) {
                apiClusters.push({
                    id: `core_${clusterIndex}`,
                    elements: new Set(cluster.core_nodes),
                    data: {
                        id: `core_${clusterIndex}`,
                        name: `Core ${clusterIndex + 1} (Z_${cluster.core_dimensions.join(',Z_')})`,
                        type: 'core',
                        originalNodes: cluster.core_nodes,
                        dimensions: cluster.core_dimensions,
                        extensionCount: cluster.extensions.length,
                        extensions: [],
                        value: 1
                    }
                });
            }
            
            // 收集扩展节点的元素
            if (cluster.extensions) {
                cluster.extensions.forEach((extension, extIndex) => {
                    if (extension.nodes && extension.nodes.length > 0) {
                        apiClusters.push({
                            id: `ext_${clusterIndex}_${extIndex}`,
                            elements: new Set(extension.nodes),
                            parentId: `core_${clusterIndex}`,
                            coreId: clusterIndex,
                            data: {
                                id: `ext_${clusterIndex}_${extIndex}`,
                                name: `Extension ${clusterIndex + 1}.${extIndex + 1}`,
                                type: 'extension',
                                originalNodes: extension.nodes,
                                dimensions: extension.dimensions,
                                parentCoreId: `core_${clusterIndex}`,
                                value: 1
                            }
                        });
                    }
                });
            }
        });
    }
    
    // 收集所有reveliogood组
    const revelioGoodClusters = [];
    
    if (revelioGoodGroups instanceof Map) {
        console.log(`共发现 ${revelioGoodGroups.size} 个RevelioGood组`);
        revelioGoodGroups.forEach((elementIds, groupKey) => {
            // 对每个组内元素进行去重
            const uniqueElementIds = [...new Set(elementIds)];
            if (uniqueElementIds.length > 0) {
                // 显示类型名称，根据组类型显示不同的名称
                let displayName = '';
                
                // 不再创建或处理"reveliogood_all"组
                if (groupKey === 'reveliogood_all') {
                    // 跳过"all"组，不创建包含所有元素的大组
                    console.log(`跳过All组: reveliogood_all, 包含${uniqueElementIds.length}个元素`);
                    return; // 跳出当前迭代
                } else if (groupKey === 'reveliogood_basic') {
                    displayName = 'Basic RevelioGood';
                } else if (groupKey.startsWith('reveliogood_')) {
                    displayName = groupKey.replace('reveliogood_', 'RevelioGood_');
                } else {
                    displayName = 'RevelioGood';
                }
                
                revelioGoodClusters.push({
                    id: `revelioGood_${groupKey}`,
                    elements: new Set(uniqueElementIds),
                    groupKey: groupKey,
                    displayName: displayName
                });
            }
        });
    } else if (Array.isArray(revelioGoodGroups) && revelioGoodGroups.length > 0) {
        // 处理数组格式的reveliogood元素
        console.log(`跳过数组格式的所有元素，不创建All组，元素数量: ${revelioGoodGroups.length}`);
    }
    
    // ==== 实现新的过滤逻辑 ====
    
    // 1. 在RevelioGood聚类内部进行去重
    console.log("第1步：RevelioGood聚类内部去重");
    const uniqueRevelioGoodClusters = [];
    const revelioGoodElementsMap = new Map(); // 用于存储已处理的元素集合
    
    revelioGoodClusters.forEach(cluster => {
        const elementsKey = [...cluster.elements].sort().join(',');
        if (!revelioGoodElementsMap.has(elementsKey)) {
            // 如果这个元素集合不存在，添加到去重后的结果中
            revelioGoodElementsMap.set(elementsKey, cluster);
            uniqueRevelioGoodClusters.push(cluster);
        } else {
            // 如果已存在相同元素集合的聚类，选择保留的聚类
            const existingCluster = revelioGoodElementsMap.get(elementsKey);
            
            // 优先级：reveliogood_n > reveliogood_basic > 其他
            if (cluster.groupKey.match(/^reveliogood_\d+$/) && 
                (existingCluster.groupKey === 'reveliogood_basic' || 
                 !existingCluster.groupKey.match(/^reveliogood_\d+$/))) {
                // 如果当前是 reveliogood_n 并且现有的不是，替换
                revelioGoodElementsMap.set(elementsKey, cluster);
                // 找到并替换 uniqueRevelioGoodClusters 中的现有聚类
                const index = uniqueRevelioGoodClusters.findIndex(c => c.id === existingCluster.id);
                if (index !== -1) {
                    uniqueRevelioGoodClusters[index] = cluster;
                }
                console.log(`RevelioGood内部去重：替换 "${existingCluster.displayName}" 为优先级更高的 "${cluster.displayName}"`);
            } else {
                console.log(`RevelioGood内部去重：跳过重复聚类 "${cluster.displayName}"，保留已存在的 "${existingCluster.displayName}"`);
            }
        }
    });
    
    console.log(`RevelioGood内部去重：原有 ${revelioGoodClusters.length} 个聚类，去重后 ${uniqueRevelioGoodClusters.length} 个聚类`);
    
    // 2. 在API聚类内部去重（外延聚类之间去重，但与所属的核心聚类不去重）
    console.log("第2步：API聚类内部去重");
    const uniqueApiClusters = [];
    const apiCoreMap = new Map();  // 存储核心聚类
    const apiExtensionMap = new Map();  // 存储所有外延聚类的元素集合
    const apiElementsMap = new Map();  // 存储所有元素集合与对应的聚类ID
    
    // 首先处理所有外延聚类
    apiClusters.forEach(cluster => {
        if (cluster.parentId) {
            // 这是外延聚类
            const elementsKey = [...cluster.elements].sort().join(',');
            
            if (!apiExtensionMap.has(elementsKey)) {
                // 如果这个元素集合尚未处理，添加到去重后的结果
                apiExtensionMap.set(elementsKey, cluster);
                apiElementsMap.set(elementsKey, cluster.id);
                uniqueApiClusters.push(cluster);
            } else {
                // 如果已存在相同元素集合的外延聚类
                const existingCluster = apiExtensionMap.get(elementsKey);
                
                // 两个外延聚类属于同一个核心聚类时，只保留一个
                if (existingCluster.coreId === cluster.coreId) {
                    console.log(`API内部去重：跳过属于同一核心聚类的重复外延聚类 "${cluster.data.name}"`);
                } else {
                    console.log(`API内部去重：发现不同核心聚类的重复外延聚类 "${cluster.data.name}" 和 "${existingCluster.data.name}"，保留第一个`);
                }
            }
        }
    });
    
    // 然后处理所有核心聚类，检查是否与外延聚类重复
    apiClusters.forEach(cluster => {
        if (!cluster.parentId) {
            // 这是核心聚类
            const elementsKey = [...cluster.elements].sort().join(',');
            
            // 检查是否与已处理的外延聚类元素重复
            if (apiElementsMap.has(elementsKey)) {
                // 如果核心聚类与某个外延聚类重复，优先保留外延聚类
                const duplicateExtensionId = apiElementsMap.get(elementsKey);
                console.log(`API内部去重：核心聚类 "${cluster.data.name}" 与外延聚类重复，优先保留外延聚类 "${duplicateExtensionId}"`);
                
                // 检查该核心聚类是否有自己的外延聚类
                const hasOwnExtensions = apiClusters.some(ext => ext.parentId === cluster.id);
                
                if (hasOwnExtensions) {
                    // 如果核心聚类有自己的外延聚类，仍然保留该核心聚类，但标记它，后续可能需要特殊处理
                    console.log(`API内部去重：尽管 "${cluster.data.name}" 与外延聚类重复，但因为它有自己的外延聚类，所以保留`);
                    cluster.hasDuplicateExtension = true;
                    apiCoreMap.set(cluster.id, cluster);
                    uniqueApiClusters.push(cluster);
                } else {
                    // 如果核心聚类没有自己的外延聚类，不保留该核心聚类
                    console.log(`API内部去重：核心聚类 "${cluster.data.name}" 没有自己的外延聚类，被外延聚类替代`);
                }
            } else {
                // 如果核心聚类不与任何外延聚类重复，正常添加
                apiCoreMap.set(cluster.id, cluster);
                apiElementsMap.set(elementsKey, cluster.id);
                uniqueApiClusters.push(cluster);
            }
        }
    });
    
    console.log(`API内部去重：原有 ${apiClusters.length} 个聚类，去重后 ${uniqueApiClusters.length} 个聚类`);
    
    // 3. 在已经内部去重后的reveliogood聚类和API聚类之间进行去重
    console.log("第3步：RevelioGood聚类和API聚类之间去重");
    const duplicateRevelioGoodIds = new Set();
    
    // 预处理API聚类的元素ID，将长格式(svg/xxx/xxx/id)转为短格式(id)
    const processedApiClusters = uniqueApiClusters.map(cluster => {
        const shortElements = new Set();
        cluster.elements.forEach(id => {
            // 提取最后一个斜杠后面的部分作为简短ID
            const shortId = id.split('/').pop();
            if (shortId) {
                shortElements.add(shortId);
            }
        });
        
        return {
            ...cluster,
            shortElements
        };
    });
    
    // 输出示例ID格式供调试
    if (processedApiClusters.length > 0 && processedApiClusters[0].elements.size > 0) {
        const firstApiElement = [...processedApiClusters[0].elements][0];
        const firstShortApiElement = [...processedApiClusters[0].shortElements][0];
        console.log(`API聚类元素ID示例: 原始格式="${firstApiElement}", 处理后="${firstShortApiElement}"`);
    }
    
    if (uniqueRevelioGoodClusters.length > 0 && uniqueRevelioGoodClusters[0].elements.size > 0) {
        const firstRevelioElement = [...uniqueRevelioGoodClusters[0].elements][0];
        console.log(`RevelioGood聚类元素ID示例: "${firstRevelioElement}"`);
    }
    
    // 比较RevelioGood聚类和API聚类
    uniqueRevelioGoodClusters.forEach(revelioCluster => {
        processedApiClusters.forEach(apiCluster => {
            // 首先检查元素数量是否相同，这是完全重复的必要条件
            if (apiCluster.shortElements.size === revelioCluster.elements.size) {
                // 检查集合元素是否完全相同
                const allElementsMatch = [...revelioCluster.elements].every(id => 
                    apiCluster.shortElements.has(id)
                );
                
                if (allElementsMatch) {
                    // 如果元素完全相同，标记该reveliogood组为重复
                    duplicateRevelioGoodIds.add(revelioCluster.id);
                    console.log(`跨类型去重：RevelioGood聚类 "${revelioCluster.displayName}" 与API聚类 "${apiCluster.data?.name || apiCluster.id}" 重复，将只保留API聚类`);
                    
                    // 输出部分匹配成功的元素ID示例，每个聚类最多显示3个
                    const revelioSample = [...revelioCluster.elements].slice(0, 3);
                    const apiSample = [...apiCluster.shortElements].slice(0, 3);
                    console.log(`匹配成功的元素ID示例 - RevelioGood: ${revelioSample.join(', ')} | API: ${apiSample.join(', ')}`);
                }
            }
        });
    });
    
    console.log(`跨类型去重：发现 ${duplicateRevelioGoodIds.size} 个重复的RevelioGood聚类将被跳过，只保留API聚类`);
    
    // 4. 移除reveliobad类
    console.log("第4步：移除RevelioBAD类");
    const filteredApiIds = new Set();
    const filteredRevelioGoodIds = new Set();
    
    // 处理revelioBad元素组，检查是否有聚类中的所有元素都共享同一个revelioBad_n
    if (revelioBadGroups instanceof Map && revelioBadGroups.size > 0) {
        console.log(`共发现 ${revelioBadGroups.size} 个RevelioBAD组，将检查是否需要过滤元素组合`);
        
        // 处理API聚类
        processedApiClusters.forEach(cluster => {
            const elementIds = [...cluster.shortElements];
            
            if (elementIds.length === 0) return;
            
            // 检查API聚类中的元素是否都在某个revelioBad_n组中
            revelioBadGroups.forEach((badElementIds, badGroupKey) => {
                // 跳过基础组
                if (badGroupKey === 'reveliobad_basic') return;
                
                // 检查当前revelioBad_n组中的元素是否包含所有API聚类中的元素
                const allElementsHaveSameBad = elementIds.every(id => 
                    badElementIds.includes(id)
                );
                
                if (allElementsHaveSameBad && elementIds.length > 0) {
                    // 如果所有元素都共享相同的revelioBad_n，标记该聚类为需要过滤
                    filteredApiIds.add(cluster.id);
                    console.log(`过滤API聚类: "${cluster.data?.name || cluster.id}" 因为其所有元素都共享revelioBad标记 "${badGroupKey}"`);
                }
            });
        });
        
        // 处理revelioGood聚类
        uniqueRevelioGoodClusters.forEach(revelioCluster => {
            const elementIds = [...revelioCluster.elements];
            
            if (elementIds.length === 0) return;
            
            // 检查revelioGood聚类中的元素是否都在某个revelioBad_n组中
            revelioBadGroups.forEach((badElementIds, badGroupKey) => {
                // 跳过基础组
                if (badGroupKey === 'reveliobad_basic') return;
                
                // 检查当前revelioBad_n组中的元素是否包含所有revelioGood聚类中的元素
                const allElementsHaveSameBad = elementIds.every(id => 
                    badElementIds.includes(id)
                );
                
                if (allElementsHaveSameBad && elementIds.length > 0) {
                    // 如果所有元素都共享相同的revelioBad_n，标记该聚类为需要过滤
                    filteredRevelioGoodIds.add(revelioCluster.id);
                    console.log(`过滤RevelioGood聚类: "${revelioCluster.displayName}" 因为其所有元素都共享revelioBad标记 "${badGroupKey}"`);
                }
            });
        });
        
        console.log(`共有 ${filteredApiIds.size} 个API聚类和 ${filteredRevelioGoodIds.size} 个RevelioGood聚类将被过滤（因为RevelioBAD）`);
    }
    
    // 5. 最后合并结果并渲染卡片、list overview
    console.log("第5步：合并最终结果并准备渲染");
    
    // 添加API聚类（不受RevelioGood重复影响）
    uniqueApiClusters.forEach(apiCluster => {
        // 跳过被标记为RevelioBAD的聚类
        if (filteredApiIds.has(apiCluster.id)) {
            return;
        }
        
        if (!apiCluster.parentId) {
            // 添加核心节点
            processedNodes.push(apiCluster.data);
            
            // 查找该核心聚类的所有扩展节点
            const extensions = uniqueApiClusters.filter(ext => 
                ext.parentId === apiCluster.id && !filteredApiIds.has(ext.id)
            );
            
            // 在核心节点数据中添加扩展节点引用
            apiCluster.data.extensions = extensions.map(ext => ext.data);
            apiCluster.data.extensionCount = extensions.length;
            
            // 如果此核心聚类被标记为与外延聚类重复，添加特殊标记
            if (apiCluster.hasDuplicateExtension) {
                apiCluster.data.hasDuplicateExtension = true;
                console.log(`标记核心聚类 "${apiCluster.data.name}" 为与外延聚类重复`);
            }
        } else if (apiCluster.parentId) {
            // 检查父核心聚类是否存在于最终结果中
            const parentExists = uniqueApiClusters.some(c => 
                !c.parentId && c.id === apiCluster.parentId && !filteredApiIds.has(c.id)
            );
            
            // 如果父核心聚类不存在（可能因重复被过滤掉），这个外延聚类应该被作为独立聚类添加
            if (!parentExists) {
                // 创建一个独立的聚类节点
                const standaloneCluster = {
                    id: `standalone_${apiCluster.id}`,
                    name: `Standalone ${apiCluster.data.name}`,
                    type: 'core', // 将其视为核心节点
                    originalNodes: [...apiCluster.data.originalNodes],
                    dimensions: apiCluster.data.dimensions || [],
                    extensionCount: 0,
                    extensions: [],
                    value: 1,
                    isStandaloneExtension: true // 标记为独立的外延聚类
                };
                
                console.log(`添加独立外延聚类: "${standaloneCluster.name}" (原父聚类不存在)`);
                processedNodes.push(standaloneCluster);
            }
        }
    });
    
    // 添加RevelioGood聚类（跳过与API聚类重复的）
    let revelioGoodClusterIndex = 0;
    
    uniqueRevelioGoodClusters.forEach(revelioCluster => {
        // 如果聚类被标记为重复或被标记为RevelioBAD，跳过
        if (duplicateRevelioGoodIds.has(revelioCluster.id) || filteredRevelioGoodIds.has(revelioCluster.id)) {
            return;
        }
        
        // 创建新的聚类ID
        const newClusterId = `revelioGood_core_${revelioGoodClusterIndex}`;
        
        // 创建一个新的聚类节点
        const newCluster = {
            id: newClusterId,
            name: `Core ${revelioGoodClusterIndex + 1} (${revelioCluster.displayName})`,
            type: 'core',
            originalNodes: [...revelioCluster.elements],
            dimensions: [],
            extensionCount: 0,
            extensions: [],
            value: 1,
            // 标记为reveliogood聚类
            isRevelioGood: true,
            // 保存原始的groupKey以便识别reveliogood_X_n类型
            groupKey: revelioCluster.groupKey
        };
        
        // 将新聚类添加到节点列表中
        processedNodes.push(newCluster);
        
        // 增加索引
        revelioGoodClusterIndex++;
    });
    
    console.log(`最终处理结果：共${processedNodes.length}个聚类，其中RevelioGood聚类 ${revelioGoodClusterIndex} 个`);
    
    // 收集最终被渲染的reveliogood节点的ID组，包括所有被额外加分的节点
    const finalRevelioGoodClusters = [];
    
    // 遍历已处理的节点，找出所有带有isRevelioGood标记的节点
    processedNodes.forEach(node => {
        if (node.isRevelioGood && node.originalNodes && node.originalNodes.length > 0) {
            // 确保不收集全部元素的总组
            const nodeName = node.name || '';
            
            if (nodeName.includes('RevelioGood_') || 
                (nodeName.includes('RevelioGood') && !nodeName.includes('All RevelioGood Elements'))) {
                
                // 收集符合条件的reveliogood组
                finalRevelioGoodClusters.push([...node.originalNodes]);
                
                // 额外记录节点信息，便于调试
                console.log(`收集到isRevelioGood节点: ${node.id}, 名称: ${node.name}, 包含${node.originalNodes.length}个元素`);
            } else {
                console.log(`跳过总组: ${node.id}, 名称: ${node.name}, 包含${node.originalNodes.length}个元素`);
            }
        }
    });
    
    // 如果有reveliogood节点，存入store
    if (finalRevelioGoodClusters.length > 0) {
        console.log(`最终收集到${finalRevelioGoodClusters.length}个会获得额外显著性分数的RevelioGood节点ID组，存入store`);
        store.dispatch('setRevelioGoodClusters', finalRevelioGoodClusters);
    } else {
        // 如果没有找到reveliogood节点，清空store中的数据
        console.log('未找到任何会获得额外显著性分数的RevelioGood节点，清空store中的数据');
        store.dispatch('setRevelioGoodClusters', []);
    }
    
    // 收集revelioBad节点ID组，存入store，用于后续过滤
    const finalRevelioGoodBadClusters = [];
    if (revelioBadGroups instanceof Map && revelioBadGroups.size > 0) {
        revelioBadGroups.forEach((elements, groupKey) => {
            // 跳过基础组
            if (groupKey === 'reveliobad_basic') return;
            
            // 只收集特定类型的reveliobad_n组
            if (groupKey.startsWith('reveliobad_') && elements.length > 0) {
                finalRevelioGoodBadClusters.push([...elements]);
                console.log(`收集到revelioBad组: ${groupKey}, 包含${elements.length}个元素`);
            }
        });
        
        if (finalRevelioGoodBadClusters.length > 0) {
            console.log(`最终收集到${finalRevelioGoodBadClusters.length}个revelioBad节点ID组，存入store`);
            store.dispatch('setRevelioBadClusters', finalRevelioGoodBadClusters);
        } else {
            console.log('未找到任何revelioBad节点，清空store中的数据');
            store.dispatch('setRevelioBadClusters', []);
        }
    } else {
        console.log('未找到任何revelioBad组，清空store中的数据');
        store.dispatch('setRevelioBadClusters', []);
    }
    
    return { nodes: processedNodes };
}

// 新增卡片点击处理函数
function handleCardClick(node, event) {
    // 如果当前正在拖动或刚拖动完毕，不处理点击
    if (isDragging.value || wasRecentlyDragging.value) {
        return;
    }

    // 防止事件冒泡
    event.stopPropagation();

    // 调用原来的点击处理逻辑
    showNodeList(node);
    
    // 计算该卡片的显著性值并存入store
    const cardSalienceValue = calculateAttentionProbability(node);
    // 将值格式化为百分比字符串，与SvgUploader中的显示格式保持一致
    const formattedSalience = (cardSalienceValue * 100).toFixed(3);
    // 将显著性值提交到Vuex store
    store.dispatch('setClickedCardSalience', formattedSalience);
    console.log(`已将卡片 ${node.id} 的显著性值 ${formattedSalience} 存入store`);
}

// 显示节点列表
function showNodeList(node) {
    try {
        // 防止任何潜在的拖动触发此函数
        if (isDragging.value || wasRecentlyDragging.value) {
            return;
        }

        // 获取当前卡片中的SVG元素
        const cardContainer = document.querySelector(`[data-node-id="${node.id}"] .card-svg-container svg`);
        if (!cardContainer) {
            console.error('Card SVG container not found');
            return;
        }

        // 获取所有不透明度为1的元素（即高亮元素）
        const highlightedElements = Array.from(cardContainer.querySelectorAll('*'))
            .filter(el => el.style.opacity === '1' && el.id && el.tagName !== 'svg' && el.tagName !== 'g');

        // 收集这些元素的ID
        const nodeNames = highlightedElements.map(el => el.id);

        store.commit('UPDATE_SELECTED_NODES', { nodeIds: nodeNames, group: null });
    } catch (error) {
        console.error('Error getting highlighted nodes:', error);
    }
}

// 判断节点是否被选中
function isNodeSelected(node) {
    const nodeElements = node.originalNodes.map(n => n.split('/').pop());
    const selectedElements = nodeElements.filter(id => selectedNodeIds.value.includes(id));
    return selectedElements.length > 0 &&
        selectedElements.length === selectedNodeIds.value.length &&
        selectedElements.every(id => selectedNodeIds.value.includes(id));
}

// 拖动相关方法
function startDrag(e) {
    // 记录拖动起点，用于判断是否真的拖动了
    startX.value = e.pageX - cardsWrapper.value.offsetLeft;
    lastX.value = e.pageX;
    scrollLeft.value = cardsWrapper.value.scrollLeft;

    // 处于拖动准备状态，但不立即标记为拖动
    isDragging.value = true;
    wasRecentlyDragging.value = false;  // 重置状态

    cardsWrapper.value.style.cursor = 'grabbing';
    cardsWrapper.value.style.userSelect = 'none';

    // 停止任何正在进行的动画
    if (animationFrame.value) {
        cancelAnimationFrame(animationFrame.value);
    }

    // 清除任何已存在的超时
    if (dragEndTimeout.value) {
        clearTimeout(dragEndTimeout.value);
    }

    // 重置速度，防止之前的惯性继续影响
    velocity.value = 0;
}

function onDrag(e) {
    if (!isDragging.value) return;
    e.preventDefault();

    const x = e.pageX;
    const diffX = x - lastX.value;

    // 计算速度但限制最大值，防止过大的速度引起过度滚动
    velocity.value = Math.min(Math.max(diffX, -15), 15);
    lastX.value = x;

    // 只有当拖动距离超过阈值时才真正进行拖动
    const dragDistance = Math.abs(x - startX.value);
    if (dragDistance > 5) {
        wasRecentlyDragging.value = true;

        // 根据拖动距离计算滚动位置
        const walk = (x - startX.value);
        cardsWrapper.value.scrollLeft = scrollLeft.value - walk;

        // 更新阴影状态
        updateScrollShadows();
    }
}

function endDrag(e) {
    if (!isDragging.value) return;

    // 检查是否真的拖动了足够的距离
    const dragDistance = Math.abs(e.pageX - startX.value);

    // 如果几乎没有移动，不执行任何滚动或标记为拖动
    if (dragDistance <= 3) {
        isDragging.value = false;
        wasRecentlyDragging.value = false;
        cardsWrapper.value.style.cursor = 'grab';
        cardsWrapper.value.style.userSelect = '';
        velocity.value = 0; // 重要：重置速度
        return;
    }

    isDragging.value = false;
    cardsWrapper.value.style.cursor = 'grab';
    cardsWrapper.value.style.userSelect = '';

    // 添加惯性滚动，但只有当速度足够大且确实拖动了时
    if (Math.abs(velocity.value) > 3 && wasRecentlyDragging.value) {
        const startTime = Date.now();
        const startVelocity = velocity.value;

        function momentumScroll() {
            const elapsed = Date.now() - startTime;
            const remaining = Math.max(0, Math.abs(startVelocity) * 500 - elapsed); // 500ms的减速时间
            const speed = (remaining / (Math.abs(startVelocity) * 500)) * startVelocity;

            if (remaining > 0 && Math.abs(speed) > 0.1) {
                cardsWrapper.value.scrollLeft -= speed;
                updateScrollShadows(); // 更新阴影状态
                animationFrame.value = requestAnimationFrame(momentumScroll);
                wasRecentlyDragging.value = true;  // 惯性滚动期间保持状态
            } else {
                // 惯性滚动结束后，延迟重置拖动状态
                dragEndTimeout.value = setTimeout(() => {
                    wasRecentlyDragging.value = false;
                }, 300);  // 延迟300ms重置状态
            }
        }

        momentumScroll();
    } else {
        // 即使没有惯性滚动，如果确实拖动了，保持短暂延迟后重置
        if (wasRecentlyDragging.value) {
            dragEndTimeout.value = setTimeout(() => {
                wasRecentlyDragging.value = false;
            }, 300);
        } else {
            // 立即重置状态，允许立即点击
            wasRecentlyDragging.value = false;
        }
    }

    // 更新阴影状态
    updateScrollShadows();

    // 无论如何都重置速度
    velocity.value = 0;
}

// 修改渲染图形的方法
function renderGraph(container, graphData, useCache = false) {
    if (!container || !graphData) return;

    // 获取节点ID
    const nodeId = container.closest('.card').getAttribute('data-node-id');
    const cacheKey = `render_${nodeId}`;

    // 如果启用缓存且缓存中存在，直接使用缓存
    if (useCache && thumbnailCache.has(cacheKey)) {
        const cachedContent = thumbnailCache.get(cacheKey);
        d3.select(container).html(cachedContent);
        return;
    }

    // 清除所有现有的SVG
    const containerElement = d3.select(container);
    containerElement.selectAll('svg').remove();

    const cardWidth = container.clientWidth;
    const cardHeight = container.clientHeight;

    const svg = containerElement
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .style('display', 'block')
        .style('max-width', '100%')
        .style('max-height', '100%');

    const g = svg.append('g');

    const nodeData = graphData.nodes[0];
    if (!nodeData) return;

    const padding = 0;
    const availableWidth = cardWidth;
    const availableHeight = cardHeight;

    const coreNode = g.append('g')
        .attr('class', 'node')
        .attr('transform', `translate(0,0)`);

    const foreignObject = coreNode.append('foreignObject')
        .attr('width', availableWidth)
        .attr('height', availableHeight)
        .attr('x', 0)
        .attr('y', 0);

    const div = foreignObject.append('xhtml:div')
        .style('width', '100%')
        .style('height', '100%')
        .style('overflow', 'hidden')
        .style('display', 'flex')
        .style('align-items', 'center')
        .style('justify-content', 'center');

    // 使用requestAnimationFrame而不是requestIdleCallback，确保优先渲染
    const thumbnailContent = createThumbnail(nodeData);
    div.html(thumbnailContent);

    const thumbnailSvg = div.select('svg').node();
    if (thumbnailSvg) {
        const bbox = thumbnailSvg.getBBox();
        const viewPadding = 5;
        thumbnailSvg.setAttribute('viewBox', `${bbox.x - viewPadding} ${bbox.y - viewPadding} ${bbox.width + viewPadding * 2} ${bbox.height + viewPadding * 2}`);
        thumbnailSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

        // 在SVG渲染完成后更新统计信息
        elementStats.value.set(nodeData.id, getHighlightedElementsStats(nodeData));
    }

    // 缓存渲染结果
    thumbnailCache.set(cacheKey, containerElement.html());
}

// 添加数据源URL
const MAPPING_DATA_URL = "http://127.0.0.1:5000/average_equivalent_mapping";
const EQUIVALENT_WEIGHTS_URL = "http://127.0.0.1:5000/equivalent_weights_by_tag";
const NORMAL_DATA_URL = "http://127.0.0.1:5000/normalized_init_json";

// 特征名称映射
const featureNameMap = {
    'tag': 'color',
    'opacity': 'opacity',
    'fill_h_cos': 'fill hue',
    'fill_h_sin': 'fill hue',
    'fill_s_n': 'fill saturation',
    'fill_l_n': 'fill lightness',
    'stroke_h_cos': 'stroke hue',
    'stroke_h_sin': 'stroke hue',
    'stroke_s_n': 'stroke saturation',
    'stroke_l_n': 'stroke lightness',
    'stroke_width': 'stroke width',
    'bbox_left_n': 'Bbox left',
    'bbox_right_n': 'Bbox right',
    'bbox_top_n': 'Bbox top',
    'bbox_bottom_n': 'Bbox bottom',
    'bbox_mds_1': 'position',
    'bbox_mds_2': 'position',
    'bbox_center_y_n': 'center y',
    'bbox_center_x_n': 'center x',
    'bbox_width_n': 'width',
    'bbox_height_n': 'height',
    'bbox_fill_area': 'area'
};

// 添加分析数据的ref
const analysisData = ref(null);
const equivalentWeightsData = ref(null);

// 生成分析文字的函数
const generateAnalysis = (nodeData) => {
    // 使用normalizedData而不是analysisData和equivalentWeightsData
    if (!normalizedData.value || normalizedData.value.length === 0) return '';

    let nodesToAnalyze = [];

    if (nodeData.type === 'extension') {
        // 对于外延节点，同时分析核心节点的内容
        const coreIndex = parseInt(nodeData.clusterId);
        const coreNode = nodes.value.find(n => n.id === `core_${coreIndex}`);
        if (coreNode) {
            nodesToAnalyze.push(...coreNode.originalNodes);
        }
        // 再添加外延节点内容
        nodesToAnalyze.push(...nodeData.originalNodes);
    } else {
        // 核心节点只分析自身
        nodesToAnalyze = [...nodeData.originalNodes];
    }

    if (nodesToAnalyze.length === 0) return '';

    // 1. 将所有节点分为高亮组和非高亮组
    const highlightedFeatures = [];
    const nonHighlightedFeatures = [];

    // 遍历normalized数据
    normalizedData.value.forEach(item => {
        // 从完整路径中提取最后的ID部分
        const normalizedItemLastId = item.id.split('/').pop();

        // 检查当前元素是否是高亮元素
        const isHighlighted = nodesToAnalyze.some(analyzeNode => {
            // 从分析节点路径中提取最后的ID部分
            const analyzeNodeLastId = analyzeNode.split('/').pop();
            return normalizedItemLastId === analyzeNodeLastId;
        });

        if (isHighlighted) {
            highlightedFeatures.push(item.features);
        } else {
            nonHighlightedFeatures.push(item.features);
        }
    });

    // 如果没有高亮元素或非高亮元素，返回空字符串
    if (highlightedFeatures.length === 0 || nonHighlightedFeatures.length === 0) {
        return '';
    }

    // 2. 计算每个特征的统计数据
    const featureScores = [];
    const featureCount = highlightedFeatures[0].length;

    for (let featureIndex = 0; featureIndex < featureCount; featureIndex++) {
        // 提取高亮组和非高亮组的当前特征值
        const highlightedValues = highlightedFeatures.map(features => features[featureIndex]);
        const nonHighlightedValues = nonHighlightedFeatures.map(features => features[featureIndex]);

        // 计算高亮组和非高亮组的平均值
        const highlightedMean = highlightedValues.reduce((sum, val) => sum + val, 0) / highlightedValues.length;
        const nonHighlightedMean = nonHighlightedValues.reduce((sum, val) => sum + val, 0) / nonHighlightedValues.length;

        // 计算差异性 - 高亮组和非高亮组平均值的差异
        const difference = Math.abs(highlightedMean - nonHighlightedMean);

        // 计算内聚性 - 高亮组内部值的方差（值越小表示内聚性越高）
        let cohesion = 0;

        if (highlightedValues.length === 1) {
            // 单个元素时，内聚性最高
            cohesion = 1;
        } else {
            // 计算高亮组的方差，并转换为内聚性分数（方差越小，内聚性越高）
            const highlightedVariance = highlightedValues.reduce((sum, val) => {
                return sum + Math.pow(val - highlightedMean, 2);
            }, 0) / highlightedValues.length;

            // 转换方差为内聚性分数（使用指数衰减）
            cohesion = Math.exp(-5 * highlightedVariance);
        }

        // 计算该特征的最终得分
        const featureScore = difference * 0.6 + cohesion * 0.4;

        // 获取特征名称
        // 定义特征索引到名称的映射
        const featureIndexToName = [
            'tag', 'opacity', 'fill_h_cos', 'fill_h_sin', 'fill_s_n', 'fill_l_n',
            'stroke_h_cos', 'stroke_h_sin', 'stroke_s_n', 'stroke_l_n', 'stroke_width',
            'bbox_left_n', 'bbox_right_n', 'bbox_top_n', 'bbox_bottom_n',
            'bbox_mds_1', 'bbox_mds_2', 'bbox_center_x_n', 'bbox_center_y_n',
            'bbox_width_n', 'bbox_height_n', 'bbox_fill_area'
        ];

        const featureName = featureIndexToName[featureIndex] || `feature_${featureIndex}`;
        const displayName = featureNameMap[featureName] || featureName;

        // 修复特征名称的格式，去除额外的空格
        const cleanDisplayName = displayName.trim();

        featureScores.push({
            index: featureIndex,
            name: cleanDisplayName,
            score: featureScore,
            difference: difference,
            cohesion: cohesion
        });
    }

    // 使用Set来保存已处理的特征名，确保唯一性
    const uniqueFeatureNames = new Set();

    // 3. 完全按重要性对特征进行排序和去重
    const sortedFeatures = featureScores
        .sort((a, b) => b.score - a.score)
        // 使用Map对象来存储每个特征名称的最高分
        .reduce((map, feature) => {
            // 如果该特征名称尚未添加或当前特征得分更高，则更新
            if (!map.has(feature.name) || feature.score > map.get(feature.name).score) {
                map.set(feature.name, feature);
            }
            return map;
        }, new Map())
        // 转换为数组
        .values();

    // 取前三个最重要的且唯一的特征
    const topFeatures = Array.from(sortedFeatures).slice(0, 3);

    // 4. 生成HTML，确保特征标签唯一
    return topFeatures.map(feature => {
        const color = '#905F29';
        return `<span class="feature-tag" style="color: ${color}; border: 1px solid ${color}20; background-color: ${color}08">
            ${feature.name}
        </span>`;
    }).join(' ');
};

// 获取分析数据
const fetchAnalysisData = async () => {
    try {
        const [responseMapping, responseEquivalentWeights] = await Promise.all([
            axios.get(MAPPING_DATA_URL),
            axios.get(EQUIVALENT_WEIGHTS_URL)
        ]);

        if (responseMapping.data && responseEquivalentWeights.data) {
            analysisData.value = responseMapping.data;
            equivalentWeightsData.value = responseEquivalentWeights.data;
        }
    } catch (error) {
        console.error('Failed to fetch analysis data:', error);
    }
};

// 首先添加一个ref来存储特征数据
const clusterFeatures = ref(null);
const normalizedData = ref(null);
// 添加一个ref来存储从SVG中提取的reveliogood元素
const revelioGoodElements = ref([]);
const revelioBadElements = ref([]);

// 在loadAndRenderGraph函数中添加获取特征数据的逻辑
async function loadAndRenderGraph() {
    try {
        loading.value = true;
        isInitialRender.value = true;

        // 移除fetchAnalysisData()调用，不再需要获取这些数据
        const [svgResponse, graphResponse, featuresResponse, normalDataResponse] = await Promise.all([
            fetch('http://127.0.0.1:5000/get_svg'),
            fetch('http://127.0.0.1:5000/static/data/subgraphs/subgraph_dimension_all.json'),
            fetch('http://127.0.0.1:5000/cluster_features'),
            fetch(NORMAL_DATA_URL)
        ]);

        const [svgContent, data, featuresData, normalData] = await Promise.all([
            svgResponse.text(),
            graphResponse.json(),
            featuresResponse.json(),
            normalDataResponse.json()
        ]);

        // 从SVG中提取包含class="reveliogood"和class="reveliobad"的元素，按组分类
        const { revelioGoodGroups, revelioBadGroups } = extractSVGElements(svgContent);
        revelioGoodElements.value = Array.from(revelioGoodGroups.values()).flat();
        revelioBadElements.value = Array.from(revelioBadGroups.values()).flat();

        // 检查数据是否更新
        if (originalSvgContent.value !== svgContent ||
            JSON.stringify(graphData.value) !== JSON.stringify(data) ||
            JSON.stringify(clusterFeatures.value) !== JSON.stringify(featuresData) ||
            JSON.stringify(normalizedData.value) !== JSON.stringify(normalData)) {

            isDataUpdated.value = true;
            clusterFeatures.value = featuresData;
            normalizedData.value = normalData;
            originalSvgContent.value = svgContent;
            graphData.value = data;

            // 清除缓存
            thumbnailCache.clear();
            elementStats.value.clear();

            // 处理数据，将扩展节点集成到核心节点中，并添加从SVG中提取的revelioGood和revelioBad元素聚类
            const processedData = processGraphData(data, revelioGoodGroups, revelioBadGroups);
            nodes.value = processedData.nodes.filter(node => node.type === 'core');

            // 等待DOM更新后，一次性渲染所有卡片
            await nextTick();

            // 统一渲染所有卡片
            renderAllCards();

            // 初始化总览条
            nextTick(() => updateOverview());
        }
    } catch (error) {
        console.error('Error loading data:', error);
    } finally {
        loading.value = false;
        // 初始化渲染完成
        isInitialRender.value = false;
        isDataUpdated.value = false;

        // 在数据加载和渲染完成后，更新视觉显著性数据
        nextTick(() => {
            updateVisualSalienceData();
            updateOverview();
        });
    }
}

// 添加一个函数用于解析SVG并提取class="reveliogood"的元素
function extractRevelioGoodElements(svgContent) {
    try {
        // 创建一个DOM解析器
        const parser = new DOMParser();
        const svgDoc = parser.parseFromString(svgContent, 'image/svg+xml');
        
        // 创建一个Map来存储不同类型的reveliogood元素组
        const revelioGoodGroups = new Map();
        
        // 不再创建"all"组，避免将所有元素混合到一个组中
        // 只保留特定reveliogood_n组和基础reveliogood组
        
        // 初始化一个普通的reveliogood组
        revelioGoodGroups.set('reveliogood_basic', []);
        
        // 查找所有包含class="reveliogood"的元素
        const revelioGoodNodes = svgDoc.querySelectorAll('[class*="reveliogood"]');
        
        // 遍历所有元素
        revelioGoodNodes.forEach(node => {
            if (!node.id) return;
            
            const classAttr = node.getAttribute('class') || '';
            const elementId = node.id;
            
            // 不再添加到"所有reveliogood元素"组
            
            // 检查是否包含普通的reveliogood类（不带数字）
            if (classAttr.match(/\breveliogood\b/)) {
                revelioGoodGroups.get('reveliogood_basic').push(elementId);
            }
            
            // 检查所有reveliogood_n格式的类
            const matches = classAttr.match(/reveliogood_\d+/g);
            // 检查所有reveliogood_X_n格式的类
            const matchesX = classAttr.match(/reveliogood_X_\d+/g);
            
            // 处理普通的reveliogood_n格式
            if (matches && matches.length > 0) {
                // 将元素添加到每一个匹配的reveliogood_n组
                matches.forEach(match => {
                    if (!revelioGoodGroups.has(match)) {
                        revelioGoodGroups.set(match, []);
                    }
                    revelioGoodGroups.get(match).push(elementId);
                });
            }
            
            // 处理reveliogood_X_n格式
            if (matchesX && matchesX.length > 0) {
                // 将元素添加到每一个匹配的reveliogood_X_n组
                matchesX.forEach(match => {
                    if (!revelioGoodGroups.has(match)) {
                        revelioGoodGroups.set(match, []);
                    }
                    revelioGoodGroups.get(match).push(elementId);
                });
            }
        });
        
        // 移除空组
        for (const [key, elements] of revelioGoodGroups.entries()) {
            if (elements.length === 0) {
                revelioGoodGroups.delete(key);
            }
        }
        
        // 打印收集到的组信息用于调试
        console.log("收集到的revelioGood组：");
        for (const [key, elements] of revelioGoodGroups.entries()) {
            console.log(`- ${key}: ${elements.length}个元素`);
        }
        
        return revelioGoodGroups;
    } catch (error) {
        console.error('Error extracting reveliogood elements:', error);
        return new Map();
    }
}

// 添加一个函数用于解析SVG并提取class="reveliogood"和class="reveliobad"的元素
function extractSVGElements(svgContent) {
    try {
        // 创建一个DOM解析器
        const parser = new DOMParser();
        const svgDoc = parser.parseFromString(svgContent, 'image/svg+xml');
        
        // 创建Map来存储不同类型的reveliogood和reveliobad元素组
        const revelioGoodGroups = new Map();
        const revelioBadGroups = new Map();
        
        // 初始化基础组
        revelioGoodGroups.set('reveliogood_basic', []);
        revelioBadGroups.set('reveliobad_basic', []);
        
        // 查找所有包含class="reveliogood"或class="reveliobad"的元素
        const revelioGoodNodes = svgDoc.querySelectorAll('[class*="reveliogood"]');
        const revelioBadNodes = svgDoc.querySelectorAll('[class*="reveliobad"]');
        
        // 处理revelioGood元素
        revelioGoodNodes.forEach(node => {
            if (!node.id) return;
            
            const classAttr = node.getAttribute('class') || '';
            const elementId = node.id;
            
            // 检查是否包含普通的reveliogood类（不带数字）
            if (classAttr.match(/\breveliogood\b/)) {
                revelioGoodGroups.get('reveliogood_basic').push(elementId);
            }
            
            // 检查所有reveliogood_n格式的类
            const matches = classAttr.match(/reveliogood_\d+/g);
            // 检查所有reveliogood_X_n格式的类
            const matchesX = classAttr.match(/reveliogood_X_\d+/g);
            
            // 处理普通的reveliogood_n格式
            if (matches && matches.length > 0) {
                // 将元素添加到每一个匹配的reveliogood_n组
                matches.forEach(match => {
                    if (!revelioGoodGroups.has(match)) {
                        revelioGoodGroups.set(match, []);
                    }
                    revelioGoodGroups.get(match).push(elementId);
                });
            }
            
            // 处理reveliogood_X_n格式
            if (matchesX && matchesX.length > 0) {
                // 将元素添加到每一个匹配的reveliogood_X_n组
                matchesX.forEach(match => {
                    if (!revelioGoodGroups.has(match)) {
                        revelioGoodGroups.set(match, []);
                    }
                    revelioGoodGroups.get(match).push(elementId);
                });
            }
        });
        
        // 处理revelioBad元素
        revelioBadNodes.forEach(node => {
            if (!node.id) return;
            
            const classAttr = node.getAttribute('class') || '';
            const elementId = node.id;
            
            // 检查是否包含普通的reveliobad类（不带数字）
            if (classAttr.match(/\breveliobad\b/)) {
                revelioBadGroups.get('reveliobad_basic').push(elementId);
            }
            
            // 检查所有reveliobad_n格式的类
            const matches = classAttr.match(/reveliobad_\d+/g);
            // 检查所有reveliobad_X_n格式的类
            const matchesX = classAttr.match(/reveliobad_X_\d+/g);
            
            // 处理普通的reveliobad_n格式
            if (matches && matches.length > 0) {
                // 将元素添加到每一个匹配的reveliobad_n组
                matches.forEach(match => {
                    if (!revelioBadGroups.has(match)) {
                        revelioBadGroups.set(match, []);
                    }
                    revelioBadGroups.get(match).push(elementId);
                });
            }
            
            // 处理reveliobad_X_n格式
            if (matchesX && matchesX.length > 0) {
                // 将元素添加到每一个匹配的reveliobad_X_n组
                matchesX.forEach(match => {
                    if (!revelioBadGroups.has(match)) {
                        revelioBadGroups.set(match, []);
                    }
                    revelioBadGroups.get(match).push(elementId);
                });
            }
        });
        
        // 移除空组
        for (const [key, elements] of revelioGoodGroups.entries()) {
            if (elements.length === 0) {
                revelioGoodGroups.delete(key);
            }
        }
        
        for (const [key, elements] of revelioBadGroups.entries()) {
            if (elements.length === 0) {
                revelioBadGroups.delete(key);
            }
        }
        
        // 打印收集到的组信息用于调试
        console.log("收集到的revelioGood组：");
        for (const [key, elements] of revelioGoodGroups.entries()) {
            console.log(`- ${key}: ${elements.length}个元素`);
        }
        
        console.log("收集到的revelioBad组：");
        for (const [key, elements] of revelioBadGroups.entries()) {
            console.log(`- ${key}: ${elements.length}个元素`);
        }
        
        return { revelioGoodGroups, revelioBadGroups };
    } catch (error) {
        console.error('Error extracting revelio elements:', error);
        return { revelioGoodGroups: new Map(), revelioBadGroups: new Map() };
    }
}

// 新增统一渲染所有卡片的函数
function renderAllCards() {
    const containers = document.querySelectorAll('.card-svg-container');
    if (containers.length === 0) {
        // 如果DOM还没准备好，等待下一帧再试
        requestAnimationFrame(renderAllCards);
        return;
    }

    // 先收集所有需要渲染的节点
    const nodesToRender = [];
    containers.forEach((container) => {
        const nodeId = container.closest('.card').getAttribute('data-node-id');
        const node = flattenedNodes.value.find(n => n.id === nodeId);
        if (node) {
            nodesToRender.push({
                container,
                node,
                nodeData: { nodes: [node] }
            });
        }
    });

    // 批量优化渲染 - 分批处理避免阻塞主线程
    const batchSize = 5;
    let currentBatch = 0;

    function processBatch() {
        const startIdx = currentBatch * batchSize;
        const endIdx = Math.min(startIdx + batchSize, nodesToRender.length);

        for (let i = startIdx; i < endIdx; i++) {
            const { container, node, nodeData } = nodesToRender[i];
            renderGraph(container, nodeData);
            // 同时更新统计信息
            elementStats.value.set(node.id, getHighlightedElementsStats(node));
        }

        currentBatch++;

        if (currentBatch * batchSize < nodesToRender.length) {
            setTimeout(processBatch, 0); // 使用setTimeout允许UI更新
        } else {
            // 所有卡片渲染完成后，更新总览条
            nextTick(() => updateOverview());
        }
    }

    processBatch();
}

// 处理滚动事件，更新滚动条位置
function handleScroll() {
    updateScrollShadows();
    updateScrollbarThumb();
    // 更新总览条中当前可见的卡片标记
    updateVisibleCards();
}

// 更新滚动条滑块位置和大小
function updateScrollbarThumb() {
    if (!cardsWrapper.value || !scrollbarThumb.value) return;

    const container = cardsWrapper.value;
    const scrollTrack = document.querySelector('.overview-scrollbar-container .custom-scrollbar-track');

    if (!scrollTrack) return;

    const trackWidth = scrollTrack.clientWidth;
    const containerWidth = container.clientWidth;
    const scrollWidth = container.scrollWidth;

    // 计算滑块宽度比例 (可视区域宽度 / 内容总宽度)
    const thumbWidthRatio = containerWidth / scrollWidth;

    // 计算滑块位置比例 (当前滚动位置 / 最大滚动距离)
    const maxScroll = scrollWidth - containerWidth;
    const scrollRatio = container.scrollLeft / maxScroll;

    // 设置滑块宽度 (最小宽度为30px)
    thumbWidth.value = Math.max(thumbWidthRatio * trackWidth, 30);

    // 计算滑块可移动的最大距离
    const maxThumbPosition = trackWidth - thumbWidth.value;

    // 设置滑块位置
    thumbPosition.value = scrollRatio * maxThumbPosition;
}

// 开始拖动滚动条
function startScrollbarDrag(e) {
    if (!cardsWrapper.value) return;

    // 如果点击的是滑块，则开始拖动
    if (e.target.classList.contains('custom-scrollbar-thumb')) {
        isScrollbarDragging.value = true;
        scrollbarStartX.value = e.clientX;
        scrollbarInitialLeft.value = thumbPosition.value;

        // 添加全局鼠标事件监听器
        document.addEventListener('mousemove', onScrollbarDrag);
        document.addEventListener('mouseup', endScrollbarDrag);
    } else {
        // 如果点击的是轨道，则直接跳转到该位置
        const track = e.currentTarget;
        const trackRect = track.getBoundingClientRect();
        const clickPosition = e.clientX - trackRect.left;

        // 计算点击位置相对于轨道的比例
        const trackWidth = track.clientWidth;
        const clickRatio = clickPosition / trackWidth;

        // 计算对应的滚动位置
        const container = cardsWrapper.value;
        const maxScroll = container.scrollWidth - container.clientWidth;
        const newScrollPosition = clickRatio * maxScroll;

        // 设置滚动位置
        container.scrollTo({
            left: newScrollPosition,
            behavior: 'smooth'
        });
    }

    e.preventDefault();
}

// 拖动滚动条
function onScrollbarDrag(e) {
    if (!isScrollbarDragging.value || !cardsWrapper.value) return;

    const container = cardsWrapper.value;
    const track = document.querySelector('.overview-scrollbar-container .custom-scrollbar-track');

    if (!track) return;

    // 计算拖动距离
    const deltaX = e.clientX - scrollbarStartX.value;

    // 计算新的滑块位置
    const trackWidth = track.clientWidth;
    const maxThumbPosition = trackWidth - thumbWidth.value;
    const newThumbPosition = Math.max(0, Math.min(scrollbarInitialLeft.value + deltaX, maxThumbPosition));

    // 计算对应的滚动位置比例
    const scrollRatio = newThumbPosition / maxThumbPosition;

    // 计算实际滚动位置
    const maxScroll = container.scrollWidth - container.clientWidth;
    const newScrollPosition = scrollRatio * maxScroll;

    // 设置滚动位置
    container.scrollLeft = newScrollPosition;

    // 更新滑块位置
    thumbPosition.value = newThumbPosition;

    e.preventDefault();
}

// 结束拖动滚动条
function endScrollbarDrag() {
    isScrollbarDragging.value = false;

    // 移除全局鼠标事件监听器
    document.removeEventListener('mousemove', onScrollbarDrag);
    document.removeEventListener('mouseup', endScrollbarDrag);
}

// 添加更新滚动阴影的方法
function updateScrollShadows() {
    if (!cardsWrapper.value) return;

    const container = cardsWrapper.value;
    const scrollLeft = container.scrollLeft;
    const maxScrollLeft = container.scrollWidth - container.clientWidth;

    // 当滚动位置大于0时显示左侧阴影
    showLeftShadow.value = scrollLeft > 10;

    // 当未滚动到最右侧时显示右侧阴影
    showRightShadow.value = scrollLeft < maxScrollLeft - 10;
}

onMounted(async () => {
    await loadAndRenderGraph();

    // 初始化阴影状态和滚动条
    nextTick(() => {
        updateScrollShadows();
        updateScrollbarThumb();

        // 计算并更新所有卡片的视觉显著性
        updateVisualSalienceData();

        // 初始化总览条
        updateOverview();
    });

    // 添加窗口调整大小时更新阴影、滚动条和总览条的监听器
    window.addEventListener('resize', handleResize);

    // 添加滚动事件监听，用于更新总览条中当前可见的卡片
    if (cardsWrapper.value) {
        cardsWrapper.value.addEventListener('scroll', updateVisibleCards);
    }
});

// 修改窗口大小改变事件处理
let resizeTimeout;
function handleResize() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        updateScrollShadows();
        updateScrollbarThumb();
        updateOverview(); // 这会调用drawConnectionLines和updateVisibleCards
    }, 100);
}

// 修改组件卸载函数，移除相关监听器
onUnmounted(() => {
    window.removeEventListener('resize', handleResize);
    clearTimeout(resizeTimeout);

    // 确保移除滚动条拖动的事件监听器
    document.removeEventListener('mousemove', onScrollbarDrag);
    document.removeEventListener('mouseup', endScrollbarDrag);

    // 移除滚动事件监听器
    if (cardsWrapper.value) {
        cardsWrapper.value.removeEventListener('scroll', updateVisibleCards);
    }
});

// 监听选中节点的变化 - 优化为仅在真正改变选中状态时才重绘
watch(selectedNodeIds, (newVal, oldVal) => {
    // 只有当选择真正变化且不是初始渲染时才更新
    const hasChanged = newVal.length !== oldVal.length ||
        newVal.some(id => !oldVal.includes(id)) ||
        oldVal.some(id => !newVal.includes(id));

    if (hasChanged && !isInitialRender.value) {
        // 使用防抖函数避免短时间内多次渲染
        if (animationFrame.value) {
            cancelAnimationFrame(animationFrame.value);
        }

        animationFrame.value = requestAnimationFrame(() => {
            const containers = document.querySelectorAll('.card-svg-container');
            containers.forEach((container) => {
                const nodeId = container.closest('.card').getAttribute('data-node-id');
                const node = flattenedNodes.value.find(n => n.id === nodeId);
                if (node) {
                    const nodeData = { nodes: [node] };
                    // 使用thumbnail缓存，减少重复渲染
                    renderGraph(container, nodeData, true);
                }
            });

            // 在卡片重新渲染后更新视觉显著性数据
            nextTick(() => {
                updateVisualSalienceData();
            });
        });
    }
});

function getHighlightedElementsStats(nodeData) {
    try {
        // 获取当前卡片中的SVG元素
        const cardContainer = document.querySelector(`[data-node-id="${nodeData.id}"] .card-svg-container svg`);
        if (!cardContainer) {
            console.error('Card SVG container not found');
            return {};
        }

        const stats = new Map();

        // 获取所有不透明度为1的元素（即高亮元素）
        const highlightedElements = Array.from(cardContainer.querySelectorAll('*'))
            .filter(el => el.style.opacity === '1' && el.id && el.tagName !== 'svg' && el.tagName !== 'g');

        // 统计每种元素的数量
        highlightedElements.forEach(element => {
            const tagName = element.tagName.toLowerCase();
            stats.set(tagName, (stats.get(tagName) || 0) + 1);
        });

        return Object.fromEntries(stats);
    } catch (error) {
        console.error('Error counting highlighted elements:', error);
        return {};
    }
}

// 修改注意力概率计算函数
const calculateAttentionProbability = (node, returnRawScore = false) => {
    if (!normalizedData.value || normalizedData.value.length === 0) return 0.1;

    try {
        let nodesToAnalyze = [];

        if (node.type === 'extension') {
            // 对于外延节点，同时分析核心节点的内容
            const coreIndex = parseInt(node.clusterId);
            const coreNode = nodes.value.find(n => n.id === `core_${coreIndex}`);
            if (coreNode) {
                nodesToAnalyze.push(...coreNode.originalNodes);
            }
            // 再添加外延节点内容
            nodesToAnalyze.push(...node.originalNodes);
        } else {
            // 核心节点只分析自身
            nodesToAnalyze = [...node.originalNodes];
        }

        if (nodesToAnalyze.length === 0) return 0.1;

        // 1. 将所有节点分为高亮组和非高亮组
        const highlightedFeatures = [];
        const nonHighlightedFeatures = [];
        const highlightedIds = [];
        const nonHighlightedIds = [];

        // 遍历normalized数据
        normalizedData.value.forEach(item => {
            // 从完整路径中提取最后的ID部分
            const normalizedItemLastId = item.id.split('/').pop();

            // 检查当前元素是否是高亮元素
            const isHighlighted = nodesToAnalyze.some(analyzeNode => {
                // 从分析节点路径中提取最后的ID部分
                const analyzeNodeLastId = analyzeNode.split('/').pop();
                return normalizedItemLastId === analyzeNodeLastId;
            });

            if (isHighlighted) {
                highlightedFeatures.push(item.features);
                highlightedIds.push(normalizedItemLastId);
            } else {
                nonHighlightedFeatures.push(item.features);
                nonHighlightedIds.push(normalizedItemLastId);
            }
        });

        // 如果没有高亮元素或没有非高亮元素，返回默认值
        if (highlightedFeatures.length === 0 || nonHighlightedFeatures.length === 0) {
            return 0.1;
        }

        // 获取元素权重信息
        const elementWeights = equivalentWeightsData.value || {};

        // 计算加权后的特征向量
        function getWeightedFeatures(features, elementId) {
            // 如果没有权重数据或找不到该元素的权重，直接返回原特征
            if (!elementWeights || !elementWeights[elementId]) {
                return [...features];
            }

            // 获取该元素的4个权重向量
            const weights = elementWeights[elementId];

            // 如果权重数据格式不正确，直接返回原特征
            if (!Array.isArray(weights) || weights.length !== 4) {
                return [...features];
            }

            // 将4个权重向量加起来
            const combinedWeights = new Array(features.length).fill(0);
            for (let i = 0; i < weights.length; i++) {
                const weightVector = weights[i];
                for (let j = 0; j < combinedWeights.length; j++) {
                    combinedWeights[j] += weightVector[j] || 0;
                }
            }

            // 将特征向量与权重向量相乘
            const weightedFeatures = features.map((value, index) => {
                return value * (combinedWeights[index] || 1); // 如果权重为0或不存在，使用1
            });

            return weightedFeatures;
        }

        // 余弦相似度计算函数
        function cosineSimilarity(vecA, vecB) {
            // 计算点积
            let dotProduct = 0;
            for (let i = 0; i < vecA.length; i++) {
                dotProduct += vecA[i] * vecB[i];
            }

            // 计算向量长度
            let vecAMagnitude = 0;
            let vecBMagnitude = 0;
            for (let i = 0; i < vecA.length; i++) {
                vecAMagnitude += vecA[i] * vecA[i];
                vecBMagnitude += vecB[i] * vecB[i];
            }
            vecAMagnitude = Math.sqrt(vecAMagnitude);
            vecBMagnitude = Math.sqrt(vecBMagnitude);

            // 避免除以零
            if (vecAMagnitude === 0 || vecBMagnitude === 0) {
                return 0;
            }

            // 计算余弦相似度
            return dotProduct / (vecAMagnitude * vecBMagnitude);
        }

        // 计算组内元素平均相似度
        let intraGroupSimilarity = 1.0; // 默认设置为最大值

        // 如果组内有多个元素，计算它们之间的平均相似度
        if (highlightedFeatures.length > 1) {
            let similaritySum = 0;
            let pairCount = 0;

            // 计算组内所有元素对之间的相似度
            for (let i = 0; i < highlightedFeatures.length; i++) {
                for (let j = i + 1; j < highlightedFeatures.length; j++) {
                    // 获取加权后的特征向量
                    const weightedFeaturesA = getWeightedFeatures(highlightedFeatures[i], highlightedIds[i]);
                    const weightedFeaturesB = getWeightedFeatures(highlightedFeatures[j], highlightedIds[j]);

                    // 计算加权特征向量之间的余弦相似度
                    similaritySum += cosineSimilarity(weightedFeaturesA, weightedFeaturesB);
                    pairCount++;
                }
            }

            // 计算平均相似度
            intraGroupSimilarity = similaritySum / pairCount;
        }

        // 计算组内与组外元素之间的平均相似度
        let interGroupSimilarity = 0;
        let interPairCount = 0;

        // 计算每个组内元素与每个组外元素之间的相似度
        for (let i = 0; i < highlightedFeatures.length; i++) {
            for (let j = 0; j < nonHighlightedFeatures.length; j++) {
                // 获取加权后的特征向量
                const weightedFeaturesA = getWeightedFeatures(highlightedFeatures[i], highlightedIds[i]);
                const weightedFeaturesB = getWeightedFeatures(nonHighlightedFeatures[j], nonHighlightedIds[j]);

                // 计算加权特征向量之间的余弦相似度
                interGroupSimilarity += cosineSimilarity(weightedFeaturesA, weightedFeaturesB);
                interPairCount++;
            }
        }

        // 计算平均相似度，避免除以零
        interGroupSimilarity = interPairCount > 0 ? interGroupSimilarity / interPairCount : 0;

        // 避免除以零，如果组间相似度为0，设置显著性为最大值
        let salienceScore = interGroupSimilarity > 0 ? intraGroupSimilarity / interGroupSimilarity : 1.0;

        // 考虑面积因素 (保留原来的面积影响因素)
        const AREA_INDEX = 19; // bbox_fill_area 在特征向量中的索引是19

        // 计算所有元素的平均面积（包括高亮和非高亮元素）
        const allFeatures = [...highlightedFeatures, ...nonHighlightedFeatures];
        const allElementsAvgArea = allFeatures.reduce((sum, features) =>
            sum + features[AREA_INDEX], 0) / allFeatures.length;

        // 计算高亮元素的平均面积
        const highlightedAvgArea = highlightedFeatures.reduce((sum, features) =>
            sum + features[AREA_INDEX], 0) / highlightedFeatures.length;

        // 使用所有元素平均面积的1.3倍作为阈值
        const areaThreshold = allElementsAvgArea * 1.1;

        // 如果高亮元素的平均面积小于阈值，显著降低显著性
        if (highlightedAvgArea < areaThreshold) {
            salienceScore = salienceScore / 3;
        }
        
        // 检查是否是reveliogood聚类，如果是则额外加5分
        if (node.isRevelioGood) {
            // 检查是否是reveliogood_X_n类型，如果是则不额外加分
            if (!node.groupKey || !node.groupKey.startsWith('reveliogood_X_')) {
                salienceScore += 0.4;
            } else {
                // salienceScore -= 0.0;
                console.log(`聚类 "${node.name}" 是reveliogood_X_n类型，不额外加显著性分数`);
            }
        }
        
        // 获取所有元素的类名信息
        const highlightedClassNames = [];
        const nonHighlightedClassNames = [];
        
        // 从normalizedData中获取每个元素的class信息
        normalizedData.value.forEach(item => {
            // 从完整路径中提取最后的ID部分
            const normalizedItemLastId = item.id.split('/').pop();
            
            // 检查当前元素是否是高亮元素
            const isHighlighted = highlightedIds.includes(normalizedItemLastId);
            
            // 如果元素有class属性，记录它
            if (item.class) {
                if (isHighlighted) {
                    highlightedClassNames.push(item.class);
                } else {
                    nonHighlightedClassNames.push(item.class);
                }
            }
        });
        
        // 检查是否所有高亮元素都包含同一个down_n类，且其他元素都不包含该类
        if (highlightedClassNames.length > 0 && nonHighlightedClassNames.length > 0) {
            // 获取所有可能的down_n类
            const downClassRegex = /\bdown_\d+\b/g;
            const allDownClasses = new Set();
            
            // 收集所有高亮元素中的down_n类
            for (const className of highlightedClassNames) {
                const matches = className.match(downClassRegex);
                if (matches) {
                    matches.forEach(match => allDownClasses.add(match));
                }
            }
            
            // 检查是否有满足条件的down_n类
            for (const downClass of allDownClasses) {
                // 检查所有高亮元素是否都包含这个down_n类
                const allHighlightedHaveClass = highlightedClassNames.every(className => 
                    className.includes(downClass));
                
                // 检查所有非高亮元素是否都不包含这个down_n类
                const noNonHighlightedHaveClass = nonHighlightedClassNames.every(className => 
                    !className.includes(downClass));
                
                // 如果同时满足这两个条件，显著性减10
                if (allHighlightedHaveClass && noNonHighlightedHaveClass) {
                    console.log(`发现所有高亮元素都包含类 ${downClass}，且其他元素都不包含该类，显著性减10`);
                    salienceScore -= 10;
                    break; // 只要找到一个满足条件的类就可以了
                }
            }
        }

        // 如果需要返回原始分数（用于排序），直接返回
        if (returnRawScore) {
            return salienceScore;
        }

        // 否则，将分数映射到0-1范围内用于显示
        // 使用sigmoid函数进行平滑映射，确保结果在0-1范围内
        const normalizedScore = Math.min(Math.max(1 / (0.8 + Math.exp(-salienceScore))));

        return normalizedScore;

    } catch (error) {
        console.error('Error calculating attention probability:', error);
        console.error('Error details:', error.stack);
        return 0.2; // 提高默认值
    }
};

// 添加计算属性来获取统计信息
const getStats = (node) => {
    return elementStats.value.get(node.id) || {};
};

// 添加计算所有卡片视觉显著性的函数
function updateVisualSalienceData() {
    if (!flattenedNodes.value || flattenedNodes.value.length === 0) {
        console.warn('没有可用的节点数据来计算视觉显著性');
        return;
    }

    try {
        // 清空颜色缓存以便重新计算
        nodeColorCache.value.clear();

        // 计算所有卡片的视觉显著性
        const salienceData = flattenedNodes.value.map(node => {
            const nodeId = node.id;
            const salienceValue = calculateAttentionProbability(node);
            const rawScore = calculateAttentionProbability(node, true);

            // 顺便计算并缓存颜色（这会调用getSalienceColor并自动缓存结果）
            getSalienceColor(node);

            return {
                nodeId,
                type: node.type,
                clusterId: node.clusterId,
                salienceValue: (salienceValue * 100).toFixed(3), // 格式化为百分比
                rawScore
            };
        });

        // 按显著性值从高到低排序
        salienceData.sort((a, b) => b.rawScore - a.rawScore);

        // 将结果存储到Vuex store中
        store.commit('SET_VISUAL_SALIENCE', salienceData);

        console.log('Visual saliency data has been updated:', salienceData);
    } catch (error) {
        console.error('Error in calculating visual saliency:', error);
    }
}

// 在数据更新后重新计算视觉显著性
watch(normalizedData, (newVal) => {
    if (newVal && newVal.length > 0 && !isInitialRender.value) {
        nextTick(() => {
            updateVisualSalienceData();
        });
    }
}, { deep: true });

// 添加获取显著性颜色的函数
const getSalienceColor = (node) => {
    // 首先检查缓存中是否已存在
    if (nodeColorCache.value.has(node.id)) {
        return nodeColorCache.value.get(node.id);
    }

    try {
        // 计算显著性值（0-1之间）
        const salienceValue = calculateAttentionProbability(node);

        // 主题色 #905F29 作为基础颜色
        const baseColor = { r: 144, g: 95, b: 41 }; // #905F29 的RGB值

        // 根据显著性值调整颜色深浅
        // 显著性越高，颜色越深
        const minOpacity = 0.15; // 最小不透明度
        const opacity = minOpacity + salienceValue * (1 - minOpacity);

        // 构建最终颜色 - 对于低显著性值，让颜色变淡而不是完全透明
        const color = `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${opacity})`;

        // 将计算结果存入缓存
        nodeColorCache.value.set(node.id, color);

        return color;
    } catch (error) {
        console.error('Error calculating salience color:', error);
        const defaultColor = 'rgba(144, 95, 41, 0.2)'; // 默认淡色
        nodeColorCache.value.set(node.id, defaultColor);
        return defaultColor;
    }
};

// 添加处理总览条点击的方法
const handleOverviewClick = (nodeId, event) => {
    console.log('Overview clicked for node:', nodeId);

    // 防止事件冒泡
    if (event) {
        event.stopPropagation();
    }

    // 确保不在拖拽状态
    if (isDragging.value || wasRecentlyDragging.value) {
        console.log('Ignoring click during drag');
        return;
    }

    // 查找对应的卡片元素
    const cardSelector = `.card[data-node-id="${nodeId}"]`;
    const cardElement = document.querySelector(cardSelector);

    if (!cardElement) {
        console.error('Card element not found for nodeId:', nodeId);
        return;
    }

    if (!cardsWrapper.value) {
        console.error('Cards wrapper not found');
        return;
    }

    try {
        // 标记正在进行跳转以防止其他拖动事件干扰
        wasRecentlyDragging.value = true;

        // 获取卡片在滚动容器中的位置
        const container = cardsWrapper.value;
        const cardLeft = cardElement.offsetLeft;
        const containerWidth = container.clientWidth;

        // 计算目标滚动位置（居中显示）
        const targetScrollLeft = cardLeft - (containerWidth / 2) + (cardElement.offsetWidth / 2);

        console.log('Scrolling to position:', Math.max(0, targetScrollLeft));

        // 使用平滑滚动
        container.scrollTo({
            left: Math.max(0, targetScrollLeft),
            behavior: 'smooth'
        });

        // 临时高亮效果 - 改为更轻微的效果
        cardElement.style.transition = 'all 0.3s ease';
        cardElement.style.boxShadow = '0 0 0 1px #905F2980'; // 更改为更轻的边框效果

        // 重置高亮
        setTimeout(() => {
            cardElement.style.boxShadow = '';
        }, 1500);

        // 滚动完成后重置状态
        setTimeout(() => {
            wasRecentlyDragging.value = false;
        }, 600);
    } catch (error) {
        console.error('Error scrolling to card:', error);
        wasRecentlyDragging.value = false;
    }
};

// 添加缺失的更新可见卡片的函数
function updateVisibleCards() {
    try {
        if (!cardsWrapper.value || !overviewSvg.value) return;
        
        // 获取所有卡片元素
        const cards = document.querySelectorAll('.card');
        if (cards.length === 0) return;
        
        // 获取总览条中的所有卡片指示器
        const clusterItems = overviewSvg.value.querySelectorAll('.cluster-item');
        if (clusterItems.length === 0) return;
        
        // 获取容器的可视范围
        const containerRect = cardsWrapper.value.getBoundingClientRect();
        const containerLeft = containerRect.left;
        const containerRight = containerRect.right;
        
        // 移除所有现有的可见标记
        clusterItems.forEach(item => {
            item.classList.remove('cluster-visible');
        });
        
        // 检查每个卡片是否在可视范围内
        cards.forEach(card => {
            const cardRect = card.getBoundingClientRect();
            const cardLeft = cardRect.left;
            const cardRight = cardRect.right;
            
            // 如果卡片有一部分在可视范围内
            if (cardRight > containerLeft && cardLeft < containerRight) {
                const nodeId = card.getAttribute('data-node-id');
                const clusterItem = overviewSvg.value.querySelector(`.cluster-item[data-node-id="${nodeId}"]`);
                
                if (clusterItem) {
                    clusterItem.classList.add('cluster-visible');
                }
            }
        });
    } catch (error) {
        console.error('Error updating visible cards:', error);
    }
}

// 添加缺失的createArcPath函数
// 创建弧线路径的辅助函数
function createArcPath(x1, y1, x2, y2, verticalOffset) {
    // 计算两点之间的距离
    const distance = Math.abs(x2 - x1);
    
    // 设置圆角半径
    const cornerRadius = 5;
    
    // 使用传入的垂直偏移量，如果没有传入则计算
    const offset = verticalOffset !== undefined ? verticalOffset : (() => {
        // 根据距离计算垂直线段的高度
        const minOffset = 8; // 最小高度
        const maxOffset = 25; // 最大高度
        const normalizedDistance = Math.min(distance / 200, 1); // 归一化距离，最大200px
        return minOffset + Math.round(normalizedDistance * normalizedDistance * (maxOffset - minOffset));
    })();
    
    // 计算垂直线段的Y坐标
    const verticalY = y1 - offset;
    
    // 创建圆角方形路径，确保圆角方向正确
    if (x1 < x2) {
        // 从左到右
        return `M ${x1} ${y1} 
                L ${x1} ${y1 - cornerRadius} 
                Q ${x1} ${verticalY} ${x1 + cornerRadius} ${verticalY} 
                L ${x2 - cornerRadius} ${verticalY} 
                Q ${x2} ${verticalY} ${x2} ${verticalY + cornerRadius} 
                L ${x2} ${y2}`;
    } else {
        // 从右到左
        return `M ${x1} ${y1} 
                L ${x1} ${y1 - cornerRadius} 
                Q ${x1} ${verticalY} ${x1 - cornerRadius} ${verticalY} 
                L ${x2 + cornerRadius} ${verticalY} 
                Q ${x2} ${verticalY} ${x2} ${verticalY + cornerRadius} 
                L ${x2} ${y2}`;
    }
}

// 添加缺失的calculateArcMidPoint和calculateArrowAngle函数
// 计算弧线中点的辅助函数
function calculateArcMidPoint(x1, y1, x2, y2, verticalOffset) {
    // 计算两点之间的距离
    const distance = Math.abs(x2 - x1);
    
    // 使用传入的垂直偏移量，如果没有传入则计算
    const offset = verticalOffset !== undefined ? verticalOffset : (() => {
        // 使用与createArcPath相同的逻辑计算高度
        const minOffset = 8;
        const maxOffset = 25;
        const normalizedDistance = Math.min(distance / 200, 1);
        return minOffset + Math.round(normalizedDistance * normalizedDistance * (maxOffset - minOffset));
    })();
    
    // 计算垂直线段的Y坐标
    const verticalY = y1 - offset;
    
    // 计算水平线段的中点
    const midX = (x1 + x2) / 2;
    
    return { x: midX, y: verticalY };
}

// 计算箭头角度的辅助函数
function calculateArrowAngle(x1, y1, x2, y2) {
    // 确定箭头方向 - 从核心指向外延
    return x1 < x2 ? 0 : 180; // 水平向右或水平向左
}
</script>

<style scoped>
.core-graph-container {
    width: 100%;
    height: calc(100%);
    /* 减去标题的高度 */
    overflow: hidden;
    border: none;
    position: relative;
}

/* 总览条样式 - 独立于卡片样式 */
.clusters-overview {
    width: 100%; /* 将宽度从100%缩小到80% */
    height: 70px;
    /* 增加高度，从70px改为85px，以容纳滚动条 */
    padding: 0;
    margin-bottom: 10px;
    margin-left: auto; /* 添加左右自动边距使其居中 */
    margin-right: auto;
    background: linear-gradient(to bottom, #f8f9fa, #f1f3f4);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    display: flex;
    align-items: center;
    justify-content: center; /* 修改为center，使内容居中 */
    overflow: visible;
    /* 修改为visible，确保内容不被裁切 */
    position: relative;
    z-index: 10;
    /* 添加z-index，确保总览条在其他元素之上 */
}

/* 添加总览条标题样式 */
.overview-title {
    position: absolute; /* 改为绝对定位 */
    font-size: 1.4em; /* 增大字号 */
    font-weight: 500;
    left: 25px; /* 距左边距 */
    top: 50%; /* 垂直居中 */
    transform: translateY(-30%); /* 垂直居中 */
    color: #333; /* 使用主题色 */
    white-space: nowrap;
    display: flex;
    align-items: center;
    font-family: 'Poppins', sans-serif;
    z-index: 2; /* 确保标题在其他内容之上 */
}

/* 添加一个新的容器来包裹overview-wrapper */
.overview-wrapper-container {
    flex: 1;
    display: flex;
    align-items: center;
    padding-left: 180px;
    height: 100%;
    padding-top: 15px;
}

.overview-wrapper {
    flex: 0 0 auto; /* 不伸缩，保持自身大小 */
    height: 100%;
    overflow-x: visible;
    /* 修改为visible，确保内容不被裁切 */
    overflow-y: visible;
    /* 修改为visible，确保内容不被裁切 */
    display: flex;
    flex-direction: column;
    /* 修改为列布局，使滚动条位于SVG下方 */
    align-items: center; /* 默认居中 */
    justify-content: center;
    padding: 0;
    scrollbar-width: none;
    -ms-overflow-style: none;
    position: relative;
}

.overview-wrapper::-webkit-scrollbar {
    display: none;
}

.overview-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: -5px; /* 稍微上移，视觉上更居中 */
}

.overview-svg {
    height: 50px;
    /* 增加高度，确保有足够的空间显示连接线 */
    width: 100%;
    margin-top: 0; /* 调整上边距 */
    overflow: visible;
    /* 确保内容可以超出SVG边界 */
}

/* 添加总览条内滚动条容器样式 */
.overview-scrollbar-container {
    height: 12px;
    width: 100%;
    box-sizing: border-box;
    margin-top: 5px;
}

/* 连接线样式 */
.connection-lines {
    overflow: visible;
    /* 确保连接线可以超出容器边界 */
}

.connection-lines path {
    transition: all 0.3s ease;
    pointer-events: none;
    /* 禁用鼠标事件，防止干扰 */
    opacity: 1;
    /* 增加不透明度 */
    stroke-linecap: round;
    stroke-linejoin: round;
    overflow: visible;
    /* 确保连接线可以超出容器边界 */
}


.cluster-item {
    cursor: pointer;
    transition: all 0.2s ease;
}


/* 临时高亮动画样式 */
@keyframes pulse-highlight-subtle {
    0% {
        box-shadow: 0 0 0 0 rgba(144, 95, 41, 0.3);
        opacity: 0.3;
    }

    50% {
        box-shadow: 0 0 0 4px rgba(144, 95, 41, 0.15);
        opacity: 0.15;
    }

    100% {
        box-shadow: 0 0 0 0 rgba(144, 95, 41, 0);
        opacity: 0;
    }
}

/* 替换原有的脉冲动画，保持一致性 */
@keyframes pulse-highlight {
    0% {
        box-shadow: 0 0 0 0 rgba(144, 95, 41, 0.3);
        opacity: 0.3;
    }

    50% {
        box-shadow: 0 0 0 4px rgba(144, 95, 41, 0.15);
        opacity: 0.15;
    }

    100% {
        box-shadow: 0 0 0 0 rgba(144, 95, 41, 0);
        opacity: 0;
    }
}

/* 卡片容器样式 - 保持原有逻辑 */
.cards-container {
    width: 100%;
    height: calc(100% - 95px);
    /* 调整高度计算，从80px改为95px，适应总览条高度的增加 */
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

/* 恢复滚动相关样式 */
.cards-wrapper {
    display: flex;
    flex: 1;
    padding: 5px 0px 0px 0px;
    gap: 16px;
    overflow-x: auto;
    scroll-behavior: smooth;
    scrollbar-width: none;
    -ms-overflow-style: none;
    cursor: grab;
    user-select: none;
    -webkit-overflow-scrolling: touch;
    width: 100%;
    box-sizing: border-box;
    flex-wrap: nowrap;
}

.cards-wrapper::-webkit-scrollbar {
    display: none;
}

/* 自定义滚动条轨道 */
.custom-scrollbar-track {
    width: 100%;
    height: 4px;
    background: rgba(144, 95, 41, 0.1);
    border-radius: 2px;
    position: relative;
    cursor: pointer;
}

/* 自定义滚动条滑块 */
.custom-scrollbar-thumb {
    height: 100%;
    background: rgba(144, 95, 41, 0.5);
    border-radius: 2px;
    position: absolute;
    top: 0;
    cursor: grab;
    transition: background 0.2s;
}

.custom-scrollbar-thumb:hover {
    background: rgba(144, 95, 41, 0.7);
}

.custom-scrollbar-thumb:active {
    background: rgba(144, 95, 41, 0.8);
    cursor: grabbing;
}

/* 卡片样式 */
.card {
    flex: 0 0 auto;
    width: min(400px, calc(100% - 32px));
    max-width: 400px;
    min-width: 280px;
    height: 100%;
    background: #ffffff;
    border-radius: 12px;
    border: 1.5px solid #905F29;
    display: flex;
    flex-direction: column;
    transition: all 0.2s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

@media (max-width: 768px) {
    .card {
        width: calc(100% - 32px);
        min-width: 240px;
    }
}

.card:hover {
    transform: translateY(-2px);
    border-color: #7F5427;
    box-shadow: 0 4px 6px rgba(144, 95, 41, 0.15);
}

.card-core {
    border-color: #905F29;
}

.card-core:hover {
    border-color: #7F5427;
    box-shadow: 0 4px 6px rgba(144, 95, 41, 0.15);
}

.card-extension {
    border-color: #905F29;
}

.card-extension:hover {
    border-color: #7F5427;
    box-shadow: 0 4px 6px rgba(144, 95, 41, 0.15);
}

.card.has-extension {
    border-color: #905F29;
}

.card.has-extension:hover {
    border-color: #7F5427;
    box-shadow: 0 4px 6px rgba(144, 95, 41, 0.15);
}

/* 移除扩展指示器和导航控制相关样式 */

.card-svg-container {
    flex: 1;
    min-height: 0;
    padding: 0;
    border-bottom: 1px solid #e8eaed;
    pointer-events: none;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.card-svg-container svg {
    width: 100%;
    height: 100%;
    max-height: 100%;
    object-fit: contain;
}

.card-info {
    flex: 0 0 auto;
    padding: 8px 12px;
    background: #f8f9fa;
    border-bottom-left-radius: 12px;
    max-height: 90px;
    overflow-y: auto;
    position: relative;
    padding-right: 120px;
    border-top: 1px solid rgba(144, 95, 41, 0.2);
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.highlight-stats {
    font-size: 14px;
    color: #5f6368;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    font-weight: 500;
    align-items: center;
}

.highlight-stats span {
    background: #f1f3f4;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 500;
}

.encodings-wrapper {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
    margin-top: -6px;
}

.visual-encodings {
    font-size: 14px;
    color: #5f6368;
    font-weight: 500;
    display: inline-block;
    width: 170px;
    min-width: 120px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-shrink: 0;
}

.analysis-content {
    margin-top: 0;
    font-size: 14px;
    line-height: 1.4;
    gap: 4px;
    padding-right: 8px;
    max-width: calc(100% - 50px);
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    color: #905F29;
    margin-left: 4px;
    flex: 1;
}

:deep(.feature-tag) {
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 14px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
    height: 20px;
    margin-right: 4px;
    margin-bottom: 2px;
    margin-top: 2px;
    background-color: rgba(144, 95, 41, 0.08);
    color: #905F29;
    border: 1px solid rgba(144, 95, 41, 0.2);
}

.loading-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.loading-spinner {
    width: 40px;
    height: 40px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #905F29;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

.attention-probability {
    position: absolute;
    bottom: 8px;
    right: 12px;
    font-size: 3em;
    font-weight: 800;
    color: #905F29;
    padding: 4px 8px;
    border-radius: 6px;
    background: rgba(144, 95, 41, 0.08);
    border: 1px solid rgba(144, 95, 41, 0.2);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-width: 90px;
    text-align: center;
}

.attention-probability-label {
    font-size: 0.4em;
    line-height: 1.2;
    margin-bottom: 2px;
    white-space: nowrap;
    opacity: 0.8;
    width: 100%;
}

.attention-probability-value {
    font-size: 0.5em;
    line-height: 1.2;
    color: #b4793a;
    white-space: nowrap;
    width: 100%;
    font-weight: 650;
}

.card-extension .attention-probability {
    color: #905F29;
    background: rgba(144, 95, 41, 0.08);
    border: 1px solid rgba(144, 95, 41, 0.2);
}
</style>
