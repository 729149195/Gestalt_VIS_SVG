<template>
    <span class="title">Graphical Patterns List</span>
    <div class="core-graph-container">
        
        <div v-if="loading" class="loading-overlay">
            <div class="loading-spinner"></div>
        </div>
        <div class="cards-container" ref="cardsContainer">
            <div class="cards-wrapper" ref="cardsWrapper" @mousedown="startDrag" @mousemove="onDrag" @mouseup="endDrag" @mouseleave="endDrag">
                <div v-for="node in sortedNodes" 
                     :key="node.id" 
                     class="card"
                     :class="{ 
                         'card-selected': isNodeSelected(node),
                         'has-extension': hasExtension(node)
                     }"
                     :data-type="node.type"
                     :data-node-id="node.id"
                     @click="!isDragging && showNodeList(node)">
                    <div v-if="hasExtension(node)" class="extension-indicator">
                        <v-icon color="primary">mdi-layers</v-icon>
                        <span class="extension-count">{{ getExtensionCount(node) }}</span>
                    </div>
                    
                    <div v-if="hasExtension(node)" class="page-controls">
                        <span class="page-info">{{ getCurrentPage(node) }}/{{ getTotalPages(node) }}</span>
                        <div class="page-buttons">
                            <v-btn 
                                icon="mdi-chevron-left" 
                                size="small"
                                @click.stop="prevPage(node)"
                                :disabled="isFirstPage(node)"
                            ></v-btn>
                            <v-btn 
                                icon="mdi-chevron-right" 
                                size="small"
                                @click.stop="nextPage(node)"
                                :disabled="isLastPage(node)"
                            ></v-btn>
                        </div>
                    </div>

                    <div class="card-svg-container" ref="graphContainer"></div>
                    <div class="card-info">
                        <div class="highlight-stats">
                            <template v-if="Object.keys(getStats(node)).length > 0">
                                Included elements: <span v-for="(count, type) in getStats(node)" :key="type">
                                    {{ count }}  <{{ type }}>
                                </span>
                            </template>
                        </div>
                        <div class="analysis-content" v-html="generateAnalysis(node)"></div>
                        <div class="attention-probability">
                            {{ (calculateAttentionProbability(node) * 100).toFixed(3) }}%
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
const graphData = ref(null);
const nodes = ref([]);
const originalSvgContent = ref('');
const loading = ref(true);
const currentPages = ref(new Map());
const elementStats = ref(new Map());

// 添加缩略图缓存
const thumbnailCache = new Map();

// 添加拖动相关的状态
const isDragging = ref(false);
const startX = ref(0);
const scrollLeft = ref(0);
const lastX = ref(0);
const velocity = ref(0);
const animationFrame = ref(null);

// 添加扩展相关的方法
const hasExtension = (node) => {
    return node.extensions && node.extensions.length > 0;
};

const getExtensionCount = (node) => {
    return node.extensions ? node.extensions.length : 0;
};

const getCurrentPage = (node) => {
    return currentPages.value.get(node.id) || 1;
};

const getTotalPages = (node) => {
    return hasExtension(node) ? node.extensions.length + 1 : 1;
};

const isFirstPage = (node) => {
    return getCurrentPage(node) === 1;
};

const isLastPage = (node) => {
    return getCurrentPage(node) === getTotalPages(node);
};

const prevPage = (node) => {
    if (!isFirstPage(node)) {
        currentPages.value.set(node.id, getCurrentPage(node) - 1);
        updateNodeDisplay(node);
    }
};

const nextPage = (node) => {
    if (!isLastPage(node)) {
        currentPages.value.set(node.id, getCurrentPage(node) + 1);
        updateNodeDisplay(node);
    }
};

const updateNodeDisplay = (node) => {
    const currentPage = getCurrentPage(node);
    let displayNodes = [];
    
    if (currentPage === 1) {
        // 第一页只显示核心节点
        displayNodes = [...node.originalNodes];
    } else {
        // 其他页面显示对应的外延节点
        const extension = node.extensions[currentPage - 2];
        if (extension) {
            displayNodes = [...extension.originalNodes];
        }
    }
    
    // 更新缩略图
    const container = document.querySelector(`[data-node-id="${node.id}"] .card-svg-container`);
    if (container) {
        const nodeData = {
            id: node.id,
            type: currentPage === 1 ? 'core' : 'extension',
            originalNodes: displayNodes
        };
        renderGraph(container, { nodes: [nodeData] });
    }
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
            console.error('SVG元素未找到');
            return '';
        }

        const clonedSvg = svgElement.cloneNode(true);
        
        clonedSvg.querySelectorAll('*').forEach(el => {
            if (el.tagName !== 'svg' && el.tagName !== 'g') {
                el.style.opacity = '0.2';
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
            const coreIndex = parseInt(nodeData.id.split('_')[1]);
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
        console.error('创建缩略图时出错:', error);
        return '';
    }
}

// 处理核心聚类数据
function processGraphData(coreData) {
    const processedNodes = [];
    
    // 处理核心节点和外延节点
    coreData.core_clusters.forEach((cluster, clusterIndex) => {
        const coreNodeId = `core_${clusterIndex}`;
        
        // 添加核心节点
        processedNodes.push({
            id: coreNodeId,
            name: `co ${clusterIndex + 1} (Z_${cluster.core_dimensions.join(',Z_')})`,
            type: 'core',
            originalNodes: cluster.core_nodes,
            dimensions: cluster.core_dimensions,
            extensionCount: cluster.extensions.length,
            extensions: [],  // 初始化为空数组
            value: 1
        });

        // 添加外延节点
        cluster.extensions.forEach((extension, extIndex) => {
            const extensionNode = {
                id: `ext_${clusterIndex}_${extIndex}`,
                name: `ext ${clusterIndex + 1}.${extIndex + 1}`,
                type: 'extension',
                originalNodes: extension.nodes,
                dimensions: extension.dimensions,
                parentCoreId: coreNodeId,
                value: 1
            };
            processedNodes.push(extensionNode);
            // 将扩展节点添加到对应核心节点的extensions数组中
            const coreNode = processedNodes.find(n => n.id === coreNodeId);
            if (coreNode) {
                coreNode.extensions.push(extensionNode);
            }
        });
    });

    return { nodes: processedNodes };
}

// 显示节点列表
function showNodeList(node) {
    try {
        // 获取当前卡片中的SVG元素
        const cardContainer = document.querySelector(`[data-node-id="${node.id}"] .card-svg-container svg`);
        if (!cardContainer) {
            console.error('找不到卡片SVG容器');
            return;
        }

        // 获取所有不透明度为1的元素（即高亮元素）
        const highlightedElements = Array.from(cardContainer.querySelectorAll('*'))
            .filter(el => el.style.opacity === '1' && el.id && el.tagName !== 'svg' && el.tagName !== 'g');

        // 收集这些元素的ID
        const nodeNames = highlightedElements.map(el => el.id);

        store.commit('UPDATE_SELECTED_NODES', { nodeIds: nodeNames, group: null });
    } catch (error) {
        console.error('获取高亮节点时出错:', error);
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
    isDragging.value = true;
    startX.value = e.pageX - cardsWrapper.value.offsetLeft;
    lastX.value = e.pageX;
    scrollLeft.value = cardsWrapper.value.scrollLeft;
    cardsWrapper.value.style.cursor = 'grabbing';
    cardsWrapper.value.style.userSelect = 'none';
    
    // 停止任何正在进行的动画
    if (animationFrame.value) {
        cancelAnimationFrame(animationFrame.value);
    }
}

function onDrag(e) {
    if (!isDragging.value) return;
    e.preventDefault();
    
    const x = e.pageX;
    velocity.value = x - lastX.value;
    lastX.value = x;
    
    const walk = (x - startX.value);
    cardsWrapper.value.scrollLeft = scrollLeft.value - walk;
}

function endDrag(e) {
    if (!isDragging.value) return;
    isDragging.value = false;
    cardsWrapper.value.style.cursor = 'grab';
    cardsWrapper.value.style.userSelect = '';

    // 添加惯性滚动
    if (Math.abs(velocity.value) > 1) {
        const startTime = Date.now();
        const startVelocity = velocity.value;
        const startScroll = cardsWrapper.value.scrollLeft;
        
        function momentumScroll() {
            const elapsed = Date.now() - startTime;
            const remaining = Math.max(0, Math.abs(startVelocity) * 500 - elapsed); // 500ms的减速时间
            const speed = (remaining / (Math.abs(startVelocity) * 500)) * startVelocity;
            
            if (remaining > 0) {
                cardsWrapper.value.scrollLeft -= speed;
                animationFrame.value = requestAnimationFrame(momentumScroll);
            }
        }
        
        momentumScroll();
    }
}

// 修改渲染图形的方法
function renderGraph(container, graphData) {
    if (!container || !graphData) return;

    // 清除所有现有的SVG
    const containerElement = d3.select(container);
    containerElement.selectAll('svg').remove();

    const cardWidth = 368;
    const cardHeight = container.clientHeight - 120;

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

    const padding = 16;
    const availableWidth = cardWidth - (padding * 2);
    const availableHeight = cardHeight - (padding * 2);

    const coreNode = g.append('g')
        .attr('class', 'node')
        .attr('transform', `translate(${padding},${padding})`);

    const foreignObject = coreNode.append('foreignObject')
        .attr('width', availableWidth)
        .attr('height', availableHeight)
        .attr('x', 0)
        .attr('y', 0);

    const div = foreignObject.append('xhtml:div')
        .style('width', '100%')
        .style('height', '100%')
        .style('overflow', 'hidden')
        .style('border-radius', '6px')
        .style('background', '#fafafa')
        .style('display', 'flex')
        .style('align-items', 'center')
        .style('justify-content', 'center');

    requestIdleCallback(() => {
        const thumbnailContent = createThumbnail(nodeData);
        div.html(thumbnailContent);

        const thumbnailSvg = div.select('svg').node();
        if (thumbnailSvg) {
            const bbox = thumbnailSvg.getBBox();
            const padding = 20;
            thumbnailSvg.setAttribute('viewBox', `${bbox.x - padding} ${bbox.y - padding} ${bbox.width + padding * 2} ${bbox.height + padding * 2}`);
            thumbnailSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
            
            // 在SVG渲染完成后更新统计信息
            elementStats.value.set(nodeData.id, getHighlightedElementsStats(nodeData));
        }
    });
}

// 添加数据源URL
const MAPPING_DATA_URL = "http://127.0.0.1:5000/average_equivalent_mapping";
const EQUIVALENT_WEIGHTS_URL = "http://127.0.0.1:5000/equivalent_weights_by_tag";

// 特征名称映射
// const featureNameMap = {
//     'tag_name': '元素名称',
//     'tag': '标签类型',
//     'opacity': '不透明度',
//     'fill_h_cos': '填充色相',
//     'fill_h_sin': '填充色相',
//     'fill_s_n': '填充饱和度',
//     'fill_l_n': '填充亮度',
//     'stroke_h_cos': '描边色相',
//     'stroke_h_sin': '描边色相',
//     'stroke_s_n': '描边饱和度',
//     'stroke_l_n': '描边亮度',
//     'stroke_width': '描边宽度',
//     'bbox_left_n': '左边界位置',
//     'bbox_right_n': '右边界位置',
//     'bbox_top_n': '上边界位置',
//     'bbox_bottom_n': '下边界位置',
//     'bbox_mds_1': '位置mds特征1',
//     'bbox_mds_2': '位置mds特征2',
//     'bbox_center_x_n': '中心X坐标',
//     'bbox_center_y_n': '中心Y坐标',
//     'bbox_width_n': '宽度',
//     'bbox_height_n': '高度',
//     'bbox_fill_area': '元素面积'
// };
const featureNameMap = {
    'tag': 'Label Type',
    'opacity': 'opacity',
    'fill_h_cos': 'fill color',
    'fill_h_sin': 'fill color',
    'fill_s_n': 'fill color ',
    'fill_l_n': 'fill color',
    'stroke_h_cos': 'stroke color',
    'stroke_h_sin': 'stroke color',
    'stroke_s_n': 'stroke color',
    'stroke_l_n': 'stroke color',
    'stroke_width': 'stroke width',
    'bbox_left_n': 'position',
    'bbox_right_n': 'position',
    'bbox_top_n': 'position',
    'bbox_bottom_n': 'position',
    'bbox_mds_1': 'position',
    'bbox_mds_2': 'position',
    'bbox_center_x_n': 'position',
    'bbox_center_y_n': 'position',
    'bbox_width_n': 'width / height',
    'bbox_height_n': 'width / height',
    'bbox_fill_area': 'area'
};

// 添加分析数据的ref
const analysisData = ref(null);
const equivalentWeightsData = ref(null);

// 生成分析文字的函数
const generateAnalysis = (nodeData) => {
    if (!analysisData.value || !equivalentWeightsData.value) return '';

    const currentPage = getCurrentPage(nodeData);
    let nodesToAnalyze = [];
    
    if (currentPage === 1) {
        // 第一页只分析核心节点
        nodesToAnalyze = [...nodeData.originalNodes];
    } else {
        // 其他页面分析对应的外延节点
        const extension = nodeData.extensions[currentPage - 2];
        if (extension) {
            nodesToAnalyze = [...extension.originalNodes];
        }
    }

    // 获取这些元素的权重数据
    const nodeWeights = {};
    nodesToAnalyze.forEach(nodeId => {
        const fullPath = nodeId.startsWith('svg/') ? nodeId : `svg/${nodeId}`;
        if (equivalentWeightsData.value[fullPath]) {
            nodeWeights[fullPath] = equivalentWeightsData.value[fullPath];
        } else {
            const altPath = nodeId.replace(/^svg\//, '');
            if (equivalentWeightsData.value[`svg/${altPath}`]) {
                nodeWeights[fullPath] = equivalentWeightsData.value[`svg/${altPath}`];
            }
        }
    });

    if (Object.keys(nodeWeights).length === 0) {
        return '';
    }

    // 使用现有的分析逻辑继续处理
    const inputDimensions = analysisData.value.input_dimensions;
    const outputDimensions = analysisData.value.output_dimensions;
    const dimensionCount = outputDimensions.length;
    const featureCount = inputDimensions.length;
    const averageWeights = Array(dimensionCount).fill().map(() => Array(featureCount).fill(0));

    Object.values(nodeWeights).forEach(weights => {
        weights.forEach((dimWeights, dimIndex) => {
            dimWeights.forEach((weight, featureIndex) => {
                averageWeights[dimIndex][featureIndex] += weight / Object.keys(nodeWeights).length;
            });
        });
    });

    const featureMap = new Map();
    averageWeights.forEach((dimensionWeights, dimIndex) => {
        dimensionWeights.forEach((weight, featureIndex) => {
            const featureName = inputDimensions[featureIndex];
            const displayName = featureNameMap[featureName] || featureName;
            const absWeight = Math.abs(weight);
            
            if (!featureMap.has(displayName) || absWeight > Math.abs(featureMap.get(displayName).weight)) {
                featureMap.set(displayName, { weight, absWeight });
            }
        });
    });

    let sortedFeatures = Array.from(featureMap.entries())
        .sort((a, b) => b[1].absWeight - a[1].absWeight)
        .filter(([_, {absWeight}]) => absWeight > 0.1);

    if (sortedFeatures.length > 1) {
        const weightDiffs = [];
        for (let i = 1; i < sortedFeatures.length; i++) {
            const diff = sortedFeatures[i-1][1].absWeight - sortedFeatures[i][1].absWeight;
            weightDiffs.push({
                index: i,
                diff: diff
            });
        }

        const maxDiff = weightDiffs.reduce((max, curr) => curr.diff > max.diff ? curr : max);
        
        if (maxDiff.diff > sortedFeatures[0][1].absWeight * 0.2 && maxDiff.index <= 3) {
            sortedFeatures = sortedFeatures.slice(0, maxDiff.index);
        } else {
            sortedFeatures = sortedFeatures.slice(0, 3);
        }
    }
    // ${name} ${absWeight}
    return sortedFeatures.map(([name, {weight}]) => {
        const absWeight = Math.abs(weight).toFixed(2);
        const color = weight > 0 ? '#1E88E5' : '#1E88E5';
        return `<span class="feature-tag" style="color: ${color}; border-color: ${color}20; background-color: ${color}08">
            ${name} 
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
        console.error('获取分析数据失败:', error);
    }
};

// 首先添加一个ref来存储特征数据
const clusterFeatures = ref(null);

// 在loadAndRenderGraph函数中添加获取特征数据的逻辑
async function loadAndRenderGraph() {
    try {
        loading.value = true;
        const [svgResponse, graphResponse, featuresResponse] = await Promise.all([
            fetch('http://127.0.0.1:5000/get_svg'),
            fetch('http://127.0.0.1:5000/static/data/subgraphs/subgraph_dimension_all.json'),
            fetch('http://127.0.0.1:5000/cluster_features'),  // 添加获取特征数据
            fetchAnalysisData()
        ]);
        
        const [svgContent, data, featuresData] = await Promise.all([
            svgResponse.text(),
            graphResponse.json(),
            featuresResponse.json()
        ]);
        
        clusterFeatures.value = featuresData;
        originalSvgContent.value = svgContent;
        graphData.value = data;
        
        // 处理数据，将扩展节点集成到核心节点中
        const processedData = processGraphData(data);
        const coreNodes = processedData.nodes.filter(node => !node.id.includes('extension_'));
        const extensionNodes = processedData.nodes.filter(node => node.id.includes('extension_'));
        
        // 将扩展节点添加到对应的核心节点中
        coreNodes.forEach(coreNode => {
            const coreIndex = coreNode.id.split('_')[1];
            const relatedExtensions = extensionNodes.filter(ext => 
                ext.id.includes(`extension_${coreIndex}_`)
            );
            if (relatedExtensions.length > 0) {
                coreNode.extensions = relatedExtensions;
            }
        });
        
        nodes.value = coreNodes;

        // 为每个卡片渲染图形
        nextTick(() => {
            const containers = document.querySelectorAll('.card-svg-container');
            containers.forEach((container, index) => {
                if (nodes.value[index]) {
                    const nodeData = { nodes: [nodes.value[index]] };
                    renderGraph(container, nodeData);
                }
            });

            // 添加一个延时以确保SVG已经完全渲染
            setTimeout(() => {
                const cards = document.querySelectorAll('.card');
                cards.forEach((card) => {
                    const nodeId = card.getAttribute('data-node-id');
                    const node = nodes.value.find(n => n.id === nodeId);
                    if (node) {
                        getHighlightedElementsStats(node);
                    }
                });
            }, 100);
        });
    } catch (error) {
        console.error('Error loading data:', error);
    } finally {
        loading.value = false;
    }
}

onMounted(async () => {
    await nextTick();
    await loadAndRenderGraph();
    // 初始化统计信息
    nextTick(() => {
        nodes.value.forEach(node => {
            elementStats.value.set(node.id, getHighlightedElementsStats(node));
        });
    });
});

// 监听选中节点的变化
watch(selectedNodeIds, () => {
    nextTick(() => {
        const containers = document.querySelectorAll('.card-svg-container');
        containers.forEach((container, index) => {
            if (nodes.value[index]) {
                const nodeData = { nodes: [nodes.value[index]] };
                renderGraph(container, nodeData);
            }
        });
    });
});

// 监听页面变化
watch(currentPages, () => {
    nextTick(() => {
        nodes.value.forEach(node => {
            elementStats.value.set(node.id, getHighlightedElementsStats(node));
        });
    });
}, { deep: true });

function getHighlightedElementsStats(nodeData) {
    try {
        // 获取当前卡片中的SVG元素
        const cardContainer = document.querySelector(`[data-node-id="${nodeData.id}"] .card-svg-container svg`);
        if (!cardContainer) {
            console.error('找不到卡片SVG容器');
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
        console.error('统计高亮元素时出错:', error);
        return {};
    }
}

// 修改注意力概率计算函数
const calculateAttentionProbability = (nodeData) => {
    if (!clusterFeatures.value) return 0;

    try {
        // 1. 获取当前需要分析的节点
        const currentPage = getCurrentPage(nodeData);
        let nodesToAnalyze = [];
        
        if (currentPage === 1) {
            nodesToAnalyze = [...nodeData.originalNodes];
        } else {
            const extension = nodeData.extensions[currentPage - 2];
            if (extension) {
                nodesToAnalyze = [...extension.originalNodes];
            }
        }

        if (nodesToAnalyze.length === 0) return 0.1;

        // 2. 将所有节点分为高亮组和非高亮组
        const highlightedFeatures = [];
        const nonHighlightedFeatures = [];

        // 修改这里的路径匹配逻辑
        clusterFeatures.value.forEach(item => {
            // 标准化路径格式
            const normalizedItemId = item.id;
            const isHighlighted = nodesToAnalyze.some(analyzeNode => {
                // 移除开头的 'svg/' 并标准化分析节点的路径
                const normalizedAnalyzeNode = analyzeNode.replace(/^svg\//, '');
                return normalizedItemId === `svg/${normalizedAnalyzeNode}` || 
                       normalizedItemId === normalizedAnalyzeNode;
            });

            if (isHighlighted) {
                highlightedFeatures.push(item.features);
            } else {
                nonHighlightedFeatures.push(item.features);
            }
        });

        if (highlightedFeatures.length === 0 || nonHighlightedFeatures.length === 0) {
            return 0.1;
        }

        // 3. 计算高亮组的平均特征向量
        const highlightedMean = highlightedFeatures[0].map((_, featureIndex) => {
            return highlightedFeatures.reduce((sum, features) => 
                sum + features[featureIndex], 0) / highlightedFeatures.length;
        });

        // 4. 计算非高亮组的平均特征向量
        const nonHighlightedMean = nonHighlightedFeatures[0].map((_, featureIndex) => {
            return nonHighlightedFeatures.reduce((sum, features) => 
                sum + features[featureIndex], 0) / nonHighlightedFeatures.length;
        });

        // 5. 计算两组之间的欧氏距离
        const euclideanDistance = Math.sqrt(
            highlightedMean.reduce((sum, value, index) => {
                const diff = value - nonHighlightedMean[index];
                return sum + (diff * diff);
            }, 0)
        );

        // 6. 计算组内的平均距离（衡量组内聚集程度）
        const avgIntraDistance = highlightedFeatures.reduce((sum, features) => {
            const distance = Math.sqrt(
                features.reduce((s, value, index) => {
                    const diff = value - highlightedMean[index];
                    return s + (diff * diff);
                }, 0)
            );
            return sum + distance;
        }, 0) / highlightedFeatures.length;

        // 7. 调整评分计算
        const score = euclideanDistance / (avgIntraDistance + 0.1);
        
        // 调整sigmoid函数的参数以获得更好的分布
        const sigmoid = x => 1 / (1 + Math.exp(-3 * (x - 0.5)));
        const normalizedScore = 0.1 + sigmoid(score) * 0.8;

        // 调整节点数量因子的影响
        const nodeCountFactor = Math.min(highlightedFeatures.length / 5, 1); // 改为5个节点达到最大影响
        const finalScore = normalizedScore * (0.7 + nodeCountFactor * 0.3); // 增加节点数量的影响权重

        // 添加调试信息
        console.log('Node Data:', {
            euclideanDistance,
            avgIntraDistance,
            score,
            normalizedScore,
            nodeCountFactor,
            finalScore
        });

        return Math.min(Math.max(finalScore, 0.1), 0.9);

    } catch (error) {
        console.error('计算注意力概率时出错:', error);
        console.error('错误详情:', error.stack);
        return 0.1;
    }
};

// 添加计算排序后节点的计算属性
const sortedNodes = computed(() => {
    if (!nodes.value) return [];
    return [...nodes.value].sort((a, b) => 
        calculateAttentionProbability(b) - calculateAttentionProbability(a)
    );
});

// 添加计算属性来获取统计信息
const getStats = (node) => {
    return elementStats.value.get(node.id) || {};
};
</script>

<style scoped>
.core-graph-container {
    width: 100%;
    height: calc(100% - 40px);  /* 减去标题的高度 */
    background: #ffffff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(151, 151, 151, 0.12), 0 1px 2px rgba(119, 119, 119, 0.08);
    overflow: hidden;
    border: none;
    position: relative;
    margin-top: 0;  /* 移除顶部边距 */
}

.cards-container {
    width: 100%;
    height: 100%;
    position: relative;
    overflow: hidden;
}

.cards-wrapper {
    display: flex;
    height: 100%;
    padding: 16px;
    gap: 16px;
    overflow-x: auto;
    scroll-behavior: auto;
    scrollbar-width: none;
    -ms-overflow-style: none;
    cursor: grab;
    user-select: none;
    -webkit-overflow-scrolling: touch;
}

.cards-wrapper::-webkit-scrollbar {
    display: none;
}

.card {
    flex: 0 0 400px;
    height: 100%;
    background: #ffffff;
    border-radius: 12px;
    border: 1.5px solid #1a73e8;
    display: flex;
    flex-direction: column;
    transition: all 0.2s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.card:hover {
    transform: translateY(-2px);
    border-color: #1967d2;
    box-shadow: 0 4px 6px rgba(60, 64, 67, 0.15);
}

.card.has-extension {
    border-color: #34A853;
}

.card.has-extension:hover {
    border-color: #2E7D32;
    box-shadow: 0 4px 6px rgba(46, 125, 50, 0.15);
}

.extension-indicator {
    position: absolute;
    top: 16px;
    left: 16px;
    background: rgba(255, 255, 255, 0.95);
    padding: 6px 12px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    z-index: 2;
    backdrop-filter: blur(4px);
    border: 1px solid rgba(52, 168, 83, 0.15);
    transition: all 0.2s ease;
}

.extension-indicator:hover {
    background: rgba(255, 255, 255, 0.98);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}

.extension-count {
    font-size: 13px;
    font-weight: 600;
    color: #34A853;
    min-width: 16px;
    text-align: center;
}

.page-controls {
    position: absolute;
    top: 16px;
    right: 16px;
    background: rgba(255, 255, 255, 0.95);
    padding: 4px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    z-index: 2;
    backdrop-filter: blur(4px);
    border: 1px solid rgba(60, 64, 67, 0.15);
    transition: all 0.2s ease;
}

.page-controls:hover {
    background: rgba(255, 255, 255, 0.98);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}

.page-info {
    font-size: 13px;
    font-weight: 500;
    color: #3c4043;
    padding: 0 8px;
    user-select: none;
}

.page-buttons {
    display: flex;
    gap: 2px;
}

.page-buttons :deep(.v-btn) {
    background: transparent !important;
    color: #3c4043 !important;
    min-width: 32px !important;
    width: 32px !important;
    height: 32px !important;
    padding: 0 !important;
    border-radius: 16px !important;
}

.page-buttons :deep(.v-btn:hover) {
    background: rgba(60, 64, 67, 0.08) !important;
}

.page-buttons :deep(.v-btn:active) {
    background: rgba(60, 64, 67, 0.12) !important;
}

.page-buttons :deep(.v-btn--disabled) {
    color: rgba(60, 64, 67, 0.38) !important;
    background: transparent !important;
}

.page-buttons :deep(.v-btn__content) {
    opacity: 0.87;
}

.card-svg-container {
    flex: 1;
    min-height: 0;
    padding: 12px;
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
    display: block;
}

.card-info {
    flex: 0 0 auto;
    padding: 8px 12px;
    background: #f8f9fa;
    border-bottom-left-radius: 12px;
    border-bottom-right-radius: 12px;
    max-height: 80px;
    overflow-y: auto;
    position: relative;
    padding-right: 120px;
}

.highlight-stats {
    font-size: 12px;
    color: #5f6368;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    align-items: center;
    margin-bottom: 4px;
}

.highlight-stats span {
    background: #f1f3f4;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 500;
}

.analysis-content {
    margin-top: 4px;
    font-size: 14px;
    line-height: 1.4;
    gap: 4px;
    padding-right: 8px;
    max-width: calc(100% - 40px);
}

:deep(.feature-tag) {
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 11px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
    height: 20px;
    margin-right: 4px;
    margin-bottom: 2px;
}

/* 删除不再需要的样式 */
:deep(.dimension-analysis) {
    display: none;
}

:deep(.dimension-analysis:last-child) {
    display: none;
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
    border-top: 3px solid #1a73e8;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.attention-probability {
    position: absolute;
    bottom: 8px;
    right: 12px;
    font-size: 20px;
    font-weight: 600;
    color: #1a73e8;
    padding: 4px 8px;
    border-radius: 6px;
    background: rgba(26, 115, 232, 0.08);
    border: 1px solid rgba(26, 115, 232, 0.2);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    display: flex;
    align-items: center;
    gap: 4px;
    min-width: 90px;
    justify-content: center;
    white-space: nowrap;
}

.eye-icon {
    opacity: 0.9;
}

.title {
  margin: 12px 16px;
  font-size: 16px;
  font-weight: bold;
  color: #1d1d1f;
  letter-spacing: -0.01em;
  opacity: 0.8;
}
</style>


