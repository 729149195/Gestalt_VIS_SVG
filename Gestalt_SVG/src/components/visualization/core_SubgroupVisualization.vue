<template>
    <div class="core-graph-container">
        <div ref="graphContainer" class="graph-container"></div>
    </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch, onUnmounted, computed } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';

const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);
const graphContainer = ref(null);
const graphData = ref(null);
const nodes = ref([]);
const originalSvgContent = ref(''); // 存储原始SVG内容

// 从后端获取原始SVG内容
async function fetchOriginalSvg() {
    try {
        const response = await fetch('http://localhost:5000/get_svg');
        const svgContent = await response.text();
        originalSvgContent.value = svgContent;
    } catch (error) {
        console.error('Error fetching original SVG:', error);
    }
}

// 创建节点的缩略图
function createThumbnail(nodeData) {
    try {
        const parser = new DOMParser();
        const svgDoc = parser.parseFromString(originalSvgContent.value, 'image/svg+xml');
        const svgElement = svgDoc.querySelector('svg');
        
        if (!svgElement) {
            console.error('SVG元素未找到');
            return '';
        }

        // 先克隆SVG以避免修改原始内容
        const clonedSvg = svgElement.cloneNode(true);
        
        // 设置所有元素透明度为0.2（增加默认透明度使非高亮元素可见）
        clonedSvg.querySelectorAll('*').forEach(el => {
            if (el.tagName !== 'svg' && el.tagName !== 'g') {
                el.style.opacity = '0.2';
                // 保持原始颜色
                if (el.hasAttribute('fill')) {
                    el.style.fill = el.getAttribute('fill');
                }
                if (el.hasAttribute('stroke')) {
                    el.style.stroke = el.getAttribute('stroke');
                }
            }
        });
        
        // 高亮当前节点包含的元素
        let nodesToHighlight = [...nodeData.originalNodes];
        
        // 如果是外延节点，添加对应核心节点的元素
        if (nodeData.type === 'extension') {
            const coreIndex = parseInt(nodeData.id.split('_')[1]);
            const coreNode = nodes.value.find(n => n.id === `core_${coreIndex}`);
            if (coreNode) {
                nodesToHighlight = [...nodesToHighlight, ...coreNode.originalNodes];
            }
        }
        
        // 高亮所有需要高亮的节点
        let highlightedCount = 0;
        nodesToHighlight.forEach(nodeId => {
            const element = clonedSvg.getElementById(nodeId.split('/').pop());
            if (element) {
                element.style.opacity = '1';
                highlightedCount++;
                // 确保保持原始颜色
                if (element.hasAttribute('fill')) {
                    element.style.fill = element.getAttribute('fill');
                }
                if (element.hasAttribute('stroke')) {
                    element.style.stroke = element.getAttribute('stroke');
                }
            }
        });

        // 如果没有找到任何要高亮的元素，将所有元素设为可见
        if (highlightedCount === 0) {
            console.warn(`未找到要高亮的元素: ${nodesToHighlight.join(', ')}`);
            clonedSvg.querySelectorAll('*').forEach(el => {
                if (el.tagName !== 'svg' && el.tagName !== 'g') {
                    el.style.opacity = '1';
                }
            });
        }
        
        // 调整SVG大小为缩略图大小
        clonedSvg.setAttribute('width', '100%');
        clonedSvg.setAttribute('height', '100%');
        clonedSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        
        return clonedSvg.outerHTML;
    } catch (error) {
        console.error('创建缩略图时出错:', error);
        return '';
    }
}

// 处理核心聚类数据
function processGraphData(coreData) {
    const processedNodes = [];
    
    // 首先计算每个核心节点及其外延需要的总空间
    coreData.core_clusters.forEach((cluster, clusterIndex) => {
        const extensionCount = cluster.extensions.length;
        const coreNodeId = `core_${clusterIndex}`;
        
        // 调整核心节点和外延节点的大小比例，使其更加方正
        const coreWidth = 320; // 增加核心节点宽度
        const coreHeight = 240; // 调整核心节点高度，使其更方正
        const extensionWidth = extensionCount > 0 ? 160 : 0; // 增加外延节点宽度
        
        // 创建组合节点（包含核心和外延）
        processedNodes.push({
            id: coreNodeId,
            name: `co ${clusterIndex + 1} (Z_${cluster.core_dimensions.join(',Z_')})`,
            type: 'core',
            originalNodes: cluster.core_nodes,
            dimensions: cluster.dimensions,
            extensionCount,
            extensions: cluster.extensions,
            width: coreWidth,
            height: coreHeight,
            extensionWidth: extensionWidth,
            extensionHeight: extensionCount > 0 ? (coreHeight / Math.min(2, extensionCount)) : 0, // 最多分成2份
            value: (coreWidth + extensionWidth) * coreHeight,
            thumbnail: null
        });
    });

    return { nodes: processedNodes };
}

// 创建上下文菜单
function createContextMenu(svg, x, y, node) {
    d3.selectAll('.context-menu').remove();

    const menu = svg.append('g')
        .attr('class', 'context-menu')
        .attr('transform', `translate(${x}, ${y})`);

    menu.append('rect')
        .attr('width', 120)
        .attr('height', 30)
        .attr('fill', 'white')
        .attr('stroke', '#ccc')
        .attr('rx', 5)
        .attr('ry', 5);

    menu.append('text')
        .attr('x', 10)
        .attr('y', 20)
        .text('查看节点列表')
        .style('font-size', '14px')
        .style('cursor', 'pointer')
        .on('click', () => {
            showNodeList(node);
            menu.remove();
        });

    svg.on('click.menu', () => {
        menu.remove();
        svg.on('click.menu', null);
    });
}

// 显示节点列表
function showNodeList(node) {
    let nodeNames = [];
    if (node.type === 'extension') {
        // 如果是外延节点，同时选中其核心节点
        // 从node.id (形如 'ext_0_1') 中提取核心聚类的索引
        const coreIndex = parseInt(node.id.split('_')[1]);
        const coreNode = nodes.value.find(n => n.id === `core_${coreIndex}`);
        
        // 添加核心节点的原始节点
        if (coreNode) {
            const coreCluster = graphData.value.core_clusters[coreIndex];
            nodeNames.push(...coreCluster.core_nodes.map(n => n.split('/').pop()));
        }
        
        // 添加外延节点的原始节点
        const extension = graphData.value.core_clusters[coreIndex].extensions[parseInt(node.id.split('_')[2])];
        nodeNames.push(...extension.nodes.map(n => n.split('/').pop()));
    } else {
        // 如果是核心节点，只选中核心节点
        nodeNames = node.originalNodes.map(n => n.split('/').pop());
    }
    
    store.commit('UPDATE_SELECTED_NODES', { nodeIds: nodeNames, group: null });
}

// 渲染图形
function renderGraph(container, graphData) {
    if (!container || !graphData) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // 清除现有内容
    d3.select(container).selectAll('svg').remove();

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', [0, 0, width, height]);

    // 设置缩放
    const zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);
    const g = svg.append('g');

    // 创建布局数据
    const hierarchyData = {
        name: "root",
        children: graphData.nodes
    };

    // 使用改进的treemap布局
    const treemap = d3.treemap()
        .size([width * 0.96, height * 0.96])
        .paddingTop(24)
        .paddingRight(24)
        .paddingBottom(24)
        .paddingLeft(24)
        .paddingInner(16)  // 减小节点间距以获得更多空间
        .round(true);

    const root = d3.hierarchy(hierarchyData)
        .sum(d => {
            // 调整节点大小计算逻辑，使其更倾向于方形
            if (d.type === 'core') {
                // 核心节点基础大小
                const baseSize = 800;
                // 根据外延节点数量适度增加大小
                return baseSize + (d.extensionCount * 300);
            }
            // 外延节点固定大小，但保持较大以避免过小
            return 500;
        });

    treemap(root);

    const allNodeData = [];

    // 处理所有节点的位置
    root.leaves().forEach(node => {
        const nodeWidth = node.x1 - node.x0;
        const nodeHeight = node.y1 - node.y0;
        const extensionCount = node.data.extensionCount;

        if (extensionCount > 0) {
            // 调整核心节点和外延节点的比例
            const totalWidth = nodeWidth;
            const totalHeight = nodeHeight;
            
            // 根据外延节点数量动态调整布局
            let coreWidth, extensionWidth, gap;
            if (extensionCount <= 2) {
                // 当外延节点较少时，采用更宽的布局
                coreWidth = totalWidth * 0.65;  // 核心节点占65%
                extensionWidth = totalWidth * 0.32;  // 外延节点占32%
                gap = totalWidth * 0.03;  // 3%间隔
            } else {
                // 当外延节点较多时，采用更窄的布局以保持方形
                coreWidth = totalWidth * 0.55;  // 核心节点占55%
                extensionWidth = totalWidth * 0.42;  // 外延节点占42%
                gap = totalWidth * 0.03;  // 3%间隔
            }

            // 计算每个外延节点的高度，确保最小高度
            const minExtensionHeight = Math.max(totalHeight / extensionCount, totalHeight / 3);
            const extensionHeight = Math.min(minExtensionHeight, totalHeight / extensionCount);

            // 添加核心节点
            const coreNode = {
                ...node,
                x0: node.x0 + extensionWidth + gap,
                x1: node.x0 + extensionWidth + gap + coreWidth,
                y0: node.y0,
                y1: node.y1,
                isCore: true
            };
            allNodeData.push(coreNode);

            // 添加外延节点，调整位置使其均匀分布
            node.data.extensions.forEach((extension, index) => {
                const yStart = node.y0 + (index * extensionHeight);
                const yEnd = Math.min(node.y0 + ((index + 1) * extensionHeight), node.y1);
                
                allNodeData.push({
                    data: {
                        id: `ext_${node.data.id.split('_')[1]}_${index}`,
                        name: `ex(${extension.dimension})`,
                        type: 'extension',
                        dimension: extension.dimension,
                        originalNodes: extension.nodes,
                        thumbnail: null
                    },
                    x0: node.x0,
                    y0: yStart,
                    x1: node.x0 + extensionWidth,
                    y1: yEnd,
                    isCore: false
                });
            });
        } else {
            // 没有外延节点的核心节点保持原样
            allNodeData.push({
                ...node,
                isCore: true
            });
        }
    });

    // 渲染节点
    const nodeGroup = g.selectAll('g.node')
        .data(allNodeData)
        .join('g')
        .attr('class', 'node')
        .attr('transform', d => `translate(${d.x0},${d.y0})`);

    // 添加阴影滤镜定义
    const defs = svg.append('defs');
    
    // 默认阴影
    defs.append('filter')
        .attr('id', 'dropShadow')
        .attr('filterUnits', 'userSpaceOnUse')
        .attr('color-interpolation-filters', 'sRGB')
        .html(`
            <feDropShadow dx="0" dy="1" stdDeviation="2" flood-opacity="0.2"/>
        `);
    
    // hover时的阴影
    defs.append('filter')
        .attr('id', 'dropShadowHover')
        .attr('filterUnits', 'userSpaceOnUse')
        .attr('color-interpolation-filters', 'sRGB')
        .html(`
            <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.25"/>
        `);

    // 添加节点背景矩形（带阴影）
    nodeGroup.append('rect')
        .attr('width', d => d.x1 - d.x0)
        .attr('height', d => d.y1 - d.y0)
        .attr('fill', 'white')
        .attr('stroke', d => {
            const nodeElements = (d.isCore ? d.data : d.data).originalNodes.map(n => n.split('/').pop());
            const selectedElements = nodeElements.filter(id => selectedNodeIds.value.includes(id));
            const selectedSet = new Set(selectedElements);
            // 检查选中的元素是否与选中集合完全匹配
            const isMatch = selectedElements.length > 0 && 
                          selectedElements.length === selectedNodeIds.value.length &&
                          selectedElements.every(id => selectedNodeIds.value.includes(id));
            return d.isCore ? 
                (isMatch ? '#1967d2' : '#1a73e8') : 
                (isMatch ? '#188038' : '#34a853');
        })
        .attr('stroke-width', d => {
            const nodeElements = (d.isCore ? d.data : d.data).originalNodes.map(n => n.split('/').pop());
            const selectedElements = nodeElements.filter(id => selectedNodeIds.value.includes(id));
            const isMatch = selectedElements.length > 0 && 
                          selectedElements.length === selectedNodeIds.value.length &&
                          selectedElements.every(id => selectedNodeIds.value.includes(id));
            return isMatch ? 2.5 : 1.5;
        })
        .attr('stroke-opacity', d => {
            const nodeElements = (d.isCore ? d.data : d.data).originalNodes.map(n => n.split('/').pop());
            const selectedElements = nodeElements.filter(id => selectedNodeIds.value.includes(id));
            const isMatch = selectedElements.length > 0 && 
                          selectedElements.length === selectedNodeIds.value.length &&
                          selectedElements.every(id => selectedNodeIds.value.includes(id));
            return isMatch ? 1 : 0.8;
        })
        .attr('rx', 8)
        .attr('ry', 8)
        .attr('filter', 'url(#dropShadow)')
        .style('transition', 'all 0.2s ease-in-out');

    // 添加连接线（从外延节点到核心节点）
    nodeGroup.each(function(d) {
        if (!d.isCore) {
            const node = d3.select(this);
            const coreNodeId = `core_${d.data.id.split('_')[1]}`;
            const coreNode = allNodeData.find(n => n.isCore && n.data.id === coreNodeId);
            
            if (coreNode) {
                const nodeElements = d.data.originalNodes.map(n => n.split('/').pop());
                const selectedElements = nodeElements.filter(id => selectedNodeIds.value.includes(id));
                const isMatch = selectedElements.length > 0 && 
                              selectedElements.length === selectedNodeIds.value.length &&
                              selectedElements.every(id => selectedNodeIds.value.includes(id));
                
                g.append('path')
                    .attr('d', `M${d.x1},${d.y0 + (d.y1 - d.y0)/2} 
                               L${coreNode.x0},${d.y0 + (d.y1 - d.y0)/2}`)
                    .attr('stroke', isMatch ? '#188038' : '#34a853')
                    .attr('stroke-width', isMatch ? 2 : 1.5)
                    .attr('stroke-opacity', isMatch ? 0.8 : 0.6)
                    .attr('stroke-dasharray', '4,4')
                    .attr('fill', 'none')
                    .style('transition', 'all 0.2s ease-in-out');
            }
        }
    });

    // 添加缩略图，调整padding使SVG更大
    nodeGroup.each(function(d) {
        const node = d3.select(this);
        const width = d.x1 - d.x0;
        const height = d.y1 - d.y0;
        
        // 减小padding，增大SVG显示区域
        const foreignObject = node.append('foreignObject')
            .attr('width', width - 16)
            .attr('height', height - 32)
            .attr('x', 8)
            .attr('y', 8);

        const div = foreignObject.append('xhtml:div')
            .style('width', '100%')
            .style('height', '100%')
            .style('overflow', 'hidden')
            .style('border-radius', '6px')
            .style('background', '#fafafa');

        div.html(createThumbnail(d.isCore ? d.data : d.data));

        // 添加标签背景
        node.append('rect')
            .attr('class', 'label-bg')
            .attr('x', 8)
            .attr('y', height - 28)
            .attr('width', width - 16)
            .attr('height', 24)
            .attr('fill', 'white')
            .attr('rx', 4)
            .attr('ry', 4);

        // 添加标签文本
        node.append('text')
            .attr('x', width / 2)
            .attr('y', height - 12)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .style('font-family', "'Google Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif")
            .style('font-size', '13px')
            .style('font-weight', '500')
            .style('fill', '#3c4043')
            .text(d.isCore ? d.data.name : d.data.name);
    });

    // 添加hover效果
    nodeGroup.on('mouseenter', function() {
        d3.select(this).select('rect')
            .attr('filter', 'url(#dropShadowHover)');
    })
    .on('mouseleave', function() {
        d3.select(this).select('rect')
            .attr('filter', 'url(#dropShadow)');
    });

    // 添加点击事件
    nodeGroup.on('click', (event, d) => {
        showNodeList(d.isCore ? d.data : d.data);
    });
}

// 计算外延节点的位置
function calculateExtensionPositions(count, coreX, coreY, coreWidth, coreHeight, extensionSize, containerWidth, containerHeight) {
    const positions = [];
    const padding = 0;
    const extensionHeight = coreHeight / count;

    for (let i = 0; i < count; i++) {
        positions.push({
            x: coreX - extensionSize - 2, // 向左偏移，确保不会遮挡其他节点
            y: coreY + (i * extensionHeight)
        });
    }

    return positions;
}

// 加载数据并渲染
async function loadAndRenderGraph() {
    try {
        await fetchOriginalSvg(); // 首先获取原始SVG内容
        const response = await fetch('http://localhost:5000/static/data/subgraphs/subgraph_dimension_all.json');
        const data = await response.json();
        console.log('Loaded core cluster data structure:', {
            hasCoreClusters: 'core_clusters' in data,
            dataKeys: Object.keys(data),
            firstCluster: data.core_clusters ? data.core_clusters[0] : null
        });
        graphData.value = data;
        const processedData = processGraphData(data);
        nodes.value = processedData.nodes;
        renderGraph(graphContainer.value, processedData);
    } catch (error) {
        console.error('Error loading core cluster data:', error);
    }
}

onMounted(async () => {
    await nextTick();
    await loadAndRenderGraph();
});

// 监听选中节点的变化
watch(selectedNodeIds, () => {
    nextTick(() => {
        const svg = d3.select(graphContainer.value).select('svg');
        
        // 更新节点边框
        svg.selectAll('.node rect').each(function(d) {
            const rect = d3.select(this);
            // 检查是否为标签背景或者数据不存在
            if (!rect.classed('label-bg') && d) {
                try {
                    const data = d.data || d;
                    const isCore = d.isCore || data.type === 'core';
                    const nodeElements = (isCore ? data : data).originalNodes.map(n => n.split('/').pop());
                    const selectedElements = nodeElements.filter(id => selectedNodeIds.value.includes(id));
                    const isMatch = selectedElements.length > 0 && 
                                  selectedElements.length === selectedNodeIds.value.length &&
                                  selectedElements.every(id => selectedNodeIds.value.includes(id));
                    
                    rect
                        .attr('stroke', isCore ? 
                            (isMatch ? '#1967d2' : '#1a73e8') : 
                            (isMatch ? '#188038' : '#34a853'))
                        .attr('stroke-width', isMatch ? 2.5 : 1.5)
                        .attr('stroke-opacity', isMatch ? 1 : 0.8);
                } catch (error) {
                    console.warn('Error updating node border:', error);
                }
            }
        });
        
        // 更新连接线
        svg.selectAll('path').each(function(d, i, nodes) {
            const path = d3.select(this);
            if (path.attr('stroke-dasharray') === '4,4') {  // 只处理连接线
                const sourceNode = d3.select(nodes[i].parentNode).datum();
                if (sourceNode && !sourceNode.isCore) {  // 确保是外延节点的连接线且数据存在
                    try {
                        const data = sourceNode.data || sourceNode;
                        const nodeElements = data.originalNodes.map(n => n.split('/').pop());
                        const selectedElements = nodeElements.filter(id => selectedNodeIds.value.includes(id));
                        const isMatch = selectedElements.length > 0 && 
                                      selectedElements.length === selectedNodeIds.value.length &&
                                      selectedElements.every(id => selectedNodeIds.value.includes(id));
                        
                        path
                            .attr('stroke', isMatch ? '#188038' : '#34a853')
                            .attr('stroke-width', isMatch ? 2 : 1.5)
                            .attr('stroke-opacity', isMatch ? 0.8 : 0.6);
                    } catch (error) {
                        console.warn('Error updating connection line:', error);
                    }
                }
            }
        });
    });
});
</script>

<style scoped>
.core-graph-container {
    width: 100%;
    height: 100%;
    background: #ffffff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.08);
    overflow: hidden;
    border: none;
}

.graph-container {
    width: 100%;
    height: 100%;
    position: relative;
    padding: 16px;
}

:deep(.node) {
    cursor: pointer;
}

:deep(.node text) {
    fill: #5f6368;
    font-family: 'Google Sans', 'Roboto', sans-serif;
    font-size: 13px;
    font-weight: 500;
}

:deep(.node path) {
    fill: #ffffff;
    transition: fill 0.1s ease;
}

:deep(.node:hover path) {
    fill: #f8f9fa;
}

:deep(.context-menu) {
    cursor: pointer;
    font-family: 'Google Sans', 'Roboto', sans-serif;
}

:deep(.context-menu rect) {
    fill: #ffffff;
    stroke: #dadce0;
    filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1));
}

:deep(.context-menu text) {
    fill: #3c4043;
    font-size: 13px;
    font-weight: 500;
}
</style>
