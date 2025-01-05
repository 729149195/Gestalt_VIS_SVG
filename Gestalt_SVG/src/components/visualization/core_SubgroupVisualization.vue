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
    const parser = new DOMParser();
    const svgDoc = parser.parseFromString(originalSvgContent.value, 'image/svg+xml');
    const svgElement = svgDoc.querySelector('svg');
    
    // 设置所有元素透明度为0.3
    svgElement.querySelectorAll('*').forEach(el => {
        if (el.tagName !== 'svg' && el.tagName !== 'g') {
            el.style.opacity = '0.02';
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
    nodesToHighlight.forEach(nodeId => {
        const element = svgDoc.getElementById(nodeId.split('/').pop());
        if (element) {
            element.style.opacity = '1';
        }
    });
    
    // 调整SVG大小为缩略图大小
    svgElement.setAttribute('width', '100%');
    svgElement.setAttribute('height', '100%');
    
    return svgElement.outerHTML;
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
            name: `核心聚类 ${clusterIndex + 1} (Z_${cluster.core_dimensions.join(',Z_')})`,
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
        .paddingInner(32)
        .round(true);

    const root = d3.hierarchy(hierarchyData)
        .sum(d => d.value);

    treemap(root);

    const allNodeData = [];

    // 处理所有节点的位置
    root.leaves().forEach(node => {
        const nodeWidth = node.x1 - node.x0;
        const nodeHeight = node.y1 - node.y0;
        const extensionCount = node.data.extensionCount;

        if (extensionCount > 0) {
            // 计算核心节点的实际位置
            const coreNode = {
                ...node,
                x0: node.x0 + node.data.extensionWidth,
                x1: node.x1,
                y0: node.y0,
                y1: node.y1,
                isCore: true
            };
            allNodeData.push(coreNode);

            // 计算外延节点的位置，确保最小高度
            const minExtensionHeight = nodeHeight / Math.min(2, extensionCount);
            node.data.extensions.forEach((extension, index) => {
                allNodeData.push({
                    data: {
                        id: `ext_${node.data.id.split('_')[1]}_${index}`,
                        name: `外延(${extension.dimension})`,
                        type: 'extension',
                        dimension: extension.dimension,
                        originalNodes: extension.nodes,
                        thumbnail: null
                    },
                    x0: node.x0,
                    y0: node.y0 + (index * minExtensionHeight),
                    x1: node.x0 + node.data.extensionWidth,
                    y1: node.y0 + ((index + 1) * minExtensionHeight),
                    isCore: false
                });
            });
        } else {
            // 没有外延节点的核心节点
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

    // 添加节点形状（带齿孔的矩形）
    nodeGroup.append('path')
        .attr('fill', 'white')
        .attr('stroke', d => d.isCore ? '#1a73e8' : '#34a853')
        .attr('stroke-width', d => {
            const nodeElements = (d.isCore ? d.data : d.data).originalNodes.map(n => n.split('/').pop());
            const allSelected = nodeElements.every(id => selectedNodeIds.value.includes(id));
            return allSelected ? 3 : 2;
        })
        .attr('stroke-opacity', d => {
            const nodeElements = (d.isCore ? d.data : d.data).originalNodes.map(n => n.split('/').pop());
            const allSelected = nodeElements.every(id => selectedNodeIds.value.includes(id));
            const someSelected = nodeElements.some(id => selectedNodeIds.value.includes(id));
            return allSelected ? 1 : (someSelected ? 0.85 : 0.7);
        })
        .attr('d', d => {
            const width = d.x1 - d.x0;
            const height = d.y1 - d.y0;
            const radius = Math.min(8, width / 4, height / 4);

            if (!d.isCore) {
                // 外延节点：右边有齿孔
                const holeCount = Math.max(3, Math.floor(height / 30)); // 减少齿孔数量，增大间距
                const holeRadius = 4; // 增大齿孔半径
                const spacing = height / (holeCount + 1);
                
                let path = `M ${radius},0 L ${width},0`;
                
                // 添加右边的齿孔
                for (let i = 1; i <= holeCount; i++) {
                    const y = i * spacing;
                    path += ` L ${width},${y - holeRadius}`;
                    path += ` A ${holeRadius},${holeRadius} 0 1,1 ${width},${y + holeRadius}`;
                }
                
                path += ` L ${width},${height} L ${radius},${height}`;
                path += ` Q 0,${height} 0,${height - radius}`;
                path += ` L 0,${radius} Q 0,0 ${radius},0`;
                
                return path + 'Z';
            } else {
                // 核心节点：左边有齿孔
                const holeCount = Math.max(3, Math.floor(height / 30));
                const holeRadius = 4;
                const spacing = height / (holeCount + 1);
                
                let path = `M ${radius},0 L ${width - radius},0`;
                path += ` Q ${width},0 ${width},${radius}`;
                path += ` L ${width},${height - radius}`;
                path += ` Q ${width},${height} ${width - radius},${height}`;
                path += ` L ${radius},${height}`;
                
                // 添加左边的齿孔
                for (let i = holeCount; i >= 1; i--) {
                    const y = i * spacing;
                    path += ` L 0,${y + holeRadius}`;
                    path += ` A ${holeRadius},${holeRadius} 0 1,1 0,${y - holeRadius}`;
                }
                
                path += ` L 0,${radius} Q 0,0 ${radius},0`;
                
                return path + 'Z';
            }
        });

    // 添加缩略图和标签
    nodeGroup.each(function(d) {
        const node = d3.select(this);
        const width = d.x1 - d.x0;
        const height = d.y1 - d.y0;
        
        // 添加缩略图
        const foreignObject = node.append('foreignObject')
            .attr('width', width - 16)
            .attr('height', height - 40)
            .attr('x', 8)
            .attr('y', 8);

        const div = foreignObject.append('xhtml:div')
            .style('width', '100%')
            .style('height', '100%')
            .style('overflow', 'hidden')
            .style('border-radius', '4px');

        div.html(createThumbnail(d.isCore ? d.data : d.data));

        // 添加标签
        node.append('text')
            .attr('x', width / 2)
            .attr('y', height - 12)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .style('font-family', "'Google Sans', 'Roboto', sans-serif")
            .style('font-size', '13px')
            .style('font-weight', '500')
            .style('fill', '#3c4043')
            .text(d.isCore ? d.data.name : d.data.name);
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
        svg.selectAll('.node-group').each(function(d) {
            const nodeGroup = d3.select(this);
            const rect = nodeGroup.select('rect');
            
            // 获取节点包含的所有元素ID
            const nodeElements = d.originalNodes.map(n => n.split('/').pop());
            // 检查是否所有元素都被选中
            const allSelected = nodeElements.every(id => selectedNodeIds.value.includes(id));
            // 检查是否有部分元素被选中
            const someSelected = nodeElements.some(id => selectedNodeIds.value.includes(id));
            
            // 更新边框样式
            rect
                .attr('stroke', d => {
                    if (allSelected) {
                        // 全部选中时使用更亮的颜色
                        return d.type === 'core' ? '#ff3333' : '#33cc33';
                    } else {
                        // 保持原有颜色
                        return d.type === 'core' ? '#ff6347' : '#69b3a2';
                    }
                })
                .attr('stroke-width', allSelected ? 4 : 2)  // 全部选中时加粗边框
                .attr('stroke-opacity', allSelected ? 1 : (someSelected ? 0.8 : 0.6));  // 调整透明度
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
