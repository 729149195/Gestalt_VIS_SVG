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
    nodeData.originalNodes.forEach(nodeId => {
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
    const links = [];
    const nodeSet = new Set();

    // 处理每个核心聚类
    coreData.core_clusters.forEach((cluster, clusterIndex) => {
        // 添加核心节点
        const coreNodeId = `core_${clusterIndex}`;
        console.log('Processing cluster:', {
            index: clusterIndex,
            clusterKeys: Object.keys(cluster),
            dimensions: cluster.dimensions,
            cluster: cluster
        });
        
        // 安全地处理维度信息
        const dimensionsStr = cluster.dimensions 
            ? `Z_${cluster.dimensions.join(',Z_')}`
            : `核心聚类 ${clusterIndex + 1}`;  // 临时回退方案
            
        processedNodes.push({
            id: coreNodeId,
            name: dimensionsStr,
            type: 'core',
            originalNodes: cluster.core_nodes,
            dimensions: cluster.dimensions,
            width: 240,
            height: 160,
            thumbnail: null
        });
        nodeSet.add(coreNodeId);

        // 添加外延节点
        cluster.extensions.forEach((extension, extIndex) => {
            const extNodeId = `ext_${clusterIndex}_${extIndex}`;
            processedNodes.push({
                id: extNodeId,
                name: `外延(${extension.dimension})`,
                type: 'extension',
                dimension: extension.dimension,
                originalNodes: extension.nodes,
                width: 200,  // 增大外延节点宽度
                height: 120,  // 增大外延节点高度
                thumbnail: null
            });
            nodeSet.add(extNodeId);

            links.push({
                source: coreNodeId,
                target: extNodeId,
                value: 1
            });
        });
    });

    return { nodes: processedNodes, links };
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

    d3.select(container).selectAll('svg').remove();

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', [-width / 2, -height / 2, width, height]);

    const zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);

    const g = svg.append('g');

    const simulation = d3.forceSimulation(graphData.nodes)
        .force('link', d3.forceLink(graphData.links).id(d => d.id)
            .distance(200)
            .strength(0.2))
        .force('charge', d3.forceManyBody()
            .strength(d => d.type === 'core' ? -2000 : -500))
        .force('center', d3.forceCenter(0, 0))
        // 移除y方向的力，改为直接固定核心节点的y坐标
        // 仅对核心节点应用x方向的力，保持均匀分布
        .force('x', d3.forceX().strength(d => {
            if (d.type === 'core') {
                return 0.5;
            }
            return 0;
        }).x(d => {
            if (d.type === 'core') {
                const coreNodes = graphData.nodes.filter(n => n.type === 'core');
                const index = coreNodes.indexOf(d);
                return (index - (coreNodes.length - 1) / 2) * 300;
            }
            return d.x;
        }))
        .force('collision', d3.forceCollide().radius(d => {
            return d.type === 'core' ? 150 : 100;
        }).strength(0.8));

    // 初始化节点位置并固定核心节点的y坐标
    graphData.nodes.forEach(node => {
        if (node.type === 'core') {
            node.y = 0;
            node.fy = 0;  // 固定y坐标为0
            const coreNodes = graphData.nodes.filter(n => n.type === 'core');
            const index = coreNodes.indexOf(node);
            node.x = (index - (coreNodes.length - 1) / 2) * 300;
        } else {
            const coreId = node.id.split('_')[1];
            const coreNode = graphData.nodes.find(n => n.id === `core_${coreId}`);
            if (coreNode) {
                node.x = coreNode.x + (Math.random() - 0.5) * 300;
                node.y = (Math.random() > 0.5 ? 1 : -1) * (Math.random() * 300 + 200);
            }
        }
    });

    const link = g.append('g')
        .selectAll('line')
        .data(graphData.links)
        .join('line')
        .attr('stroke', '#aaa')
        .attr('stroke-width', 2);

    // 创建节点组
    const nodeGroup = g.append('g')
        .selectAll('g')
        .data(graphData.nodes)
        .join('g')
        .attr('class', 'node-group')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));

    // 添加矩形框
    nodeGroup.append('rect')
        .attr('width', d => d.width)
        .attr('height', d => d.height)
        .attr('x', d => -d.width / 2)
        .attr('y', d => -d.height / 2)
        .attr('fill', 'white')
        .attr('stroke', d => d.type === 'core' ? '#ff6347' : '#69b3a2')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', d => d.type === 'extension' ? '5,5' : 'none')
        .attr('rx', 8)  // 添加水平圆角
        .attr('ry', 8)  // 添加垂直圆角
        .style('cursor', 'pointer');

    // 添加SVG缩略图容器
    const foreignObjects = nodeGroup.append('foreignObject')
        .attr('width', d => d.width - 10)
        .attr('height', d => d.height - 10)
        .attr('x', d => -d.width / 2 + 5)
        .attr('y', d => -d.height / 2 + 5);

    // 添加缩略图div容器
    const thumbnailContainers = foreignObjects.append('xhtml:div')
        .style('width', '100%')
        .style('height', '100%')
        .style('overflow', 'hidden');

    // 添加缩略图
    thumbnailContainers.each(function(d) {
        this.innerHTML = createThumbnail(d);
    });

    // 添加标签
    nodeGroup.append('text')
        .attr('dy', d => d.height / 2 + 25)  // 增加标签与节点的距离
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')  // 增大字体大小
        .style('pointer-events', 'none')
        .each(function(d) {
            const text = d3.select(this);
            console.log('Node data for label:', d); // 调试输出
            text.append('tspan')
                .attr('x', 0)
                .text(d.name)
                .style('font-weight', 'bold');  // 加粗标题
        });

    nodeGroup.on('click', (event, d) => {
        showNodeList(d);
    });

    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        nodeGroup.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        if (d.type === 'core') {
            d.fy = 0;  // 核心节点固定在y=0
        } else {
            d.fy = d.y;  // 外延节点可以自由移动
        }
    }

    function dragged(event, d) {
        d.fx = event.x;
        if (d.type === 'core') {
            d.fy = 0;  // 核心节点固定在y=0
        } else {
            d.fy = event.y;  // 外延节点可以自由移动
        }
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        if (d.type === 'core') {
            d.fx = null;
            d.fy = 0;  // 核心节点保持在y=0
        } else {
            d.fx = null;
            d.fy = null;  // 外延节点完全自由
        }
    }
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
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(200, 200, 200, 0.3);
    overflow: hidden;
    transition: all 0.3s ease;
}

.core-graph-container:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
    border: 1px solid rgba(180, 180, 180, 0.4);
}

.graph-container {
    width: 100%;
    height: 100%;
    position: relative;
    padding: 16px;
}

:deep(.context-menu) {
    cursor: pointer;
}
</style>
