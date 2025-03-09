<template>
    <div class="force-graph-container">
        <div class="title-container">
            <span class="title">Graphical Patterns List</span>
        </div>
        <div class="core-view-container">
            <CoreSubgroupVisualization />
        </div>

    </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
import CoreSubgroupVisualization from './core_SubgroupVisualization.vue';

const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);
const checkbox = ref(false);
const originalSvgContent = ref(''); // 存储原始SVG内容的ref

// 获取原始SVG内容的函数
async function fetchOriginalSvg() {
    try {
        const response = await fetch('http://127.0.0.1:5000/get_svg');
        const svgContent = await response.text();
        originalSvgContent.value = svgContent;
    } catch (error) {
        console.error('Error fetching original SVG:', error);
    }
}

// 创建缩略图的函数
function createThumbnail(nodeData) {
    const parser = new DOMParser();
    const svgDoc = parser.parseFromString(originalSvgContent.value, 'image/svg+xml');
    const svgElement = svgDoc.querySelector('svg');

    // 设置所有元素透明度为0.02
    svgElement.querySelectorAll('*').forEach(el => {
        if (el.tagName !== 'svg' && el.tagName !== 'g') {
            el.style.opacity = '0.02';
        }
    });

    // 高亮当前节点包含的元素
    const nodeIds = nodeData.isGroup ?
        nodeData.originalNodes.map(n => n.name.split('/').pop()) :
        [nodeData.name.split('/').pop()];

    nodeIds.forEach(nodeId => {
        const element = svgDoc.getElementById(nodeId);
        if (element) {
            element.style.opacity = '1';
        }
    });

    // 调整SVG大小为缩略图大小
    svgElement.setAttribute('width', '100%');
    svgElement.setAttribute('height', '100%');

    return svgElement.outerHTML;
}

// 在processGraphData函数之前添加新的处理函数
function processGraphData(graphData) {
    // 深拷贝输入数据以避免修改原始数据
    const data = JSON.parse(JSON.stringify(graphData));

    // 使用并查集来找到连通分量
    const uf = new Map();

    // 并查集的查找函数
    function find(x) {
        if (!uf.has(x)) {
            uf.set(x, x);
        }
        if (uf.get(x) !== x) {
            uf.set(x, find(uf.get(x)));
        }
        return uf.get(x);
    }

    // 并查集的合并函数
    function union(x, y) {
        uf.set(find(x), find(y));
    }

    // 初始化并查集
    data.nodes.forEach(node => {
        uf.set(node.id, node.id);
    });

    // 根据边来合并节点
    data.links.forEach(link => {
        union(link.source, link.target);
    });

    // 收集每个组的节点
    const groups = new Map();
    data.nodes.forEach(node => {
        const root = find(node.id);
        if (!groups.has(root)) {
            groups.set(root, []);
        }
        groups.get(root).push(node);
    });

    // 处理大于20个节点的组
    const newNodes = [];
    const nodeMapping = new Map(); // 用于记录原始节点到新节点的映射

    groups.forEach((nodes, root) => {
        if (nodes.length > 1) {
            // 创建一个大节点
            const groupNode = {
                id: `group_${root}`,
                name: `Group (${nodes.length} nodes)`,
                originalNodes: nodes,
                originalLinks: data.links.filter(link =>
                    nodes.some(n => n.id === link.source || n.id === link.source.id) &&
                    nodes.some(n => n.id === link.target || n.id === link.target.id)
                ),
                isGroup: true,
                groupId: root,  // 添加groupId用于重新聚合
                size: Math.sqrt(nodes.length) * 8
            };
            newNodes.push(groupNode);
            // 记录映射关系
            nodes.forEach(node => {
                nodeMapping.set(node.id, groupNode.id);
                // 为原始节点添加组信息
                node.groupId = root;
            });
        } else {
            // 保持原始节点
            nodes.forEach(node => {
                node.size = 8;
                node.groupId = root;  // 为所有节点添加组信息
                newNodes.push(node);
                nodeMapping.set(node.id, node.id);
            });
        }
    });

    // 处理边
    const newLinks = [];
    const linkSet = new Set(); // 用于去重

    data.links.forEach(link => {
        const sourceGroup = nodeMapping.get(link.source);
        const targetGroup = nodeMapping.get(link.target);

        if (sourceGroup !== targetGroup) {
            const linkKey = `${sourceGroup}-${targetGroup}`;
            if (!linkSet.has(linkKey)) {
                newLinks.push({
                    source: sourceGroup,
                    target: targetGroup,
                    value: 1
                });
                linkSet.add(linkKey);
            }
        }
    });

    return {
        nodes: newNodes,
        links: newLinks
    };
}

// 创建上下文菜单
function createContextMenu(svg, x, y, node, simulation) {
    // 移除可能存在的旧菜单
    d3.selectAll('.context-menu').remove();

    // 如果是组节点，不显示上下文菜单
    if (node.isGroup) return;

    // 创建菜单容器
    const menu = svg.append('g')
        .attr('class', 'context-menu')
        .attr('transform', `translate(${x}, ${y})`);

    // 添加菜单背景
    menu.append('rect')
        .attr('width', 100)
        .attr('height', 30)  // 统一高度
        .attr('fill', 'white')
        .attr('stroke', '#ccc')
        .attr('rx', 5)
        .attr('ry', 5);

    if (node.groupId) {  // 检查节点是否属于某个组
        // 重新聚合选项
        menu.append('text')
            .attr('x', 10)
            .attr('y', 20)
            .text('重新聚合')
            .style('font-size', '14px')
            .style('cursor', 'pointer')
            .on('click', () => {
                remergeNodes(node, simulation);
                menu.remove();
            });
    }

    // 点击其他地方时关闭菜单
    svg.on('click.menu', () => {
        menu.remove();
        svg.on('click.menu', null);
    });
}

// 重新聚合节点函数
function remergeNodes(node, simulation) {
    const currentNodes = simulation.nodes();
    const currentLinks = simulation.force('link').links();

    // 找到同组的所有点
    const groupNodes = currentNodes.filter(n => n.groupId === node.groupId);
    if (groupNodes.length <= 5) return; // 如果节点数量不够，不进行合并

    // 创建新的组节点
    const groupNode = {
        id: `group_${node.groupId}`,
        name: `Group (${groupNodes.length} nodes)`,
        originalNodes: groupNodes,
        originalLinks: currentLinks.filter(link =>
            groupNodes.some(n => n.id === link.source.id) &&
            groupNodes.some(n => n.id === link.target.id)
        ),
        isGroup: true,
        groupId: node.groupId,
        size: Math.sqrt(groupNodes.length) * 8,
        x: d3.mean(groupNodes, d => d.x),
        y: d3.mean(groupNodes, d => d.y)
    };

    // 移除原有节点
    groupNodes.forEach(n => {
        const index = currentNodes.indexOf(n);
        if (index > -1) {
            currentNodes.splice(index, 1);
        }
    });

    // 添加新的组节点
    currentNodes.push(groupNode);

    // 更新连接
    const newLinks = currentLinks.filter(link =>
        !groupNodes.some(n => n.id === link.source.id || n.id === link.target.id)
    );

    // 添加组节点的外部连接
    currentLinks.forEach(link => {
        const sourceInGroup = groupNodes.some(n => n.id === link.source.id);
        const targetInGroup = groupNodes.some(n => n.id === link.target.id);

        if (sourceInGroup !== targetInGroup) {
            newLinks.push({
                source: sourceInGroup ? groupNode : link.source,
                target: targetInGroup ? groupNode : link.target,
                value: 1
            });
        }
    });

    // 更新仿真器
    simulation.nodes(currentNodes);
    simulation.force('link').links(newLinks);
    simulation.alpha(1).restart();

    // 重新渲染
    updateVisualization(simulation);
}

// 更新可视化
function updateVisualization(simulation) {
    const svg = d3.select(simulation.container);
    const g = svg.select('g');

    // 更新连接
    const link = g.select('.links')
        .selectAll('line')
        .data(simulation.force('link').links())
        .join('line')
        .attr('stroke', '#aaa')
        .attr('stroke-width', d => Math.sqrt(d.value));

    // 更新节点
    const node = g.select('.nodes')
        .selectAll('circle')
        .data(simulation.nodes())
        .join('circle')
        .attr('r', d => d.size || 8)
        .attr('fill', d => {
            if (d.isGroup) {
                return d.originalNodes.every(originalNode =>
                    selectedNodeIds.value.includes(originalNode.name.split('/').pop())
                ) ? '#ff6347' : '#69b3a2';
            } else {
                return selectedNodeIds.value.includes(d.name.split('/').pop()) ? '#ff6347' : '#69b3a2';
            }
        })
        .attr('stroke-width', '1')
        .style('cursor', 'pointer')
        .on('click', function (event, d) {
            if (checkbox.value) return;

            if (d.isGroup) {
                d.originalNodes.forEach(originalNode => {
                    const nodeName = originalNode.name.split('/').pop();
                    if (!selectedNodeIds.value.includes(nodeName)) {
                        store.commit('ADD_SELECTED_NODE', nodeName);
                    }
                });
                d3.select(this).attr('fill', '#ff6347');
            } else {
                const nodeName = d.name.split('/').pop();
                if (selectedNodeIds.value.includes(nodeName)) {
                    store.commit('REMOVE_SELECTED_NODE', nodeName);
                    d3.select(this).attr('fill', '#69b3a2');
                } else {
                    store.commit('ADD_SELECTED_NODE', nodeName);
                    d3.select(this).attr('fill', '#ff6347');
                }
            }
        })
        .on('contextmenu', function (event, d) {
            event.preventDefault();
            event.stopPropagation();
            const [x, y] = d3.pointer(event, svg.node());
            createContextMenu(svg, x, y, d, simulation);
        })
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended)
        );

    // 更新标签
    const labels = g.select('.labels')
        .selectAll('text')
        .data(simulation.nodes())
        .join('text')
        .attr('dy', -10)
        .attr('text-anchor', 'middle')
        .text((d) => d.name.split('/').pop())
        .style('font-size', '14px')
        .style('pointer-events', 'none'); // 确保标签不会干扰鼠标事件

    simulation.on('tick', () => {
        link
            .attr('x1', (d) => d.source.x)
            .attr('y1', (d) => d.source.y)
            .attr('x2', (d) => d.target.x)
            .attr('y2', (d) => d.target.y);

        node.attr('cx', (d) => d.x).attr('cy', (d) => d.y);

        labels
            .attr('x', (d) => d.x)
            .attr('y', (d) => d.y);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        const width = event.sourceEvent.target.ownerSVGElement.clientWidth / 2;
        const height = event.sourceEvent.target.ownerSVGElement.clientHeight / 2;

        d.fx = Math.max(-width, Math.min(width, event.x));
        d.fy = Math.max(-height, Math.min(height, event.y));
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

function renderGraph(container, graphData) {
    // 处理图数据
    const processedData = processGraphData(graphData);

    // 确保容器和数据都存在
    if (!container || !processedData || !processedData.nodes || !processedData.links) {
        console.error('Invalid container or graph data');
        return;
    }

    // 确保容器有有效的尺寸
    const width = container.clientWidth || 600;
    const height = container.clientHeight || 400;

    if (width <= 0 || height <= 0) {
        console.error('Invalid container dimensions:', container);
        return;
    }

    // 清除可能存在的旧SVG
    d3.select(container).selectAll('svg').remove();

    // 创建新的SVG
    const svg = d3
        .select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', [-width / 2, -height / 2, width, height]);

    // 添加缩放功能
    const zoom = d3.zoom()
        .filter((event) => {
            // Allow zooming only when selection mode is off
            return !checkbox.value;
        })
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    // 如果不在框选模式下,启用缩放
    if (!checkbox.value) {
        svg.call(zoom);
    }

    const g = svg.append('g');

    // 力导引仿真器
    const simulation = d3
        .forceSimulation(processedData.nodes)
        .force('link', d3.forceLink(processedData.links).id((d) => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-50))
        .force('center', d3.forceCenter(0, 0));

    simulation.container = container;

    // 绘制连线
    const link = g
        .append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(processedData.links)
        .join('line')
        .attr('stroke', '#aaa')
        .attr('stroke-width', (d) => Math.sqrt(d.value));

    // 修改节点组的渲染
    const nodeGroup = g
        .append('g')
        .attr('class', 'nodes')
        .selectAll('g')
        .data(processedData.nodes)
        .join('g')
        .attr('class', 'node-group')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended)
        );

    // 添加矩形框
    nodeGroup.append('rect')
        .attr('width', d => d.isGroup ? 240 : 120)
        .attr('height', d => d.isGroup ? 160 : 80)
        .attr('x', d => d.isGroup ? -120 : -60)
        .attr('y', d => d.isGroup ? -80 : -40)
        .attr('fill', 'white')
        .attr('stroke', d => {
            if (d.isGroup) {
                return d.originalNodes.every(originalNode =>
                    selectedNodeIds.value.includes(originalNode.name.split('/').pop())
                ) ? '#ff6347' : '#69b3a2';
            } else {
                return selectedNodeIds.value.includes(d.name.split('/').pop()) ? '#ff6347' : '#69b3a2';
            }
        })
        .attr('stroke-width', 2)
        .attr('rx', 8)
        .attr('ry', 8)
        .style('cursor', 'pointer');

    // 添加SVG缩略图容器
    const foreignObjects = nodeGroup.append('foreignObject')
        .attr('width', d => d.isGroup ? 230 : 110)
        .attr('height', d => d.isGroup ? 150 : 70)
        .attr('x', d => d.isGroup ? -115 : -55)
        .attr('y', d => d.isGroup ? -75 : -35);

    // 添加缩略图div容器
    const thumbnailContainers = foreignObjects.append('xhtml:div')
        .style('width', '100%')
        .style('height', '100%')
        .style('overflow', 'hidden');

    // 添加缩略图
    thumbnailContainers.each(function (d) {
        this.innerHTML = createThumbnail(d);
    });

    // 修改事件处理
    nodeGroup.on('click', function (event, d) {
        if (checkbox.value) return;

        if (d.isGroup) {
            d.originalNodes.forEach(originalNode => {
                const nodeName = originalNode.name.split('/').pop();
                if (!selectedNodeIds.value.includes(nodeName)) {
                    store.commit('ADD_SELECTED_NODE', nodeName);
                }
            });
            d3.select(this).select('rect').attr('stroke', '#ff6347');
        } else {
            const nodeName = d.name.split('/').pop();
            if (selectedNodeIds.value.includes(nodeName)) {
                store.commit('REMOVE_SELECTED_NODE', nodeName);
                d3.select(this).select('rect').attr('stroke', '#69b3a2');
            } else {
                store.commit('ADD_SELECTED_NODE', nodeName);
                d3.select(this).select('rect').attr('stroke', '#ff6347');
            }
        }
    })
        .on('contextmenu', function (event, d) {
            event.preventDefault();
            event.stopPropagation();
            const [x, y] = d3.pointer(event, svg.node());
            createContextMenu(svg, x, y, d, simulation);
        });

    // 修改标签位置
    const labels = nodeGroup.append('text')
        .attr('dy', d => d.isGroup ? 100 : 60)
        .attr('text-anchor', 'middle')
        .text(d => d.name.split('/').pop())
        .style('font-size', '14px')
        .style('pointer-events', 'none');

    // 修改tick函数
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
        d.fy = d.y;
    }

    function dragged(event, d) {
        const width = event.sourceEvent.target.ownerSVGElement.clientWidth / 2;
        const height = event.sourceEvent.target.ownerSVGElement.clientHeight / 2;

        d.fx = Math.max(-width, Math.min(width, event.x));
        d.fy = Math.max(-height, Math.min(height, event.y));
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

// 处理键盘按键事件
function handleKeyDown(event) {
    if (event.key === 'c' || event.key === 'C') {
        checkbox.value = !checkbox.value;
    }
}

// 修改onMounted钩子
onMounted(async () => {
    try {
        await fetchOriginalSvg(); // 首先获取原始SVG内容
        window.addEventListener('keydown', handleKeyDown);
    } catch (error) {
        console.error('Error in onMounted:', error);
    }
});

onUnmounted(() => {
    window.removeEventListener('keydown', handleKeyDown);
});
</script>
<style scoped>
.force-graph-container {
    position: relative;
    width: 100%;
    margin: 0 auto;
    height: calc(100vh - 120px);
    max-height: 900px;
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(200, 200, 200, 0.3);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

.force-graph-container:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
    border: 1px solid rgba(180, 180, 180, 0.4);
}

/* 添加标题容器样式 */
.title-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 12px 16px 0 16px;
}

/* 添加图例样式 */
.legend {
    display: flex;
    align-items: center;
    gap: 16px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
}

.legend-probability-sample {
    font-size: 14px;
    font-weight: 600;
    color: #885F35;
    padding: 4px 8px;
    border-radius: 6px;
    background: rgba(136, 95, 53, 0.08);
    border: 1px solid rgba(136, 95, 53, 0.2);
    min-width: 70px;
    text-align: center;
}

.legend-text {
  font-size: 16px;
  font-weight: 500;
  color: #666;
}

/* 添加提示容器样式 */
.hints-container {
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 4px 0 12px 0;
    gap: 4px;
}

/* 修改滑动提示样式 */
.scroll-hint {
    font-size: 13px;
    color: #666;
    letter-spacing: 0.01em;
    opacity: 0.75;
}

.scroll-hint:first-child {
    margin-bottom: 2px;
}

.core-view-container {
    flex: 1;
    overflow: auto;
    height: 100%;
    min-height: 0;
    width: 100%;
    padding: 0 16px;
}

.title {
  font-size: 1.5em;
  font-weight: bold;
  color: #1d1d1f;
  letter-spacing: -0.01em;
  opacity: 0.8;
}
</style>
