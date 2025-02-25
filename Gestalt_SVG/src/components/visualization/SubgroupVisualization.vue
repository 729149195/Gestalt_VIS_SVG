<template>
    <div class="force-graph-container">
        <span class="title">Graphical Patterns List</span>
        <div class="hints-container">
            <span class="scroll-hint">← Scroll horizontally to see more patterns →</span>
            <span class="scroll-hint">Click to view the model pattern</span>
        </div>
        <div v-if="currentPage === 1" class="core-view-container">
            <CoreSubgroupVisualization />
        </div>

        <!-- 显示维度组合视图 -->
        <!-- <div v-else class="graph-grid">
            <div v-for="(dims, index) in currentDimensions" :key="getDimensionKey(dims)" class="graph-item">
                <div class="graph-title">{{ formatDimensions(dims) }}</div>
                <div :ref="el => { if (el) graphContainers[index] = el }" class="graph-container"></div>
            </div>
        </div> -->
    </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch, onUnmounted, computed } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
import CoreSubgroupVisualization from './core_SubgroupVisualization.vue';

// 添加延迟函数
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);
const checkbox = ref(false);
const currentPage = ref(1);
const graphContainers = ref([]);
const isSelecting = ref(false);
let selectionRect = null;
let selectionStart = { x: 0, y: 0 };
const originalSvgContent = ref(''); // 添加存储原始SVG内容的ref

// 添加获取原始SVG内容的函数
async function fetchOriginalSvg() {
    try {
        const response = await fetch('http://127.0.0.1:5000/get_svg');
        const svgContent = await response.text();
        originalSvgContent.value = svgContent;
    } catch (error) {
        console.error('Error fetching original SVG:', error);
    }
}

// 添加创建缩略图的函数
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

// 生成所有可能的维度组合
const allDimensions = computed(() => {
    const baseDimensions = [0, 1, 2, 3];
    let combinations = [];

    // 添加单维度组合
    combinations.push(...baseDimensions.map(d => [d]));

    // 添加双维度组合
    for (let i = 0; i < baseDimensions.length; i++) {
        for (let j = i + 1; j < baseDimensions.length; j++) {
            combinations.push([baseDimensions[i], baseDimensions[j]]);
        }
    }

    // 添加三维度组合
    for (let i = 0; i < baseDimensions.length; i++) {
        for (let j = i + 1; j < baseDimensions.length; j++) {
            for (let k = j + 1; k < baseDimensions.length; k++) {
                combinations.push([baseDimensions[i], baseDimensions[j], baseDimensions[k]]);
            }
        }
    }

    // 添加四维度组合
    combinations.push(baseDimensions);

    return combinations;
});

// 计算当前页面显示的维度组合
const currentDimensions = computed(() => {
    if (currentPage.value === 1) return [];
    const startIndex = (currentPage.value - 2) * 6;
    return allDimensions.value.slice(startIndex, startIndex + 6);
});

// 格式化维度显示
function formatDimensions(dims) {
    return dims.map(dim => `z_${dim + 1}`).join('，');
}

// 获取维度组合的唯一键
function getDimensionKey(dims) {
    return dims.join('');
}

// 处理页面变化
async function handlePageChange(page) {
    currentPage.value = page;
    if (page === 1) return;

    // 清空现有的图表容器
    graphContainers.value = [];
    // 等待 DOM 更新
    await nextTick();
    // 重新加载当前页的图表
    await loadAndRenderAllGraphs();
}

// 修改加载和渲染函数
async function loadAndRenderAllGraphs() {
    try {
        await nextTick();
        await delay(300);

        for (let i = 0; i < currentDimensions.value.length; i++) {
            try {
                const dims = currentDimensions.value[i];
                const dimKey = getDimensionKey(dims);
                const data = await d3.json(`http://127.0.0.1:5000/static/data/subgraphs/subgraph_dimension_${dimKey}.json`);
                const container = graphContainers.value[i];

                if (!container) {
                    console.error(`Container for dimensions ${dims} not found`);
                    continue;
                }

                await delay(100);

                const width = container.clientWidth || 600;
                const height = container.clientHeight || 400;

                if (width <= 0 || height <= 0) {
                    console.warn(`Container ${i} dimensions not ready, using default values`);
                }

                renderGraph(container, data);
            } catch (error) {
                console.error(`Error loading data for dimensions ${currentDimensions.value[i]}:`, error);
            }
        }
    } catch (error) {
        console.error('Error in loadAndRenderAllGraphs:', error);
    }
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

// 修改createContextMenu函数，移除展开功能
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

// 修改remergeNodes函数
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
    // 理图数据
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

    // 添加鼠标事件监听
    svg.on('mousedown', (event) => onMouseDown(event, svg))
        .on('mousemove', (event) => onMouseMove(event, svg))
        .on('mouseup', (event) => onMouseUp(event, svg));

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

function clearSelectedNodes() {
    store.dispatch('clearSelectedNodes');

    graphContainers.value.forEach((container) => {
        d3.select(container)
            .selectAll('.node-group rect')
            .attr('stroke', '#69b3a2');
    });
}

watch(selectedNodeIds, () => {
    nextTick(() => {
        graphContainers.value.forEach((container) => {
            const svg = d3.select(container).select('svg');
            svg.selectAll('.node-group')
                .select('rect')
                .attr('stroke', (d) => {
                    if (d.isGroup) {
                        return d.originalNodes.every(originalNode =>
                            selectedNodeIds.value.includes(originalNode.name.split('/').pop())
                        ) ? '#ff6347' : '#69b3a2';
                    } else {
                        const nodeName = d.name.split('/').pop();
                        return selectedNodeIds.value.includes(nodeName) ? '#ff6347' : '#69b3a2';
                    }
                });
        });
    });
});

function onMouseDown(event, svg) {
    if (!checkbox.value) return;

    isSelecting.value = true;
    const point = d3.pointer(event);
    selectionStart = { x: point[0], y: point[1] };

    if (!selectionRect) {
        selectionRect = svg.append('rect')
            .attr('class', 'selection')
            .attr('x', selectionStart.x)
            .attr('y', selectionStart.y)
            .attr('width', 0)
            .attr('height', 0);
    }
}

function onMouseMove(event, svg) {
    if (!isSelecting.value) return;

    const point = d3.pointer(event);
    const x = Math.min(selectionStart.x, point[0]);
    const y = Math.min(selectionStart.y, point[1]);
    const width = Math.abs(selectionStart.x - point[0]);
    const height = Math.abs(selectionStart.y - point[1]);

    selectionRect
        .attr('x', x)
        .attr('y', y)
        .attr('width', width)
        .attr('height', height);
}

function onMouseUp(event, svg) {
    if (!isSelecting.value) return;

    isSelecting.value = false;
    const selectionBox = selectionRect.node().getBBox();
    selectNodesInBox(selectionBox, svg);
    selectionRect.remove();
    selectionRect = null;
}

function selectNodesInBox(selectionBox, svg) {
    const selectedNodes = [];
    const transform = d3.zoomTransform(svg.node());
    const adjustedSelectionBox = {
        x: (selectionBox.x - transform.x) / transform.k,
        y: (selectionBox.y - transform.y) / transform.k,
        width: selectionBox.width / transform.k,
        height: selectionBox.height / transform.k,
    };

    svg.selectAll('circle').each(function (d) {
        const cx = d.x;
        const cy = d.y;

        if (cx >= adjustedSelectionBox.x && cx <= adjustedSelectionBox.x + adjustedSelectionBox.width &&
            cy >= adjustedSelectionBox.y && cy <= adjustedSelectionBox.y + adjustedSelectionBox.height) {
            selectedNodes.push(d);
        }
    });

    const nodeIds = selectedNodes.map(node => node.name.split('/').pop());
    store.commit('UPDATE_SELECTED_NODES', { nodeIds, group: null });
}

function handleKeyDown(event) {
    if (event.key === 'c' || event.key === 'C') {
        checkbox.value = !checkbox.value;
        graphContainers.value.forEach((container) => {
            const svg = d3.select(container).select('svg');
            if (checkbox.value) {
                disableZoom(svg);
            } else {
                enableZoom(svg);
            }
        });
    }
}

function disableZoom(svg) {
    svg.on('.zoom', null);
    svg.style('cursor', 'crosshair');
}

function enableZoom(svg) {
    const zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            svg.select('g').attr('transform', event.transform);
        });

    svg.call(zoom);
    svg.style('cursor', 'default');
}

// 添加工具提示相关的响应式变量
const tooltipStyle = ref({
    opacity: 0,
    top: '0px',
    left: '0px'
});
const tooltipText = ref('');
const pageTooltip = ref(null);

// 添加工具提示显示函数
function showPageTooltip(event, page) {
    const text = page === 1 ? '核心聚类视图' : `第 ${page-1} 页维度组合`;
    tooltipText.value = text;
    tooltipStyle.value = {
        opacity: 1,
        top: `${event.clientY}px`,
        left: `${event.clientX - 100}px`
    };
}

// 添加工具提示隐藏函数
function hidePageTooltip() {
    tooltipStyle.value.opacity = 0;
}

// 修改onMounted钩子
onMounted(async () => {
    try {
        await fetchOriginalSvg(); // 首先获取原始SVG内容
        await nextTick();
        await delay(100);
        await loadAndRenderAllGraphs();
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
    width: 100%;  /* 设置固定宽度 */
    margin: 0 auto;  /* 水平居中 */
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

.controls {
    position: absolute;
    top: 20px;
    right: 20px;
    z-index: 10;
}

.control-button {
    position: absolute;
    top: 5px;
    right: 5px;
    background-color: #55C000 !important;
    border-color: #55C000 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

.control-button:hover {
    background-color: #4CAF00 !important;
    border-color: #4CAF00 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

.graph-grid {
    display: grid;
    grid-template-columns: repeat(3, 400px);  /* 设置固定列宽 */
    justify-content: center;  /* 网格水平居中 */
    gap: 20px;
    padding: 10px;
    height: 100%;
    overflow: auto;
    min-height: 0;
}

.graph-item {
    width: 400px;  /* 设置固定宽度 */
    display: flex;
    flex-direction: column;
    border: 1px solid rgba(200, 200, 200, 0.3);
    border-radius: 8px;
    background: #ffffff;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
    height: 350px;
    min-height: 350px;
}

.graph-item:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
    border: 1px solid rgba(180, 180, 180, 0.4);
}

.graph-title {
    padding: 12px;
    font-weight: bold;
    background-color: #f5f7fa;
    color: #303133;
    border-bottom: 1px solid rgba(200, 200, 200, 0.3);
}

.graph-container {
    flex: 1;
    min-height: 300px;
    position: relative;
}

.core-view-container {
    flex: 1;
    overflow: auto;
    height: 100%;
    min-height: 0;
    width: 1260px;  /* 设置固定宽度，留出padding空间 */
    margin: 0 auto;  /* 水平居中 */
}

.pagination-wrapper {
    display: none;
}

/* 添加侧边书签导航样式 */
.side-pagination {
    position: fixed;
    right: 5px;
    top: 50%;
    transform: translateY(-50%);
    z-index: 100;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.pagination-dots {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 8px;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(200, 200, 200, 0.3);
}

.pagination-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: rgba(0, 0, 0, 0.2);
    cursor: pointer;
    transition: all 0.3s ease;
}

.pagination-dot:hover {
    background-color: rgba(0, 0, 0, 0.4);
    transform: scale(1.2);
}

.pagination-dot.active {
    background-color: #55C000;
    width: 6px;
    height: 6px;
}

/* 页面提示工具提示样式 */
.page-tooltip {
    position: fixed;
    background-color: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s ease;
    white-space: nowrap;
}

.title {
  margin: 12px 16px 0 16px;
  font-size: 16px;
  font-weight: bold;
  color: #1d1d1f;
  letter-spacing: -0.01em;
  opacity: 0.8;
}
</style>
