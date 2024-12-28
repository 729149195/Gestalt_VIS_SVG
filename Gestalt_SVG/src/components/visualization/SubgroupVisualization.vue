<template>
    <div class="force-graph-container">
        <div class="controls">
            <el-button class="clear-button" @click="clearSelectedNodes">清空并高亮所有节点</el-button>
            <v-switch v-model="checkbox" inset color="#55C000"
                :label="checkbox ? '框选模式开启(缩放已禁用)' : '框选模式关闭(缩放已启用)'" />
            
        </div>
        <div class="graph-grid">
            <div v-for="(dims, index) in currentDimensions" :key="getDimensionKey(dims)" class="graph-item">
                <div class="graph-title">{{ formatDimensions(dims) }}</div>
                <div :ref="el => { if (el) graphContainers[index] = el }" class="graph-container"></div>
            </div>
        </div>
        <div class="pagination-wrapper">
            <v-pagination
                v-model="currentPage"
                :length="Math.ceil(allDimensions.length / 6)"
                :total-visible="7"
                color="#55C000"
                @update:model-value="handlePageChange"
            ></v-pagination>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch, onUnmounted, computed } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
import { ElButton } from 'element-plus'

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
    const startIndex = (currentPage.value - 1) * 6;
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
                const data = await d3.json(`http://localhost:5000/static/data/subgraphs/subgraph_dimension_${dimKey}.json`);
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

// 在renderGraph函数之前添加新的处理函数
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
        if (nodes.length > 5) {
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

// 在renderGraph函数开始前添加新的函数
function createContextMenu(svg, x, y, node, simulation) {
    // 移除可能存在的旧菜单
    d3.selectAll('.context-menu').remove();

    // 创建菜单容器
    const menu = svg.append('g')
        .attr('class', 'context-menu')
        .attr('transform', `translate(${x}, ${y})`);

    // 添加菜单背景
    menu.append('rect')
        .attr('width', 100)
        .attr('height', node.isGroup ? 30 : 60)  // 根据节点类型调整高度
        .attr('fill', 'white')
        .attr('stroke', '#ccc')
        .attr('rx', 5)
        .attr('ry', 5);

    if (node.isGroup) {
        // 展开节点选项
        menu.append('text')
            .attr('x', 10)
            .attr('y', 20)
            .text('展开节点')
            .style('font-size', '14px')
            .style('cursor', 'pointer')
            .on('click', () => {
                expandNode(node, simulation);
                menu.remove();
            });
    } else {
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

// 添加重新聚合函数
function remergeNodes(node, simulation) {
    const currentNodes = simulation.nodes();
    const currentLinks = simulation.force('link').links();
    
    // 找到同组的所有点
    const groupNodes = currentNodes.filter(n => n.groupId === node.groupId);
    if (groupNodes.length <= 20) return; // 如果节点数量不够，不进行合并
    
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
                source: sourceInGroup ? groupNode.id : link.source.id,
                target: targetInGroup ? groupNode.id : link.target.id,
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

// 修改expandNode函数，确保保留所有连接
function expandNode(groupNode, simulation) {
    if (!groupNode.isGroup) return;

    const currentNodes = simulation.nodes();
    const currentLinks = simulation.force('link').links();

    // 移除组节点
    const nodeIndex = currentNodes.indexOf(groupNode);
    if (nodeIndex > -1) {
        currentNodes.splice(nodeIndex, 1);
    }

    // 添加原始节点，保持组信息
    groupNode.originalNodes.forEach(node => {
        node.x = groupNode.x + (Math.random() - 0.5) * 50;
        node.y = groupNode.y + (Math.random() - 0.5) * 50;
        node.size = 8;
        node.groupId = groupNode.groupId;  // 保持组信息
        currentNodes.push(node);
    });

    // 更新连接
    const newLinks = currentLinks.filter(link => 
        link.source.id !== groupNode.id && link.target.id !== groupNode.id
    );

    // 添加组内原始连接
    if (groupNode.originalLinks) {
        groupNode.originalLinks.forEach(link => {
            newLinks.push({
                source: link.source.id || link.source,
                target: link.target.id || link.target,
                value: link.value || 1
            });
        });
    }

    // 重建与其他节点的连接
    currentLinks.forEach(link => {
        if (link.source.id === groupNode.id) {
            groupNode.originalNodes.forEach(node => {
                newLinks.push({
                    source: node.id,
                    target: link.target.id,
                    value: 1
                });
            });
        } else if (link.target.id === groupNode.id) {
            groupNode.originalNodes.forEach(node => {
                newLinks.push({
                    source: link.source.id,
                    target: node.id,
                    value: 1
                });
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
                // 如果是组节点，检查其所有原始节点是否都被选中
                return d.originalNodes.every(originalNode => 
                    selectedNodeIds.value.includes(originalNode.name.split('/').pop())
                ) ? '#ff6347' : '#69b3a2';
            } else {
                // 如果是普通节点，直接检查是否被选中
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
        .on('contextmenu', function(event, d) {
            event.preventDefault();
            const [x, y] = d3.pointer(event, svg.node());
            createContextMenu(svg, x, y, d, simulation);
        })
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended)
        );

    // 节点标签
    g.append('g')
        .attr('class', 'labels')
        .selectAll('text')
        .data(simulation.nodes())
        .join('text')
        .attr('dy', -10)
        .attr('text-anchor', 'middle')
        .text((d) => d.name.split('/').pop())
        .style('font-size', '14px');

    simulation.on('tick', () => {
        link
            .attr('x1', (d) => d.source.x)
            .attr('y1', (d) => d.source.y)
            .attr('x2', (d) => d.target.x)
            .attr('y2', (d) => d.target.y);

        node.attr('cx', (d) => d.x).attr('cy', (d) => d.y);

        g.selectAll('.labels text')
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
    const width = container.clientWidth || 600; // 提供默认值
    const height = container.clientHeight || 400; // 提供默认值

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

    // 绘制节点
    const node = g
        .append('g')
        .attr('class', 'nodes')
        .selectAll('circle')
        .data(processedData.nodes)
        .join('circle')
        .attr('r', d => d.size || 8)
        .attr('fill', d => {
            if (d.isGroup) {
                // 如果是组节点，检查其所有原始节点是否都被选中
                return d.originalNodes.every(originalNode => 
                    selectedNodeIds.value.includes(originalNode.name.split('/').pop())
                ) ? '#ff6347' : '#69b3a2';
            } else {
                // 如果是普通节点，直接检查是否被选中
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
        .on('contextmenu', function(event, d) {
            event.preventDefault();
            const [x, y] = d3.pointer(event, svg.node());
            createContextMenu(svg, x, y, d, simulation);
        })
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended)
        );

    // 节点标签
    g.append('g')
        .attr('class', 'labels')
        .selectAll('text')
        .data(processedData.nodes)
        .join('text')
        .attr('dy', -10)
        .attr('text-anchor', 'middle')
        .text((d) => d.name.split('/').pop())
        .style('font-size', '14px');

    simulation.on('tick', () => {
        link
            .attr('x1', (d) => d.source.x)
            .attr('y1', (d) => d.source.y)
            .attr('x2', (d) => d.target.x)
            .attr('y2', (d) => d.target.y);

        node.attr('cx', (d) => d.x).attr('cy', (d) => d.y);

        g.selectAll('.labels text')
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

function clearSelectedNodes() {
    store.dispatch('clearSelectedNodes');

    graphContainers.value.forEach((container) => {
        d3.select(container)
            .selectAll('circle')
            .attr('fill', '#69b3a2');
    });
}

watch(selectedNodeIds, () => {
    nextTick(() => {
        graphContainers.value.forEach((container) => {
            const svg = d3.select(container).select('svg');
            svg.selectAll('circle')
                .attr('fill', (d) => {
                    const nodeName = d.name.split('/').pop();
                    return selectedNodeIds.value.includes(nodeName) ? '#ff6347' : '#69b3a2';
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

// 修改onMounted钩子
onMounted(async () => {
    try {
        // 等待DOM完全渲染
        await nextTick();
        // 等待一小段时间确保布局计算完成
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
.graph-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    width: 100%;
    padding: 20px;
}

.graph-item {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 400px;
    border: 1px solid #dcdfe6;
    border-radius: 8px;
    background: #ffffff;
    overflow: hidden;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.graph-title {
    padding: 12px;
    font-weight: bold;
    background-color: #f5f7fa;
    color: #303133;
    border-bottom: 1px solid #dcdfe6;
}

.graph-container {
    flex: 1;
    width: 100%;
    height: calc(100% - 40px);
    position: relative;
}

.controls {
    position: absolute;
    top: 530px;
    z-index: 1000;
}


.clear-button {
    padding: 8px 20px;
    background-color: #ff6347;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.clear-button:hover {
    background-color: #ff4f2b;
    transform: translateY(-1px);
}

.selection {
    fill: #55C000;
    fill-opacity: 0.2;
    stroke: #55C000;
    stroke-width: 1px;
}

.checkbox-control span {
    color: #606266;
    font-size: 14px;
    font-weight: 500;
}

.pagination-wrapper {
    display: flex;
    justify-content: center;
    padding: 20px;
    margin: 20px;
}

:deep(.v-pagination__item--active) {
    background-color: #55C000 !important;
}

:deep(.v-pagination__item:hover) {
    color: #55C000;
}

:deep(.v-pagination__navigation:hover) {
    color: #55C000;
}
</style>
