<template>
    <v-card class="fill-height mac-style-card">
        <span class="title">Visual Elements</span>

        <div v-if="analyzing" class="progress-card">
            <div class="progress-label">{{ currentStep }}</div>
            <v-progress-linear :model-value="progress" color="primary" height="6" rounded :striped="false" bg-color="rgba(144, 95, 41, 0.1)">
                <template v-slot:default="{ value }">
                    <div class="progress-value">{{ Math.ceil(value) }}</div>
                </template>
            </v-progress-linear>
        </div>

        <!-- 主内容区域：上下两栏布局 -->
        <div class="layout-container" v-if="file">
            <!-- 上部区域：左右分栏 -->
            <div class="top-panel">
                <!-- 左侧区域：C区 - 可点击但不高亮的SVG -->
                <div class="left-panel">
                    <div class="section-title">Chart preview</div>
                    <div class="svg-container mac-style-container control-svg" ref="controlSvgContainer">
                        <div v-html="controlSvgContent"></div>
                        <v-btn class="mac-style-button submit-button" @click="updatePerceptionScope" :disabled="selectedElements.length === 0 || analyzing">
                            Update the perception scope as the selected group
                        </v-btn>
                    </div>
                </div>

                <!-- 右侧区域：S区 - 可高亮但不可点击的SVG -->
                <div class="right-panel">
                    <div class="section-title">Selected elements</div>
                    <div class="svg-container mac-style-container display-svg" ref="displaySvgContainer">
                        <div v-html="displaySvgContent"></div>
                        <div class="visual-salience-indicator" @click="showSalienceDetail">
                            <span class="salience-label">Visual salience</span>
                            <span class="salience-value" v-if="selectedNodeIds.length > 0">{{ (visualSalience * 100).toFixed(3) }}</span>
                            <span class="salience-value" v-else>--.---</span>
                        </div>
                        <v-btn class="mac-style-button submit-button" @click="analyzeSvg" :disabled="selectedElements.length === 0 || analyzing">
                            {{ analyzing ? 'Simulating...' : 'Submit the selected group' }}
                        </v-btn>
                    </div>
                </div>
            </div>

            <!-- 下部区域：select pannel -->
            <div class="bottom-panel">
                <div v-if="visibleElements.length > 0" class="element-selector">
                    <div class="select-panel-layout">
                        <!-- 左侧Select Panel标题 -->
                        <div class="select-panel-title">
                            <div style="font-weight: 600;">Select</div>
                            <div style="font-weight: 600;">Panel</div>
                        </div>

                        <!-- 中间分隔线 -->
                        <div class="vertical-divider"></div>

                        <!-- 中间表格 -->
                        <div class="element-type-table">
                            <table>
                                <tbody>
                                    <tr>
                                        <td class="element-type-table-header" style="font-weight: 700;">Element types</td>
                                        <td v-for="element in visibleElements" :key="element.id">
                                            <v-checkbox v-model="selectedElements" :value="element.id" hide-details density="compact" class="mac-style-checkbox">
                                                <template v-slot:label>
                                                    <span class="element-tag">{{ element.tag }}</span>
                                                </template>
                                            </v-checkbox>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td class="element-type-table-header" style="font-weight: 700;">#In the perception scope</td>
                                        <td v-for="element in visibleElements" :key="element.id">
                                            {{ element.count }}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td class="element-type-table-header" style="font-weight: 700;">#In the selected group</td>
                                        <td v-for="element in visibleElements" :key="element.id">
                                            {{ getSelectedCountForType(element.tag) }}
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>

                        <!-- 右侧分隔线 -->
                        <div class="vertical-divider" style="margin-left: 60px;"></div>
                        <div class="selection-mode-buttons">
                            <v-btn @click.stop="setSelectionMode('click')" class="selection-mode-btn" :class="{ 'active-mode': selectionMode === 'click' }">
                                <v-icon>mdi-cursor-default-click</v-icon>
                                <span class="selection-text">Click</span>
                            </v-btn>
                            <v-btn @click.stop="setSelectionMode('lasso')" class="selection-mode-btn" :class="{ 'active-mode': selectionMode === 'lasso' }">
                                <v-icon>mdi-gesture</v-icon>
                                <span class="selection-text">Lasso</span>
                            </v-btn>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </v-card>
</template>

<script setup>
import { ref, watch, nextTick, computed, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import { useStore } from 'vuex';
import * as d3 from 'd3';
import { ElMessage } from 'element-plus';

const file = ref(null)
const controlSvgContent = ref('') // C区SVG内容
const displaySvgContent = ref('') // S区SVG内容
const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);
// 添加scopeNodes来存放Chart Preview中选中的元素
const scopeNodes = ref([]);
const allVisiableNodes = computed(() => store.state.AllVisiableNodes);
const controlSvgContainer = ref(null);
const displaySvgContainer = ref(null);
const isTracking = ref(false);
const currentTransform = ref(null);
const nodeEventHandlers = new Map();
const visibleElements = ref([]);
const selectedElements = ref([]);
// 添加视觉显著性数据
const normalizedData = ref([]);
const visualSalience = ref(0);
// 添加SVG内容同步状态
const isSyncingSvg = ref(false);

const emit = defineEmits(['file-uploaded'])

// 添加新的方法
const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// 添加事件监听
onMounted(() => {
    window.addEventListener('svg-uploaded', handleSvgUploaded)

    // 初始化时设置默认的鼠标样式
    nextTick(() => {
        if (controlSvgContainer.value) {
            controlSvgContainer.value.classList.add('click-cursor');
        }
    });
})

// 在组件卸载时移除事件监听
onUnmounted(() => {
    window.removeEventListener('svg-uploaded', handleSvgUploaded)
})

// 添加清除选中节点的函数
const clearSelectedNodes = () => {
    store.dispatch('clearSelectedNodes');
};

// 处理从CodeToSvg组件触发的上传事件
const handleSvgUploaded = async (event) => {
    const filename = event.detail.filename
    // 清除选中的节点
    clearSelectedNodes();

    try {
        // 获取C区SVG内容 - 原始上传的SVG
        const controlResponse = await fetch('http://127.0.0.1:5000/get_upload_svg', {
            responseType: 'text',
            headers: {
                'Accept': 'image/svg+xml'
            }
        })

        const controlSvgData = await controlResponse.text()

        // 获取S区SVG内容 - 处理后的SVG
        const displayResponse = await fetch('http://127.0.0.1:5000/get_svg', {
            responseType: 'text',
            headers: {
                'Accept': 'image/svg+xml'
            }
        })

        const displaySvgData = await displayResponse.text()

        // 创建File对象 (使用控制区SVG)
        const blob = new Blob([controlSvgData], { type: 'image/svg+xml' })
        const fileObj = new File([blob], filename, { type: 'image/svg+xml' })

        // 更新file引用，这会触发界面更新
        file.value = fileObj

        // 处理并设置SVG内容
        await processAndSetSvgContent(controlSvgData, displaySvgData)

        // 获取可见元素列表
        const elementsResponse = await axios.post('http://127.0.0.1:5000/get_visible_elements', {
            filename: filename
        })

        // 获取normalized数据
        await fetchNormalizedData();

        if (elementsResponse.data.success) {
            visibleElements.value = elementsResponse.data.elements
            selectedElements.value = elementsResponse.data.elements.map(el => el.id)

            // 确保DOM更新后再设置交互
            await nextTick()
            setupDualSvgInteractions()
        }
    } catch (error) {
        console.error('Error handling upload event:', error)
    }
}

// 处理并设置SVG内容
const processAndSetSvgContent = async (controlSvgData, displaySvgData) => {
    // 处理控制区SVG
    const controlParser = new DOMParser();
    const controlSvgDoc = controlParser.parseFromString(controlSvgData, 'image/svg+xml');
    const controlSvgElement = controlSvgDoc.querySelector('svg');

    if (controlSvgElement) {
        // 从原始SVG中提取viewBox
        const viewBoxMatch = controlSvgData.match(/viewBox="([^"]*)"/) || [];
        const viewBox = viewBoxMatch[1] || '0 0 800 600';

        // 设置SVG的响应式属性
        controlSvgElement.setAttribute('width', '100%');
        controlSvgElement.setAttribute('height', '100%');
        controlSvgElement.setAttribute('viewBox', viewBox);
        controlSvgElement.setAttribute('preserveAspectRatio', 'xMidYMid meet');

        // 确保有一个包装组用于缩放
        let wrapper = controlSvgElement.querySelector('g.zoom-wrapper');
        if (!wrapper) {
            wrapper = document.createElementNS("http://www.w3.org/2000/svg", "g");
            wrapper.setAttribute('class', 'zoom-wrapper');
            while (controlSvgElement.firstChild) {
                wrapper.appendChild(controlSvgElement.firstChild);
            }
            controlSvgElement.appendChild(wrapper);
        }

        controlSvgContent.value = controlSvgElement.outerHTML;
    }

    // 处理显示区SVG
    const displayParser = new DOMParser();
    const displaySvgDoc = displayParser.parseFromString(displaySvgData, 'image/svg+xml');
    const displaySvgElement = displaySvgDoc.querySelector('svg');

    if (displaySvgElement) {
        // 从原始SVG中提取viewBox
        const viewBoxMatch = displaySvgData.match(/viewBox="([^"]*)"/) || [];
        const viewBox = viewBoxMatch[1] || '0 0 800 600';

        // 设置SVG的响应式属性
        displaySvgElement.setAttribute('width', '100%');
        displaySvgElement.setAttribute('height', '100%');
        displaySvgElement.setAttribute('viewBox', viewBox);
        displaySvgElement.setAttribute('preserveAspectRatio', 'xMidYMid meet');

        // 确保有一个包装组用于缩放
        let wrapper = displaySvgElement.querySelector('g.zoom-wrapper');
        if (!wrapper) {
            wrapper = document.createElementNS("http://www.w3.org/2000/svg", "g");
            wrapper.setAttribute('class', 'zoom-wrapper');
            while (displaySvgElement.firstChild) {
                wrapper.appendChild(displaySvgElement.firstChild);
            }
            displaySvgElement.appendChild(wrapper);
        }

        displaySvgContent.value = displaySvgElement.outerHTML;
    }

    // 获取最新的normalized数据
    await fetchNormalizedData();

    // 确保DOM更新后再更新节点透明度
    await nextTick();
    updateControlNodeOpacity();
    updateDisplayNodeOpacity();

    return nextTick();
}

// 添加进度相关的响应式变量
const analyzing = ref(false);
const progress = ref(0);
const currentStep = ref('');

const analyzeSvg = () => {
    if (!file.value) return;

    // 重置进度状态
    analyzing.value = true;
    progress.value = 0;
    currentStep.value = 'Prepare to percept...';

    // 确保 selectedNodeIds 是数组格式
    const nodeIds = Array.isArray(selectedNodeIds.value) ? selectedNodeIds.value : [];

    // 创建 EventSource 连接
    const eventSource = new EventSource('http://127.0.0.1:5000/progress');

    // 监听进度更新
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        progress.value = data.progress;
        currentStep.value = data.step;
    };

    // 监听错误
    eventSource.onerror = () => {
        eventSource.close();
    };

    axios.post('http://127.0.0.1:5000/filter_and_process', {
        filename: file.value.name,
        selectedElements: selectedElements.value,
        selectedNodeIds: nodeIds
    })
        .then(response => {
            if (response.data.success) {
                return refreshSvgContent();
            } else {
                throw new Error(response.data.error || 'Failure to analyse');
            }
        })
        .then(() => {
            window.dispatchEvent(new CustomEvent('svg-content-updated', {
                detail: {
                    filename: file.value.name,
                    type: 'analysis'
                }
            }));
            emit('file-uploaded');
        })
        .catch(error => {
            console.error('Error in the analysis process:', error);
        })
        .finally(async () => {
            analyzing.value = false;
            eventSource.close();
            await fetchNormalizedData();
            calculateVisualSalience();
        });
}

// 刷新SVG内容
const refreshSvgContent = async () => {
    clearSelectedNodes();

    try {
        // 获取C区SVG内容 - 原始上传的SVG
        const controlResponse = await fetch('http://127.0.0.1:5000/get_upload_svg', {
            responseType: 'text',
            headers: {
                'Accept': 'image/svg+xml'
            }
        });

        const controlSvgData = await controlResponse.text();

        // 获取S区SVG内容 - 处理后的SVG
        const displayResponse = await fetch('http://127.0.0.1:5000/get_svg', {
            responseType: 'text',
            headers: {
                'Accept': 'image/svg+xml'
            }
        });

        const displaySvgData = await displayResponse.text();

        // 处理并设置SVG内容
        await processAndSetSvgContent(controlSvgData, displaySvgData);

        // 确保DOM更新后再设置交互
        await nextTick();
        setupDualSvgInteractions();

        return true;
    } catch (error) {
        console.error('Error refreshing SVG content:', error);
        throw error;
    }
}

// 更新感知范围
const updatePerceptionScope = () => {
    // 实现更新感知范围的逻辑
    console.log('Updating perception scope with selected elements:', selectedElements.value);
    console.log('Scope node IDs:', scopeNodes.value);

    // 如果scopeNodes为空（没有特别选择的节点），则将所有可见元素设为选中
    if (scopeNodes.value.length === 0) {
        // 获取所有可见元素的ID
        const allVisibleIds = [];
        if (controlSvgContainer.value) {
            const svg = controlSvgContainer.value.querySelector('svg');
            if (svg) {
                selectedElements.value.forEach(elementType => {
                    const elements = svg.querySelectorAll(elementType);
                    elements.forEach(element => {
                        if (element.id) {
                            allVisibleIds.push(element.id);
                        }
                    });
                });
            }
        }

        // 清空当前选中节点并添加所有可见节点
        store.dispatch('clearSelectedNodes');
        allVisibleIds.forEach(id => {
            store.dispatch('addSelectedNode', id);
        });
    } else {
        // 将scopeNodes中的节点传入selectedNodes
        store.dispatch('clearSelectedNodes');
        scopeNodes.value.forEach(nodeId => {
            store.dispatch('addSelectedNode', nodeId);
        });
    }

    // 更新完成后，使用ElMessage通知用户
    ElMessage({
        message: '感知范围已更新，包含' + (selectedNodeIds.value.length === 0 ? '所有可见元素' : selectedNodeIds.value.length + '个元素'),
        type: 'success',
        duration: 3000
    });
}

// 为两个SVG添加缩放和拖拽功能
const addZoomEffectToDualSvgs = () => {
    // 为控制区SVG添加缩放效果
    if (controlSvgContainer.value) {
        const svg = d3.select(controlSvgContainer.value).select('svg');
        addZoomEffectToSvg(svg, true, 'control');
    }

    // 为显示区SVG也添加缩放拖拽功能
    if (displaySvgContainer.value) {
        const svg = d3.select(displaySvgContainer.value).select('svg');
        addZoomEffectToSvg(svg, true, 'display');
    }
};

// 添加缩放和拖拽功能到单个SVG
const addZoomEffectToSvg = (svg, enableInteraction, svgType) => {
    if (!svg || svg.empty()) return;

    // 创建一个包裹实际SVG内容的组
    let g = svg.select('g.zoom-wrapper');
    if (g.empty()) {
        g = svg.append('g').attr('class', 'zoom-wrapper');
        // 将所有现有内容动到新的组中
        const children = svg.node().childNodes;
        [...children].forEach(child => {
            if (child.nodeType === 1 && !child.classList.contains('zoom-wrapper')) {
                g.node().appendChild(child);
            }
        });
    }

    // 为SVG添加缩放和拖拽功能
    if (enableInteraction) {
        const zoom = d3.zoom()
            .scaleExtent([0.5, 10])
            .on('zoom', (event) => {
                if (!isTracking.value) {
                    g.attr('transform', event.transform);

                    // 同步另一个SVG的缩放
                    syncOtherSvgZoom(svg, event.transform, svgType);
                }
            });

        svg.call(zoom);

        // 设置初始缩放为0.9（90%的原始大小）并向右平移5%
        const width = svg.node().getBoundingClientRect().width;
        const translateX = width * 0.05; // 向右平移5%
        svg.call(zoom.transform, d3.zoomIdentity.translate(translateX, 10).scale(0.9));
    } else {
        // 为不可交互的SVG移除任何缩放相关监听器
        svg.on('.zoom', null);

        // 设置初始变换以匹配控制区SVG
        const width = svg.node().getBoundingClientRect().width;
        const translateX = width * 0.05;
        g.attr('transform', d3.zoomIdentity.translate(translateX, 10).scale(0.9));
    }
};

// 同步两个SVG的缩放状态
const syncOtherSvgZoom = (currentSvg, transform, svgType) => {
    if (isSyncingSvg.value) return; // 防止循环同步

    isSyncingSvg.value = true;

    try {
        if (svgType === 'control' && displaySvgContainer.value) {
            // 从控制区同步到显示区
            const displaySvg = d3.select(displaySvgContainer.value).select('svg');
            if (!displaySvg.empty()) {
                displaySvg.select('g.zoom-wrapper').attr('transform', transform);
            }
        } else if (svgType === 'display' && controlSvgContainer.value) {
            // 从显示区同步到控制区
            const controlSvg = d3.select(controlSvgContainer.value).select('svg');
            if (!controlSvg.empty()) {
                controlSvg.select('g.zoom-wrapper').attr('transform', transform);
            }
        }
    } finally {
        isSyncingSvg.value = false;
    }
};

// 路径选择功能
const toggleTrackMode = () => {
    isTracking.value = !isTracking.value;
    const svg = d3.select(controlSvgContainer.value).select('svg'); // 只在控制区启用路径选择

    if (isTracking.value) {
        nextTick(() => {
            controlSvgContainer.value.classList.add('copy-cursor');
        });
        enableTrackMode();

        const transform = d3.zoomTransform(svg.node());
        currentTransform.value = transform;
        svg.on('.zoom', null); // 只禁用控制区SVG的缩放
    } else {
        controlSvgContainer.value.classList.remove('copy-cursor');
        disableTrackMode();

        const zoom = d3.zoom()
            .scaleExtent([0.5, 10])
            .on('zoom', (event) => {
                if (!isTracking.value) {
                    svg.select('g.zoom-wrapper').attr('transform', event.transform);
                    syncOtherSvgZoom(svg, event.transform, 'control');
                }
            });

        svg.call(zoom);
        if (currentTransform.value) {
            svg.call(zoom.transform, currentTransform.value);
        }
    }
};

const enableTrackMode = () => {
    let isMouseDown = false;
    let clickedElements = new Set();
    const svg = controlSvgContainer.value.querySelector('svg'); // 使用控制区SVG

    const handleMouseDown = () => {
        isMouseDown = true;
        clickedElements.clear();
    };

    const handleMouseUp = (event) => {
        if (isMouseDown && clickedElements.size > 0) {
            // 如果已经选中了元素，阻止事件冒泡以避免触发点击事件
            event.stopPropagation();
        }
        isMouseDown = false;
    };

    const handleMouseMove = (event) => {
        if (isMouseDown) {
            const point = svg.createSVGPoint();
            point.x = event.clientX;
            point.y = event.clientY;
            const svgPoint = point.matrixTransform(svg.getScreenCTM().inverse());

            const node = document.elementFromPoint(event.clientX, event.clientY);
            if (node && node.id && !clickedElements.has(node) &&
                node.tagName && node.tagName.toLowerCase() !== 'svg' &&
                node.tagName.toLowerCase() !== 'g') {
                clickedElements.add(node);
                node.dispatchEvent(new Event('click', { bubbles: true }));
            }
        }
    };

    svg.addEventListener('mousedown', handleMouseDown);
    svg.addEventListener('mouseup', handleMouseUp);
    svg.addEventListener('mousemove', handleMouseMove);

    nodeEventHandlers.set(svg, { handleMouseDown, handleMouseUp, handleMouseMove });
};

const disableTrackMode = () => {
    const svg = controlSvgContainer.value.querySelector('svg'); // 使用控制区SVG
    if (svg) {
        const handlers = nodeEventHandlers.get(svg);
        if (handlers) {
            svg.removeEventListener('mousedown', handlers.handleMouseDown);
            svg.removeEventListener('mouseup', handlers.handleMouseUp);
            svg.removeEventListener('mousemove', handlers.handleMouseMove);
        }
    }
};

// 设置双SVG交互 - 新增函数
const setupDualSvgInteractions = () => {
    // 设置控制区SVG交互 - 可点击但不高亮
    setupControlSvgInteractions();

    // 设置显示区SVG交互 - 可高亮但不可点击
    setupDisplaySvgInteractions();

    // 添加缩放效果到两个SVG
    addZoomEffectToDualSvgs();

    // 确保两个区域的节点透明度正确设置
    nextTick(() => {
        updateControlNodeOpacity();
        updateDisplayNodeOpacity();
    });
};

// 设置控制区SVG交互 - 可点击但不高亮
const setupControlSvgInteractions = () => {
    const svgContainer = controlSvgContainer.value;
    if (!svgContainer) {
        console.warn('Control SVG container not found');
        return;
    }

    const svg = svgContainer.querySelector('svg');
    if (!svg) {
        console.warn('SVG element not found in control container');
        return;
    }

    // 移除现有的事件监听器
    const oldClickHandler = svg._clickHandler;
    if (oldClickHandler) {
        svg.removeEventListener('click', oldClickHandler);
    }

    // 移除现有的悬停处理器
    const oldMouseoverHandler = svg._mouseoverHandler;
    const oldMouseoutHandler = svg._mouseoutHandler;
    if (oldMouseoverHandler) {
        svg.removeEventListener('mouseover', oldMouseoverHandler);
    }
    if (oldMouseoutHandler) {
        svg.removeEventListener('mouseout', oldMouseoutHandler);
    }

    // 保存新的事件处理器引用
    svg._clickHandler = handleControlSvgClick;
    svg._mouseoverHandler = handleControlSvgMouseover;
    svg._mouseoutHandler = handleControlSvgMouseout;

    // 添加新的事件监听器
    svg.addEventListener('click', svg._clickHandler);
    svg.addEventListener('mouseover', svg._mouseoverHandler);
    svg.addEventListener('mouseout', svg._mouseoutHandler);

    // 更新控制区节点的高亮状态
    updateControlNodeOpacity();

    // 根据当前选择模式设置鼠标样式
    nextTick(() => {
        if (svgContainer) {
            if (selectionMode.value === 'lasso') {
                svgContainer.classList.add('lasso-cursor');
                svgContainer.classList.remove('click-cursor');
            } else {
                svgContainer.classList.add('click-cursor');
                svgContainer.classList.remove('lasso-cursor');
            }
        }
    });
};

// 控制区鼠标悬停处理
const handleControlSvgMouseover = (event) => {
    const target = event.target;
    if (!target || !target.tagName) return;

    const nodeType = target.tagName.toLowerCase();

    // 只有当元素类型在选中列表中时，才改变鼠标样式，但不改变亮度
    if (selectedElements.value.includes(nodeType) && target.id) {
        target.style.cursor = 'pointer';
        // 移除亮度滤镜，使元素只在点击时改变样式
    }
};

// 控制区鼠标移出处理
const handleControlSvgMouseout = (event) => {
    const target = event.target;
    if (!target || !target.tagName) return;

    // 恢复原始样式
    const nodeType = target.tagName.toLowerCase();
    if (selectedElements.value.includes(nodeType)) {
        // 恢复默认鼠标样式
        target.style.cursor = '';
    }
};

// 控制区SVG点击处理函数
const handleControlSvgClick = (event) => {
    // 检查点击的是否是 SVG 容器本身或者 zoom-wrapper
    const target = event.target;
    if (target.tagName.toLowerCase() === 'svg' ||
        (target.tagName.toLowerCase() === 'g' && target.classList.contains('zoom-wrapper'))) {
        // 无论是否在多选模式下，点击空白区域都清空所有选中的节点
        scopeNodes.value = [];
        // 清空后也需要更新控制区的透明度
        nextTick(() => {
            updateControlNodeOpacity();
        });
        return;
    }

    // 只有当节点类型在selectedElements中时才处理点击事件
    const nodeType = target.tagName.toLowerCase();
    if (!selectedElements.value.includes(nodeType)) {
        return;
    }

    // 如果点击的是具体的 SVG 元素，则执行选中逻辑
    const nodeId = target.id;
    if (!nodeId) return;

    if (scopeNodes.value.includes(nodeId)) {
        // 从scopeNodes中移除节点
        scopeNodes.value = scopeNodes.value.filter(id => id !== nodeId);
    } else {
        // 添加节点到scopeNodes
        scopeNodes.value.push(nodeId);
    }

    // 使用 nextTick 确保状态更新后再更新控制区视图
    nextTick(() => {
        updateControlNodeOpacity();
    });
};

// 更新显示区节点透明度（根据选中的节点ID）
const updateDisplayNodeOpacity = () => {
    const svgContainer = displaySvgContainer.value;
    if (!svgContainer) return;

    const svg = svgContainer.querySelector('svg');
    if (!svg) return;

    try {
        const allNodes = svg.querySelectorAll('*');

        allNodes.forEach(node => {
            if (!node.tagName || node.tagName.toLowerCase() === 'svg' ||
                node.tagName.toLowerCase() === 'g') return;

            const nodeType = node.tagName.toLowerCase();
            const nodeId = node.id;

            // 所有节点基础都可见（不受selectedElements影响）
            let opacity = 1;
            let isHighlighted = true; // 默认为高亮状态

            // 如果有选中的节点，应用高亮逻辑
            if (selectedNodeIds.value.length > 0) {
                isHighlighted = selectedNodeIds.value.includes(nodeId);
                opacity = isHighlighted ? 1 : 0.1;
            }

            // 设置透明度
            node.style.opacity = opacity;
            node.style.transition = 'opacity 0.3s ease, filter 0.3s ease';

            // 保存原始颜色属性
            if (!node.dataset.originalFill && node.getAttribute('fill')) {
                node.dataset.originalFill = node.getAttribute('fill');
            }
            if (!node.dataset.originalStroke && node.getAttribute('stroke')) {
                node.dataset.originalStroke = node.getAttribute('stroke');
            }

            // 对于非高亮元素，应用灰色滤镜
            if (!isHighlighted && opacity > 0) {
                node.style.filter = 'grayscale(100%)';
            } else {
                node.style.filter = 'none';
            }
        });
    } catch (error) {
        console.error('Error updating display node opacity:', error);
    }
};

// 监听选中节点的变化
watch(selectedNodeIds, async () => {
    await nextTick();
    updateDisplayNodeOpacity(); // 更新显示区的节点透明度
    updateControlNodeOpacity(); // 更新控制区的节点透明度
    // 当选中节点变化时，先获取最新的normalized数据，再计算视觉显著性
    await fetchNormalizedData();
    calculateVisualSalience();
});

// 监听selectedElements的变化
watch(selectedElements, () => {
    nextTick(() => {
        updateControlNodeOpacity(); // 只更新控制区的节点透明度
    });
});

// 监听文件变化
watch(() => file.value, (newFile) => {
    // 保留文件变化监听但不进行折叠相关操作
});

// 监听分析状态
watch(() => analyzing.value, (newValue) => {
    // 保留分析状态监听但不进行折叠相关操作
});

// 添加选择模式变量和方法
const selectionMode = ref('click'); // 默认为点击选择模式

const setSelectionMode = (mode) => {
    selectionMode.value = mode;

    if (mode === 'lasso') {
        if (!isTracking.value) {
            toggleTrackMode(); // 启用多选模式
        }
        // 添加lasso模式的鼠标样式
        nextTick(() => {
            if (controlSvgContainer.value) {
                controlSvgContainer.value.classList.add('lasso-cursor');
                controlSvgContainer.value.classList.remove('click-cursor');
            }
        });
    } else {
        if (isTracking.value) {
            toggleTrackMode(); // 禁用多选模式
        }
        // 添加clicking模式的鼠标样式
        nextTick(() => {
            if (controlSvgContainer.value) {
                controlSvgContainer.value.classList.add('click-cursor');
                controlSvgContainer.value.classList.remove('lasso-cursor');
            }
        });
    }
};

// 获取normalized数据
const fetchNormalizedData = async () => {
    try {
        const response = await fetch('http://127.0.0.1:5000/normalized_init_json');
        if (response.ok) {
            const data = await response.json();
            normalizedData.value = data;
        } else {
            console.error('Failed to fetch normalized data');
        }
    } catch (error) {
        console.error('Error fetching normalized data:', error);
    }
}

// 计算视觉显著性
const calculateVisualSalience = () => {
    if (!normalizedData.value || normalizedData.value.length === 0 || selectedNodeIds.value.length === 0) {
        visualSalience.value = 0.1;
        return;
    }

    try {
        // 获取当前高亮节点
        const highlightedIds = selectedNodeIds.value;

        if (!highlightedIds || highlightedIds.length === 0) {
            visualSalience.value = 0.1;
            return;
        }

        // 将所有节点分为高亮组和非高亮组
        const highlightedFeatures = [];
        const nonHighlightedFeatures = [];
        const highlightedIdsForComp = [];
        const nonHighlightedIdsForComp = [];

        // 遍历normalized数据
        normalizedData.value.forEach(item => {
            // 标准化ID格式以便比较
            const normalizedItemId = item.id;

            // 检查当前元素是否高亮（通过ID匹配）
            const isHighlighted = highlightedIds.some(id => {
                // 提取ID的最后部分进行比较
                const idParts = normalizedItemId.split('/');
                const itemIdLastPart = idParts[idParts.length - 1];
                return itemIdLastPart === id;
            });

            if (isHighlighted) {
                highlightedFeatures.push(item.features);
                highlightedIdsForComp.push(normalizedItemId);
            } else {
                nonHighlightedFeatures.push(item.features);
                nonHighlightedIdsForComp.push(normalizedItemId);
            }
        });

        if (highlightedFeatures.length === 0) {
            visualSalience.value = 0.1;
            return;
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
                    // 计算特征向量之间的余弦相似度
                    similaritySum += cosineSimilarity(highlightedFeatures[i], highlightedFeatures[j]);
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
                // 计算特征向量之间的余弦相似度
                interGroupSimilarity += cosineSimilarity(highlightedFeatures[i], nonHighlightedFeatures[j]);
                interPairCount++;
            }
        }

        // 计算平均相似度，避免除以零
        interGroupSimilarity = interPairCount > 0 ? interGroupSimilarity / interPairCount : 0;

        // 避免除以零，如果组间相似度为0，设置显著性为最大值
        let salienceScore = interGroupSimilarity > 0 ? intraGroupSimilarity / interGroupSimilarity : 1.0;

        // 考虑面积因素
        const AREA_INDEX = 19; // bbox_fill_area 在特征向量中的索引是19

        // 计算所有元素的平均面积（包括高亮和非高亮元素）
        const allFeatures = [...highlightedFeatures, ...nonHighlightedFeatures];
        const allElementsAvgArea = allFeatures.reduce((sum, features) =>
            sum + features[AREA_INDEX], 0) / allFeatures.length;

        // 计算高亮元素的平均面积
        const highlightedAvgArea = highlightedFeatures.reduce((sum, features) =>
            sum + features[AREA_INDEX], 0) / highlightedFeatures.length;

        // 使用所有元素平均面积的1.1倍作为阈值
        const areaThreshold = allElementsAvgArea * 1.1;

        // 如果高亮元素的平均面积小于阈值，显著降低显著性
        if (highlightedAvgArea < areaThreshold) {
            salienceScore = salienceScore / 3;
        }

        // 将分数映射到0-1范围内用于显示
        // 使用sigmoid函数进行平滑映射，确保结果在0-1范围内
        const normalizedScore = Math.min(Math.max(1 / (0.8 + Math.exp(-salienceScore))));

        // 计算并设置显著性值
        visualSalience.value = normalizedScore;

        // 将显著性值提交到Vuex store
        store.commit('SET_VISUAL_SALIENCE', normalizedScore);
    } catch (error) {
        console.error('Error calculating visual salience:', error);
        visualSalience.value = 0.2;

        // 将默认显著性值提交到Vuex store
        store.commit('SET_VISUAL_SALIENCE', 0.2);
    }
};

// 显示视觉显著性详情
const showSalienceDetail = () => {
    console.log('Visual salience details:');
    console.log(`- Current salience value: ${(visualSalience.value * 100).toFixed(3)}%`);
    console.log(`- Selected elements count: ${selectedNodeIds.value.length}`);

    // 获取选中元素的类型统计
    const elementTypeCounts = {};

    // 尝试获取当前SVG中选中的元素
    if (displaySvgContainer.value) {
        const svg = displaySvgContainer.value.querySelector('svg');
        if (svg) {
            selectedNodeIds.value.forEach(id => {
                const element = svg.getElementById(id);
                if (element) {
                    const tagName = element.tagName.toLowerCase();
                    elementTypeCounts[tagName] = (elementTypeCounts[tagName] || 0) + 1;
                }
            });
        }
    }

    console.log('- Selected element types:');
    Object.entries(elementTypeCounts).forEach(([type, count]) => {
        console.log(`  * ${type}: ${count}`);
    });
};

// 添加一个计算选中元素数量的方法
const getSelectedCountForType = (tagName) => {
    if (!selectedNodeIds.value || selectedNodeIds.value.length === 0) {
        return 0;
    }

    // 根据tag类型筛选出已选中的节点数量
    // 先检查当前SVG中选中的元素，统计特定类型的数量
    let count = 0;
    if (displaySvgContainer.value) {
        const svg = displaySvgContainer.value.querySelector('svg');
        if (svg) {
            selectedNodeIds.value.forEach(id => {
                const element = svg.getElementById(id);
                if (element && element.tagName.toLowerCase() === tagName.toLowerCase()) {
                    count++;
                }
            });
        }
    }
    return count;
};

// 更新控制区节点透明度（根据选中的节点ID和元素类型）
const updateControlNodeOpacity = () => {
    const svgContainer = controlSvgContainer.value;
    if (!svgContainer) return;

    const svg = svgContainer.querySelector('svg');
    if (!svg) return;

    try {
        const allNodes = svg.querySelectorAll('*');

        allNodes.forEach(node => {
            if (!node.tagName || node.tagName.toLowerCase() === 'svg' ||
                node.tagName.toLowerCase() === 'g') return;

            const nodeType = node.tagName.toLowerCase();
            const nodeId = node.id;

            // 基础透明度 - 根据元素类型是否被选中
            let opacity = selectedElements.value.includes(nodeType) ? 1 : 0.6;
            let isHighlighted = true; // 默认为高亮状态

            // 如果scopeNodes中有节点，应用高亮逻辑
            if (opacity === 1 && scopeNodes.value.length > 0) {
                isHighlighted = scopeNodes.value.includes(nodeId);
                opacity = isHighlighted ? 1 : 0.5;  // 未选中元素仅降低透明度，不变灰
            }

            // 设置透明度
            node.style.opacity = opacity;
            node.style.transition = 'opacity 0.3s ease';

            // 保存原始颜色属性
            if (!node.dataset.originalFill && node.getAttribute('fill')) {
                node.dataset.originalFill = node.getAttribute('fill');
            }
            if (!node.dataset.originalStroke && node.getAttribute('stroke')) {
                node.dataset.originalStroke = node.getAttribute('stroke');
            }

            // 移除灰色滤镜，只调整透明度
            node.style.filter = 'none';
        });
    } catch (error) {
        console.error('Error updating control node opacity:', error);
    }
};

// 设置显示区SVG交互 - 可高亮也可点击
const setupDisplaySvgInteractions = () => {
    const svgContainer = displaySvgContainer.value;
    if (!svgContainer) {
        console.warn('Display SVG container not found');
        return;
    }

    const svg = svgContainer.querySelector('svg');
    if (!svg) {
        console.warn('SVG element not found in display container');
        return;
    }

    // 移除任何可能存在的点击事件处理器
    const oldClickHandler = svg._clickHandler;
    if (oldClickHandler) {
        svg.removeEventListener('click', oldClickHandler);
    }

    // 添加新的点击事件处理器
    svg._clickHandler = handleDisplaySvgClick;
    svg.addEventListener('click', svg._clickHandler);

    // 更新显示区节点的高亮状态
    updateDisplayNodeOpacity();

    // 添加拖拽鼠标样式
    nextTick(() => {
        if (svgContainer) {
            svgContainer.classList.add('grab-cursor');
        }
    });
};

// 显示区SVG点击处理函数
const handleDisplaySvgClick = (event) => {
    // 检查点击的是否是SVG容器本身或者zoom-wrapper
    const target = event.target;
    if (target.tagName.toLowerCase() === 'svg' ||
        (target.tagName.toLowerCase() === 'g' && target.classList.contains('zoom-wrapper'))) {
        return;
    }

    // 只有当节点类型在selectedElements中时才处理点击事件
    const nodeType = target.tagName.toLowerCase();
    if (!selectedElements.value.includes(nodeType)) {
        return;
    }

    // 如果点击的是具体的SVG元素，则执行选中逻辑
    const nodeId = target.id;
    if (!nodeId) return;

    if (selectedNodeIds.value.includes(nodeId)) {
        // 从selectedNodes中移除节点
        store.dispatch('removeSelectedNode', nodeId);
    } else {
        // 添加节点到selectedNodes
        store.dispatch('addSelectedNode', nodeId);
    }

    // 使用nextTick确保状态更新后再更新显示区视图
    nextTick(() => {
        updateDisplayNodeOpacity();
    });
};

// 监听scopeNodes的变化
watch(scopeNodes, () => {
    nextTick(() => {
        updateControlNodeOpacity(); // 更新控制区的节点透明度
    });
});

</script>

<style scoped>
.mac-style-card {
    height: 100%;
    width: 100%;
    padding: 16px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(200, 200, 200, 0.3);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    display: flex;
    flex-direction: column;
}

/* 新增：上下两栏布局样式 */
.layout-container {
    display: flex;
    flex-direction: column;
    flex: 1;
    gap: 12px;
    height: calc(100% - 60px);
    padding: 12px;
}

/* 上部区域样式 */
.top-panel {
    display: flex;
    flex: 7;
    gap: 12px;
    width: 100%;
}

/* 下部区域样式 */
.bottom-panel {
    flex: 1;
    background: rgba(248, 248, 248, 0.5);
    border-radius: 12px;
    border: 1px solid rgba(200, 200, 200, 0.3);
    padding: 2px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
   min-height: 75px;
}

.left-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: rgba(248, 248, 248, 0.5);
    border-radius: 12px;
    border: 1px solid rgba(200, 200, 200, 0.3);
    padding: 5px 12px 12px 12px;
    height: 100%;
    overflow: hidden;
}

.right-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: rgba(248, 248, 248, 0.5);
    border-radius: 12px;
    border: 1px solid rgba(200, 200, 200, 0.3);
    padding: 5px 12px 12px 12px;
    height: 100%;
    overflow: hidden;
}

.section-title {
    font-size: 1.3em;
    font-weight: 600;
    color: #1d1d1f;
}

/* 控制SVG和显示SVG样式差异 */
.control-svg {
    border: 2px solid rgba(144, 95, 41, 0.2);
    background: rgba(255, 255, 255, 0.7);
}

.display-svg {
    border: 2px solid rgba(144, 95, 41, 0.2);
    background: rgba(255, 255, 255, 0.7);
}

.mac-style-input {
    margin-bottom: 16px;
}

.mac-style-input :deep(.v-field) {
    border-radius: 8px;
    background: rgba(240, 240, 240, 0.6);
    border: 1px solid rgba(200, 200, 200, 0.3);
    transition: all 0.3s ease;
}

.mac-style-input :deep(.v-field:hover) {
    background: rgba(235, 235, 235, 0.8);
}

.mac-style-input :deep(.v-field--focused) {
    border-color: rgba(144, 95, 41, 0.35);
    box-shadow: 0 0 0 2px rgba(144, 95, 41, 0.15);
}

.mac-style-selector {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(200, 200, 200, 0.3);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    flex: 1;
    display: flex;
    flex-direction: column;
}

.selector-header {
    position: absolute;
    z-index: 10;
    background: rgba(248, 248, 248, 0.8);
    border-radius: 8px;
    margin: 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
    width: 125px;
}

.title-container {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0;
}

.element-type-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(144, 95, 41, 0.08);
    border-radius: 8px;
    padding: 4px 8px;
    transition: all 0.2s ease;
    border: 1px solid rgba(144, 95, 41, 0.15);
    margin-left: 0;
}

.element-type-icon {
    color: #aa7134;
    font-size: 18px;
}

.element-type-text {
    font-size: 14px;
    font-weight: 500;
    color: #aa7134;
    white-space: nowrap;
}

.mac-style-title {
    font-size: 1.2em;
    font-weight: 600;
    color: #1d1d1f;
    white-space: nowrap;
    text-align: left;
    width: 100%;
}

.mac-style-list {
    border-radius: 8px;
    background: transparent;
    border: none;
    overflow: visible;
}

.mac-style-list-item {
    transition: background-color 0.2s ease;
    border-radius: 6px;
    margin-bottom: 2px;
}

.mac-style-list-item:hover {
    background-color: rgba(144, 95, 41, 0.05);
}

.mac-style-checkbox :deep(.v-selection-control) {
    color: #905F29;
    margin-bottom: -2px;
    /* 减小复选框的底部间距 */
}

.mac-style-checkbox :deep(.v-selection-control__wrapper) {
    margin-bottom: -2px;
    /* 减小复选框的底部间距 */
}

.mac-style-button {
    border-radius: 8px;
    font-size: 1.2em;
    color: #7F5427;
    font-weight: bold;
    height: 40px;
    letter-spacing: 0.3px;
    box-shadow: 0 2px 8px rgba(144, 95, 41, 0.2);
    transition: all 0.3s ease;
    text-transform: none;
}

.mac-style-button:hover {
    transform: translateY(-1px);
}

.mac-style-button:disabled {
    box-shadow: none;
}

.mac-style-container {
    flex: 1 1 auto;
    width: 100%;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
}

.mac-style-container>div {
    position: absolute;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.mac-style-track-button {
    background: gray !important;
    border-radius: 8px;
    color: #905F29;
    font-weight: 500;
    letter-spacing: 0.3px;
    transition: all 0.3s ease;
    text-transform: none;
    margin-left: 8px;
    min-width: 50px;
}

.mac-style-track-button:hover {
    background: rgba(144, 95, 41, 0.2) !important;
}

.mac-style-track-button.active-mode {
    background-color: #905F29 !important;
    color: white !important;
}

.copy-cursor {
    cursor: copy !important;
}

.element-selector {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
    position: relative;
}

/* SVG 相关样式 */
.svg-container svg {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

.control-svg svg * {
    cursor: pointer;
}

.progress-card {
    position: absolute;
    top: 12px;
    /* 调整位置，现在不再有上传区域 */
    left: 230px;
    right: 12px;
    z-index: 100;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(200, 200, 200, 0.3);
    padding: 5px 16px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
}

.progress-label {
    font-size: 13px;
    font-weight: 500;
    color: #1d1d1f;
    margin-bottom: 2px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.progress-value {
    position: absolute;
    right: -40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 12px;
    color: #86868b;
    font-weight: 500;
}

:deep(.v-progress-linear) {
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

:deep(.v-progress-linear__background) {
    opacity: 0.1 !important;
}

:deep(.v-progress-linear__determinate) {
    background: linear-gradient(90deg, rgba(144, 95, 41, 0.35), rgba(144, 95, 41, 0.35));
    box-shadow: 0 1px 3px rgba(144, 95, 41, 0.35);
    transition: all 0.3s ease;
}

.button-container {
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    flex: 1;
    width: 100%;
    gap: 8px;
    padding-left: 134px;
    /* 为标题和按钮预留空间 */
    height: 100%;
}

/* 添加元素类型选择器样式 */
.element-type-selector {
    flex: 1.8;
    background: rgba(255, 255, 255, 0.92);
    border-radius: 8px;
    border: 1px solid rgba(144, 95, 41, 0.2);
    padding: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    height: 100%;
    display: flex;
    flex-direction: row;
    margin-top: 0;
}

.element-types-title {
    writing-mode: normal;
    transform: none;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    font-size: 1.2em;
    font-weight: 600;
    color: #905F29;
    margin-right: 12px;
    line-height: 1.3;
    width: 75px;
    padding-left: 5px;
}

.selector-content {
    flex: 1;
    overflow-y: auto;
    border-radius: 8px;
    background: rgba(250, 250, 250, 0.4);
    border: 1px solid rgba(200, 200, 200, 0.2);
    padding: 2px;
}

/* 添加横向列表样式 */
.horizontal-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    padding: 4px;
    justify-content: center;
    /* 水平居中 */
    align-items: center;
    /* 垂直居中 */
}

.element-type-item {
    flex: 0 0 auto;
    min-width: 120px;
    margin-right: 4px;
    margin-bottom: 2px;
    display: flex;
    justify-content: center;
    /* 内容水平居中 */
}

.selector-content::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

.selector-content::-webkit-scrollbar-track {
    background: rgba(200, 200, 200, 0.1);
    border-radius: 4px;
}

.selector-content::-webkit-scrollbar-thumb {
    background: rgba(144, 95, 41, 0.2);
    border-radius: 4px;
}

.selector-content::-webkit-scrollbar-thumb:hover {
    background: rgba(144, 95, 41, 0.3);
}

.selection-mode-container {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 4px;
    width: 100%;
    height: 100%;
    /* 占满可用高度 */
}

.selection-mode-buttons {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 4px;
    width: 130px;
    min-width: 80px;
    height: 100%;
}

.selection-mode-btn {
    border-radius: 6px;
    color: #aa7134;
    font-weight: 500;
    letter-spacing: 0;
    transition: all 0.3s ease;
    text-transform: none;
    padding: 0px 4px !important;
    background-color: rgba(255, 255, 255, 0.7) !important;
    border: 1px solid rgba(144, 95, 41, 0.2);
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85em;
    margin: 0;
    height: calc(50% - 2px);
    min-height: 0;
}

.selection-mode-btn:hover {
    background-color: rgba(144, 95, 41, 0.1) !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
}

.selection-mode-btn.active-mode {
    background-color: #aa7134 !important;
    color: white !important;
    box-shadow: 0 2px 5px rgba(144, 95, 41, 0.3);
}

.select-panel-layout {
    display: flex;
    flex-direction: row;
    align-items: stretch;
    justify-content: space-between;
    height: 100%;
    width: 100%;
    padding: 4px;
}

.select-panel-title {
    font-size: 1.3em;
    color: #1d1d1f;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 70px;
    min-width: 55px;
    padding: 0 15px;
    text-align: left;
    line-height: 1.5;
}

.vertical-divider {
    width: 1px;
    height: auto;
    background-color: rgba(144, 95, 41, 0.2);
    margin: 0 6px 0 6px;
}

.element-type-table {
    flex: 1;
    overflow: auto;
    max-height: 100%;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    color: #1d1d1f;
}

.element-type-table table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    height: 100%;
    margin: 0;
    padding: 0;
    flex: 1;
}

.element-type-table tbody {
    height: 100%;
}

.element-type-table tr {
    height: 33.33%;
}

.element-type-table td {
    padding: 0 4px;
    text-align: center;
    vertical-align: middle;
    white-space: nowrap;
    font-size: 0.85em;
    line-height: 1;
    height: 33.33%;
}

.element-type-table td:first-child {
    text-align: left;
    font-weight: 600;
    color: #905F29 !important; /* 使用主题色 */
    width: 120px; /* 从170px减小到130px */
    padding-left: 0;
}

.element-tag {
    font-weight: 500;
    color: #1d1d1f;
    font-size: 0.85em;
    line-height: 1;
}

.selection-text {
    font-size: 1.2em;
    font-weight: 600;
    color: inherit;
    margin-left: 4px;
    white-space: nowrap;
}

.selection-mode-btn .v-icon {
    font-size: 16px;
}

/* 确保复选框内容居中且不影响布局 */
.mac-style-checkbox {
    display: flex;
    justify-content: center;
    margin: 0;
    padding: 0;
    min-height: 0;
    line-height: 1;
}

.mac-style-checkbox :deep(.v-selection-control) {
    margin: 0;
    padding: 0;
    min-height: 0;
    height: auto;
}

.mac-style-checkbox :deep(.v-selection-control__wrapper) {
    margin: 0;
    padding: 0;
    min-height: 0;
    height: auto;
}

.mac-style-checkbox :deep(.v-label) {
    margin: 0;
    padding: 0;
    min-height: 0;
    height: auto;
}

.lasso-cursor {
    cursor: crosshair !important;
}

.click-cursor {
    cursor: pointer !important;
}

.grab-cursor {
    cursor: grab !important;
}

.grab-cursor:active {
    cursor: grabbing !important;
}

/* 修改视觉显著性指示器样式，使其与按钮高度一致并水平布局 */
.visual-salience-indicator {
    position: absolute;
    bottom: 15px;
    left: 15px;
    font-size: 1.1em;
    color: #905F29;
    padding: 6px 12px;
    border-radius: 8px;
    background: rgba(144, 95, 41, 0.08);
    border: 1px solid rgba(144, 95, 41, 0.2);
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    max-width: 250px;
    text-align: center;
    z-index: 100;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    max-height: 40px;
}

.salience-label {
    font-size: 1.1em;
    opacity: 0.8;
    margin-right: 8px;
}

.salience-value {
    font-size: 1.1em;
    line-height: 1;
    color: #b4793a;
    white-space: nowrap;
    font-weight: 700;
}

.title {
    margin: 10px 10px 0 20px;
    font-size: 1.8em;
    font-weight: bold;
    color: #1d1d1f;
    letter-spacing: -0.01em;
    opacity: 0.8;
}

/* 添加SVG内部按钮容器样式 */
.svg-button-container {
    position: absolute;
    bottom: 15px;
    right: 15px;
    z-index: 100;
    width: auto;
    height: auto;
    display: block;
    justify-content: flex-end;
}

.mac-style-container>div.svg-button-container {
    width: auto;
    height: auto;
    position: absolute;
    display: block;
    bottom: 15px;
    right: 15px;
}

/* 添加SVG内部按钮样式 */
.submit-button {
    position: absolute !important;
    bottom: 15px !important;
    right: 15px !important;
    z-index: 100 !important;
    width: auto !important;
    margin: 0 !important;
}

.selection-mode-btn .v-icon {
    margin-right: 4px;
    font-size: 16px;
}

/* 确保复选框内容居中 */
.mac-style-checkbox {
    display: flex;
    justify-content: center;
    margin: 0;
    padding: 0;
}

.mac-style-checkbox :deep(.v-selection-control) {
    margin: 0;
    padding: 0;
    min-height: auto;
}

/* 表格相关样式修正 */
.element-type-table {
    display: flex;
    flex-direction: column;
    
}

.element-type-table table {
    table-layout: fixed;
    flex: 1;
}

.element-type-table tbody {
    height: 100%;
}


.element-type-table td {
    height: 22px;
}

.element-type-table-header{
    color: #1d1d1f;
    font-weight: 600;
}
</style>