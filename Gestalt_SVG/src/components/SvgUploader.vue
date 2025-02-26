<template>
    <v-card class="fill-height mac-style-card">
        <div class="mac-upload-zone">
            <div class="mac-upload-container" @click="triggerFileInput" @dragover.prevent @drop.prevent="handleDrop">
                <input type="file" ref="fileInput" accept=".svg" class="hidden-input" @change="handleFileChange">
                <div class="upload-content">
                    <v-icon size="32" class="upload-icon">mdi-cloud-upload-outline</v-icon>
                    <div class="upload-text">
                        <span class="primary-text">Drag and drop SVG file here</span>
                        <span class="secondary-text">Or click to select a file</span>
                    </div>
                    <div v-if="file" class="file-info">
                        <span class="file-name">{{ file.name }}</span>
                        <span class="file-size">{{ formatFileSize(file.size) }}</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- 进度条组件 - 移到元素选择列表前面 -->
        <div v-if="analyzing" class="progress-card">
            <div class="progress-label">{{ currentStep }}</div>
            <v-progress-linear :model-value="progress" color="primary" height="6" rounded :striped="false" bg-color="rgba(85, 192, 0, 0.1)">
                <template v-slot:default="{ value }">
                    <div class="progress-value">{{ Math.ceil(value) }}%</div>
                </template>
            </v-progress-linear>
        </div>

        <!-- 添加元素类型选择列表 -->
        <div v-if="visibleElements.length > 0" class="element-selector mac-style-selector" :class="{ 'collapsed': !isListExpanded }">
            <div class="selector-header" @click="toggleList">
                <h3 class="mac-style-title">Select elements:</h3>
                <v-icon class="expand-icon" :class="{ 'rotated': isListExpanded }">
                    mdi-chevron-down
                </v-icon>
            </div>
            <div class="selector-content" :class="{ 'hidden': !isListExpanded }">
                <v-list density="compact" class="mac-style-list">
                    <v-list-item v-for="element in visibleElements" :key="element.id" class="mac-style-list-item">
                        <v-checkbox v-model="selectedElements" :label="`${element.tag} (${element.count})`" :value="element.id" hide-details class="mac-style-checkbox"></v-checkbox>
                    </v-list-item>
                </v-list>
            </div>
            <div class="button-container">
                <v-btn color="primary" class="mac-style-button" @click="analyzeSvg" :disabled="selectedElements.length === 0 || analyzing">
                    {{ analyzing ? 'Simulating...' : 'Simulated perception' }}
                </v-btn>
                <v-btn @click="toggleTrackMode" class="track mac-style-track-button" :class="{ 'active-mode': isTracking }" :title="'Multiple Selection Mode'">
                    <span class="selection-text">Multi</span>
                </v-btn>
            </div>
        </div>

        <div v-if="file" class="svg-container mac-style-container" ref="svgContainer">
            <div v-html="processedSvgContent"></div>

        </div>
    </v-card>
</template>

<script setup>
import { ref, watch, nextTick, computed, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import { useStore } from 'vuex';
import * as d3 from 'd3';

const file = ref(null)
const processedSvgContent = ref('')
const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);
const allVisiableNodes = computed(() => store.state.AllVisiableNodes);
const svgContainer = ref(null);
const isTracking = ref(false);
const currentTransform = ref(null);
const nodeEventHandlers = new Map();
const visibleElements = ref([]);
const selectedElements = ref([]);

const emit = defineEmits(['file-uploaded'])

// 添加新的方法
const fileInput = ref(null);

const triggerFileInput = () => {
    fileInput.value.click();
};

const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
        file.value = selectedFile;
        uploadFile();
    }
};

const handleDrop = (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === 'image/svg+xml') {
        file.value = droppedFile;
        uploadFile();
    }
};

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
        // 设置file值，这样可以触发界面更新
        const response = await fetch('http://127.0.0.1:5000/get_svg', {
            responseType: 'text',
            headers: {
                'Accept': 'image/svg+xml'
            }
        })

        const svgContent = await response.text()

        // 创建File对象
        const blob = new Blob([svgContent], { type: 'image/svg+xml' })
        const fileObj = new File([blob], filename, { type: 'image/svg+xml' })

        // 更新file引用，这会触发界面更新
        file.value = fileObj

        // 获取并显示SVG内容
        await fetchProcessedSvg()

        // 获取可见元素列表
        const elementsResponse = await axios.post('http://127.0.0.1:5000/get_visible_elements', {
            filename: filename
        })

        if (elementsResponse.data.success) {
            visibleElements.value = elementsResponse.data.elements
            selectedElements.value = elementsResponse.data.elements.map(el => el.id)

            // 确保DOM更新后再设置交互
            await nextTick()
            setupSvgInteractions()
            updateNodeOpacity()
        }
    } catch (error) {
        console.error('Error handling upload event:', error)
    }
}

const uploadFile = () => {
    if (!file.value) return
    const formData = new FormData()

    // 创建新的File对象，添加uploaded_前缀
    const newFile = new File([file.value], `uploaded_${file.value.name}`, { type: file.value.type })
    formData.append('file', newFile)

    // 清除选中的节点
    clearSelectedNodes();

    // 先上传文件
    axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })
        .then(response => {
            if (response.data.success) {
                // 触发事件通知CodeToSvg组件
                window.dispatchEvent(new CustomEvent('svg-content-updated', {
                    detail: { filename: newFile.name }
                }))
                // 立即获取并显示SVG内容
                return fetchProcessedSvg()
            }
        })
        .then(() => {
            // 获取可见元素列表
            return axios.post('http://127.0.0.1:5000/get_visible_elements', {
                filename: newFile.name
            })
        })
        .then(async response => {
            if (response.data.success) {
                visibleElements.value = response.data.elements;
                selectedElements.value = response.data.elements.map(el => el.id);

                // 确保DOM更新后再设置交互
                await nextTick();
                setupSvgInteractions();
                updateNodeOpacity();

                // 不再在这里触发file-uploaded事件
                // 而是等待用户点击分析按钮后触发
            }
        })
        .catch(error => {
            console.error('Error in upload process:', error)
        })
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
                return fetchProcessedSvg();
            } else {
                throw new Error(response.data.error || '分析失败');
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
            console.error('分析过程出错:', error);
        })
        .finally(() => {
            analyzing.value = false;
            eventSource.close();
        });
}

const fetchProcessedSvg = () => {
    // 清除选中的节点
    clearSelectedNodes();

    return axios.get('http://127.0.0.1:5000/get_svg', {
        responseType: 'text',
        headers: {
            'Accept': 'image/svg+xml'
        }
    })
        .then(svgResponse => {
            let svgContent = svgResponse.data;

            // 确保SVG内容是有效的
            if (!svgContent.includes('<svg')) {
                throw new Error('Invalid SVG content received');
            }

            // 处理SVG内容
            const parser = new DOMParser();
            const svgDoc = parser.parseFromString(svgContent, 'image/svg+xml');
            const svgElement = svgDoc.querySelector('svg');

            if (!svgElement) {
                throw new Error('No SVG element found in response');
            }

            // 从原始SVG中提取viewBox
            const viewBoxMatch = svgContent.match(/viewBox="([^"]*)"/) || [];
            const viewBox = viewBoxMatch[1] || '0 0 800 600';

            // 设置SVG的响应式属性
            svgElement.setAttribute('width', '100%');
            svgElement.setAttribute('height', '100%');
            svgElement.setAttribute('viewBox', viewBox);
            svgElement.setAttribute('preserveAspectRatio', 'xMidYMid meet');

            // 确保有一个包装组用于缩放
            let wrapper = svgElement.querySelector('g.zoom-wrapper');
            if (!wrapper) {
                wrapper = document.createElementNS("http://www.w3.org/2000/svg", "g");
                wrapper.setAttribute('class', 'zoom-wrapper');
                while (svgElement.firstChild) {
                    wrapper.appendChild(svgElement.firstChild);
                }
                svgElement.appendChild(wrapper);
            }

            processedSvgContent.value = svgElement.outerHTML;

            return nextTick(() => {
                setupSvgInteractions();
                addZoomEffectToSvg();
            });
        })
        .catch(error => {
            console.error('Error fetching SVG:', error);
            throw error;
        });
};

// 添加缩放和拖拽功能
const addZoomEffectToSvg = () => {
    const container = svgContainer.value;
    if (!container) return;
    const svg = d3.select(container).select('svg');
    if (!svg.empty()) {
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

        const zoom = d3.zoom()
            .scaleExtent([0.5, 10])
            .on('zoom', (event) => {
                if (!isTracking.value) {
                    g.attr('transform', event.transform);
                }
            });

        svg.call(zoom);

        // 设置初始缩放为0.8（80%的原始大小）并向右平移10%
        const width = svg.node().getBoundingClientRect().width;
        const translateX = width * 0.05; // 向右平移10%
        svg.call(zoom.transform, d3.zoomIdentity.translate(translateX, 10).scale(0.8));
    }
};

// 路径选择功能
const toggleTrackMode = () => {
    isTracking.value = !isTracking.value;
    const svg = d3.select(svgContainer.value).select('svg');

    if (isTracking.value) {
        nextTick(() => {
            svgContainer.value.classList.add('copy-cursor');
        });
        enableTrackMode();

        const transform = d3.zoomTransform(svg.node());
        currentTransform.value = transform;
        svg.on('.zoom', null);
    } else {
        svgContainer.value.classList.remove('copy-cursor');
        disableTrackMode();

        const zoom = d3.zoom()
            .scaleExtent([0.5, 10])
            .on('zoom', (event) => {
                if (!isTracking.value) {
                    svg.select('g.zoom-wrapper').attr('transform', event.transform);
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
    const svg = svgContainer.value.querySelector('svg');

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
    const svg = svgContainer.value.querySelector('svg');
    if (svg) {
        const handlers = nodeEventHandlers.get(svg);
        if (handlers) {
            svg.removeEventListener('mousedown', handlers.handleMouseDown);
            svg.removeEventListener('mouseup', handlers.handleMouseUp);
            svg.removeEventListener('mousemove', handlers.handleMouseMove);
        }
    }
};

// 修改setupSvgInteractions函数
const setupSvgInteractions = () => {
    const svgContainer = document.querySelector('.svg-container svg');
    if (!svgContainer) {
        console.warn('SVG container not found');
        return;
    }

    // 移除现有的事件监听器
    const oldClickHandler = svgContainer._clickHandler;
    if (oldClickHandler) {
        svgContainer.removeEventListener('click', oldClickHandler);
    }

    // 保存新的事件处理器引用
    svgContainer._clickHandler = handleSvgClick;

    // 添加新的事件监听器
    svgContainer.addEventListener('click', svgContainer._clickHandler);

    // 更新节点透明度
    updateNodeOpacity();

    // 添加缩放效果
    addZoomEffectToSvg();
};

// 更新节点透明度
const updateNodeOpacity = () => {
    const svgContainer = document.querySelector('.svg-container');
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
            let opacity = selectedElements.value.includes(nodeType) ? 1 : 0;

            // 如果有选中的节点，无论是否路径选择模式下，都使用相同的选中逻辑
            if (opacity === 1 && selectedNodeIds.value.length > 0) {
                opacity = selectedNodeIds.value.includes(nodeId) ? 1 : 0.3;
            }

            node.style.opacity = opacity;
            node.style.transition = 'opacity 0.3s ease';
        });
    } catch (error) {
        console.error('更新节点透明度时出错:', error);
    }
};

// 点击 SVG 节点的处理函数
const handleSvgClick = (event) => {
    // 检查点击的是否是 SVG 容器本身或者 zoom-wrapper
    const target = event.target;
    if (target.tagName.toLowerCase() === 'svg' ||
        (target.tagName.toLowerCase() === 'g' && target.classList.contains('zoom-wrapper'))) {
        // 只有在非多选模式下才清空所有选中的节点
        if (!isTracking.value) {
            store.dispatch('clearSelectedNodes');
        }
        return;
    }

    // 如果点击的是具体的 SVG 元素，则执行原有的选中逻辑
    const nodeId = target.id;
    if (!nodeId) return;

    if (selectedNodeIds.value.includes(nodeId)) {
        store.commit('REMOVE_SELECTED_NODE', nodeId);
    } else {
        store.commit('ADD_SELECTED_NODE', nodeId);
    }

    // 使用 nextTick 确保状态更新后再更新视图
    nextTick(() => {
        updateNodeOpacity();
    });
};

// 监听选中节点的变化
watch(selectedNodeIds, () => {
    nextTick(() => {
        updateNodeOpacity();
    });
});

// 监听selectedElements的变化
watch(selectedElements, () => {
    nextTick(() => {
        updateNodeOpacity();
    });
});

const isListExpanded = ref(false);

const toggleList = () => {
    isListExpanded.value = !isListExpanded.value;
};

// 监听文件变化，当有新文件时自动展开列表
watch(() => file.value, (newFile) => {
    if (newFile) {
        isListExpanded.value = true;
    }
});

// 监听分析状态，当开始分析时自动收起列表
watch(() => analyzing.value, (newValue) => {
    if (newValue) {
        isListExpanded.value = false;
    }
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
    border-color: #55C000;
    box-shadow: 0 0 0 2px rgba(85, 192, 0, 0.2);
}

.mac-style-selector {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(200, 200, 200, 0.3);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    max-width: 400px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.mac-style-selector.collapsed {
    max-height: 120px;
}

.selector-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    user-select: none;
}

.expand-icon {
    transition: transform 0.3s ease;
    opacity: 0.6;
}

.expand-icon.rotated {
    transform: rotate(-180deg);
}

.selector-content {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    opacity: 1;
    max-height: 200px;
    overflow-y: auto;
    margin: 12px 0;
}

.selector-content.hidden {
    opacity: 0;
    max-height: 0;
    margin: 0;
    overflow: hidden;
}

.mac-style-title {
    font-size: 16px;
    font-weight: 500;
    color: #1d1d1f;
    margin-bottom: 12px;
}

.mac-style-list {
    border-radius: 8px;
    background: rgba(250, 250, 250, 0.6);
    border: 1px solid rgba(200, 200, 200, 0.2);
    overflow: hidden;
}

.mac-style-list-item {
    transition: background-color 0.2s ease;
}

.mac-style-list-item:hover {
    background-color: rgba(85, 192, 0, 0.05);
}

.mac-style-checkbox :deep(.v-selection-control) {
    color: #55C000;
}

.mac-style-button {
    background: #55C000 !important;
    border-radius: 8px;
    color: white;
    font-weight: 500;
    letter-spacing: 0.3px;
    box-shadow: 0 2px 8px rgba(85, 192, 0, 0.2);
    transition: all 0.3s ease;
    text-transform: none;
    height: 36px;
    flex: 1;
}

.mac-style-button:hover {
    background: #4CAF00 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(85, 192, 0, 0.3);
}

.mac-style-button:disabled {
    background: rgba(85, 192, 0, 0.5) !important;
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
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.9) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(200, 200, 200, 0.3);
    transition: all 0.3s ease;
    height: 36px;
    padding: 0 10px;
    min-width: 70px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
}

.selection-text {
    font-size: 13px;
}

.mac-style-track-button:hover {
    background: rgba(250, 250, 250, 0.95) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.mac-style-track-button.active-mode {
    background: #55C000 !important;
    border-color: #55C000;
    color: white;
}

.copy-cursor {
    cursor: copy !important;
}

.element-selector {
    position: absolute;
    left: 16px;
    bottom: 16px;
    z-index: 90;
    max-height: calc(100vh - 280px);
    /* 调整最大高度，留出进度条的空间 */
    overflow-y: auto;
}

/* SVG 相关样式 */
.svg-container svg {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

.svg-container svg * {
    cursor: pointer;
}

.mac-upload-zone {
    flex: 0 0 auto;
    position: relative;
    margin-top: 16px;
    margin-left: 16px;
    margin-right: 16px;
    z-index: 10;
}

.mac-upload-container {
    background: rgba(255, 255, 255, 0.95);
    border: 1px dashed rgba(85, 192, 0, 0.3);
    border-radius: 8px;
    padding: 8px 12px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.mac-upload-container:hover {
    border-color: rgba(85, 192, 0, 0.6);
    background: rgba(255, 255, 255, 0.98);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.hidden-input {
    display: none;
}

.upload-content {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
}

.upload-icon {
    color: #55C000;
    opacity: 0.8;
    transition: all 0.3s ease;
    font-size: 30px !important;
}

.upload-text {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0px;
}

.primary-text {
    font-size: 16px;
    font-weight: 500;
    color: #1d1d1f;
}

.secondary-text {
    font-size: 14px;
    color: #86868b;
}

.file-info {
    margin-left: auto;
    padding: 4px 8px;
    background: rgba(85, 192, 0, 0.1);
    border-radius: 4px;
    display: flex;
    gap: 6px;
    align-items: center;
}

.file-name {
    font-weight: 500;
    color: #1d1d1f;
    font-size: 12px;
}

.file-size {
    color: #86868b;
    font-size: 11px;
}

.progress-card {
    position: absolute;
    top: 90px;
    /* 调整位置，确保在上传区域下方 */
    left: 16px;
    right: 16px;
    z-index: 100;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(200, 200, 200, 0.3);
    padding: 12px 16px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
}

.progress-label {
    font-size: 13px;
    font-weight: 500;
    color: #1d1d1f;
    margin-bottom: 8px;
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
    background: linear-gradient(90deg, #55C000, #4CAF00);
    box-shadow: 0 1px 3px rgba(85, 192, 0, 0.2);
    transition: all 0.3s ease;
}

.button-container {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-top: 16px;
}
</style>