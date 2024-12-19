<template>
    <v-card class="fill-height">
        <h2 class="title">
            Parse SVG by Gestalt
        </h2>
        <v-file-input v-model="file" prepend-icon="mdi-paperclip" label="Select Svg File" density="compact"
            accept=".svg" show-size @change="uploadFile"></v-file-input>

        <!-- 添加元素类型选择列表 -->
        <div v-if="visibleElements.length > 0" class="element-selector">
            <h3>选择要分析的元素类型：</h3>
            <v-list density="compact">
                <v-list-item v-for="element in visibleElements" :key="element.id">
                    <v-checkbox v-model="selectedElements" :label="`${element.tag} (${element.count}个)`"
                        :value="element.id" hide-details></v-checkbox>
                </v-list-item>
            </v-list>
            <v-btn color="primary" class="mt-4" @click="analyzeSvg" :disabled="selectedElements.length === 0">
                分析选中元素类型
            </v-btn>
        </div>

        <div v-if="file" class="svg-container" ref="svgContainer">
            <div v-html="processedSvgContent"></div>
            <v-btn @click="toggleTrackMode" class="track" :class="{ 'active-mode': isTracking }">
                <v-icon>mdi-cursor-pointer</v-icon>
            </v-btn>
        </div>
     
    </v-card>
</template>

<script setup>
import { ref, watch, nextTick, computed } from 'vue'
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

const uploadFile = () => {
    if (!file.value) return

    console.log('开始上传文件:', file.value.name)
    const formData = new FormData()
    formData.append('file', file.value)

    // 先上传文件
    axios.post('http://localhost:5000/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })
        .then(response => {
            console.log('文件上传成功:', response.data)
            if (response.data.success) {
                // 立即获取并显示SVG内容
                return fetchProcessedSvg()
            }
        })
        .then(() => {
            // 获取可见元素列表
            return axios.post('http://localhost:5000/get_visible_elements', {
                filename: file.value.name
            })
        })
        .then(response => {
            if (response.data.success) {
                visibleElements.value = response.data.elements;
                selectedElements.value = response.data.elements.map(el => el.id);
                // 确保DOM更新后再设置交互
                return nextTick(() => {
                    setupSvgInteractions();
                    updateNodeOpacity();
                });
            }
        })
        .catch(error => {
            console.error('Error in upload process:', error)
        })
}

const analyzeSvg = () => {
    if (!file.value) return;

    // 禁用分析按钮
    const analyzing = ref(true);

    // 确保 selectedNodeIds 是数组格式
    const nodeIds = Array.isArray(selectedNodeIds.value) ? selectedNodeIds.value : [];
    
    console.log('选中的节点ID:', nodeIds); // 添加调试日志

    axios.post('http://localhost:5000/filter_and_process', {
        filename: file.value.name,
        selectedElements: selectedElements.value,
        selectedNodeIds: nodeIds
    })
        .then(response => {
            if (response.data.success) {
                console.log('分析成功，开始获取处理后的SVG');
                return fetchProcessedSvg();
            } else {
                throw new Error(response.data.error || '分析失败');
            }
        })
        .then(() => {
            console.log('SVG更新完成');
            emit('file-uploaded');
        })
        .catch(error => {
            console.error('分析过程出错:', error);
        })
        .finally(() => {
            analyzing.value = false;
        });
}

const fetchProcessedSvg = () => {
    console.log('开始获取SVG内容')
    return axios.get('http://localhost:5000/get_svg', {
        responseType: 'text',
        headers: {
            'Accept': 'image/svg+xml'
        }
    })
        .then(svgResponse => {
            console.log('获取到SVG响应')
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
            console.log('SVG内容已更新');

            return nextTick(() => {
                console.log('设置SVG交互');
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

    const handleMouseUp = () => {
        isMouseDown = false;
    };

    const handleMouseMove = (event) => {
        if (isMouseDown) {
            const point = svg.createSVGPoint();
            point.x = event.clientX;
            point.y = event.clientY;
            const svgPoint = point.matrixTransform(svg.getScreenCTM().inverse());

            const node = document.elementFromPoint(event.clientX, event.clientY);
            if (node && allVisiableNodes.value.includes(node.id) && !clickedElements.has(node)) {
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
    svgContainer.removeEventListener('click', handleSvgClick);

    // 添加新的事件监听器
    svgContainer.addEventListener('click', handleSvgClick);

    // 更新节点透明度
    updateNodeOpacity();

    // 添加缩放效果
    addZoomEffectToSvg();

    console.log('SVG交互设置完成');
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

            // 如果有选中的节点，无论是否在路径选择模式下，都使用相同的选中逻辑
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
    const nodeId = event.target.id;
    if (!nodeId || !allVisiableNodes.value.includes(nodeId)) return;

    console.log('点击节点:', nodeId);

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

</script>

<style scoped>
.title {
    font-size: 1.5rem;
    font-weight: bold;
    letter-spacing: 2px;
    text-align: left;
    color: black;
    margin-left: 8px;
    margin-bottom: 8px
}

.fill-height {
    height: 100%;
    width: 100%;
    padding: 8px;
    overflow: hidden;
}

.svg-container {
    width: 100%;
    height: calc(100% - 70px);
    margin: auto;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}

.svg-container>div {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.svg-container svg {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.svg-container svg * {
    cursor: pointer;
}

.track {
    position: absolute;
    top: 10px;
    right: 10px;
}

.copy-cursor {
    cursor: copy !important;
}

.active-mode {
    background-color: var(--v-theme-primary) !important;
}

.element-selector {
    overflow-y: auto;
    margin: 16px 0;
    position: absolute;
    left: 30px;
    bottom: 10px;
    z-index: 1000;
}
</style>