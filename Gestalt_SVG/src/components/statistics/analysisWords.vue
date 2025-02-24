<template>
  <div class="analysis-words-container">
    <!-- 修改按钮图标为更合适的展开图标 -->
    <button class="apple-button-corner" @click="showDrawer = true">
      <div class="arrow-wrapper">
        <svg class="arrow-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M9 18l6-6-6-6"/>
        </svg>
      </div>
    </button>
    
    <div class="sections-container">
      <span class="title">Analysis and Suggestions：</span>
      <div class="section feature-section">
        <div class="analysis-content" @scroll="handleScroll" v-html="analysisContent"></div>
      </div>
      <div class="section middle-section">
        <div class="analysis-content" @scroll="handleScroll" v-html="selectedNodesAnalysis"></div>
      </div>
    </div>
    
    <!-- 使用 Teleport 将抽屉传送到 body -->
    <Teleport to="body">
      <div class="drawer-overlay" v-if="showDrawer" @click="showDrawer = false"></div>
      <div class="side-drawer" :class="{ 'drawer-open': showDrawer }">
        <button class="close-button" @click="showDrawer = false">×</button>
        <div class="drawer-body">
          <maxstic :key="componentKey" />
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import axios from 'axios'
import maxstic from '../visualization/maxstic.vue'
import { useStore } from 'vuex'

// 数据源URL
const MAPPING_DATA_URL = "http://127.0.0.1:5000/average_equivalent_mapping";
const EQUIVALENT_WEIGHTS_URL = "http://127.0.0.1:5000/equivalent_weights_by_tag";

// 特征名称映射
// const featureNameMap = {
//     'tag': '标签类型',
//     'opacity': '不透明度',
//     'fill_h_cos': '色相',
//     'fill_h_sin': '色相',
//     'fill_s_n': '饱和度',
//     'fill_l_n': '亮度',
//     'stroke_h_cos': '色相',
//     'stroke_h_sin': '色相',
//     'stroke_s_n': '饱和度',
//     'stroke_l_n': '亮度',
//     'stroke_width': '描边',
//     'bbox_left_n': '位置',
//     'bbox_right_n': '位置',
//     'bbox_top_n': '位置',
//     'bbox_bottom_n': '位置',
//     'bbox_mds_1': '位置',
//     'bbox_mds_2': '位置',
//     'bbox_center_x_n': '位置',
//     'bbox_center_y_n': '位置',
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

const props = defineProps({
  title: {
    type: String,
    default: 'analysis'
  },
  updateKey: {
    type: Number,
    default: 0
  }
})

// 添加一个计算组件key的ref
const componentKey = ref(0)

// 监听 updateKey 的变化
watch(() => props.updateKey, (newVal) => {
  componentKey.value = newVal
  // 重新获取数据
  fetchDataAndGenerateAnalysis()
})

const emit = defineEmits(['scroll'])

const analysisContent = ref('等待分析...')
const selectedNodesAnalysis = ref('等待选中节点...')

// 将 showDialog 改名为 showDrawer
const showDrawer = ref(false)

const store = useStore()

// 生成分析文字的函数
const generateAnalysis = (dataMapping, dataEquivalentWeights, isSelectedNodes = false) => {
    if (!dataMapping || !dataEquivalentWeights) return isSelectedNodes ? '等待选中节点...' : '等待分析...';

    const inputDimensions = dataMapping.input_dimensions;
    const outputDimensions = dataMapping.output_dimensions;
    const weights = dataMapping.weights;

    // 检查数据结构是否完整
    if (!Array.isArray(inputDimensions) || !Array.isArray(weights) || weights.length === 0) {
        return isSelectedNodes ? '无法分析选中节点的数据' : '数据结构不完整';
    }

    // 创建一个Map来存储每个特征的最大绝对权重
    const featureMaxWeights = new Map();

    // 遍历权重数组
    weights.forEach(dimensionWeights => {
        if (Array.isArray(dimensionWeights)) {
            dimensionWeights.forEach((weight, featureIndex) => {
                if (featureIndex < inputDimensions.length) {
                    const featureName = inputDimensions[featureIndex];
                    const displayName = featureNameMap[featureName] || featureName;
                    const absWeight = Math.abs(weight);
                    
                    // 如果当前权重更大，则更新该特征的最大权重和符号
                    if (!featureMaxWeights.has(displayName) || absWeight > featureMaxWeights.get(displayName).absWeight) {
                        featureMaxWeights.set(displayName, {
                            weight: weight,
                            absWeight: absWeight
                        });
                    }
                }
            });
        }
    });

    // 转换为数组并排序，分成正相关和负相关两组
    const features = Array.from(featureMaxWeights.entries())
        .filter(([_, {absWeight}]) => absWeight > 0.1); // 只保留权重绝对值大于0.1的特征

    const positiveFeatures = features
        .filter(([_, {weight}]) => weight > 0)
        .sort((a, b) => b[1].absWeight - a[1].absWeight);

    const negativeFeatures = features
        .filter(([_, {weight}]) => weight < 0)
        .sort((a, b) => b[1].absWeight - a[1].absWeight);

    // 如果没有找到任何特征，返回提示信息
    if (features.length === 0) {
        return '<div class="no-selection">没有找到显著的特征关系</div>';
    }

    // 生成HTML
    let analysis = '<div class="feature-columns">';
    
    // 正相关列
    analysis += '<div class="feature-column positive">';
    analysis += `<div class="column-title">${isSelectedNodes ? 'Used features' : 'Available features'}</div>`;
    
    // 存储超出显示限制的正相关特征
    const hiddenPositiveFeatures = positiveFeatures.slice(3).map(([name, {absWeight}]) => 
        `${name} (${absWeight.toFixed(2)})`
    ).join('\n');
    
    positiveFeatures.forEach(([name, {absWeight}], index) => {
        if (index >= 3) {
            if (index === 3) {
                analysis += `
                    <div class="feature-item ellipsis" title="${hiddenPositiveFeatures}">
                        <span class="feature-tag tooltip" style="color: #E53935; border-color: #E5393520; background-color: #E5393508">
                            ...
                        </span>
                    </div>
                `;
            }
            return;
        }
        
        const strengthValue = absWeight.toFixed(2);
        let strengthSymbol = '★';
        if (absWeight > 1) strengthSymbol = '★★★★';
        else if (absWeight > 0.8) strengthSymbol = '★★★';
        else if (absWeight > 0.5) strengthSymbol = '★★';
        
        analysis += `
            <div class="feature-item">
                <span class="feature-tag" style="color: #E53935; border-color: #E5393520; background-color: #E5393508">
                    ${name}
                </span>
                <span class="feature-influence" style="color: #E53935">
                    ${strengthValue} ${strengthSymbol}
                </span>
            </div>
        `;
    });
    analysis += '</div>';
    
    // 负相关列
    analysis += '<div class="feature-column negative">';
    analysis += `<div class="column-title">${isSelectedNodes ? 'Used Selected Features' : 'Used Feature'}</div>`;
    
    // 存储超出显示限制的负相关特征
    const hiddenNegativeFeatures = negativeFeatures.slice(3).map(([name, {absWeight}]) => 
        `${name} (${absWeight.toFixed(2)})`
    ).join('\n');
    
    negativeFeatures.forEach(([name, {absWeight}], index) => {
        if (index >= 3) {
            if (index === 3) {
                analysis += `
                    <div class="feature-item ellipsis" title="${hiddenNegativeFeatures}">
                        <span class="feature-tag tooltip" style="color: #1E88E5; border-color: #1E88E520; background-color: #1E88E508">
                            ...
                        </span>
                    </div>
                `;
            }
            return;
        }
        
        const strengthValue = absWeight.toFixed(2);
        let strengthSymbol = '•';
        if (absWeight > 1) strengthSymbol = '★★★';
        else if (absWeight > 0.8) strengthSymbol = '★★';
        else if (absWeight > 0.5) strengthSymbol = '★';
        
        analysis += `
            <div class="feature-item">
                <span class="feature-tag" style="color: #1E88E5; border-color: #1E88E520; background-color: #1E88E508">
                    ${name}
                </span>
                <span class="feature-influence" style="color: #1E88E5">
                    ${strengthValue} ${strengthSymbol}
                </span>
            </div>
        `;
    });
    analysis += '</div>';
    
    analysis += '</div>';
    return analysis;
};

// 获取数据并生成分析
const fetchDataAndGenerateAnalysis = async () => {
    try {
        const [responseMapping, responseEquivalentWeights] = await Promise.all([
            axios.get(MAPPING_DATA_URL),
            axios.get(EQUIVALENT_WEIGHTS_URL)
        ]);

        if (!responseMapping.data || !responseEquivalentWeights.data) {
            throw new Error('网络响应有问题');
        }

        // 生成全局分析文字
        analysisContent.value = generateAnalysis(responseMapping.data, responseEquivalentWeights.data, false);
        
        // 获取选中节点的分析数据
        const selectedNodeIds = store.state.selectedNodes.nodeIds;
        if (selectedNodeIds && selectedNodeIds.length > 0) {
            // 从equivalent_weights_by_tag中获取选中节点的权重数据
            const selectedNodesWeights = {};
            selectedNodeIds.forEach(nodeId => {
                // 在所有权重数据中查找匹配的节点ID
                const matchingKey = Object.keys(responseEquivalentWeights.data).find(key => 
                    key.endsWith(`/${nodeId}`)  // 使用endsWith来匹配节点ID
                );
                
                if (matchingKey) {
                    selectedNodesWeights[matchingKey] = responseEquivalentWeights.data[matchingKey];
                }
            });

            if (Object.keys(selectedNodesWeights).length > 0) {
                // 计算每个特征的最大绝对值权重
                const maxWeights = [];
                const dimensions = responseMapping.data.input_dimensions.length;
                
                // 初始化最大权重数组
                for (let i = 0; i < dimensions; i++) {
                    maxWeights[i] = {
                        value: 0,  // 实际权重值
                        absValue: 0  // 绝对值
                    };
                }

                // 遍历所有选中节点的权重
                Object.values(selectedNodesWeights).forEach(nodeWeights => {
                    nodeWeights.forEach(row => {
                        row.forEach((weight, index) => {
                            const absWeight = Math.abs(weight);
                            // 如果当前权重的绝对值大于已记录的最大绝对值
                            if (absWeight > maxWeights[index].absValue) {
                                maxWeights[index] = {
                                    value: weight,
                                    absValue: absWeight
                                };
                            }
                        });
                    });
                });

                // 创建选中节点的分析数据
                const selectedData = {
                    ...responseMapping.data,
                    weights: [maxWeights.map(w => w.value)]  // 使用最大权重值
                };

                selectedNodesAnalysis.value = generateAnalysis(selectedData, responseEquivalentWeights.data, true);
            } else {
                selectedNodesAnalysis.value = '<div class="no-selection">无法找到选中节点的权重数据</div>';
            }
        } else {
            selectedNodesAnalysis.value = '<div class="no-selection">请选择节点查看分析...</div>';
        }
    } catch (error) {
        console.error('获取数据失败:', error);
        analysisContent.value = '分析生成失败，请重试';
        selectedNodesAnalysis.value = '分析生成失败，请重试';
    }
};

// 修改监听逻辑，当选中节点变化时重新获取数据
watch(() => store.state.selectedNodes.nodeIds, () => {
    fetchDataAndGenerateAnalysis();
}, { deep: true, immediate: true });

const handleScroll = (event) => {
    emit('scroll', {
        scrollTop: event.target.scrollTop,
        scrollHeight: event.target.scrollHeight
    })
}

// 监听 SVG 内容更新事件
const handleSvgUpdate = () => {
    fetchDataAndGenerateAnalysis();
};

onMounted(() => {
    window.addEventListener('svg-content-updated', handleSvgUpdate);
    // 初始获取数据
    fetchDataAndGenerateAnalysis();
});

onUnmounted(() => {
    window.removeEventListener('svg-content-updated', handleSvgUpdate);
});

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
</script>

<style scoped>
.analysis-words-container {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
  border: 1px solid rgba(200, 200, 200, 0.2);
  padding: 14px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  height: 400px;
  display: flex;
  flex-direction: column;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
  position: relative;
}

/* 添加标题样式 */
.title {
  position: absolute;
  top: 12px;
  left: 16px;
  font-size: 16px;
  font-weight: bold;
  color: #1d1d1f;
  margin: 0;
  padding: 0;
  z-index: 10;
  letter-spacing: -0.01em;
  opacity: 0.8;
}

.sections-container {
  display: flex;
  gap: 16px;
  height: 100%;
  overflow: hidden;
  margin-top: 24px; /* 为标题留出空间 */
}

.section {
  flex: 1;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.5);
  border: 1px solid rgba(200, 200, 200, 0.2);
  padding: 12px;
  overflow: auto;
}

.feature-section {
  min-width: 0;
}

.middle-section, .right-section {
  background: rgba(240, 240, 240, 0.3);
}

.analysis-content {
  height: 100%;
  overflow: auto;
}

.analysis-header {
  margin-bottom: 16px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  padding-bottom: 16px;
  flex-shrink: 0;
}

:deep(.feature-tag) {
  background-color: rgba(0, 122, 255, 0.08);
  border: 1px solid rgba(0, 122, 255, 0.15);
  border-radius: 4px;
  padding: 1px 6px;
  margin: 0 1px;
  color: #007AFF;
  font-size: 13px;
  display: inline-block;
  font-weight: 500;
  letter-spacing: -0.016em;
}

:deep(.dimension-analysis) {
  margin-bottom: 10px;
  padding: 6px 10px;
  border-radius: 6px;
}

:deep(.dimension-analysis:last-child) {
  margin-bottom: 0;
}

/* 右上角按钮样式 */
.apple-button-corner {
  position: absolute;
  top: 12px;
  right: 12px;
  background: rgba(0, 122, 255, 0.08);
  border: 1px solid rgba(0, 122, 255, 0.1);
  border-radius: 8px;
  width: 36px;
  height: 36px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  color: #007AFF;
  z-index: 10;
  overflow: visible;
}

.apple-button-corner:hover {
  background: rgba(0, 122, 255, 0.12);
  transform: scale(1.05);
  box-shadow: 0 2px 8px rgba(0, 122, 255, 0.15);
}

.apple-button-corner:active {
  transform: scale(0.98);
  background: rgba(0, 122, 255, 0.15);
}

.apple-button-corner:hover{
  opacity: 1;
  transform: translateX(0);
}

/* 添加遮罩层样式 */
.drawer-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(2px);
  z-index: 998;
  animation: fadeIn 0.3s ease-out;
}

/* 抽屉基础样式 */
.side-drawer {
  position: fixed;
  top: 0;
  right: -100%;
  width: 60%;
  height: 100vh;
  background: #fff;
  z-index: 999;
  display: flex;
  flex-direction: column;
  transition: right 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: -2px 0 8px rgba(0, 0, 0, 0.15);
}

/* 抽屉打开状态 */
.side-drawer.drawer-open {
  right: 0;
}

/* 抽屉头部样式 */
.drawer-header {
  padding: 16px 24px;
  background: rgba(255, 255, 255, 0.98);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  top: 0;
  z-index: 10;
}

.drawer-header h3 {
  margin: 0;
  font-size: 18px;
  color: #1d1d1f;
  font-weight: 500;
}

/* 抽屉内容区域样式 */
.drawer-body {
  flex: 1;
  overflow: auto;
  padding: 24px;
  height: calc(100vh - 70px);
}

/* 关闭按钮样式 */
.close-button {
  background: none;
  border: none;
  font-size: 24px;
  color: #86868b;
  cursor: pointer;
  border-radius: 50%;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  position: absolute;
  top: 5px;
  right: 5px;
  z-index: 1000;
}

.close-button:hover {
  background: rgba(0, 0, 0, 0.05);
  color: #1d1d1f;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

/* 移除旧的对话框相关样式 */
.fullscreen-dialog,
.dialog-overlay,
.dialog-content {
  display: none;
}

/* 箭头容器和动画样式 */
.arrow-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.arrow-icon {
  transition: transform 0.3s ease;
}

.apple-button-corner:hover .arrow-icon {
  transform: rotate(180deg);
}

:deep(.feature-columns) {
    display: flex;
    gap: 24px;
    padding: 12px;
}

:deep(.feature-column) {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

:deep(.column-title) {
    font-size: 14px;
    font-weight: 600;
    padding: 8px;
    margin-bottom: 4px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    color: #333;
}

:deep(.feature-item) {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 8px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.6);
}

:deep(.feature-influence) {
    font-size: 13px;
    font-weight: 500;
    margin-left: auto;
}

:deep(.feature-tag) {
    min-width: 80px;
    text-align: center;
    padding: 2px 8px;
    border-radius: 4px;
    border-width: 1px;
    border-style: solid;
}

:deep(.feature-item.ellipsis) {
    opacity: 0.6;
    justify-content: center;
    cursor: help;
    position: relative;
}

:deep(.feature-item.ellipsis:hover) {
    opacity: 1;
}

:deep([title]) {
    position: relative;
}

:deep([title]:hover::before) {
    content: attr(title);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    padding: 8px 12px;
    background: rgba(0, 0, 0, 0.8);
    color: white;
    border-radius: 6px;
    font-size: 12px;
    white-space: pre-line;
    z-index: 1000;
    max-width: 300px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
}

:deep([title]:hover::after) {
    content: '';
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: rgba(0, 0, 0, 0.8);
    margin-bottom: -12px;
    z-index: 1000;
}

:deep(.no-selection) {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #86868b;
    font-size: 14px;
    font-style: italic;
}
</style>