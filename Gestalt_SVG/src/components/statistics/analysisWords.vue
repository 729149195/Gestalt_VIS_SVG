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
      <div class="section-wrapper">
        <div class="title">Over All Features</div>
        <div class="section feature-section" ref="featureSection">
          <!-- 添加阴影遮盖器 -->
          <div class="shadow-overlay top" :class="{ active: featureSectionScrollTop > 10 }"></div>
          <div class="shadow-overlay bottom" :class="{ active: isFeatureSectionScrollable && !isFeatureSectionScrolledToBottom }"></div>
          <div class="analysis-content" @scroll="handleFeatureSectionScroll" v-html="analysisContent"></div>
        </div>
      </div>
      <div class="section-wrapper">
        <div class="title">Suggests For Tagert</div>
        <div class="section middle-section" ref="middleSection">
          <!-- 添加阴影遮盖器 -->
          <div class="shadow-overlay top" :class="{ active: middleSectionScrollTop > 10 }"></div>
          <div class="shadow-overlay bottom" :class="{ active: isMiddleSectionScrollable && !isMiddleSectionScrolledToBottom }"></div>
          <div class="analysis-content" @scroll="handleMiddleSectionScroll" v-html="selectedNodesAnalysis"></div>
        </div>
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
// const MAPPING_DATA_URL = "http://127.0.0.1:5000/average_equivalent_mapping";
// const EQUIVALENT_WEIGHTS_URL = "http://127.0.0.1:5000/equivalent_weights_by_tag";
const NORMAL_DATA_URL = "http://127.0.0.1:5000/normalized_init_json";

// 特征名称映射
const featureNameMap = {
    'tag': 'shape',
    'opacity': 'opacity',
    'fill_h_cos': 'fill color',
    'fill_h_sin': 'fill color',
    'fill_s_n': 'fill color',
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

// 添加一个变量来存储原始特征数据
const rawFeatureData = ref(null);

// 添加滚动相关的状态
const featureSection = ref(null);
const middleSection = ref(null);
const featureSectionScrollTop = ref(0);
const middleSectionScrollTop = ref(0);
const isFeatureSectionScrollable = ref(false);
const isMiddleSectionScrollable = ref(false);
const isFeatureSectionScrolledToBottom = ref(true);
const isMiddleSectionScrolledToBottom = ref(true);

// 生成分析文字的函数
const generateAnalysis = (normalData, isSelectedNodes = false, selectedNodeIds = []) => {
    if (!normalData || !Array.isArray(normalData) || normalData.length === 0) {
        return isSelectedNodes ? '等待选中节点...' : '等待分析...';
    }

    // 如果是选中节点分析但没有选中节点
    if (isSelectedNodes && (!selectedNodeIds || selectedNodeIds.length === 0)) {
        return '<div class="no-selection">请选择节点查看分析...</div>';
    }

    // 获取特征数量
    const featureCount = normalData[0]?.features?.length || 0;
    if (featureCount === 0) {
        return '<div class="no-selection">找不到有效的特征数据</div>';
    }
    
    // 创建特征索引到名称的映射
    // 因为normalized_init_json.json中没有特征名称，所以我们使用featureNameMap中的索引位置映射
    const featureIndices = Object.keys(featureNameMap).map((key, index) => ({
        index,
        name: featureNameMap[key] || key
    }));
    
    // 将相同特征名称的索引分组
    const featureGroups = {};
    featureIndices.forEach(({index, name}) => {
        if (!featureGroups[name]) {
            featureGroups[name] = [];
        }
        featureGroups[name].push(index);
    });
    
    // 创建特征统计对象
    let featureStats = {};
    
    // 初始化特征统计数据
    Object.keys(featureGroups).forEach(displayName => {
        featureStats[displayName] = {
            values: [],
            selectedValues: [],
            unselectedValues: [],
            featureIndices: featureGroups[displayName],
            hasNonZeroValues: false,          // 添加标记，记录该特征是否存在非零值
            hasNonZeroSelectedValues: false,  // 添加标记，记录选中元素中该特征是否存在非零值
            hasNonZeroUnselectedValues: false // 添加标记，记录未选中元素中该特征是否存在非零值
        };
    });
    
    // 收集所有特征值 - 对于每个特征组，我们取各索引特征值的平均值
    normalData.forEach(node => {
        const isSelected = selectedNodeIds.some(id => 
            node.id === id || node.id.endsWith(`/${id}`)
        );
        
        Object.keys(featureStats).forEach(displayName => {
            const featureIndices = featureStats[displayName].featureIndices;
            
            // 只有在索引有效时才计算
            if (featureIndices.length > 0 && featureIndices[0] < node.features.length) {
                // 计算特征组的平均值
                const sum = featureIndices.reduce((acc, index) => {
                    // 确保索引有效
                    if (index < node.features.length) {
                        return acc + Math.abs(node.features[index]);
                    }
                    return acc;
                }, 0);
                
                const value = sum / featureIndices.length;
                
                // 添加到所有值数组
                featureStats[displayName].values.push(value);
                
                // 检查是否有非零值
                if (value > 0) {
                    featureStats[displayName].hasNonZeroValues = true;
                }
                
                // 根据是否选中添加到相应数组
                if (isSelected) {
                    featureStats[displayName].selectedValues.push(value);
                    // 检查选中元素中是否有非零值
                    if (value > 0) {
                        featureStats[displayName].hasNonZeroSelectedValues = true;
                    }
                } else {
                    featureStats[displayName].unselectedValues.push(value);
                    // 检查未选中元素中是否有非零值
                    if (value > 0) {
                        featureStats[displayName].hasNonZeroUnselectedValues = true;
                    }
                }
            }
        });
    });
    
    // 计算每个特征的统计数据
    Object.keys(featureStats).forEach(displayName => {
        const feature = featureStats[displayName];
        
        // 跳过没有值的特征
        if (feature.values.length === 0) {
            delete featureStats[displayName];
            return;
        }
        
        // 计算全局统计量
        const allValues = feature.values;
        const mean = allValues.reduce((sum, val) => sum + val, 0) / allValues.length;
        const variance = allValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / allValues.length;
        const stdDev = Math.sqrt(variance);
        
        feature.mean = mean;
        feature.variance = variance;
        feature.stdDev = stdDev;
        // 计算区分度，并限制最大值为1，确保不会导致超过5颗星
        feature.distinctiveness = Math.min(1, stdDev / (Math.abs(mean) + 0.0001)); // 避免除以0，并限制最大值
        
        // 添加特征类型优先级，用于相同星级时的排序
        feature.typePriority = getFeatureTypePriority(displayName);
        
        // 如果是选中节点分析，还需要计算选中和未选中之间的差异
        if (isSelectedNodes && feature.selectedValues.length > 0 && feature.unselectedValues.length > 0) {
            const selectedMean = feature.selectedValues.reduce((sum, val) => sum + val, 0) / feature.selectedValues.length;
            const unselectedMean = feature.unselectedValues.reduce((sum, val) => sum + val, 0) / feature.unselectedValues.length;
            
            feature.selectedMean = selectedMean;
            feature.unselectedMean = unselectedMean;
            
            // 计算选中和未选中之间的差异
            feature.meanDifference = selectedMean - unselectedMean;
            
            // 计算差异的显著性 - 简化版本，降低显著性阈值
            // 使用简化方法，用差异值直接作为重要程度的指标
            feature.significance = Math.min(5, Math.abs(feature.meanDifference) * 5); // 限制最大值为5
            
            // 设置一个最小显著性值，确保所有特征都有一定的显著性
            if (feature.significance < 0.5) {
                feature.significance = 0.5;
            }
        }
    });
    
    // 转换特征统计为数组，以便排序
    let featureArray = Object.entries(featureStats)
        // 不在初始阶段过滤全为0的特征，我们会在后续针对不同部分应用不同的过滤规则
        //.filter(([_, stats]) => stats.hasNonZeroValues) 
        .map(([name, stats]) => ({
            name,
            ...stats
        }));
    
    // 根据是否为选中节点分析选择不同的排序标准
    if (isSelectedNodes) {
        // 根据选中和未选中之间差异的显著性排序
        // 移除significance过滤，确保所有特征都能被显示
        //featureArray = featureArray.filter(feature => feature.significance !== undefined);
        
        // 分成正差异和负差异，为Used features部分过滤全为0的特征，但Suggest Features部分不过滤
        const positiveFeatures = featureArray
            .filter(feature => feature.meanDifference > 0 && feature.hasNonZeroSelectedValues) // Used features需要过滤全零特征
            .sort((a, b) => {
                // 首先按显著性排序
                if (Math.abs(a.significance - b.significance) > 0.001) {
                    return b.significance - a.significance;
                }
                // 显著性相同时，按特征类型优先级排序
                return b.typePriority - a.typePriority;
            })
            .slice(0, 20); // 增加到20个
            
        const negativeFeatures = featureArray
            .filter(feature => feature.meanDifference < 0) // Suggest Features不过滤全零特征
            .sort((a, b) => {
                // 首先按显著性排序
                if (Math.abs(a.significance - b.significance) > 0.001) {
                    return b.significance - a.significance;
                }
                // 显著性相同时，按特征类型优先级排序
                return b.typePriority - a.typePriority;
            })
            .slice(0, 20); // 增加到20个
            
        // 如果某一类特征太少，尝试补充显示更多的另一类特征
        if (positiveFeatures.length < 5 && negativeFeatures.length > 10) {
            // 如果正差异特征少于5个，则多显示一些负差异特征
            const moreNegativeFeatures = featureArray
                .filter(feature => feature.meanDifference < 0) // Suggest Features不过滤全零特征
                .sort((a, b) => {
                    // 首先按显著性排序
                    if (Math.abs(a.significance - b.significance) > 0.001) {
                        return b.significance - a.significance;
                    }
                    // 显著性相同时，按特征类型优先级排序
                    return b.typePriority - a.typePriority;
                })
                .slice(0, 25); // 增加到25个
                
            // 替换原有的负差异特征数组
            negativeFeatures.splice(0, negativeFeatures.length, ...moreNegativeFeatures);
        } else if (negativeFeatures.length < 5 && positiveFeatures.length > 10) {
            // 如果负差异特征少于5个，则多显示一些正差异特征
            const morePositiveFeatures = featureArray
                .filter(feature => feature.meanDifference > 0 && feature.hasNonZeroSelectedValues) // Used features需要过滤全零特征
                .sort((a, b) => {
                    // 首先按显著性排序
                    if (Math.abs(a.significance - b.significance) > 0.001) {
                        return b.significance - a.significance;
                    }
                    // 显著性相同时，按特征类型优先级排序
                    return b.typePriority - a.typePriority;
                })
                .slice(0, 25); // 增加到25个
                
            // 替换原有的正差异特征数组
            positiveFeatures.splice(0, positiveFeatures.length, ...morePositiveFeatures);
        }
        
        // 如果仍然没有足够的特征(特例情况)，则取全部特征并平分
        if (positiveFeatures.length + negativeFeatures.length < 10) {
            // 按显著性对所有特征排序，为Used features准备的特征需要过滤全零值，为Suggest Features准备的不需要
            const allPositiveFeatures = featureArray
                .filter(feature => feature.meanDifference > 0 && feature.hasNonZeroSelectedValues)
                .sort((a, b) => {
                    // 首先按显著性排序
                    if (Math.abs(a.significance - b.significance) > 0.001) {
                        return b.significance - a.significance;
                    }
                    // 显著性相同时，按特征类型优先级排序
                    return b.typePriority - a.typePriority;
                });
                
            const allNegativeFeatures = featureArray
                .filter(feature => feature.meanDifference < 0)
                .sort((a, b) => {
                    // 首先按显著性排序
                    if (Math.abs(a.significance - b.significance) > 0.001) {
                        return b.significance - a.significance;
                    }
                    // 显著性相同时，按特征类型优先级排序
                    return b.typePriority - a.typePriority;
                });
                
            // 替换原有特征数组
            positiveFeatures.splice(0, positiveFeatures.length, ...allPositiveFeatures);
            negativeFeatures.splice(0, negativeFeatures.length, ...allNegativeFeatures);
        }

        // 生成HTML
        let analysis = '<div class="feature-columns">';
        
        // 正差异特征（选中元素特有的特征）- Used Features
        analysis += '<div class="feature-column positive">';
        analysis += `<div class="column-title">Used features</div>`;
        
        if (positiveFeatures.length > 0) {
            positiveFeatures.forEach(feature => {
                // 根据显著性计算星星数量
                const filledStars = Math.min(5, Math.ceil(feature.significance));
                const emptyStars = 5 - filledStars;
                
                let starsHtml = '';
                // 添加实心星星 - 确保不超过5个星星
                for (let i = 0; i < filledStars && i < 5; i++) {
                    starsHtml += '<span class="star filled">★</span>';
                }
                // 添加空心星星
                for (let i = 0; i < emptyStars && i < (5 - filledStars); i++) {
                    starsHtml += '<span class="star empty">☆</span>';
                }
                
                analysis += `
                    <div class="feature-item">
                        <span class="feature-tag" style="color: #666666; border-color: #66666620; background-color: #66666608">
                            ${feature.name}
                        </span>
                        <span class="feature-influence" style="color: #666666">
                            ${starsHtml}
                        </span>
                    </div>
                `;
            });
        } else {
            analysis += `<div class="no-selection">没有发现显著特征</div>`;
        }
        
        analysis += '</div>';
        
        // 负差异特征（选中元素缺乏的特征）- Suggest Features
        analysis += '<div class="feature-column negative">';
        analysis += `<div class="column-title">Suggest Features</div>`;
        
        if (negativeFeatures.length > 0) {
            negativeFeatures.forEach(feature => {
                // 根据显著性计算星星数量
                const filledStars = Math.min(5, Math.ceil(feature.significance));
                const emptyStars = 5 - filledStars;
                
                let starsHtml = '';
                // 添加实心星星 - 确保不超过5个星星
                for (let i = 0; i < filledStars && i < 5; i++) {
                    starsHtml += '<span class="star filled">★</span>';
                }
                // 添加空心星星
                for (let i = 0; i < emptyStars && i < (5 - filledStars); i++) {
                    starsHtml += '<span class="star empty">☆</span>';
                }
                
                analysis += `
                    <div class="feature-item">
                        <span class="feature-tag" style="color: #666666; border-color: #66666620; background-color: #66666608">
                            ${feature.name}
                        </span>
                        <span class="feature-influence" style="color: #666666">
                            ${starsHtml}
                        </span>
                    </div>
                `;
            });
        } else {
            analysis += `<div class="no-selection">没有发现建议特征</div>`;
        }
        
        analysis += '</div>';
        analysis += '</div>';
        
        return analysis;
    } else {
        // 全局分析：根据特征的突出程度（区分度）排序
        featureArray.sort((a, b) => {
            // 首先按区分度排序
            if (Math.abs(a.distinctiveness - b.distinctiveness) > 0.001) {
                return b.distinctiveness - a.distinctiveness;
            }
            // 区分度相同时，按特征类型优先级排序
            return b.typePriority - a.typePriority;
        });
        
        // 分离最突出和最不突出的特征，Used Features过滤全零特征，Available features不过滤
        const mostDistinctive = featureArray
            .filter(feature => feature.hasNonZeroValues) // Used Features需要过滤全零特征
            .slice(0, 10); // 修改为显示10个
            
        const leastDistinctive = featureArray
            //.filter(feature => feature.hasNonZeroValues) // Available features不过滤全零特征
            .slice(-10).reverse(); // 修改为显示10个
        
        // 生成HTML
        let analysis = '<div class="feature-columns">';
        
        // 最突出的特征 - Used Features
        analysis += '<div class="feature-column negative">';
        analysis += `<div class="column-title">Used Features</div>`;
        
        if (mostDistinctive.length > 0) {
            mostDistinctive.forEach(feature => {
                // 根据区分度计算星星数量
                const filledStars = Math.min(5, Math.ceil(feature.distinctiveness * 5));
                const emptyStars = 5 - filledStars;
                
                let starsHtml = '';
                // 添加实心星星 - 确保不超过5个星星
                for (let i = 0; i < filledStars && i < 5; i++) {
                    starsHtml += '<span class="star filled">★</span>';
                }
                // 添加空心星星
                for (let i = 0; i < emptyStars && i < (5 - filledStars); i++) {
                    starsHtml += '<span class="star empty">☆</span>';
                }
                
                analysis += `
                    <div class="feature-item">
                        <span class="feature-tag" style="color: #666666; border-color: #66666620; background-color: #66666608">
                            ${feature.name}
                        </span>
                        <span class="feature-influence" style="color: #666666">
                            ${starsHtml}
                        </span>
                    </div>
                `;
            });
        } else {
            analysis += `<div class="no-selection">没有发现显著特征</div>`;
        }
        
        analysis += '</div>';
        
        // 最不突出的特征 - Available Features
        analysis += '<div class="feature-column positive">';
        analysis += `<div class="column-title">Available features</div>`;
        
        if (leastDistinctive.length > 0) {
            leastDistinctive.forEach(feature => {
                // 对于最不突出的特征，使用反向计算星星，多的星星表示更值得利用
                const reverseDistinctiveness = 1 - feature.distinctiveness;
                const filledStars = Math.min(5, Math.ceil(reverseDistinctiveness * 5));
                const emptyStars = 5 - filledStars;
                
                let starsHtml = '';
                // 添加实心星星 - 确保不超过5个星星
                for (let i = 0; i < filledStars && i < 5; i++) {
                    starsHtml += '<span class="star filled">★</span>';
                }
                // 添加空心星星
                for (let i = 0; i < emptyStars && i < (5 - filledStars); i++) {
                    starsHtml += '<span class="star empty">☆</span>';
                }
                
                analysis += `
                    <div class="feature-item">
                        <span class="feature-tag" style="color: #666666; border-color: #66666620; background-color: #66666608">
                            ${feature.name}
                        </span>
                        <span class="feature-influence" style="color: #666666">
                            ${starsHtml}
                        </span>
                    </div>
                `;
            });
        } else {
            analysis += `<div class="no-selection">没有发现可用特征</div>`;
        }
        
        analysis += '</div>';
        analysis += '</div>';
        
        return analysis;
    }
};

// 获取数据并生成分析
const fetchDataAndGenerateAnalysis = async () => {
    try {
        // 只获取正则化数据，不再需要其他两个API数据
        const response = await axios.get(NORMAL_DATA_URL);

        if (!response.data) {
            throw new Error('网络响应有问题');
        }

        // 保存原始特征数据
        rawFeatureData.value = response.data;

        // 获取选中节点的ID
        const selectedNodeIds = store.state.selectedNodes.nodeIds || [];

        // 生成全局分析文字 - 不需要传入选中节点ID
        analysisContent.value = generateAnalysis(response.data, false);
        
        // 生成选中节点的分析 - 传入选中节点ID
        if (selectedNodeIds && selectedNodeIds.length > 0) {
            selectedNodesAnalysis.value = generateAnalysis(response.data, true, selectedNodeIds);
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

// 更新滚动处理函数，分别处理两个滚动区域
const handleFeatureSectionScroll = (event) => {
    const target = event.target;
    featureSectionScrollTop.value = target.scrollTop;
    
    // 检查是否可滚动
    isFeatureSectionScrollable.value = target.scrollHeight > target.clientHeight;
    
    // 检查是否滚动到底部
    isFeatureSectionScrolledToBottom.value = Math.abs(
        target.scrollHeight - target.clientHeight - target.scrollTop
    ) < 2;
    
    // 保持原有的滚动事件传递
    emit('scroll', {
        scrollTop: target.scrollTop,
        scrollHeight: target.scrollHeight
    });
}

const handleMiddleSectionScroll = (event) => {
    const target = event.target;
    middleSectionScrollTop.value = target.scrollTop;
    
    // 检查是否可滚动
    isMiddleSectionScrollable.value = target.scrollHeight > target.clientHeight;
    
    // 检查是否滚动到底部
    isMiddleSectionScrolledToBottom.value = Math.abs(
        target.scrollHeight - target.clientHeight - target.scrollTop
    ) < 2;
    
    // 保持原有的滚动事件传递
    emit('scroll', {
        scrollTop: target.scrollTop,
        scrollHeight: target.scrollHeight
    });
}

// 初始化滚动检测
const initScrollDetection = () => {
    // 为两个区域设置初始滚动状态
    if (featureSection.value) {
        const featureContent = featureSection.value.querySelector('.analysis-content');
        if (featureContent) {
            isFeatureSectionScrollable.value = featureContent.scrollHeight > featureContent.clientHeight;
            isFeatureSectionScrolledToBottom.value = Math.abs(
                featureContent.scrollHeight - featureContent.clientHeight - featureContent.scrollTop
            ) < 2;
        }
    }
    
    if (middleSection.value) {
        const middleContent = middleSection.value.querySelector('.analysis-content');
        if (middleContent) {
            isMiddleSectionScrollable.value = middleContent.scrollHeight > middleContent.clientHeight;
            isMiddleSectionScrolledToBottom.value = Math.abs(
                middleContent.scrollHeight - middleContent.clientHeight - middleContent.scrollTop
            ) < 2;
        }
    }
}

// 兼容旧的handleScroll函数，实际不再使用
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

// 监听内容变化，重新检测滚动状态
watch([analysisContent, selectedNodesAnalysis], () => {
    // 在下一个渲染周期检测滚动状态
    setTimeout(() => {
        initScrollDetection();
    }, 0);
});

onMounted(() => {
    window.addEventListener('svg-content-updated', handleSvgUpdate);
    // 初始获取数据
    fetchDataAndGenerateAnalysis();
    
    // 初始化滚动检测
    setTimeout(() => {
        initScrollDetection();
    }, 100); // 给足够时间让内容渲染
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

// 添加获取特征类型优先级的函数
function getFeatureTypePriority(featureName) {
    // 颜色特征优先级最高
    if (featureName.includes('color')) {
        return 3;
    }
    // 位置特征优先级次之
    else if (featureName.includes('position')) {
        return 2;
    }
    // 其他特征优先级最低
    else {
        return 1;
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

/* 修改标题样式，移除绝对定位 */
.title {
  font-size: 16px;
  font-weight: bold;
  color: #1d1d1f;
  margin: 0 0 8px 0;
  padding: 0;
  letter-spacing: -0.01em;
  opacity: 0.8;
}

.sections-container {
  display: flex;
  gap: 16px;
  height: 100%;
  overflow: hidden;
}

.section-wrapper {
  display: flex;
  flex-direction: column;
  flex: 1;
}

.section {
  flex: 1;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.5);
  border: 1px solid rgba(200, 200, 200, 0.2);
  padding: 12px;
  overflow: hidden; /* 修改为hidden，防止与阴影遮盖器冲突 */
  position: relative; /* 添加相对定位，作为阴影遮盖器的参考 */
}

.feature-section {
  min-width: 0;
}

.analysis-content {
  height: 100%;
  overflow: auto;
  position: relative;
  z-index: 1; /* 确保内容在阴影之上可以滚动 */
}

/* 添加滚动阴影遮盖器样式 */
.shadow-overlay {
  position: absolute;
  left: 0;
  right: 0;
  height: 20px;
  pointer-events: none; /* 允许鼠标事件穿透到下面的内容 */
  z-index: 2;
  opacity: 0;
  transition: opacity 0.3s ease;
}

.shadow-overlay.top {
  top: 0;
  background: linear-gradient(to bottom, 
    rgba(255, 255, 255, 0.8) 0%, 
    rgba(255, 255, 255, 0) 100%);
  border-top-left-radius: 12px;
  border-top-right-radius: 12px;
}

.shadow-overlay.bottom {
  bottom: 0;
  background: linear-gradient(to top, 
    rgba(255, 255, 255, 0.8) 0%, 
    rgba(255, 255, 255, 0) 100%);
  border-bottom-left-radius: 12px;
  border-bottom-right-radius: 12px;
}

.shadow-overlay.active {
  opacity: 1;
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
    max-height: 100%;
    overflow-y: auto;
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
    position: sticky;
    top: 0;
    background: rgba(255, 255, 255, 0.9);
    z-index: 1;
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
    white-space: nowrap;
}

:deep(.star) {
    display: inline-block;
    margin: 0 1px;
}

:deep(.star.filled) {
    font-weight: bold;
}

:deep(.star.empty) {
    opacity: 0.5;
}

:deep(.feature-tag) {
    min-width: 80px;
    text-align: center;
    padding: 2px 8px;
    border-radius: 4px;
    border-width: 1px;
    border-style: solid;
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