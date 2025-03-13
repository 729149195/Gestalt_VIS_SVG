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
      
      <!-- 保持大标题在顶部 -->
      <div class="title">Assessment of Visual effects</div>
      
      <div class="sections-container">
        <div class="section-wrapper">
          <!-- 将侧边标题放在section外部，添加旋转类 -->
          <div class="section-header">
            <div class="aside-label">All elements</div>
          </div>
          <div class="section feature-section" ref="featureSection">
            <!-- 添加阴影遮盖器 -->
            <div class="shadow-overlay top" :class="{ active: featureSectionScrollTop > 10 }"></div>
            <div class="shadow-overlay bottom" :class="{ active: isFeatureSectionScrollable && !isFeatureSectionScrolledToBottom }"></div>
            <div class="analysis-content" @scroll="handleFeatureSectionScroll" v-html="analysisContent"></div>
          </div>
        </div>
        <div class="section-wrapper">
          <!-- 将侧边标题放在section外部，添加旋转类 -->
          <div class="section-header">
            <div class="aside-label">Selected elements</div>
          </div>
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
  import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
  import axios from 'axios'
  import maxstic from '../visualization/maxstic.vue'
  import { useStore } from 'vuex'
  
  const NORMAL_DATA_URL = "http://127.0.0.1:5000/normalized_init_json";
  
  // 特征名称映射
  const featureNameMap = {
      'tag': 'shape',
      'opacity': 'opacity',
      'fill_h_cos': 'fill h',
      'fill_h_sin': 'fill h',
      'fill_s_n': 'fill s',
      'fill_l_n': 'fill l',
      'stroke_h_cos': 'stroke h',
      'stroke_h_sin': 'stroke h',
      'stroke_s_n': 'stroke s',
      'stroke_l_n': 'stroke l',
      'stroke_width': 'stroke width',
      'bbox_left_n': 'left',
      'bbox_right_n': 'right',
      'bbox_top_n': 'top',
      'bbox_bottom_n': 'bottom',
      'bbox_mds_1': 'mds1',
      'bbox_mds_2': 'mds2',
      'bbox_width_n': 'width',
      'bbox_height_n': 'height',
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
  
  const analysisContent = ref('Waiting for analysis...')
  const selectedNodesAnalysis = ref('Waiting for selected nodes...')
  
  // 将 showDialog 改名为 showDrawer
  const showDrawer = ref(false)
  
  const store = useStore()
  
  // 添加一个变量来获取高亮元素的visual salience值
  const visualSalienceValue = computed(() => {
    // 从store中获取visualSalience值，如果不存在则默认为100
    return store.state.visualSalience ? store.state.visualSalience * 100 : 100;
  })
  
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
          return isSelectedNodes ? 'Waiting for selected nodes...' : 'Waiting for analysis...';
      }
  
      // 如果是选中节点分析但没有选中节点
      if (isSelectedNodes && (!selectedNodeIds || selectedNodeIds.length === 0)) {
          return '<div class="no-selection">Please select a node to view the analysis...</div>';
      }
  
      // 获取特征数量
      const featureCount = normalData[0]?.features?.length || 0;
      if (featureCount === 0) {
          return '<div class="no-selection">Can not find valid feature data</div>';
      }
      
      // 创建特征索引到名称的映射
      // 因为normalized_init_json.json中没有特征名称，所以我们使用featureNameMap中的索引位置映射
      const featureIndices = Object.keys(featureNameMap).map((key, index) => ({
          index,
          name: featureNameMap[key] || key,
          key: key // 保存原始key
      }));
      
      // 将相同特征名称的索引分组
      const featureGroups = {};
      featureIndices.forEach(({index, name, key}) => {
          if (!featureGroups[name]) {
              featureGroups[name] = {
                  indices: [],
                  keys: []
              };
          }
          featureGroups[name].indices.push(index);
          featureGroups[name].keys.push(key);
      });
      
      // 创建特征统计对象
      let featureStats = {};
      
      // 初始化特征统计数据
      Object.keys(featureGroups).forEach(displayName => {
          featureStats[displayName] = {
              values: [],
              selectedValues: [],
              unselectedValues: [],
              featureIndices: featureGroups[displayName].indices,
              featureKeys: featureGroups[displayName].keys,
              hasNonZeroValues: false,          // 添加标记，记录该特征是否存在非零值
              hasNonZeroSelectedValues: false,  // 添加标记，记录选中元素中该特征是否存在非零值
              hasNonZeroUnselectedValues: false, // 添加标记，记录未选中元素中该特征是否存在非零值
              variance: 0,                      // 添加方差，用于衡量多样性
              meanDifference: 0                 // 添加均值差异，用于衡量选中与未选中的差异
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
              feature.meanDifference = Math.abs(selectedMean - unselectedMean);
              
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
          .map(([name, stats]) => ({
              name,
              ...stats
          }));
      
      // 根据是否为选中节点分析选择不同的排序标准
      if (isSelectedNodes) {
          // 筛选出高亮元素相对于非高亮元素有差异的特征
          // 只保留有差异的特征（meanDifference > 0）且选中元素中有非零值的特征
          const significantFeatures = featureArray
              .filter(feature => 
                  feature.meanDifference > 0.001 && // 确保有明显差异
                  feature.hasNonZeroSelectedValues // 确保选中元素中有非零值
              )
              .sort((a, b) => {
                  // 按差异大小排序
                  return b.meanDifference - a.meanDifference;
              })
              .slice(0, 20); // 最多显示20个
          
          // 筛选出可能的改进特征，这些特征在未选中元素中有非零值，但在选中元素中没有或很少
          const negativeFeatures = featureArray
              .filter(feature => 
                  feature.hasNonZeroUnselectedValues && // 未选中元素中有非零值
                  (!feature.hasNonZeroSelectedValues || // 选中元素中没有非零值
                   (feature.selectedMean < feature.unselectedMean)) // 或者选中元素的均值小于未选中元素
              )
              .sort((a, b) => {
                  // 按差异大小排序
                  if (a.hasNonZeroSelectedValues && b.hasNonZeroSelectedValues) {
                      return (b.unselectedMean - b.selectedMean) - (a.unselectedMean - a.selectedMean);
                  } else if (a.hasNonZeroSelectedValues) {
                      return 1; // a 有非零值，排后面
                  } else if (b.hasNonZeroSelectedValues) {
                      return -1; // b 有非零值，排后面
                  } else {
                      return b.unselectedMean - a.unselectedMean; // 都没有非零值，按未选中均值排序
                  }
              })
              .slice(0, 10); // 最多显示10个
          
          // 生成HTML
          let analysis = '<div class="feature-columns">';
          
          // 正差异特征（选中元素特有的特征）- Used Features
          analysis += '<div class="feature-column positive">';
          analysis += `<div class="column-title all-elements-title">Used visual effects</div>`;
          
          if (significantFeatures.length > 0) {
              // 创建一个包装容器用于两列布局
              analysis += `<div class="two-column-wrapper">`;
              
              // 将特征分成两组，以便两列显示
              for (let i = 0; i < significantFeatures.length; i += 2) {
                  analysis += `<div class="two-column-row">`;
                  
                  // 添加第一个特征
                  analysis += `
                      <div class="feature-item two-column-item">
                          <span class="feature-tag all-elements-tag" style="color: #555555; border-color: #55555530; background-color: #f5f5f5">
                              ${significantFeatures[i].name}
                          </span>
                      </div>
                  `;
                  
                  // 如果有第二个特征，也添加它
                  if (i + 1 < significantFeatures.length) {
                      analysis += `
                          <div class="feature-item two-column-item">
                              <span class="feature-tag all-elements-tag" style="color: #555555; border-color: #55555530; background-color: #f5f5f5">
                                  ${significantFeatures[i + 1].name}
                              </span>
                          </div>
                      `;
                  }
                  
                  analysis += `</div>`;
              }
              
              analysis += `</div>`;
          } else {
              analysis += `<div class="no-selection">No visual effects found</div>`;
          }
          
          analysis += '</div>';
          
          // 负差异特征（选中元素缺乏的特征）- Suggest Features
          analysis += '<div class="feature-column negative">';
          analysis += `<div class="column-title">Suggestions for improvement</div>`;
          
          // 检查visual salience值是否小于70
          if (visualSalienceValue.value < 85) {
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
                  analysis += `<div class="no-selection">No recommended features found</div>`;
              }
          } else {
              // 当visual salience值大于等于70时显示的信息，使用更紧凑的样式
              analysis += `
                  <div class="high-salience-notice">
                      <div class="salience-icon">✓</div>
                      <div class="salience-content">
                          <div class="salience-title">Visual salience is already good</div>
                          <div class="salience-value">${visualSalienceValue.value.toFixed(1)}%</div>
                      </div>
                  </div>
              `;
          }
          
          analysis += '</div>';
          analysis += '</div>';
          
          return analysis;
      } else {
          // 全局分析：根据特征的多样性（方差）排序
          // 筛选出有变化的特征（方差大于0）
          const diverseFeatures = featureArray
              .filter(feature => 
                  feature.variance > 0.001 && // 确保有明显变化
                  feature.hasNonZeroValues // 确保有非零值
              )
              .sort((a, b) => {
                  // 按方差大小排序
                  return b.variance - a.variance;
              })
              .slice(0, 20); // 最多显示20个
              
          // 筛选出方差较小的特征，作为可用但未充分利用的特征
          const leastDistinctive = featureArray
              .filter(feature => 
                  feature.variance <= 0.001 || // 方差小，变化不明显
                  !feature.hasNonZeroValues // 或者没有非零值
              )
              .sort((a, b) => {
                  // 按方差从小到大排序
                  return a.variance - b.variance;
              })
              .slice(0, 10); // 最多显示10个
          
          // 生成HTML
          let analysis = '<div class="feature-columns">';
          
          // 最突出的特征 - Used Features
          analysis += '<div class="feature-column negative">';
          analysis += `<div class="column-title all-elements-title">Used visual effects</div>`;
          
          if (diverseFeatures.length > 0) {
              // 创建一个包装容器用于两列布局
              analysis += `<div class="two-column-wrapper">`;
              
              // 将特征分成两组，以便两列显示
              for (let i = 0; i < diverseFeatures.length; i += 2) {
                  analysis += `<div class="two-column-row">`;
                  
                  // 添加第一个特征
                  analysis += `
                      <div class="feature-item two-column-item">
                          <span class="feature-tag all-elements-tag" style="color: #555555; border-color: #55555530; background-color: #f5f5f5">
                              ${diverseFeatures[i].name}
                          </span>
                      </div>
                  `;
                  
                  // 如果有第二个特征，也添加它
                  if (i + 1 < diverseFeatures.length) {
                      analysis += `
                          <div class="feature-item two-column-item">
                              <span class="feature-tag all-elements-tag" style="color: #555555; border-color: #55555530; background-color: #f5f5f5">
                                  ${diverseFeatures[i + 1].name}
                              </span>
                          </div>
                      `;
                  }
                  
                  analysis += `</div>`;
              }
              
              analysis += `</div>`;
          } else {
              analysis += `<div class="no-selection">No distinguishing features found</div>`;
          }
          
          analysis += '</div>';
          
          // 最不突出的特征 - Available Features
          analysis += '<div class="feature-column positive">';
          analysis += `<div class="column-title">Available visual effects <span class="rank-tag">rank</span></div>`;
          
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
              analysis += `<div class="no-selection">No usable features found</div>`;
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
              throw new Error('Problems with network response');
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
              selectedNodesAnalysis.value = '<div class="no-selection">Please select a node to view the analysis...</div>';
          }
      } catch (error) {
          console.error('Failed to get data:', error);
          analysisContent.value = 'Analysis generation failed, please try again';
          selectedNodesAnalysis.value = 'Analysis generation failed, please try again';
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
    font-size: 1.8em;
    font-weight: bold;
    color: #1d1d1f;
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
    position: relative; /* 添加相对定位 */
  }
  
  /* 新增：section-header样式 */
  .section-header {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 30px;
    z-index: 10;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f5f5f7;
    border-right: 1px solid rgba(200, 200, 200, 0.5);
    border-top-left-radius: 12px;
    border-bottom-left-radius: 12px;
  }
  
  /* 新增：aside-label样式 */
  .aside-label {
    transform: rotate(-90deg);
    white-space: nowrap;
    font-size: 1.4em;
    font-weight: 700;
    color: #333;
    letter-spacing: 0.04em;
    width: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .section {
    flex: 1;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.5);
    border: 1px solid rgba(200, 200, 200, 0.2);
    padding: 12px 12px 12px 40px; /* 增加左侧padding，为旋转的侧边栏留出空间 */
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
    left: 30px; /* 修改左侧位置，避免覆盖侧边栏 */
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
    border-top-right-radius: 12px;
  }
  
  .shadow-overlay.bottom {
    bottom: 0;
    background: linear-gradient(to top, 
      rgba(255, 255, 255, 0.8) 0%, 
      rgba(255, 255, 255, 0) 100%);
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
      font-size: 16px;
      font-weight: 600;
      padding: 8px;
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
  
  /* 为 All elements 部分的 Used visual effects 添加特殊的 feature-item 样式 */
  :deep(.feature-column.negative .feature-item),
  :deep(.feature-column.positive .feature-item) {
      padding: 2px 4px;
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
  
  /* 添加自定义滚动条样式 */
  .analysis-content::-webkit-scrollbar,
  :deep(.feature-columns::-webkit-scrollbar) {
      width: 6px;
      height: 6px;
  }
  
  .analysis-content::-webkit-scrollbar-track,
  :deep(.feature-columns::-webkit-scrollbar-track) {
      background: rgba(0, 0, 0, 0.03);
      border-radius: 3px;
  }
  
  .analysis-content::-webkit-scrollbar-thumb,
  :deep(.feature-columns::-webkit-scrollbar-thumb) {
      background: rgba(0, 0, 0, 0.15);
      border-radius: 3px;
      transition: background 0.3s;
  }
  
  .analysis-content::-webkit-scrollbar-thumb:hover,
  :deep(.feature-columns::-webkit-scrollbar-thumb:hover) {
      background: rgba(0, 0, 0, 0.25);
  }
  
  .section:hover .analysis-content::-webkit-scrollbar-thumb {
      background: rgba(0, 0, 0, 0.2);
  }
  
  :deep(.feature-rank) {
      font-size: 13px;
      font-weight: 500;
      margin-left: auto;
      white-space: nowrap;
  }
  
  :deep(.rank-tag) {
      background-color: rgba(102, 102, 102, 0.08);
      border: 1px solid rgba(102, 102, 102, 0.15);
      border-radius: 4px;
      padding: 1px 6px;
      margin-left: 8px;
      color: #666666;
      font-size: 12px;
      display: inline-block;
      font-weight: 500;
  }
  
  /* 添加高显著性提示的样式 */
  :deep(.high-salience-notice) {
      display: flex;
      align-items: center;
      background: linear-gradient(135deg, rgba(76, 175, 80, 0.08) 0%, rgba(76, 175, 80, 0.15) 100%);
      border: 1px solid rgba(76, 175, 80, 0.2);
      border-radius: 12px;
      padding: 16px;
      margin: 12px 0;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
      transition: all 0.3s ease;
  }
  
  :deep(.high-salience-notice:hover) {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  }
  
  :deep(.salience-icon) {
      background-color: #4CAF50;
      color: white;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      margin-right: 16px;
      flex-shrink: 0;
      box-shadow: 0 2px 6px rgba(76, 175, 80, 0.3);
  }
  
  :deep(.salience-content) {
      flex: 1;
  }
  
  :deep(.salience-title) {
      font-size: 16px;
      font-weight: 600;
      color: #2E7D32;
      margin-bottom: 4px;
  }
  
  :deep(.salience-value) {
      font-size: 24px;
      font-weight: 700;
      color: #4CAF50;
      margin-bottom: 4px;
  }
  
  :deep(.salience-description) {
      font-size: 14px;
      color: #555;
      opacity: 0.9;
  }
  
  /* 添加 All elements 部分的 Used visual effects 下的 tag 特殊样式 */
  :deep(.all-elements-tag) {
      width: 100%;
      min-width: 100%;
      font-size: 16px;
      padding: 0 15px;
      text-align: center;
      font-weight: 500;
      box-sizing: border-box;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0;
      transition: all 0.2s ease;
      border-width: 1px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
      height: 100%;
      border-radius: 8px;
      letter-spacing: 0.02em;
      background-color: #f5f5f5;
  }
  
  :deep(.all-elements-tag:hover) {
      background-color: #ebebeb !important;
      transform: translateY(-1px);
      box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08);
      color: #333 !important;
      border-color: #55555540 !important;
  }
  
  /* 添加 All elements 部分的 Used visual effects 标题样式 */
  :deep(.all-elements-title) {
      font-size: 18px;
      padding: 10px 8px;
      margin-bottom: 0;
      font-weight: 600;
      color: #444;
  }
  
  /* 添加两列布局容器 */
  :deep(.two-column-wrapper) {
      display: flex;
      flex-direction: column;
      width: 100%;
      padding: 8px 0;
  }
  
  :deep(.two-column-row) {
      display: flex;
      width: 100%;
      gap: 12px;
      margin-bottom: 12px;
  }
  
  :deep(.two-column-item) {
      width: calc(50% - 6px);
      height: 46px;
      flex: 1;
  }
  </style>