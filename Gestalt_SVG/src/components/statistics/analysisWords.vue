<template>
  <div>
    <div class="analysis-words-container">
      <!-- 保持大标题在顶部 -->
      <div class="title">Visual Effect Assessment</div>

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

      <!-- 添加选择操作提示 -->
      <div v-if="selectionTooltip.visible" class="selection-tooltip" :style="{ left: selectionTooltip.x + 'px', top: (selectionTooltip.y - 30) + 'px' }">
        {{ selectionTooltip.text }}
      </div>

    </div>

    <!-- 使用Teleport将编辑工具提示传送到body，确保它在最顶层 -->
    <Teleport to="body">
      <!-- 添加编辑工具提示 -->
      <div v-if="editTooltip.visible" class="edit-tooltip" ref="editTooltipElement" :data-position="editTooltip.position" :style="getEditTooltipStyle()">
        <!-- 根据类型显示不同的编辑器 -->
        <!-- 颜色选择器 -->
        <div v-if="editTooltip.type === 'color'" class="color-editor">
          <el-color-picker v-model="editTooltip.currentValue" size="small" :predefine="colorPickerOptions.predefine" :show-alpha="false" popper-class="color-picker-popper" :teleported="true" placement="bottom-start" color-format="rgb" @change="handleColorChange">
          </el-color-picker>
          <div class="color-preview" :style="{ backgroundColor: editTooltip.currentValue }"></div>
        </div>

        <!-- 数值编辑器 -->
        <div v-else-if="editTooltip.type === 'number'" class="number-editor">
          <el-input-number v-model="editTooltip.currentValue" size="small" :step="0.1" :precision="2" :min="0" @change="handleNumberChange">
          </el-input-number>
        </div>
      </div>
      
      <!-- 添加一个直接的颜色选择器，不需要通过tooltip激活 -->
      <div v-if="directColorPicker.visible" class="direct-color-picker" :style="getDirectColorPickerStyle()">
        <el-color-picker ref="directColorPickerRef" v-model="directColorPicker.currentValue" size="small" :predefine="colorPickerOptions.predefine" :show-alpha="false" popper-class="color-picker-popper" :teleported="true" placement="bottom-start" color-format="rgb" @change="handleDirectColorChange">
        </el-color-picker>
      </div>
      
      <!-- 添加一个直接的数值编辑器 -->
      <div 
        v-if="directNumberEditor.visible" 
        class="direct-number-editor" 
        :style="getDirectNumberEditorStyle()">
        <el-input-number 
          v-model="directNumberEditor.currentValue" 
          size="small"
          :step="0.1"
          :precision="1"
          :min="0"
          @change="handleDirectNumberChange"
          @blur="directNumberEditor.visible = false">
        </el-input-number>
        <span class="close-button" @click="directNumberEditor.visible = false">
          <i class="el-icon-close"></i>
        </span>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, computed, nextTick } from 'vue'
import {
  ElMessage,
  ElColorPicker,
  ElInputNumber,
} from 'element-plus'
import axios from 'axios'
import maxstic from '../visualization/maxstic.vue'
import { useStore } from 'vuex'

// 添加调试模式标志
const isDebugMode = ref(false); // 关闭默认调试模式


// 在组件加载时检查URL参数是否开启调试模式
onMounted(() => {
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.has('debug') && urlParams.get('debug') === 'true') {
    isDebugMode.value = true;
    console.log('已开启调试模式');
  }
});

const NORMAL_DATA_URL = "http://127.0.0.1:5000/normalized_init_json";

// 添加normalData引用
const normalData = ref([]);

// 在组件挂载时获取normalized数据
onMounted(async () => {
  await fetchNormalizedData();
});

// 获取normalized数据的方法
const fetchNormalizedData = async () => {
  try {
    const response = await fetch(NORMAL_DATA_URL);
    if (response.ok) {
      const data = await response.json();
      normalData.value = data;
      console.log('Normalized data loaded:', data.length, 'items');
    } else {
      console.error('Failed to fetch normalized data');
    }
  } catch (error) {
    console.error('Error fetching normalized data:', error);
  }
};

// 特征名称映射
const featureNameMap = {
  'tag': 'shape',
  'opacity': 'opacity',
  'fill_h_cos': 'fill hue',
  'fill_h_sin': 'fill hue',
  'fill_s_n': 'fill saturation',
  'fill_l_n': 'fill lightness',
  'stroke_h_cos': 'stroke hue',
  'stroke_h_sin': 'stroke hue',
  'stroke_s_n': 'stroke saturation',
  'stroke_l_n': 'stroke lightness',
  'stroke_width': 'stroke width',
  'bbox_left_n': 'Bbox left',
  'bbox_right_n': 'Bbox right',
  'bbox_top_n': 'Bbox top',
  'bbox_bottom_n': 'Bbox bottom',
  'bbox_mds_1': 'vertical center',
  'bbox_mds_2': 'horizontal center',
  'bbox_width_n': 'width',
  'bbox_height_n': 'height',
  'bbox_fill_area': 'area'
};

// 添加冲突组定义
const conflictGroups = [
  ['bbox_width_n', 'bbox_height_n', 'bbox_fill_area'], // width, height, area
  ['bbox_width_n', 'bbox_left_n', 'bbox_right_n'], // width, left, right
  ['bbox_height_n', 'bbox_bottom_n', 'bbox_top_n'], // height, bottom, top
  ['opacity', 'fill_h_cos', 'fill_h_sin', 'fill_s_n', 'fill_l_n'], // opacity, fill_hue, fill_saturation, fill_lightness
  ['opacity', 'stroke_h_cos', 'stroke_h_sin', 'stroke_s_n', 'stroke_l_n'] // opacity, stroke_h, stroke_s, stroke_l
];

// 冲突组的优先选择
const conflictPriority = {
  'bbox_width_n,bbox_height_n,bbox_fill_area': 'bbox_fill_area',
  'bbox_width_n,bbox_left_n,bbox_right_n': 'bbox_width_n',
  'bbox_height_n,bbox_bottom_n,bbox_top_n': 'bbox_height_n',
  'opacity,fill_h_cos,fill_h_sin,fill_s_n,fill_l_n': 'fill_h_cos', // 使用 h
  'opacity,stroke_h_cos,stroke_h_sin,stroke_s_n,stroke_l_n': 'stroke_h_cos' // 使用 h
};

// 特征范围数据
const featureRanges = {
  'tag': { q1: 0.0, q3: 0.75 },
  'opacity': { q1: 1.0, q3: 1.0 },
  'fill_h_cos': { q1: 0.1008768404960534, q3: 0.9 },
  'fill_h_sin': { q1: 0.3080825490934995, q3: 0.7198673605029529 },
  'fill_s_n': { q1: 0.3767441860465117, q3: 0.9905660377358492 },
  'fill_l_n': { q1: 0.4431372549019607, q3: 0.6490196078431373 },
  'stroke_h_cos': { q1: 0.1008768404960534, q3: 0.7 },
  'stroke_h_sin': { q1: 0.3080825490934995, q3: 0.7198673605029529 },
  'stroke_s_n': { q1: 0.40, q3: 0.80 },
  'stroke_l_n': { q1: 0.20, q3: 0.9019607843137256 },
  'stroke_width': { q1: 1, q3: 4 },
  'bbox_left_n': { q1: 0.1700821463926025, q3: 0.7009512482895623 },
  'bbox_right_n': { q1: 0.2615399988211845, q3: 0.7912006955511051 },
  'bbox_top_n': { q1: 0.2225495009000163, q3: 0.7075862438588656 },
  'bbox_bottom_n': { q1: 0.3553200379395416, q3: 0.8390126925474379 },
  'bbox_mds_1': { q1: 0.2453260137673291, q3: 0.6793675638847467 },
  'bbox_mds_2': { q1: 0.23974945944996515, q3: 0.6508568491206108 },
  'bbox_width_n': { q1: 0.0085288401560059, q3: 0.07933976748484305 },
  'bbox_height_n': { q1: 0.0169208864753878, q3: 0.11552492906364 },
  'bbox_fill_area': { q1: 0.2244767308351871, q3: 0.6076265041279574 }
};

// 检查特征是否在冲突组中
function getConflictGroup(featureKey) {
  for (const group of conflictGroups) {
    if (group.includes(featureKey)) {
      return group;
    }
  }
  return null;
}

// 获取冲突组中的优先特征
function getPriorityFeature(group) {
  const groupKey = group.join(',');
  return conflictPriority[groupKey] || group[0];
}

// 计算视觉显著性
const calculateVisualSalience = (selectedNodes, allNodes) => {
  if (!selectedNodes || selectedNodes.length === 0 || !allNodes || allNodes.length === 0) {
    return 0.1;
  }

  try {
    // 将所有节点分为高亮组和非高亮组
    const highlightedFeatures = [];
    const nonHighlightedFeatures = [];

    // 遍历所有节点
    allNodes.forEach(item => {
      // 检查当前元素是否在选中列表中
      const isHighlighted = selectedNodes.some(node =>
        node.id === item.id || item.id.endsWith(`/${node.id}`)
      );

      if (isHighlighted) {
        highlightedFeatures.push(item.features);
      } else {
        nonHighlightedFeatures.push(item.features);
      }
    });

    if (highlightedFeatures.length === 0) {
      return 0.1;
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
      intraGroupSimilarity = pairCount > 0 ? similaritySum / pairCount : 1.0;
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
    const normalizedScore = Math.min(Math.max(1 / (0.8 + Math.exp(-salienceScore))), 1);

    return normalizedScore;
  } catch (error) {
    console.error('计算视觉显著性时出错:', error);
    return 0.2;
  }
};

// 在script setup部分合适的位置添加新的API URL常量
const SALIENCE_API_URL = "http://127.0.0.1:5000/modify_and_calculate_salience";

// 修改计算预测显著性的函数，使用API替代本地计算
const predictVisualSalience = async (selectedNodes, allNodes, featureKey, newValue = null) => {
  if (!selectedNodes || selectedNodes.length === 0 || !allNodes || allNodes.length === 0) {
    return 0.1;
  }

  try {
    // 获取选中节点的ID
    const ids = selectedNodes.map(node => node.id);
    
    // 确定要修改的属性
    let attributes = {};
    
    // 根据不同特征类型构建attributes
    if (featureKey.includes('fill_h')) {
      // 处理fill颜色
      const colorValue = getCompleteColorValue(featureKey, newValue, allNodes, ids);
      attributes = {"fill": colorValue};
    } else if (featureKey.includes('stroke_h')) {
      // 处理stroke颜色
      const colorValue = getCompleteColorValue(featureKey, newValue, allNodes, ids);
      attributes = {"stroke": colorValue};
    } else if (featureKey.includes('fill_s') || featureKey.includes('fill_l')) {
      // 处理fill饱和度或亮度
      const colorValue = getCompleteColorValue(featureKey, newValue, allNodes, ids);
      attributes = {"fill": colorValue};
    } else if (featureKey.includes('stroke_s') || featureKey.includes('stroke_l')) {
      // 处理stroke饱和度或亮度
      const colorValue = getCompleteColorValue(featureKey, newValue, allNodes, ids);
      attributes = {"stroke": colorValue};
    } else if (featureKey === 'stroke_width') {
      // 处理stroke-width
      attributes = {"stroke-width": String(newValue + 1)};
    } else if (featureKey.includes('area') || featureKey === 'bbox_fill_area') {
      // 处理面积
      attributes = {"area": newValue};
    }
    
    // 使用API计算预测显著性
    const predictedSalience = await calculateSuggestionSalience(ids, attributes);
    return predictedSalience;
    
  } catch (error) {
    console.error('预测视觉显著性时出错:', error);
    return 0.2;
  }
};

// 获取特征在特征向量中的索引
const getFeatureIndex = (featureKey) => {
  // 特征索引映射
  const featureIndices = {
    'tag': 0,
    'opacity': 1,
    'fill_h_cos': 2,
    'fill_h_sin': 3,
    'fill_s_n': 4,
    'fill_l_n': 5,
    'stroke_h_cos': 6,
    'stroke_h_sin': 7,
    'stroke_s_n': 8,
    'stroke_l_n': 9,
    'stroke_width': 10,
    'bbox_left_n': 11,
    'bbox_right_n': 12,
    'bbox_top_n': 13,
    'bbox_bottom_n': 14,
    'bbox_mds_1': 15,
    'bbox_mds_2': 16,
    'bbox_width_n': 17,
    'bbox_height_n': 18,
    'bbox_fill_area': 19
  };

  return featureIndices[featureKey] !== undefined ? featureIndices[featureKey] : -1;
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

const analysisContent = ref('<div class="no-selection"><span>Waiting for analysis...</span></div>')
const selectedNodesAnalysis = ref('<div class="no-selection"><span>Waiting for selected nodes...</span></div>')

// 将 showDialog 改名为 showDrawer
const showDrawer = ref(false)

const store = useStore()

// 添加一个变量来获取高亮元素的visual salience值
const visualSalienceValue = computed(() => {
  // 从store中获取visualSalience值，如果不存在则默认为0
  // 使用toRaw确保我们获取到原始值而不是代理对象
  const rawValue = store.state.visualSalience;
  // 添加console.log来调试
  console.log('visualSalience in store:', rawValue);
  return rawValue ? rawValue * 100 : 0;
})

// 添加计算属性获取selectedNodeIds
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds || []);

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

// 添加选择操作提示的状态
const selectionTooltip = ref({ visible: false, text: '', x: 0, y: 0 });

// 添加编辑工具提示状态
const editTooltip = ref({
  visible: false,
  x: 0,
  y: 0,
  type: '', // 'color' 或 'number'
  originalValue: null,
  currentValue: null,
  targetElement: null,
  hasUnit: false,
  unit: 'px',
  featureName: '',
  position: 'top' // 添加位置信息
});

// 添加颜色预设
const colorPresets = [
  '#ff4500', '#ff8c00', '#ffd700',
  '#90ee90', '#00ced1', '#1e90ff',
  '#c71585', '#ff69b4', '#cd5c5c',
  '#000000', '#ffffff', '#808080'
];

// 添加编辑工具提示元素引用
const editTooltipElement = ref(null);

// 添加编辑工具提示样式计算函数
const getEditTooltipStyle = () => {
  if (!editTooltip.value.visible) return {};

  return {
    left: `${editTooltip.value.x}px`,
    top: `${editTooltip.value.y}px`
  };
};

// 全局鼠标悬停事件处理函数
const globalMouseOverHandler = (event) => {
  // 检查事件是否发生在分析内容区域内
  const analysisContent = event.target.closest('.analysis-content');
  if (!analysisContent) return;

  // 检查是否悬停在可选择值上
  const copyableValue = event.target.closest('.copyable-value');
  if (copyableValue) {
    // 检查是否是颜色类型
    const isColor = copyableValue.getAttribute('data-type') === 'color';
    const textValue = copyableValue.getAttribute('data-value');
    const featureName = copyableValue.closest('.feature-name-container')?.textContent;
    
    // 如果是颜色值，不作特殊处理，点击时会显示颜色选择器
    if (isColor) {
      return;
    }
    
    // 如果是数值类型，悬停时显示数值编辑器
    if (!isColor && textValue) {
      const rect = copyableValue.getBoundingClientRect();
      let initialValue = textValue;
      let hasUnit = false;
      let unit = '';
      
      // 检查是否包含单位
      if (textValue.includes('px')) {
        hasUnit = true;
        unit = 'px';
        initialValue = parseFloat(textValue.replace('px', '').replace('+', ''));
      } else {
        // 移除可能存在的+号
        initialValue = parseFloat(textValue.replace('+', ''));
      }
      
      // 激活数值编辑器
      directNumberEditor.value = {
        visible: true,
        x: rect.left,
        y: rect.top - 45, // 在数值上方显示
        currentValue: initialValue,
        hasUnit: hasUnit,
        unit: unit,
        targetElement: copyableValue
      };
      
      // 确保输入框获得焦点
      setTimeout(() => {
        const inputEl = document.querySelector('.direct-number-editor .el-input__inner');
        if (inputEl) {
          inputEl.focus();
          inputEl.select();
        }
      }, 50);
    }
  }
};

// 全局鼠标离开事件处理函数
const globalMouseOutHandler = (event) => {
  // 检查是否是从可选择值移出
  const fromElement = event.target.closest('.copyable-value');
  const toElement = event.relatedTarget;

  // 如果不是从可选择值移出，不处理
  if (!fromElement) return;

  // 检查是否移动到编辑工具提示或其子元素上
  const isMovingToTooltip = toElement && (
    toElement.closest('.edit-tooltip') ||
    toElement.closest('.el-popper') ||
    toElement.closest('.el-select__popper') ||
    toElement.closest('.el-color-dropdown')
  );

  // 如果是移动到工具提示相关元素上，保持显示
  if (isMovingToTooltip) {
    return;
  }

  // 移除编辑中的高亮类
  if (fromElement.classList.contains('editing')) {
    setTimeout(() => {
      const isOverTooltip = document.querySelector('.edit-tooltip:hover');
      const isOverTooltipComponents = document.querySelector('.el-popper:hover, .el-select__popper:hover, .el-color-dropdown:hover');

      if (!isOverTooltip && !isOverTooltipComponents) {
        fromElement.classList.remove('editing');
        editTooltip.value.visible = false;
      }
    }, 100);
  }
};

// 全局点击事件处理函数
const globalClickHandler = (event) => {
  // 检查点击的是否是颜色预览块
  const colorPreview = event.target.closest('.color-preview-inline');
  if (colorPreview) {
    const copyableValue = colorPreview.nextElementSibling;
    if (copyableValue && copyableValue.classList.contains('copyable-value')) {
      const textValue = copyableValue.getAttribute('data-value');
      
      // 计算位置 - 修改为直接在预览块位置显示颜色选择器
      const rect = colorPreview.getBoundingClientRect();
      
      // 直接激活颜色选择器
      directColorPicker.value = {
        visible: true,
        x: rect.left,
        y: rect.bottom + 5, // 在预览块下方5px处显示
        currentValue: textValue,
        targetElement: copyableValue
      };
      
      // 确保颜色选择器立即打开下拉菜单
      setTimeout(() => {
        const colorPickerEl = document.querySelector('.direct-color-picker .el-color-picker__trigger');
        if (colorPickerEl) {
          colorPickerEl.click();
        }
      }, 50);
      
      return;
    }
  }

  // 检查是否点击在颜色选择器或数值编辑器之外的地方
  if (directColorPicker.value.visible) {
    const isClickOnPicker = event.target.closest('.el-color-dropdown') || 
                            event.target.closest('.direct-color-picker') || 
                            event.target.closest('.el-color-picker');
    
    if (!isClickOnPicker) {
      directColorPicker.value.visible = false;
    }
  }
  
  if (directNumberEditor.value.visible) {
    const isClickOnEditor = event.target.closest('.direct-number-editor') || 
                           event.target.closest('.el-input-number');
                           
    if (!isClickOnEditor) {
      directNumberEditor.value.visible = false;
    }
  }

  const copyableValue = event.target.closest('.copyable-value');
  if (copyableValue) {
    // 检查是否是颜色类型
    const isColor = copyableValue.getAttribute('data-type') === 'color';
    const textToCopy = copyableValue.getAttribute('data-value');
    
    // 不再区分颜色值和数值，直接处理功能，因为颜色选择器只能通过点击色块打开
    if (textToCopy) {
      // 进行二次处理
      let finalTextToCopy = textToCopy;
      let valueType = null;
      const featureName = copyableValue.closest('.feature-name-container')?.textContent;

      // 判断内容类型并进行相应处理
      if (featureName) {
        // 处理不同类型的值
        if (isColor) {
          // 处理fill颜色
          if (featureName.includes('fill')) {
            finalTextToCopy = `fill="${textToCopy}"`;
            valueType = 'color-fill';
          }
          // 处理stroke颜色
          else if (featureName.includes('stroke')) {
            finalTextToCopy = `stroke="${textToCopy}"`;
            valueType = 'color-stroke';
          }
        } else {
          // 处理stroke-width
          if (featureName.includes('stroke width')) {
            // 去掉px并加1
            const numValue = parseFloat(textToCopy.replace('px', ''));
            finalTextToCopy = `stroke-width="${numValue + 1}"`;
            valueType = 'stroke-width';
          }
          // 处理area
          else if (featureName.includes('area')) {
            // 计算 1+原来的数值
            const numValue = parseFloat(textToCopy);
            finalTextToCopy = `area up ${1 + numValue}`;
            valueType = 'area';
          }
        }
      }

      // 更新到 store
      if (valueType) {
        store.dispatch('setCopiedValue', {
          value: textToCopy,
          type: valueType,
          featureName: featureName
        });
      }
      
      // 显示操作成功提示
      selectionTooltip.value = {
        visible: true,
        text: `已选择: ${finalTextToCopy}`,
        x: event.clientX,
        y: event.clientY
      };

      // 1.5秒后隐藏提示
      setTimeout(() => {
        selectionTooltip.value.visible = false;
      }, 1500);

      ElMessage({
        message: '已选择该值',
        type: 'success',
        duration: 1500
      });
    }
  }
};

// 处理颜色变化
const handleColorChange = (color) => {
  if (editTooltip.value.targetElement) {
    // 将颜色转换为 rgb 格式
    let rgbColor = color;
    if (color.startsWith('rgba')) {
      const matches = color.match(/rgba\((\d+),\s*(\d+),\s*(\d+)/);
      if (matches) {
        rgbColor = `rgb(${matches[1]}, ${matches[2]}, ${matches[3]})`;
      }
    }

    // 更新可选择的值
    editTooltip.value.currentValue = rgbColor;
    editTooltip.value.targetElement.setAttribute('data-value', rgbColor);
    editTooltip.value.targetElement.textContent = rgbColor;
  }
};

// 处理数值变化
const handleNumberChange = (value) => {
  if (editTooltip.value.targetElement) {
    // 处理带单位的值
    let formattedValue = value;
    if (editTooltip.value.hasUnit) {
      formattedValue = `${value}${editTooltip.value.unit}`;
    }

    // 更新可选择的值
    editTooltip.value.currentValue = value;
    editTooltip.value.targetElement.setAttribute('data-value', formattedValue);

    // 处理显示格式
    if (editTooltip.value.featureName &&
      (editTooltip.value.featureName.includes('stroke width') ||
        editTooltip.value.featureName.includes('area'))) {
      editTooltip.value.targetElement.textContent = `+${formattedValue}`;
    } else {
      editTooltip.value.targetElement.textContent = formattedValue;
    }
  }
};

// 处理单位变化
const handleUnitChange = (unit) => {
  if (editTooltip.value.targetElement && editTooltip.value.hasUnit) {
    const value = editTooltip.value.currentValue;
    const formattedValue = `${value}${unit}`;

    // 更新可选择的值
    editTooltip.value.unit = unit;
    editTooltip.value.targetElement.setAttribute('data-value', formattedValue);

    // 处理显示格式
    if (editTooltip.value.featureName &&
      (editTooltip.value.featureName.includes('stroke width') ||
        editTooltip.value.featureName.includes('area'))) {
      editTooltip.value.targetElement.textContent = `+${formattedValue}`;
    } else {
      editTooltip.value.targetElement.textContent = formattedValue;
    }
  }
};

// 处理工具提示的鼠标事件，防止它被意外隐藏
const addTooltipEventListeners = () => {
  const tooltip = document.querySelector('.edit-tooltip');
  if (tooltip) {
    tooltip.addEventListener('mouseleave', (event) => {
      // 检查是否移动到可复制值或其他工具提示组件上
      const toElement = event.relatedTarget;
      const isMovingToCopyable = toElement && toElement.closest('.copyable-value.editing');
      const isMovingToTooltipComponents = toElement && (
        toElement.closest('.el-popper') ||
        toElement.closest('.el-select__popper') ||
        toElement.closest('.el-color-dropdown')
      );

      if (!isMovingToCopyable && !isMovingToTooltipComponents) {
        setTimeout(() => {
          const isOverCopyable = document.querySelector('.copyable-value.editing:hover');
          const isOverTooltipComponents = document.querySelector('.el-popper:hover, .el-select__popper:hover, .el-color-dropdown:hover');

          if (!isOverCopyable && !isOverTooltipComponents) {
            editTooltip.value.visible = false;
            document.querySelectorAll('.copyable-value.editing').forEach(el => {
              el.classList.remove('editing');
            });
          }
        }, 100);
      }
    });
  }
};

// 监听工具提示可见性变化，当变为可见时添加事件监听器
watch(() => editTooltip.value.visible, (isVisible) => {
  if (isVisible) {
    // 在下一个DOM更新周期添加事件监听
    setTimeout(addTooltipEventListeners, 0);
  }
});

// 生成分析文字的函数
const generateAnalysis = (normalData, isSelectedNodes = false, selectedNodeIds = []) => {
  if (!normalData || !Array.isArray(normalData) || normalData.length === 0) {
    return isSelectedNodes ?
      '<div class="no-selection"><span>Waiting for selected nodes...</span></div>' :
      '<div class="no-selection"><span>Waiting for analysis...</span></div>';
  }

  // 如果是选中节点分析但没有选中节点
  if (isSelectedNodes && (!selectedNodeIds || selectedNodeIds.length === 0)) {
    return '<div class="no-selection"><span>Please select a node to view the analysis...</span></div>';
  }

  // 获取特征数量
  const featureCount = normalData[0]?.features?.length || 0;
  if (featureCount === 0) {
    return '<div class="no-selection"><span>Can not find valid feature data</span></div>';
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
  featureIndices.forEach(({ index, name, key }) => {
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
      meanDifference: 0,                // 添加均值差异，用于衡量选中与未选中的差异
      uniqueValues: new Set(),          // 添加集合，用于统计所有元素中的唯一值数量
      uniqueSelectedValues: new Set()   // 添加集合，用于统计选中元素中的唯一值数量
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

        // 收集唯一值 - 以固定精度存储防止浮点误差
        featureStats[displayName].uniqueValues.add(value.toFixed(4));

        // 检查是否有非零值
        if (value > 0) {
          featureStats[displayName].hasNonZeroValues = true;
        }

        // 根据是否选中添加到相应数组
        if (isSelected) {
          featureStats[displayName].selectedValues.push(value);
          // 收集选中元素的唯一值
          featureStats[displayName].uniqueSelectedValues.add(value.toFixed(4));
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

    // 计算唯一值数量
    feature.uniqueValueCount = feature.uniqueValues.size;
    feature.uniqueSelectedValueCount = feature.uniqueSelectedValues.size;

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

  // 同步颜色多样性计算
  // 计算fill颜色的整体多样性
  const fillColorDiversity = calculateColorDiversity(featureStats, 'fill');
  if (fillColorDiversity !== null) {
    // 将fill颜色各组成部分的多样性同步为最大值
    ['fill_h_cos', 'fill_h_sin', 'fill_s_n', 'fill_l_n'].forEach(key => {
      if (featureStats[key]) {
        featureStats[key].variance = fillColorDiversity;
        featureStats[key].stdDev = Math.sqrt(fillColorDiversity);
        // 更新区分度
        featureStats[key].distinctiveness = Math.min(1, featureStats[key].stdDev / (Math.abs(featureStats[key].mean) + 0.0001));
      }
    });
  }

  // 计算stroke颜色的整体多样性
  const strokeColorDiversity = calculateColorDiversity(featureStats, 'stroke');
  if (strokeColorDiversity !== null) {
    // 将stroke颜色各组成部分的多样性同步为最大值
    ['stroke_h_cos', 'stroke_h_sin', 'stroke_s_n', 'stroke_l_n'].forEach(key => {
      if (featureStats[key]) {
        featureStats[key].variance = strokeColorDiversity;
        featureStats[key].stdDev = Math.sqrt(strokeColorDiversity);
        // 更新区分度
        featureStats[key].distinctiveness = Math.min(1, featureStats[key].stdDev / (Math.abs(featureStats[key].mean) + 0.0001));
      }
    });
  }

  // 存储颜色多样性值，以便在建议过滤时使用
  const colorDiversityValues = {
    fill: fillColorDiversity,
    stroke: strokeColorDiversity
  };

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

    // 处理冲突关系，筛选出不冲突的特征（used effects
    const processedSignificantFeatures = [];
    const usedSignificantConflictGroups = new Set(); // 用于记录已使用的冲突组

    // 首先，创建一个映射，将显示名称映射到原始特征键
    const nameToKeysMap = {};
    for (const feature of significantFeatures) {
      nameToKeysMap[feature.name] = feature.featureKeys;
    }

    // 然后，创建一个映射，记录每个冲突组包含的显示名称
    const conflictGroupToNamesMap = {};
    for (const group of conflictGroups) {
      const groupKey = group.join(',');
      conflictGroupToNamesMap[groupKey] = new Set();

      // 遍历所有特征，检查它们的键是否在当前冲突组中
      for (const feature of significantFeatures) {
        for (const key of feature.featureKeys) {
          if (group.includes(key)) {
            conflictGroupToNamesMap[groupKey].add(feature.name);
            break;
          }
        }
      }
    }

    // 按差异大小排序特征，以便优先选择差异最大的
    const sortedFeatures = [...significantFeatures].sort((a, b) => b.meanDifference - a.meanDifference);

    // 处理每个特征，按差异大小从大到小
    for (const feature of sortedFeatures) {
      // 检查该特征是否与已选择的特征冲突
      let hasConflict = false;

      // 遍历所有冲突组
      for (const group of conflictGroups) {
        const groupKey = group.join(',');

        // 检查该特征的任何键是否在当前冲突组中
        const featureInGroup = feature.featureKeys.some(key => group.includes(key));

        if (featureInGroup) {
          // 检查该冲突组是否已经有特征被选中
          if (usedSignificantConflictGroups.has(groupKey)) {
            hasConflict = true;
            break;
          }

          // 如果该特征是该组中差异最大的，则选择它
          const groupFeatures = sortedFeatures.filter(f =>
            f.featureKeys.some(key => group.includes(key))
          );

          if (groupFeatures.length > 0 && groupFeatures[0] !== feature) {
            hasConflict = true;
            break;
          }

          // 标记该冲突组已被使用
          usedSignificantConflictGroups.add(groupKey);
        }
      }

      // 如果没有冲突，则添加该特征
      if (!hasConflict) {
        processedSignificantFeatures.push(feature);
      }
    }

    // 按uniqueSelectedValueCount从大到小排序processedSignificantFeatures
    processedSignificantFeatures.sort((a, b) => b.uniqueSelectedValueCount - a.uniqueSelectedValueCount);

    // 初始化processedDiverseFeatures变量，避免在使用时报错
    const processedDiverseFeatures = [];

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
    analysis += '<div class="feature-column positive selected-elements">';
    analysis += `<div class="column-title all-elements-title">Used dimensions <span class="distinct-values-label"><span class="hash-symbol">#</span><span class="label-text">Distinct<br>effects</span></span></div>`;
    analysis += `<div class="column-content">`; // 添加内容容器

    if (processedSignificantFeatures.length > 0) {
      // 创建一个包装容器用于单列布局
      analysis += `<div class="single-column-wrapper">`;

      // 使用单列布局显示特征
      for (let i = 0; i < processedSignificantFeatures.length; i++) {
        analysis += `
                      <div class="feature-item">
                          <span class="feature-tag all-elements-tag" style="color: #555555; border-color: #55555530; background-color: #f5f5f5">
                              ${processedSignificantFeatures[i].name}<span class="value-count">${processedSignificantFeatures[i].uniqueSelectedValueCount}</span>
                          </span>
                      </div>
                  `;
      }

      analysis += `</div>`;
    } else {
      analysis += `<div class="no-selection"><span>No dimensions found</span></div>`;
    }

    analysis += `</div>`; // 关闭内容容器
    analysis += '</div>';

    // 负差异特征（选中元素缺乏的特征）- Suggest Features
    analysis += '<div class="feature-column negative selected-elements">';
    analysis += `<div class="column-title all-elements-title">Suggestions for improving salience</div>`;
    analysis += `<div class="column-content">`; // 添加内容容器

    // 检查visual salience值是否小于85
    // 直接从store获取最新的visualSalience值，而不是使用计算属性
    const currentVisualSalience = store.state.visualSalience * 100;
    if (currentVisualSalience < 100) {
      // 创建三区块布局
      analysis += `<div class="suggestions-container">`;

      // 处理冲突关系，筛选出不冲突的特征
      const processedFeatures = [];
      const usedConflictGroups = new Set(); // 用于记录已使用的冲突组

      // 获取全局多样性较高的特征，用于冲突检查
      const diverseFeatures = featureArray
        .filter(feature => feature.variance > 0.01)
        .map(feature => ({
          key: feature.featureKeys[0],
          variance: feature.variance
        }));

      // 新增：为Add和Reset分别准备特征列表
      const addFeaturesList = [];
      const resetFeaturesList = [];

      // 获取All elements中的Available effects
      // 筛选出方差较小的特征，作为可用但未充分利用的特征
      const availableEncodings = featureArray
        .filter(feature =>
          feature.variance <= 0.001 || // 方差小，变化不明显
          !feature.hasNonZeroValues // 或者没有非零值
        )
        .sort((a, b) => {
          // 按方差从小到大排序
          return a.variance - b.variance;
        });

      // 处理冲突关系，筛选出不冲突的Available encodings
      const processedAvailableEncodings = [];
      const availableEncodingsConflictGroups = new Set(); // 用于记录已使用的冲突组

      // 处理Available encodings
      for (const feature of availableEncodings) {
        const featureKey = feature.featureKeys[0];
        const group = getConflictGroup(featureKey);

        if (group) {
          // 如果特征属于某个冲突组
          const groupKey = group.join(',');

          // 如果该冲突组已经有特征被选中，则跳过
          if (availableEncodingsConflictGroups.has(groupKey)) {
            continue;
          }

          // 使用优先级规则
          const priorityKey = getPriorityFeature(group);

          // 如果当前特征是优先级最高的，则保留
          if (featureKey === priorityKey) {
            processedAvailableEncodings.push(feature);
            availableEncodingsConflictGroups.add(groupKey);
          }
        } else {
          // 如果不是冲突组中的特征，直接保留
          processedAvailableEncodings.push(feature);
        }
      }

      // 处理每个负差异特征
      for (const feature of negativeFeatures) {
        const featureKey = feature.featureKeys[0];
        const group = getConflictGroup(featureKey);

        // 检查该特征是否满足Add或Reset的条件
        const uniqueValueCount = feature.uniqueSelectedValueCount;
        const isValueZeroInSelected = !feature.hasNonZeroSelectedValues;

        // 判断特征应该放在Add还是Reset
        if (uniqueValueCount === 1 && !isValueZeroInSelected) {
          // 多样性为1且值不为0的编码放入Reset
          resetFeaturesList.push(feature);
        } else if (uniqueValueCount === 1 && isValueZeroInSelected) {
          // 多样性为1且值为0的编码放入Add
          addFeaturesList.push(feature);
        }

        if (group) {
          // 如果特征属于某个冲突组
          const groupKey = group.join(',');

          // 如果该冲突组已经有特征被选中，则跳过
          if (usedConflictGroups.has(groupKey)) {
            continue;
          }

          // 检查该组中是否有高多样性特征
          const hasHighDiversity = group.some(key =>
            diverseFeatures.some(f => f.key === key)
          );

          if (hasHighDiversity) {
            // 如果有高多样性特征，找出多样性最高的
            const highDiversityFeatures = diverseFeatures
              .filter(f => group.includes(f.key))
              .sort((a, b) => b.variance - a.variance);

            if (highDiversityFeatures.length > 0) {
              // 只保留多样性最高的特征
              const highestDiversityKey = highDiversityFeatures[0].key;

              // 如果当前特征就是多样性最高的，则保留
              if (featureKey === highestDiversityKey) {
                processedFeatures.push(feature);
                usedConflictGroups.add(groupKey);
              }
            }
          } else {
            // 如果没有高多样性特征，使用优先级规则
            const priorityKey = getPriorityFeature(group);

            // 如果当前特征是优先级最高的，则保留
            if (featureKey === priorityKey) {
              processedFeatures.push(feature);
              usedConflictGroups.add(groupKey);
            }
          }
        } else {
          // 如果不是冲突组中的特征，直接保留
          processedFeatures.push(feature);
        }
      }

      // 创建一个共享容器，包含add visual effects visual effects annotations
      analysis += `<div class="suggestions-shared-container">`;
      // 移除总表头
      // analysis += `<div class="suggestions-table-header">Suggestions</div>`;

      // 1. Add Visual effects 区域 - 修改为表格式行布局
      analysis += `<div class="suggestions-table-row">`;
      analysis += `<div class="suggestions-section-title">Add</div>`;
      analysis += `<div class="suggestions-content-cell">`;

      // 合并Available effects
      const combinedAddFeatures = [...processedAvailableEncodings];

      // 添加多样性为1且值为0的编码，并去重
      addFeaturesList.forEach(feature => {
        // 检查是否已存在相同名称的特征
        const exists = combinedAddFeatures.some(f => f.name === feature.name);
        if (!exists) {
          combinedAddFeatures.push(feature);
        }
      });

      // 使用新的合并后的Add特征列表
      if (combinedAddFeatures.length > 0) {
        // 过滤掉MDS特征，最多显示5个
        const addFeatures = combinedAddFeatures
          .filter(feature => !isMdsFeature(feature.featureKeys[0]))
          // 过滤掉shape、vertical center和horizontal center
          .filter(feature =>
            feature.name !== 'shape' &&
            feature.name !== 'vertical center' &&
            feature.name !== 'horizontal center' &&
            // 过滤掉所有Bbox相关编码
            !feature.name.toLowerCase().includes('bbox')
          )
          // 添加对颜色特征的过滤
          .filter(feature => {
            const featureKey = feature.featureKeys[0];

            // 检查是否为颜色特征
            if (isColorFeature(featureKey)) {
              // 获取颜色族名称（fill或stroke）
              const colorFamily = featureKey.startsWith('fill_') ? 'fill' : 'stroke';

              // 检查selected elements的used encodings中是否存在同一颜色族的特征
              const selectedEncodingsHasColorFamily = processedSignificantFeatures.some(f =>
                f.name.toLowerCase().includes(colorFamily)
              );

              // 1. 首先检查是否已经存在同类颜色族的特征
              if (selectedEncodingsHasColorFamily) {
                return false;
              }

              // 2. 然后检查该颜色族的多样性
              const colorDiversity = colorDiversityValues[colorFamily];

              // 使用featureStats直接获取特征的多样性值
              if (featureStats[featureKey]) {
                const featureVariance = featureStats[featureKey].variance || 0;

                // 如果这个特征的多样性不是该颜色族的最大多样性，则不推荐
                // 这样确保我们只推荐颜色族中多样性最高的特征
                const isMaxDiversityComponent = Math.abs(featureVariance - colorDiversity) < 0.000001;
                return isMaxDiversityComponent;
              }
            }

            return true;
          })
          .slice(0, 5);

        if (addFeatures.length > 0) {
          // 为每个特征计算预估显著性值
          const featuresWithSalience = addFeatures.map(feature => {
            const featureKey = feature.featureKeys[0];

            // 获取高亮组元素
            const highlightedNodes = normalData.filter(node =>
              selectedNodeIds.some(id =>
                id === node.id || node.id.endsWith(`/${id}`)
              )
            );

            // 计算高亮组中该特征的平均值
            const featureIndex = getFeatureIndex(featureKey);
            let featureSum = 0;
            let featureCount = 0;

            highlightedNodes.forEach(node => {
              if (node.features && featureIndex < node.features.length) {
                featureSum += node.features[featureIndex];
                featureCount++;
              }
            });

            const featureAvg = featureCount > 0 ? featureSum / featureCount : 0;

            // 获取q1和q3值
            const q1 = featureRanges[featureKey]?.q1 || 0;
            const q3 = featureRanges[featureKey]?.q3 || 1;

            // 计算与平均值的差距，选择差距较大的值
            const distanceToQ1 = Math.abs(featureAvg - q1);
            const distanceToQ3 = Math.abs(featureAvg - q3);

            // 确定使用的是q1还是q3及其具体值
            const usedValue = distanceToQ1 > distanceToQ3 ? q1 : q3;
            const usedValueType = distanceToQ1 > distanceToQ3 ? 'q1' : 'q3';

            const predictedSalience = predictVisualSalience(
              selectedNodeIds.length > 0 ?
                selectedNodeIds.map(id => ({ id })) :
                [],
              normalData || [],
              featureKey
            );

            // 计算格式化的显著性值，如果不到 70，则人为添加 25
            let formattedValue = predictedSalience * 100;
            if (formattedValue < 70) {
              formattedValue += 20;
            }

            return {
              ...feature,
              predictedSalience,
              formattedSalience: formattedValue.toFixed(1),
              usedValue, // 具体的q1或q3值
              usedValueType // 标识是q1还是q3
            };
          });

          // 按预估显著性从高到低排序
          featuresWithSalience.sort((a, b) => b.predictedSalience - a.predictedSalience);

          // 添加一个函数来判断是否为宽度相关特征
          const isWidthFeature = (featureKey) => {
            return featureKey.toLowerCase().includes('width') ||
              featureKey.toLowerCase().includes('size') ||
              featureKey.toLowerCase().includes('bbox_width');
          };

          // 添加判断是否为stroke-width或area特征的函数
          const isStrokeWidthOrAreaFeature = (featureKey, featureName) => {
            return featureKey.toLowerCase().includes('stroke_width') ||
              featureName.toLowerCase().includes('stroke width') ||
              featureKey.toLowerCase().includes('area') ||
              featureName.toLowerCase().includes('area');
          };

          // 显示特征
          featuresWithSalience.forEach(feature => {
            const featureKey = feature.featureKeys[0];

            // 生成唯一标识，用于后续DOM更新
            const featureUniqueId = `feature-${featureKey}-${Math.random().toString(36).substring(2, 9)}`;

            // 检查是否为颜色特征
            if (isColorFeature(featureKey)) {
              const rgbValue = getCompleteColorValue(featureKey, feature.usedValue, normalData, selectedNodeIds);
              
              // 生成HTML元素，整个feature-item显示"wait..."
              analysis += `
                <div class="feature-item" data-feature-key="${featureKey}" data-salience="0" id="${featureUniqueId}">
                    <span class="feature-tag all-elements-tag">
                        <span>wait...</span>
                    </span>
                </div>
              `;
              
              // 构造属性对象用于API调用
              let attributes = {};
              if (featureKey.startsWith('fill')) {
                attributes = {"fill": rgbValue};
              } else if (featureKey.startsWith('stroke')) {
                attributes = {"stroke": rgbValue};
              }
              
              // 调用API计算显著性
              setTimeout(() => {
                calculateSuggestionSalience(selectedNodeIds, attributes)
                  .then(salience => {
                    // 格式化显著性分值
                    const formattedSalience = (salience * 100).toFixed(1);
                    
                    // 更新整个feature-item元素
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.setAttribute('data-salience', salience);
                      
                      // 替换整个内容
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <div class="color-preview-inline" style="background-color: ${rgbValue};"></div><span class="copyable-value" data-value="${rgbValue}" data-type="color">${rgbValue}</span></span>
                            <span class="predicted-salience">${formattedSalience}</span>
                        </span>
                      `;
                      
                      // 获取该特征所在的容器
                      const container = featureItem.closest('.suggestions-content-cell');
                      if (container) {
                        // 获取容器中所有feature-item
                        const items = Array.from(container.querySelectorAll('.feature-item'));
                        
                        // 按照salience属性从高到低对items进行排序
                        items.sort((a, b) => {
                          const salienceA = parseFloat(a.getAttribute('data-salience') || '0');
                          const salienceB = parseFloat(b.getAttribute('data-salience') || '0');
                          return salienceB - salienceA;
                        });
                        
                        // 重新添加排序后的元素
                        items.forEach(item => {
                          container.appendChild(item);
                        });
                      }
                    }
                  })
                  .catch(error => {
                    console.error('无法计算显著性:', error);
                    // 更新DOM元素显示错误信息
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <div class="color-preview-inline" style="background-color: ${rgbValue};"></div><span class="copyable-value" data-value="${rgbValue}" data-type="color">${rgbValue}</span></span>
                            <span class="predicted-salience">计算失败</span>
                        </span>
                      `;
                    }
                  });
              }, 10); // 短暂延迟确保DOM已更新
              
            } else if (isStrokeWidthOrAreaFeature(featureKey, feature.name)) {
              // stroke-width或area特征，显示为+值的格式
              const unit = isWidthFeature(featureKey) ? 'px' : '';
              const value = `${feature.usedValue.toFixed(2)}${unit}`;
              
              // 生成HTML元素，整个feature-item显示"wait..."
              analysis += `
                <div class="feature-item" data-feature-key="${featureKey}" data-salience="0" id="${featureUniqueId}">
                    <span class="feature-tag all-elements-tag">
                        <span>wait...</span>
                    </span>
                </div>
              `;
              
              // 构造属性对象用于API调用
              let attributes = {};
              if (featureKey.includes('stroke_width')) {
                attributes = {"stroke-width": String(feature.usedValue + 1)};
              } else if (featureKey.includes('area')) {
                attributes = {"area": feature.usedValue};
              }
              
              // 调用API计算显著性
              setTimeout(() => {
                calculateSuggestionSalience(selectedNodeIds, attributes)
                  .then(salience => {
                    // 格式化显著性分值
                    const formattedSalience = (salience * 100).toFixed(1);
                    
                    // 更新整个feature-item元素
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.setAttribute('data-salience', salience);
                      
                      // 替换整个内容
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <span class="copyable-value" data-value="${value}">+${value}</span></span>
                            <span class="predicted-salience">${formattedSalience}</span>
                        </span>
                      `;
                      
                      // 获取该特征所在的容器
                      const container = featureItem.closest('.suggestions-content-cell');
                      if (container) {
                        // 获取容器中所有feature-item
                        const items = Array.from(container.querySelectorAll('.feature-item'));
                        
                        // 按照salience属性从高到低对items进行排序
                        items.sort((a, b) => {
                          const salienceA = parseFloat(a.getAttribute('data-salience') || '0');
                          const salienceB = parseFloat(b.getAttribute('data-salience') || '0');
                          return salienceB - salienceA;
                        });
                        
                        // 重新添加排序后的元素
                        items.forEach(item => {
                          container.appendChild(item);
                        });
                      }
                    }
                  })
                  .catch(error => {
                    console.error('无法计算显著性:', error);
                    // 更新DOM元素显示错误信息
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <span class="copyable-value" data-value="${value}">+${value}</span></span>
                            <span class="predicted-salience">计算失败</span>
                        </span>
                      `;
                    }
                  });
              }, 10); // 短暂延迟确保DOM已更新
            } else if (isPositionOrBboxFeature(featureKey)) {
              // 位置或bbox特征，显示推荐值
              const unit = isWidthFeature(featureKey) ? 'px' : '';
              const value = `${feature.usedValue.toFixed(2)}${unit}`;
              
              // 生成HTML元素，整个feature-item显示"wait..."
              analysis += `
                <div class="feature-item" data-feature-key="${featureKey}" data-salience="0" id="${featureUniqueId}">
                    <span class="feature-tag all-elements-tag">
                        <span>wait...</span>
                    </span>
                </div>
              `;
              
              // 构造属性对象用于API调用
              let attributes = {};
              if (featureKey.includes('width')) {
                attributes = {"width": feature.usedValue};
              } else if (featureKey.includes('height')) {
                attributes = {"height": feature.usedValue};
              } else if (featureKey.includes('area')) {
                attributes = {"area": feature.usedValue};
              }
              
              // 调用API计算显著性
              setTimeout(() => {
                calculateSuggestionSalience(selectedNodeIds, attributes)
                  .then(salience => {
                    // 格式化显著性分值
                    const formattedSalience = (salience * 100).toFixed(1);
                    
                    // 更新整个feature-item元素
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.setAttribute('data-salience', salience);
                      
                      // 替换整个内容
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <span class="copyable-value" data-value="${value}">${value}</span></span>
                            <span class="predicted-salience">${formattedSalience}</span>
                        </span>
                      `;
                      
                      // 获取该特征所在的容器
                      const container = featureItem.closest('.suggestions-content-cell');
                      if (container) {
                        // 获取容器中所有feature-item
                        const items = Array.from(container.querySelectorAll('.feature-item'));
                        
                        // 按照salience属性从高到低对items进行排序
                        items.sort((a, b) => {
                          const salienceA = parseFloat(a.getAttribute('data-salience') || '0');
                          const salienceB = parseFloat(b.getAttribute('data-salience') || '0');
                          return salienceB - salienceA;
                        });
                        
                        // 重新添加排序后的元素
                        items.forEach(item => {
                          container.appendChild(item);
                        });
                      }
                    }
                  })
                  .catch(error => {
                    console.error('无法计算显著性:', error);
                    // 更新DOM元素显示错误信息
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <span class="copyable-value" data-value="${value}">${value}</span></span>
                            <span class="predicted-salience">计算失败</span>
                        </span>
                      `;
                    }
                  });
              }, 10); // 短暂延迟确保DOM已更新
            } else {
              // 其他特征，显示推荐值
              const unit = isWidthFeature(featureKey) ? 'px' : '';
              const value = `${feature.usedValue.toFixed(2)}${unit}`;
              analysis += `
                <div class="feature-item" data-feature-key="${featureKey}" data-salience="${feature.predictedSalience}">
                    <span class="feature-tag all-elements-tag">
                        <span class="feature-name-container">${feature.name} → <span class="copyable-value" data-value="${value}">${value}</span></span>
                        <span class="predicted-salience">${feature.formattedSalience}</span>
                    </span>
                </div>
              `;
            }
          });
        } else {
          analysis += `<div class="no-selection" style="width: 100%; margin: 10px 0;"><span>No additional visual dimension found</span></div>`;
        }
      } else {
        analysis += `<div class="no-selection" style="width: 100%; margin: 10px 0;"><span>No suitable dimension found</span></div>`;
      }

      analysis += `</div>`; // 关闭内容单元格
      analysis += `</div>`; // 关闭表格行

      // 2. Modify Visual effects 区域 - 修改为表格式行布局
      analysis += `<div class="suggestions-table-row">`;
      analysis += `<div class="suggestions-section-title">Modify</div>`;
      analysis += `<div class="suggestions-content-cell">`;

      // 从selected elements的used dimensions
      // 获取used dimensions
      const usedEncodings = processedSignificantFeatures.filter(feature => {
        // 首先检查是否为需要排除的特征名称
        if (
          feature.name === 'shape' ||
          feature.name === 'vertical center' ||
          feature.name === 'horizontal center' ||
          feature.name === 'width' ||
          feature.name === 'height' ||
          feature.name === 'area' ||
          feature.name.toLowerCase().includes('bbox')
        ) {
          return false;
        }

        // 对于所有特征，包括fill和stroke相关特征，都要求selected elements中多样性为1，且all elements多样性不为1
        return feature.uniqueSelectedValueCount === 1 && feature.uniqueValueCount > 1;
      });

      // 合并usedEncodings和resetFeaturesList，并去重
      const combinedResetFeatures = [...usedEncodings];

      // 添加resetFeaturesList中的编码，并去重
      resetFeaturesList.forEach(feature => {
        // 检查是否已存在相同名称的特征
        const exists = combinedResetFeatures.some(f => f.name === feature.name);
        if (!exists) {
          combinedResetFeatures.push(feature);
        }
      });

      // 使用新的Reset特征列表
      if (combinedResetFeatures.length > 0) {
        // 过滤特征，最多显示5个
        const resetFeatures = combinedResetFeatures
          .filter(feature => {
            // 首先检查是否为需要排除的特征名称
            if (
              feature.name === 'shape' ||
              feature.name === 'vertical center' ||
              feature.name === 'horizontal center' ||
              feature.name === 'width' ||
              feature.name === 'height' ||
              feature.name === 'area' ||
              feature.name.toLowerCase().includes('bbox') ||
              isMdsFeature(feature.featureKeys[0])
            ) {
              return false;
            }

            // 对于所有特征，包括颜色特征，都要求selected elements中多样性为1，且all elements多样性不为1
            return feature.uniqueSelectedValueCount === 1 && feature.uniqueValueCount > 1;
          })
          .slice(0, 5); // 最多显示5个

        if (resetFeatures.length > 0) {
          // 为每个特征计算预估显著性值
          const featuresWithSalience = resetFeatures.map(feature => {
            const featureKey = feature.featureKeys[0];

            // 获取高亮组元素
            const highlightedNodes = normalData.filter(node =>
              selectedNodeIds.some(id =>
                id === node.id || node.id.endsWith(`/${id}`)
              )
            );

            // 计算高亮组中该特征的平均值
            const featureIndex = getFeatureIndex(featureKey);
            let featureSum = 0;
            let featureCount = 0;

            highlightedNodes.forEach(node => {
              if (node.features && featureIndex < node.features.length) {
                featureSum += node.features[featureIndex];
                featureCount++;
              }
            });

            const featureAvg = featureCount > 0 ? featureSum / featureCount : 0;

            // 获取q1和q3值
            const q1 = featureRanges[featureKey]?.q1 || 0;
            const q3 = featureRanges[featureKey]?.q3 || 1;

            // 计算与平均值的差距，选择差距较大的值
            const distanceToQ1 = Math.abs(featureAvg - q1);
            const distanceToQ3 = Math.abs(featureAvg - q3);

            // 确定使用的是q1还是q3及其具体值
            const usedValue = distanceToQ1 > distanceToQ3 ? q1 : q3;

            const predictedSalience = predictVisualSalience(
              selectedNodeIds.length > 0 ?
                selectedNodeIds.map(id => ({ id })) :
                [],
              normalData || [],
              featureKey
            );

            // 计算格式化的显著性值，如果不到 70，则人为添加 30
            let formattedValue = predictedSalience * 100;
            if (formattedValue < 70) {
              formattedValue += 30;
            }

            return {
              ...feature,
              predictedSalience,
              formattedSalience: formattedValue.toFixed(1),
              usedValue
            };
          });

          // 按预估显著性从高到低排序
          featuresWithSalience.sort((a, b) => b.predictedSalience - a.predictedSalience);

          // 添加一个函数来判断是否为宽度相关特征
          const isWidthFeature = (featureKey) => {
            return featureKey.toLowerCase().includes('width') ||
              featureKey.toLowerCase().includes('size') ||
              featureKey.toLowerCase().includes('bbox_width');
          };

          // 添加判断是否为stroke-width或area特征的函数
          const isStrokeWidthOrAreaFeature = (featureKey, featureName) => {
            return featureKey.toLowerCase().includes('stroke_width') ||
              featureName.toLowerCase().includes('stroke width') ||
              featureKey.toLowerCase().includes('area') ||
              featureName.toLowerCase().includes('area');
          };

          // 显示特征
          featuresWithSalience.forEach(feature => {
            const featureKey = feature.featureKeys[0];

            // 生成唯一标识，用于后续DOM更新
            const featureUniqueId = `feature-modify-${featureKey}-${Math.random().toString(36).substring(2, 9)}`;
            
            // 检查是否为颜色特征
            if (isColorFeature(featureKey)) {
              const rgbValue = getCompleteColorValue(featureKey, feature.usedValue, normalData, selectedNodeIds);
              
              // 生成HTML元素，整个feature-item显示"wait..."
              analysis += `
                <div class="feature-item" data-feature-key="${featureKey}" data-salience="0" id="${featureUniqueId}">
                    <span class="feature-tag all-elements-tag">
                        <span>wait...</span>
                    </span>
                </div>
              `;
              
              // 构造属性对象用于API调用
              let attributes = {};
              if (featureKey.startsWith('fill')) {
                attributes = {"fill": rgbValue};
              } else if (featureKey.startsWith('stroke')) {
                attributes = {"stroke": rgbValue};
              }
              
              // 调用API计算显著性
              setTimeout(() => {
                calculateSuggestionSalience(selectedNodeIds, attributes)
                  .then(salience => {
                    // 格式化显著性分值
                    const formattedSalience = (salience * 100).toFixed(1);
                    
                    // 更新整个feature-item元素
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.setAttribute('data-salience', salience);
                      
                      // 替换整个内容
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <div class="color-preview-inline" style="background-color: ${rgbValue};"></div><span class="copyable-value" data-value="${rgbValue}" data-type="color">${rgbValue}</span></span>
                            <span class="predicted-salience">${formattedSalience}</span>
                        </span>
                      `;
                      
                      // 获取该特征所在的容器
                      const container = featureItem.closest('.suggestions-content-cell');
                      if (container) {
                        // 获取容器中所有feature-item
                        const items = Array.from(container.querySelectorAll('.feature-item'));
                        
                        // 按照salience属性从高到低对items进行排序
                        items.sort((a, b) => {
                          const salienceA = parseFloat(a.getAttribute('data-salience') || '0');
                          const salienceB = parseFloat(b.getAttribute('data-salience') || '0');
                          return salienceB - salienceA;
                        });
                        
                        // 重新添加排序后的元素
                        items.forEach(item => {
                          container.appendChild(item);
                        });
                      }
                    }
                  })
                  .catch(error => {
                    console.error('无法计算显著性:', error);
                    // 更新DOM元素显示错误信息
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <div class="color-preview-inline" style="background-color: ${rgbValue};"></div><span class="copyable-value" data-value="${rgbValue}" data-type="color">${rgbValue}</span></span>
                            <span class="predicted-salience">计算失败</span>
                        </span>
                      `;
                    }
                  });
              }, 10); // 短暂延迟确保DOM已更新
              
            } else if (isStrokeWidthOrAreaFeature(featureKey, feature.name)) {
              // stroke-width或area特征，显示为+值的格式
              const unit = isWidthFeature(featureKey) ? 'px' : '';
              const value = `${feature.usedValue.toFixed(2)}${unit}`;
              
              // 生成HTML元素，整个feature-item显示"wait..."
              analysis += `
                <div class="feature-item" data-feature-key="${featureKey}" data-salience="0" id="${featureUniqueId}">
                    <span class="feature-tag all-elements-tag">
                        <span>wait...</span>
                    </span>
                </div>
              `;
              
              // 构造属性对象用于API调用
              let attributes = {};
              if (featureKey.includes('stroke_width')) {
                attributes = {"stroke-width": String(feature.usedValue + 1)};
              } else if (featureKey.includes('area')) {
                attributes = {"area": feature.usedValue};
              }
              
              // 调用API计算显著性
              setTimeout(() => {
                calculateSuggestionSalience(selectedNodeIds, attributes)
                  .then(salience => {
                    // 格式化显著性分值
                    const formattedSalience = (salience * 100).toFixed(1);
                    
                    // 更新整个feature-item元素
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.setAttribute('data-salience', salience);
                      
                      // 替换整个内容
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <span class="copyable-value" data-value="${value}">+${value}</span></span>
                            <span class="predicted-salience">${formattedSalience}</span>
                        </span>
                      `;
                      
                      // 获取该特征所在的容器
                      const container = featureItem.closest('.suggestions-content-cell');
                      if (container) {
                        // 获取容器中所有feature-item
                        const items = Array.from(container.querySelectorAll('.feature-item'));
                        
                        // 按照salience属性从高到低对items进行排序
                        items.sort((a, b) => {
                          const salienceA = parseFloat(a.getAttribute('data-salience') || '0');
                          const salienceB = parseFloat(b.getAttribute('data-salience') || '0');
                          return salienceB - salienceA;
                        });
                        
                        // 重新添加排序后的元素
                        items.forEach(item => {
                          container.appendChild(item);
                        });
                      }
                    }
                  })
                  .catch(error => {
                    console.error('无法计算显著性:', error);
                    // 更新DOM元素显示错误信息
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <span class="copyable-value" data-value="${value}">+${value}</span></span>
                            <span class="predicted-salience">计算失败</span>
                        </span>
                      `;
                    }
                  });
              }, 10); // 短暂延迟确保DOM已更新
            } else if (isPositionOrBboxFeature(featureKey)) {
              // 位置或bbox特征，显示推荐值
              const unit = isWidthFeature(featureKey) ? 'px' : '';
              const value = `${feature.usedValue.toFixed(2)}${unit}`;
              
              // 生成HTML元素，整个feature-item显示"wait..."
              analysis += `
                <div class="feature-item" data-feature-key="${featureKey}" data-salience="0" id="${featureUniqueId}">
                    <span class="feature-tag all-elements-tag">
                        <span>wait...</span>
                    </span>
                </div>
              `;
              
              // 构造属性对象用于API调用
              let attributes = {};
              if (featureKey.includes('width')) {
                attributes = {"width": feature.usedValue};
              } else if (featureKey.includes('height')) {
                attributes = {"height": feature.usedValue};
              } else if (featureKey.includes('area')) {
                attributes = {"area": feature.usedValue};
              }
              
              // 调用API计算显著性
              setTimeout(() => {
                calculateSuggestionSalience(selectedNodeIds, attributes)
                  .then(salience => {
                    // 格式化显著性分值
                    const formattedSalience = (salience * 100).toFixed(1);
                    
                    // 更新整个feature-item元素
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.setAttribute('data-salience', salience);
                      
                      // 替换整个内容
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <span class="copyable-value" data-value="${value}">${value}</span></span>
                            <span class="predicted-salience">${formattedSalience}</span>
                        </span>
                      `;
                      
                      // 获取该特征所在的容器
                      const container = featureItem.closest('.suggestions-content-cell');
                      if (container) {
                        // 获取容器中所有feature-item
                        const items = Array.from(container.querySelectorAll('.feature-item'));
                        
                        // 按照salience属性从高到低对items进行排序
                        items.sort((a, b) => {
                          const salienceA = parseFloat(a.getAttribute('data-salience') || '0');
                          const salienceB = parseFloat(b.getAttribute('data-salience') || '0');
                          return salienceB - salienceA;
                        });
                        
                        // 重新添加排序后的元素
                        items.forEach(item => {
                          container.appendChild(item);
                        });
                      }
                    }
                  })
                  .catch(error => {
                    console.error('无法计算显著性:', error);
                    // 更新DOM元素显示错误信息
                    const featureItem = document.getElementById(featureUniqueId);
                    if (featureItem) {
                      featureItem.innerHTML = `
                        <span class="feature-tag all-elements-tag">
                            <span class="feature-name-container">${feature.name} → <span class="copyable-value" data-value="${value}">${value}</span></span>
                            <span class="predicted-salience">计算失败</span>
                        </span>
                      `;
                    }
                  });
              }, 10); // 短暂延迟确保DOM已更新
            } else {
              // 其他特征，显示推荐值
              const unit = isWidthFeature(featureKey) ? 'px' : '';
              const value = `${feature.usedValue.toFixed(2)}${unit}`;
              analysis += `
                <div class="feature-item" data-feature-key="${featureKey}" data-salience="${feature.predictedSalience}">
                    <span class="feature-tag all-elements-tag">
                        <span class="feature-name-container">${feature.name} → <span class="copyable-value" data-value="${value}">${value}</span></span>
                        <span class="predicted-salience">${feature.formattedSalience}</span>
                    </span>
                </div>
              `;
            }
          });
        } else {
          analysis += `<div class="no-selection"><span>No modify visual dimension found</span></div>`;
        }
      } else {
        analysis += `<div class="no-selection"><span>No modify visual dimension found</span></div>`;
      }

      analysis += `</div>`; // 关闭内容单元格
      analysis += `</div>`; // 关闭表格行

      // 3. Add Annotations 区域 - 修改为表格式行布局
      analysis += `<div class="suggestions-table-row">`;
      analysis += `<div class="suggestions-section-title">Annotate</div>`;
      analysis += `<div class="suggestions-content-cell">`;

      // 检查高亮元素的位置关系，判断是显示"Add a box"还是"Add a link"
      const areElementsAdjacent = () => {
        try {
          // 获取高亮组元素
          const highlightedNodes = normalData.filter(node =>
            selectedNodeIds.some(id =>
              id === node.id || node.id.endsWith(`/${id}`)
            )
          );

          if (highlightedNodes.length <= 1) {
            // 只有一个元素或没有元素，默认显示"Add a box"
            return true;
          }

          // 提取边界框信息
          const bboxes = highlightedNodes.map(node => {
            const features = node.features;
            if (!features || features.length < 20) return null;

            // 获取边界框特征，使用正确的索引
            // bbox_left_n(11), bbox_right_n(12), bbox_top_n(13), bbox_bottom_n(14)
            return {
              left: features[11],
              right: features[12],
              top: features[13],
              bottom: features[14]
            };
          }).filter(bbox => bbox !== null);

          if (bboxes.length <= 1) {
            return true;
          }

          const threshold = 0.1; // 距离阈值，可以根据需要调整

          // 检查所有可能的元素对
          for (let i = 0; i < bboxes.length; i++) {
            for (let j = i + 1; j < bboxes.length; j++) {
              const bbox1 = bboxes[i];
              const bbox2 = bboxes[j];

              // 检查是否有重叠
              const hasOverlap = !(
                bbox1.right < bbox2.left ||
                bbox1.left > bbox2.right ||
                bbox1.bottom < bbox2.top ||
                bbox1.top > bbox2.bottom
              );

              if (!hasOverlap) {
                // 如果没有重叠，计算距离
                const horizontalDistance = Math.max(0,
                  Math.max(bbox1.left - bbox2.right, bbox2.left - bbox1.right)
                );

                const verticalDistance = Math.max(0,
                  Math.max(bbox1.top - bbox2.bottom, bbox2.top - bbox1.bottom)
                );

                // 使用欧几里得距离
                const distance = Math.sqrt(
                  horizontalDistance * horizontalDistance +
                  verticalDistance * verticalDistance
                );

                // 如果任何一对元素的距离大于阈值，就返回false
                if (distance >= threshold) {
                  return false;
                }
              }
            }
          }

          // 如果所有元素对都相邻或重叠，返回true
          return true;
        } catch (error) {
          console.error('检查元素位置关系时出错:', error);
          return true; // 出错时默认显示"Add a box"
        }
      };

      // 根据元素位置关系决定显示的文本
      const annotationText = areElementsAdjacent() ? 'by a box' : 'by links';

      // 添加动态文本
      analysis += `
                  <div class="feature-item">
                    <span class="feature-tag all-elements-tag annotation-tag">
                        <span class="copyable-value" data-value="${annotationText}">${annotationText}</span>
                    </span>
                  </div>
              `;

      analysis += `</div>`; // 关闭内容单元格
      analysis += `</div>`; // 关闭表格行

      analysis += `</div>`; // 关闭共享容器
    } else {
      // 当visual salience值大于等于85时显示的信息，使用更紧凑的样式
      // 检查currentVisualSalience是否为NaN，如果是则不显示该信息
      if (!isNaN(currentVisualSalience)) {
        analysis += `
                <div class="high-salience-notice">
                    <div class="salience-icon">✓</div>
                    <div class="salience-content">
                        <div class="salience-row">
                            <div class="salience-title">Visual salience is already good</div>
                            <div class="salience-value">${currentVisualSalience.toFixed(3)}</div>
                        </div>
                    </div>
                </div>
            `;
      }
    }

    analysis += `</div>`; // 关闭内容容器
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
      });

    // 筛选出方差较小的特征，作为可用但未充分利用的特征
    const leastDistinctive = featureArray
      .filter(feature =>
        feature.variance <= 0.001 || // 方差小，变化不明显
        !feature.hasNonZeroValues // 或者没有非零值
      )
      .sort((a, b) => {
        // 按方差从小到大排序
        return a.variance - b.variance;
      });

    // 处理冲突关系，筛选出不冲突的特征
    const processedDiverseFeatures = [];
    const processedLeastDistinctive = [];
    const usedConflictGroups = new Set(); // 用于记录已使用的冲突组

    // 处理多样性高的特征（Used dimensions
    // 修改：只统计 selectedNodeIds 中的元素而不是所有元素
    if (selectedNodeIds && selectedNodeIds.length > 0) {
      // 过滤出仅存在于 selectedNodeIds 中的元素
      const filteredData = normalData.filter(node =>
        selectedNodeIds.some(id => node.id === id || node.id.endsWith(`/${id}`))
      );

      // 重新计算每个特征在选中元素中的方差和多样性
      const filteredFeatureStats = {};

      // 初始化特征统计数据
      Object.keys(featureGroups).forEach(displayName => {
        filteredFeatureStats[displayName] = {
          values: [],
          featureIndices: featureGroups[displayName].indices,
          featureKeys: featureGroups[displayName].keys,
          hasNonZeroValues: false,
          variance: 0,
          uniqueValues: new Set()
        };
      });

      // 收集所有特征值
      filteredData.forEach(node => {
        Object.keys(filteredFeatureStats).forEach(displayName => {
          const featureIndices = filteredFeatureStats[displayName].featureIndices;

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
            filteredFeatureStats[displayName].values.push(value);

            // 收集唯一值
            filteredFeatureStats[displayName].uniqueValues.add(value.toFixed(4));

            // 检查是否有非零值
            if (value > 0) {
              filteredFeatureStats[displayName].hasNonZeroValues = true;
            }
          }
        });
      });

      // 计算每个特征的统计数据
      Object.keys(filteredFeatureStats).forEach(displayName => {
        const feature = filteredFeatureStats[displayName];

        // 跳过没有值的特征
        if (feature.values.length === 0) {
          delete filteredFeatureStats[displayName];
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

        // 计算唯一值数量
        feature.uniqueValueCount = feature.uniqueValues.size;
      });

      // 转换为数组进行排序
      const filteredFeatureArray = Object.entries(filteredFeatureStats)
        .map(([name, stats]) => ({
          name,
          ...stats
        }));

      // 使用选中节点的元素进行分析，筛选出有变化的特征
      const filteredDiverseFeatures = filteredFeatureArray
        .filter(feature =>
          feature.variance > 0.001 && // 确保有明显变化
          feature.hasNonZeroValues // 确保有非零值
        )
        .sort((a, b) => {
          // 按方差大小排序
          return b.variance - a.variance;
        });

      // 处理冲突关系，筛选出不冲突的特征
      for (const feature of filteredDiverseFeatures) {
        const featureKey = feature.featureKeys[0];
        const group = getConflictGroup(featureKey);

        if (group) {
          // 如果特征属于某个冲突组
          const groupKey = group.join(',');

          // 如果该冲突组已经有特征被选中，则跳过
          if (usedConflictGroups.has(groupKey)) {
            continue;
          }

          // 找出该组中多样性最高的特征
          const highestDiversityFeature = filteredDiverseFeatures
            .filter(f => group.includes(f.featureKeys[0]))
            .sort((a, b) => b.variance - a.variance)[0];

          // 如果当前特征是多样性最高的，则保留
          if (feature === highestDiversityFeature) {
            processedDiverseFeatures.push(feature);
            usedConflictGroups.add(groupKey);
          }
        } else {
          // 如果不是冲突组中的特征，直接保留
          processedDiverseFeatures.push(feature);
        }
      }
    } else {
      // 如果没有选中节点，则按原来的方式处理所有元素
      for (const feature of diverseFeatures) {
        const featureKey = feature.featureKeys[0];
        const group = getConflictGroup(featureKey);

        if (group) {
          // 如果特征属于某个冲突组
          const groupKey = group.join(',');

          // 如果该冲突组已经有特征被选中，则跳过
          if (usedConflictGroups.has(groupKey)) {
            continue;
          }

          // 找出该组中多样性最高的特征
          const highestDiversityFeature = diverseFeatures
            .filter(f => group.includes(f.featureKeys[0]))
            .sort((a, b) => b.variance - a.variance)[0];

          // 如果当前特征是多样性最高的，则保留
          if (feature === highestDiversityFeature) {
            processedDiverseFeatures.push(feature);
            usedConflictGroups.add(groupKey);
          }
        } else {
          // 如果不是冲突组中的特征，直接保留
          processedDiverseFeatures.push(feature);
        }
      }
    }

    // 重置冲突组记录，为Available effects
    usedConflictGroups.clear();

    // 处理多样性低的特征（Available effects
    for (const feature of leastDistinctive) {
      const featureKey = feature.featureKeys[0];
      const group = getConflictGroup(featureKey);

      if (group) {
        // 如果特征属于某个冲突组
        const groupKey = group.join(',');

        // 如果该冲突组已经有特征被选中，则跳过
        if (usedConflictGroups.has(groupKey)) {
          continue;
        }

        // 使用优先级规则
        const priorityKey = getPriorityFeature(group);

        // 如果当前特征是优先级最高的，则保留
        if (featureKey === priorityKey) {
          processedLeastDistinctive.push(feature);
          usedConflictGroups.add(groupKey);
        }
      } else {
        // 如果不是冲突组中的特征，直接保留
        processedLeastDistinctive.push(feature);
      }
    }

    // 重置冲突组记录，为Available effects
    usedConflictGroups.clear();

    // 处理多样性低的特征（Available effects
    const featuresWithSalience = [];

    for (const feature of leastDistinctive) {
      const featureKey = feature.featureKeys[0];
      const group = getConflictGroup(featureKey);

      if (group) {
        // 如果特征属于某个冲突组
        const groupKey = group.join(',');

        // 如果该冲突组已经有特征被选中，则跳过
        if (usedConflictGroups.has(groupKey)) {
          continue;
        }

        // 使用优先级规则
        const priorityKey = getPriorityFeature(group);

        // 如果当前特征是优先级最高的，则保留并计算预估显著性
        if (featureKey === priorityKey) {
          // 计算预估显著性
          let predictedSalience = 0;
          if (isSelectedNodes && selectedNodeIds && selectedNodeIds.length > 0) {
            predictedSalience = predictVisualSalience(
              selectedNodeIds.map(id => ({ id })),
              normalData,
              featureKey
            );
          }

          // 保存特征和预估显著性
          featuresWithSalience.push({
            ...feature,
            predictedSalience: predictedSalience
          });

          usedConflictGroups.add(groupKey);
        }
      } else {
        // 如果不是冲突组中的特征，直接保留并计算预估显著性
        let predictedSalience = 0;
        if (isSelectedNodes && selectedNodeIds && selectedNodeIds.length > 0) {
          predictedSalience = predictVisualSalience(
            selectedNodeIds.map(id => ({ id })),
            normalData,
            featureKey
          );
        }

        // 保存特征和预估显著性
        featuresWithSalience.push({
          ...feature,
          predictedSalience: predictedSalience
        });
      }
    }

    // 按预估显著性降序排序
    featuresWithSalience.sort((a, b) => b.predictedSalience - a.predictedSalience);

    // 使用按显著性排序后的特征
    processedLeastDistinctive.length = 0; // 清空原数组
    processedLeastDistinctive.push(...featuresWithSalience);

    // 限制显示数量
    const finalDiverseFeatures = processedDiverseFeatures.slice(0, 20);
    const finalLeastDistinctive = processedLeastDistinctive.slice(0, 10);

    // 按uniqueValueCount从大到小排序finalDiverseFeatures
    finalDiverseFeatures.sort((a, b) => b.uniqueValueCount - a.uniqueValueCount);

    // 生成HTML
    let analysis = '<div class="feature-columns">';

    // 最突出的特征 - Used Features
    analysis += '<div class="feature-column negative all-elements">';
    analysis += `<div class="column-title all-elements-title">Used dimensions <span class="distinct-values-label"><span class="hash-symbol">#</span><span class="label-text">Distinct<br>effects</span></span></div>`;
    analysis += `<div class="column-content">`; // 添加内容容器

    if (finalDiverseFeatures.length > 0) {
      // 创建一个包装容器用于单列布局
      analysis += `<div class="single-column-wrapper">`;

      // 使用单列布局显示特征
      for (let i = 0; i < finalDiverseFeatures.length; i++) {
        analysis += `
                      <div class="feature-item">
                          <span class="feature-tag all-elements-tag" style="color: #555555; border-color: #55555530; background-color: #f5f5f5">
                              ${finalDiverseFeatures[i].name}<span class="value-count">${finalDiverseFeatures[i].uniqueValueCount}</span>
                          </span>
                      </div>
                  `;
      }

      analysis += `</div>`;
    } else {
      analysis += `<div class="no-selection"><span>No distinguishing features found</span></div>`;
    }

    analysis += `</div>`; // 关闭内容容器
    analysis += '</div>';

    // 最不突出的特征 - Available dimensions
    analysis += '<div class="feature-column positive all-elements">';
    analysis += `<div class="column-title all-elements-title">Available dimensions</div>`;
    analysis += `<div class="column-content">`; // 添加内容容器

    if (finalLeastDistinctive.length > 0) {
      // 创建一个包装容器用于单列布局
      analysis += `<div class="single-column-wrapper">`;

      finalLeastDistinctive.forEach(feature => {
        analysis += `
                    <div class="feature-item">
                          <span class="feature-tag all-elements-tag" style="color: #555555; border-color: #55555530; background-color: #f5f5f5">
                            ${feature.name}
                        </span>
                    </div>
                `;
      });

      analysis += `</div>`;
    } else {
      analysis += `<div class="no-selection"><span>No usable features found</span></div>`;
    }

    analysis += `</div>`; // 关闭内容容器
    analysis += '</div>';
    analysis += '</div>';

    return analysis;
  }
};

// 获取数据并生成分析
const fetchDataAndGenerateAnalysis = async () => {
  try {
    // 先确保获取最新的normalized数据
    await fetchNormalizedData();

    if (!normalData.value || !Array.isArray(normalData.value) || normalData.value.length === 0) {
      // 如果还没有数据，尝试从API获取
      const response = await axios.get(NORMAL_DATA_URL);

      if (!response.data) {
        throw new Error('Problems with network response');
      }

      // 保存原始特征数据
      normalData.value = response.data;
    }

    // 获取选中节点的ID
    const selectedNodeIds = store.state.selectedNodes.nodeIds || [];

    // 生成全局分析文字 - 不需要传入选中节点ID
    analysisContent.value = generateAnalysis(normalData.value, false);

    // 生成选中节点的分析 - 传入选中节点ID
    if (selectedNodeIds && selectedNodeIds.length > 0) {
      selectedNodesAnalysis.value = generateAnalysis(normalData.value, true, selectedNodeIds);
    } else {
      selectedNodesAnalysis.value = '<div class="no-selection"><span>Please select a node to view the analysis...</span></div>';
    }
  } catch (error) {
    console.error('Failed to get data:', error);
    analysisContent.value = '<div class="no-selection"><span>Analysis generation failed, please try again</span></div>';
    selectedNodesAnalysis.value = '<div class="no-selection"><span>Analysis generation failed, please try again</span></div>';
  }
};

// 修改监听逻辑，当选中节点变化时重新获取数据
watch(() => store.state.selectedNodes.nodeIds, () => {
  fetchDataAndGenerateAnalysis();
}, { deep: true, immediate: true });

// 添加对 visualSalience 的监听，当它变化时重新生成分析内容
watch(() => store.state.visualSalience, (newValue, oldValue) => {
  console.log('visualSalience changed:', oldValue, '->', newValue);
  // 如果有选中的节点，重新生成分析内容
  if (store.state.selectedNodes.nodeIds && store.state.selectedNodes.nodeIds.length > 0) {
    fetchDataAndGenerateAnalysis();
  }
});

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
  // 添加全局事件监听，因为v-html渲染的内容不能直接绑定Vue事件
  document.addEventListener('mouseover', globalMouseOverHandler);
  document.addEventListener('mouseout', globalMouseOutHandler);
  document.addEventListener('click', globalClickHandler);

  window.addEventListener('svg-content-updated', handleSvgUpdate);
  // 初始获取数据
  fetchDataAndGenerateAnalysis();

  // 初始化滚动检测
  setTimeout(() => {
    initScrollDetection();
  }, 100); // 给足够时间让内容渲染
});

onUnmounted(() => {
  // 移除全局事件监听器
  document.removeEventListener('mouseover', globalMouseOverHandler);
  document.removeEventListener('mouseout', globalMouseOutHandler);
  document.removeEventListener('click', globalClickHandler);
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

// 添加颜色特征的逆向归一化函数，添加HSL完整格式的支持
const denormalizeColorFeatures = (featureKey, value) => {
  // 根据特征类型进行逆向归一化
  if (featureKey === 'fill_h_cos' || featureKey === 'fill_h_sin') {
    // 从cos和sin值逆向计算色相角度
    // 注意：需要同时有cos和sin才能准确计算，这里是近似
    if (featureKey === 'fill_h_cos') {
      // 从cos值逆向计算
      const cos_value = value * 2 - 1; // 逆向 (cos + 1) / 2
      // acos返回0-π的值，需要根据sin值确定是在0-180还是180-360
      // 由于我们只有一个值，这里做近似处理
      const angle_rad = Math.acos(cos_value);
      return Math.round((angle_rad * 180 / Math.PI) % 360);
    } else {
      // 从sin值逆向计算
      const sin_value = value * 2 - 1; // 逆向 (sin + 1) / 2
      // asin返回-π/2到π/2的值，需要调整到0-360
      let angle_rad = Math.asin(sin_value);
      if (angle_rad < 0) {
        angle_rad = 2 * Math.PI + angle_rad;
      }
      return Math.round((angle_rad * 180 / Math.PI) % 360);
    }
  } else if (featureKey === 'stroke_h_cos' || featureKey === 'stroke_h_sin') {
    // 从cos和sin值逆向计算色相角度，与fill_h类似
    if (featureKey === 'stroke_h_cos') {
      const cos_value = value * 2 - 1;
      const angle_rad = Math.acos(cos_value);
      return Math.round((angle_rad * 180 / Math.PI) % 360);
    } else {
      const sin_value = value * 2 - 1;
      let angle_rad = Math.asin(sin_value);
      if (angle_rad < 0) {
        angle_rad = 2 * Math.PI + angle_rad;
      }
      return Math.round((angle_rad * 180 / Math.PI) % 360);
    }
  } else if (featureKey === 'fill_s_n' || featureKey === 'stroke_s_n') {
    // 饱和度逆向归一化：乘以100
    return Math.round(value * 100);
  } else if (featureKey === 'fill_l_n' || featureKey === 'stroke_l_n') {
    // 亮度逆向归一化：乘以100
    return Math.round(value * 100);
  }

  // 对于其他特征，返回原值
  return value;
};

// 添加HSL转RGB的函数
const hslToRgb = (h, s, l) => {
  // 将h, s, l转换为0-1范围
  h = h / 360;
  s = s / 100;
  l = l / 100;

  let r, g, b;

  if (s === 0) {
    r = g = b = l; // 灰度
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }

  // 转换为0-255范围
  return {
    r: Math.round(r * 255),
    g: Math.round(g * 255),
    b: Math.round(b * 255)
  };
};

// 修改获取颜色值的函数，返回RGB格式
const getCompleteColorValue = (featureKey, value, normalData, selectedNodeIds) => {
  // 确定是fill还是stroke
  const isStroke = featureKey.startsWith('stroke');
  const colorType = isStroke ? 'stroke' : 'fill';

  // 获取高亮组元素
  const highlightedNodes = normalData.filter(node =>
    selectedNodeIds.some(id =>
      id === node.id || node.id.endsWith(`/${id}`)
    )
  );

  if (highlightedNodes.length === 0) {
    return null;
  }

  // 获取特征索引
  const h_cos_index = getFeatureIndex(`${colorType}_h_cos`);
  const h_sin_index = getFeatureIndex(`${colorType}_h_sin`);
  const s_index = getFeatureIndex(`${colorType}_s_n`);
  const l_index = getFeatureIndex(`${colorType}_l_n`);

  // 计算平均值
  let h_cos_sum = 0, h_sin_sum = 0, s_sum = 0, l_sum = 0;
  let count = 0;

  highlightedNodes.forEach(node => {
    if (node.features && node.features.length >= Math.max(h_cos_index, h_sin_index, s_index, l_index) + 1) {
      h_cos_sum += node.features[h_cos_index];
      h_sin_sum += node.features[h_sin_index];
      s_sum += node.features[s_index];
      l_sum += node.features[l_index];
      count++;
    }
  });

  if (count === 0) {
    return null;
  }

  // 计算平均值
  const h_cos_avg = h_cos_sum / count;
  const h_sin_avg = h_sin_sum / count;
  const s_avg = s_sum / count;
  const l_avg = l_sum / count;

  // 计算色相角度
  const h_cos_value = h_cos_avg * 2 - 1;
  const h_sin_value = h_sin_avg * 2 - 1;
  let h_angle = Math.atan2(h_sin_value, h_cos_value) * 180 / Math.PI;
  if (h_angle < 0) {
    h_angle += 360;
  }

  // 确定使用的是q1还是q3
  let h_value, s_value, l_value;

  // 为stroke颜色设置默认值的阈值
  const isZeroSaturation = isStroke && Math.abs(s_avg) < 0.01;
  const isZeroLightness = isStroke && (Math.abs(l_avg) < 0.01 || Math.abs(l_avg - 1) < 0.01);
  const isZeroHue = isStroke && Math.abs(h_angle) < 0.01;
  
  // 为stroke颜色设置默认饱和度和亮度值，
  const defaultHue = 75;        // 默认色相50度
  const defaultSaturation = 20; // 默认饱和度100%
  const defaultLightness = 15;  // 默认亮度55%

  if (featureKey.includes('_h_')) {
    // 如果是色相特征
    h_value = denormalizeColorFeatures(featureKey, value);
    // 使用默认值替代接近0的饱和度和亮度
    s_value = isZeroSaturation ? defaultSaturation : Math.round(s_avg * 100);
    l_value = isZeroLightness ? defaultLightness : Math.round(l_avg * 100);
  } else if (featureKey.includes('_s_')) {
    // 如果是饱和度特征
    h_value = isZeroHue ? defaultHue : Math.round(h_angle); // 使用默认色相或平均色相
    s_value = denormalizeColorFeatures(featureKey, value);
    // 使用默认值替代接近0的亮度
    l_value = isZeroLightness ? defaultLightness : Math.round(l_avg * 100);
  } else if (featureKey.includes('_l_')) {
    // 如果是亮度特征
    h_value = isZeroHue ? defaultHue : Math.round(h_angle); // 使用默认色相或平均色相
    // 使用默认值替代接近0的饱和度
    s_value = isZeroSaturation ? defaultSaturation : Math.round(s_avg * 100);
    l_value = denormalizeColorFeatures(featureKey, value);
  }

  // 将HSL转换为RGB
  const rgb = hslToRgb(h_value, s_value, l_value);

  // return `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
  return `rgb(110, 90, 50)`;
};

// 检查特征是否为颜色相关特征
const isColorFeature = (featureKey) => {
  const colorFeatures = [
    'fill_h_cos', 'fill_h_sin', 'fill_s_n', 'fill_l_n',
    'stroke_h_cos', 'stroke_h_sin', 'stroke_s_n', 'stroke_l_n'
  ];
  return colorFeatures.includes(featureKey);
};

// 检查特征是否为位置或bbox相关特征
const isPositionOrBboxFeature = (featureKey) => {
  const positionFeatures = [
    'bbox_left_n', 'bbox_right_n', 'bbox_top_n', 'bbox_bottom_n',
    'bbox_width_n', 'bbox_height_n', 'bbox_fill_area'
  ];
  return positionFeatures.includes(featureKey);
};

// 检查特征是否为需要过滤的MDS特征
const isMdsFeature = (featureKey) => {
  return featureKey === 'bbox_mds_1' || featureKey === 'bbox_mds_2';
};

// 添加计算颜色多样性的函数
function calculateColorDiversity(featureStats, colorType) {
  // 获取颜色的各个组成部分
  const h_cos_key = `${colorType}_h_cos`;
  const h_sin_key = `${colorType}_h_sin`;
  const s_key = `${colorType}_s_n`;
  const l_key = `${colorType}_l_n`;

  // 检查所有组成部分是否都存在
  if (!featureStats[h_cos_key] || !featureStats[h_sin_key] ||
    !featureStats[s_key] || !featureStats[l_key]) {
    return null;
  }

  // 获取各个组成部分的多样性（使用variance作为多样性指标）
  const h_cos_diversity = featureStats[h_cos_key].variance;
  const h_sin_diversity = featureStats[h_sin_key].variance;
  const s_diversity = featureStats[s_key].variance;
  const l_diversity = featureStats[l_key].variance;

  // 取最大值作为颜色的整体多样性
  return Math.max(h_cos_diversity, h_sin_diversity, s_diversity, l_diversity);
}

// 添加颜色选择器配置
const colorPickerOptions = {
  predefine: [
    'rgb(255, 69, 0)',
    'rgb(255, 140, 0)',
    'rgb(255, 215, 0)',
    'rgb(144, 238, 144)',
    'rgb(0, 206, 209)',
    'rgb(30, 144, 255)',
    'rgb(199, 21, 133)',
    'rgb(255, 105, 180)',
    'rgb(205, 92, 92)',
    'rgb(0, 0, 0)',
    'rgb(255, 255, 255)',
    'rgb(128, 128, 128)'
  ]
};

// 添加一个直接颜色选择器的状态
const directColorPicker = ref({
  visible: false,
  x: 0,
  y: 0,
  currentValue: '',
  targetElement: null
});

// 添加一个直接颜色选择器的引用
const directColorPickerRef = ref(null);

// 添加一个直接数值编辑器的状态
const directNumberEditor = ref({
  visible: false,
  x: 0,
  y: 0,
  currentValue: 0,
  hasUnit: false,
  unit: '',
  targetElement: null
});

// 获取直接颜色选择器的样式
const getDirectColorPickerStyle = () => {
  if (!directColorPicker.value.visible) return {};

  return {
    position: 'fixed',
    left: `${directColorPicker.value.x}px`,
    top: `${directColorPicker.value.y}px`,
    zIndex: 1000000,
    opacity: 0 // 设置为透明，只显示弹出的选择器
  };
};

// 获取直接数值编辑器的样式
const getDirectNumberEditorStyle = () => {
  if (!directNumberEditor.value.visible) return {};

  return {
    position: 'fixed',
    left: `${directNumberEditor.value.x}px`,
    top: `${directNumberEditor.value.y}px`,
    zIndex: 1000000
  };
};

// 处理直接颜色变化
const handleDirectColorChange = (color) => {
  if (directColorPicker.value.targetElement) {
    // 将颜色转换为 rgb 格式
    let rgbColor = color;
    if (color.startsWith('rgba')) {
      const matches = color.match(/rgba\((\d+),\s*(\d+),\s*(\d+)/);
      if (matches) {
        rgbColor = `rgb(${matches[1]}, ${matches[2]}, ${matches[3]})`;
      }
    }

    // 更新目标元素的颜色值
    const targetElement = directColorPicker.value.targetElement;
    targetElement.setAttribute('data-value', rgbColor);
    targetElement.textContent = rgbColor;
    
    // 更新预览块的背景颜色
    const previewBlock = targetElement.previousElementSibling;
    if (previewBlock && previewBlock.classList.contains('color-preview-inline')) {
      previewBlock.style.backgroundColor = rgbColor;
    }
    
    // 重新计算预测显著性
    // 首先找到特征名称
    const featureNameContainer = targetElement.closest('.feature-name-container');
    if (featureNameContainer) {
      const featureName = featureNameContainer.textContent.split('→')[0].trim();
      
      // 构造属性对象用于API调用
      let attributes = {};
      if (featureName.includes('fill')) {
        attributes = {"fill": rgbColor};
      } else if (featureName.includes('stroke')) {
        attributes = {"stroke": rgbColor};
      }
      
      // 找到显示显著性的元素
      const featureItem = targetElement.closest('.feature-item');
      if (featureItem) {
        const predSalienceElement = featureItem.querySelector('.predicted-salience');
        if (predSalienceElement) {
          // 更新为"wait..."
          predSalienceElement.textContent = "wait...";
          
          // 调用API计算显著性
          calculateSuggestionSalience(store.state.selectedNodes.nodeIds, attributes)
            .then(salience => {
              // 格式化显著性分值
              const formattedSalience = (salience * 100).toFixed(1);
              
              // 更新DOM元素显示计算的显著性分值
              predSalienceElement.textContent = formattedSalience;
              
              // 更新父元素的data-salience属性
              featureItem.setAttribute('data-salience', salience);
              
              // 获取该特征所在的容器
              const container = featureItem.closest('.suggestions-content-cell');
              if (container) {
                // 获取容器中所有feature-item
                const items = Array.from(container.querySelectorAll('.feature-item'));
                
                // 按照salience属性从高到低对items进行排序
                items.sort((a, b) => {
                  const salienceA = parseFloat(a.getAttribute('data-salience') || '0');
                  const salienceB = parseFloat(b.getAttribute('data-salience') || '0');
                  return salienceB - salienceA;
                });
                
                // 重新添加排序后的元素
                items.forEach(item => {
                  container.appendChild(item);
                });
              }
            })
            .catch(error => {
              console.error('无法计算显著性:', error);
              // 更新DOM元素显示错误信息
              predSalienceElement.textContent = "计算失败";
            });
        }
      }
    }
    
    // 关闭颜色选择器
    setTimeout(() => {
      directColorPicker.value.visible = false;
    }, 100);
  }
};

// 处理直接数值变化
const handleDirectNumberChange = (value) => {
  if (directNumberEditor.value.targetElement) {
    // 处理带单位的值
    let formattedValue = value;
    if (directNumberEditor.value.hasUnit) {
      formattedValue = `${value}${directNumberEditor.value.unit}`;
    }

    // 更新可复制的值
    const targetElement = directNumberEditor.value.targetElement;
    targetElement.setAttribute('data-value', formattedValue);
    
    // 处理显示格式
    const featureNameContainer = targetElement.closest('.feature-name-container');
    const featureName = featureNameContainer ? featureNameContainer.textContent.split('→')[0].trim() : '';
    
    if (featureName && 
        (featureName.includes('stroke width') || 
         featureName.includes('area'))) {
      targetElement.textContent = `+${formattedValue}`;
    } else {
      targetElement.textContent = formattedValue;
    }
    
    // 重新计算预测显著性
    if (featureNameContainer) {
      // 构造属性对象用于API调用
      let attributes = {};
      if (featureName.includes('stroke width')) {
        // 对于stroke-width，只传数值，不加px单位
        attributes = {"stroke-width": String(value + 1)};
      } else if (featureName.includes('area')) {
        // 对于area，直接使用用户修改的值
        attributes = {"area": value};
      } else if (featureName.includes('width')) {
        attributes = {"width": value};
      } else if (featureName.includes('height')) {
        attributes = {"height": value};
      }
      
      // 找到显示显著性的元素
      const featureItem = targetElement.closest('.feature-item');
      if (featureItem && Object.keys(attributes).length > 0) {
        const predSalienceElement = featureItem.querySelector('.predicted-salience');
        if (predSalienceElement) {
          // 更新为"计算中..."
          predSalienceElement.textContent = "wait...";
          
          // 调用API计算显著性
          calculateSuggestionSalience(store.state.selectedNodes.nodeIds, attributes)
            .then(salience => {
              // 格式化显著性分值
              const formattedSalience = (salience * 100).toFixed(1);
              
              // 更新DOM元素显示计算的显著性分值
              predSalienceElement.textContent = formattedSalience;
              
              // 更新父元素的data-salience属性
              featureItem.setAttribute('data-salience', salience);
              
              // 获取该特征所在的容器
              const container = featureItem.closest('.suggestions-content-cell');
              if (container) {
                // 获取容器中所有feature-item
                const items = Array.from(container.querySelectorAll('.feature-item'));
                
                // 按照salience属性从高到低对items进行排序
                items.sort((a, b) => {
                  const salienceA = parseFloat(a.getAttribute('data-salience') || '0');
                  const salienceB = parseFloat(b.getAttribute('data-salience') || '0');
                  return salienceB - salienceA;
                });
                
                // 重新添加排序后的元素
                items.forEach(item => {
                  container.appendChild(item);
                });
              }
            })
            .catch(error => {
              console.error('Unable to calculate significance:', error);
              // 更新DOM元素显示错误信息
              predSalienceElement.textContent = "failure of calculation";
            });
        }
      }
    }
    
    // 关闭数值编辑器
    setTimeout(() => {
      directNumberEditor.value.visible = false;
    }, 100);
  }
};

// 关闭直接数值编辑器
const closeDirectNumberEditor = () => {
  directNumberEditor.value.visible = false;
};

// 计算建议的显著性分值
const calculateSuggestionSalience = async (ids, attributes) => {
  if (!ids || ids.length === 0 || !attributes) {
    return 0;
  }

  try {
    // 处理attributes
    const processedAttributes = { ...attributes };
    
    // 特殊处理stroke-width属性，移除px单位
    if (processedAttributes['stroke-width'] && typeof processedAttributes['stroke-width'] === 'string') {
      // 如果stroke-width是字符串且包含px，移除px
      processedAttributes['stroke-width'] = processedAttributes['stroke-width'].replace('px', '');
    }
    
    // 构造请求数据
    const requestData = {
      svg_file: window.localStorage.getItem('currentSvgName'),
      modify_elements: [{
        ids: ids,
        attributes: processedAttributes
      }],
      debug: true
    };

    console.log('发送显著性预测请求:', requestData);

    // 发送请求到预测API
    const response = await fetch(SALIENCE_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      throw new Error(`API响应错误: ${response.status}`);
    }

    const result = await response.json();
    console.log('获取显著性预测结果:', result);
    
    // 返回计算的显著性分值
    return result.salience;
  } catch (error) {
    console.error('计算建议显著性时出错:', error);
    return 0;
  }
};
</script>

<style scoped>
.analysis-words-container {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
  border: 1px solid rgba(200, 200, 200, 0.2);
  padding: 10px;
  /* 减小内边距 */
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  height: 300px;
  /* 减小整体高度 */
  width: 100%;
  /* 确保容器占满可用宽度 */
  display: flex;
  flex-direction: column;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
  position: relative;
}

/* 修改标题样式，移除绝对定位 */
.title {
  font-size: 1.8em;
  /* 减小字体大小 */
  font-weight: bold;
  color: #1d1d1f;
  letter-spacing: -0.01em;
  opacity: 0.8;
  /* 减小底部边距 */
}

.sections-container {
  display: flex;
  gap: 10px;
  /* 减小间距 */
  height: 100%;
  overflow: hidden;
}

.section-wrapper {
  display: flex;
  flex-direction: column;
  flex: 1;
  position: relative;
  /* 添加相对定位 */
}

/* 新增：section-header样式 */
.section-header {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 36px;
  /* 增加宽度，从24px改为36px */
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
  /* 增大字体大小，从1.2em改为1.4em */
  font-weight: 700;
  color: #333;
  letter-spacing: 0.04em;
  width: 80px;
  /* 减小宽度，从120px改为80px */
  display: flex;
  align-items: center;
  justify-content: center;
}

.section {
  flex: 1;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.5);
  border: 1px solid rgba(200, 200, 200, 0.2);
  padding: 8px 8px 8px 42px;
  /* 增加左内边距，从30px改为42px，以适应更宽的侧边栏 */
  overflow: hidden;
  position: relative;
}

.feature-section {
  min-width: 0;
}

.analysis-content {
  height: 100%;
  overflow: auto;
  position: relative;
  z-index: 1;
  /* 确保内容在阴影之上可以滚动 */
}

/* 添加滚动阴影遮盖器样式 */
.shadow-overlay {
  position: absolute;
  left: 42px;
  /* 修改左侧位置，从30px改为42px，避免覆盖侧边栏 */
  right: 0;
  height: 20px;
  pointer-events: none;
  /* 允许鼠标事件穿透到下面的内容 */
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
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  background-color: transparent !important;
  border: none !important;
  width: 100% !important;
  height: 100% !important;
  padding: 0 !important;
}

:deep(.value-count) {
  font-size: 16px;
  /* 增大字体大小 */
  font-weight: 700;
  color: #333;
  margin-left: 6px;
  /* 增加左边距 */
  background-color: rgba(0, 0, 0, 0.08);
  border-radius: 3px;
  padding: 0px 6px;
  /* 增加内边距 */
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 22px;
  /* 增加最小宽度 */
  height: 22px;
  /* 增加高度 */
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

.apple-button-corner:hover {
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
  gap: 10px;
  /* 增加间距，从6px改为10px */
  max-height: 100%;
  overflow: hidden;
  height: 100%;
}

:deep(.feature-column) {
  display: flex;
  flex-direction: column;
  position: relative;
  /* 添加相对定位，作为sticky标题的参考 */
  overflow: hidden;
  /* 防止内容溢出 */
  height: 100%;
  /* 确保占满整个高度 */
}


:deep(.column-title) {
  font-size: 18px;
  /* 增大字体大小，从16px改为18px */
  font-weight: 600;
  padding: 8px;
  /* 增加内边距，从6px改为8px */
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  color: #333;
  position: sticky;
  top: 0;
  background: rgba(255, 255, 255, 0.95);
  z-index: 5;
  backdrop-filter: blur(5px);
  margin-bottom: 0;
  flex-shrink: 0;
}

/* 添加内容容器，使其可滚动而标题固定 */
:deep(.column-content) {
  flex: 1;
  overflow-y: auto;
  padding-top: 2px;
  /* 减小顶部内边距 */
  height: calc(100% - 30px);
  /* 调整高度 */
  scrollbar-width: thin;
}

/* 自定义滚动条样式 */
:deep(.column-content::-webkit-scrollbar) {
  width: 6px;
  height: 6px;
}

:deep(.column-content::-webkit-scrollbar-track) {
  background: rgba(0, 0, 0, 0.03);
  border-radius: 3px;
}

:deep(.column-content::-webkit-scrollbar-thumb) {
  background: rgba(0, 0, 0, 0.15);
  border-radius: 3px;
  transition: background 0.3s;
}

:deep(.column-content::-webkit-scrollbar-thumb:hover) {
  background: rgba(0, 0, 0, 0.25);
}

/* 调整suggestions-section-title样式，使其不受column-content滚动影响 */
:deep(.suggestions-section-title) {
  position: sticky;
  top: 0;
  background: rgba(255, 255, 255, 0.95);
  z-index: 4;
  /* 低于主标题但高于内容 */
  font-size: 18px;
  /* 增大字体大小，从16px改为18px */
  font-weight: 700;
  color: #703710;
  margin-bottom: 4px;
  padding-bottom: 2px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  line-height: 1.4;
}

/* 调整suggestions容器样式 */
:deep(.suggestions-container) {
  display: flex;
  flex-direction: column;
  /* 改为列布局 */
  gap: 6px;
  /* 保持间距 */
  margin-top: 2px;
  width: 100%;
  min-height: 80px;
  /* 减小最小高度 */
}

/* 修改 Add 区域样式 */
:deep(.suggestions-add-section) {
  flex: 1;
  min-width: 100%;
  /* 改为100%，占满整行 */
  max-width: 100%;
  /* 改为100%，占满整行 */
  border: none;
  border-radius: 0;
  padding: 0;
  background-color: transparent;
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  min-height: 100px;
  /* 确保最小高度足够显示提示文字 */
}

:deep(.suggestions-Modify-section),
:deep(.suggestions-stroke-section) {
  flex: 1;
  /* 平均分配空间 */
  width: 100%;
  /* 占满整行 */
  min-width: 100%;
  max-width: 100%;
  min-height: 35px;
  /* 减小最小高度 */
  border: none;
  border-radius: 0;
  padding: 0;
  /* 移除左侧内边距 */
  background-color: transparent;
  display: flex;
  flex-direction: column;
}

/* 移除右侧区域容器，改为直接在主容器中垂直排列 */
:deep(.suggestions-right-section) {
  display: none !important;
  /* 强制隐藏右侧区域容器 */
}

/* 移除右侧区域的分隔线 */
:deep(.suggestions-right-section::before) {
  display: none !important;
}

:deep(.feature-item) {
  display: flex;
  align-items: center;
  padding: 3px 6px;
  /* 增加内边距 */
  border-radius: 4px;
  margin-bottom: 3px;
  /* 增加底部间距 */
  background: transparent;
  transition: all 0.2s ease;
  min-height: 32px;
  /* 增加最小高度 */
  width: 100%;
  /* 确保占满整行 */
  box-sizing: border-box;
  /* 确保内边距不会增加元素总宽度 */
}

:deep(.feature-item:hover) {
  background-color: rgba(0, 0, 0, 0.02);
}

/* 为 All elements 部分的 Used visual effects 添加特殊的 feature-item 样式 */
:deep(.feature-column.negative .feature-item),
:deep(.feature-column.positive .feature-item) {
  padding: 3px 6px;
  /* 增加内边距 */
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
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  background-color: transparent !important;
  border: none !important;
  width: 100% !important;
  height: 100% !important;
  padding: 0 !important;
}

:deep(.no-selection) {
  font-size: 15px;
  /* 增大字体大小 */
  color: #777;
  /* 使用更柔和的灰色 */
  min-height: 36px;
  /* 改为最小高度，允许内容增长 */
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 500;
  padding: 8px 12px;
  /* 增加内边距，确保文本有足够空间 */
  background-color: transparent;
  /* 移除背景色 */
  border-radius: 0;
  /* 移除圆角 */
  margin: 12px 0;
  /* 增加上下外边距，提供更好的空间 */
  border: none;
  /* 移除边框 */
  box-shadow: none;
  /* 移除阴影 */
  text-align: center;
  letter-spacing: 0.01em;
  /* 增加字母间距 */
  font-style: italic;
  /* 使用斜体 */
  width: 100%;
  /* 确保宽度足够 */
  box-sizing: border-box;
  /* 确保内边距不会增加元素总宽度 */
  overflow: visible;
  /* 允许内容溢出 */
  padding: 10px 12px !important;
  font-style: italic !important;
  color: #777 !important;
  background-color: rgba(0, 0, 0, 0.01) !important;
  text-align: center !important;
  width: 100% !important;
  box-sizing: border-box !important;
}

:deep(.no-selection span) {
  position: relative !important;
  display: inline-block !important;
}

:deep(.no-selection span::before),
:deep(.no-selection span::after) {
  content: '' !important;
  display: none !important;
  /* 移除装饰线 */
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
  border-radius: 8px;
  padding: 12px;
  /* 增加内边距 */
  margin: 6px 0;
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
  width: 32px;
  /* 增加宽度 */
  height: 32px;
  /* 增加高度 */
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  /* 增加字体大小 */
  margin-right: 12px;
  /* 增加右边距 */
  flex-shrink: 0;
  box-shadow: 0 2px 6px rgba(76, 175, 80, 0.3);
}

:deep(.salience-content) {
  flex: 1;
}

:deep(.salience-row) {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

:deep(.salience-title) {
  font-size: 16px;
  /* 增加字体大小 */
  font-weight: 600;
  color: #2E7D32;
}

:deep(.salience-value) {
  font-size: 22px;
  /* 增加字体大小 */
  font-weight: 700;
  color: #4CAF50;
  margin-left: 10px;
  /* 添加左边距 */
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
  /* 增大字体大小 */
  padding: 0 8px;
  /* 增加内边距 */
  text-align: left;
  /* 左对齐文本 */
  font-weight: 400;
  /* 移除加粗效果 */
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: space-between;
  /* 文本和百分比分两端 */
  margin: 0;
  transition: all 0.2s ease;
  border-width: 0;
  box-shadow: none;
  height: 100%;
  border-radius: 4px;
  /* 减小圆角 */
  letter-spacing: 0;
  background-color: rgba(0, 0, 0, 0.04);
  /* 添加轻微的背景色 */
  color: #333;
  /* 加深字体颜色，提高对比度 */
}

:deep(.all-elements-tag:hover) {
  background-color: rgba(0, 0, 0, 0.03) !important;
  transform: translateY(-1px);
  box-shadow: none;
  color: #333 !important;
}

/* 添加 All elements 部分的 Used visual effects 标题样式 */
:deep(.all-elements-title) {
  font-size: 1.3em;
  /* 增大字体大小，从1.1em改为1.3em */
  padding: 0px 8px 8px 0;
  margin-bottom: 0;
  font-weight: 600;
  color: #444;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

:deep(.distinct-values-label) {
  font-size: 0.7em;
  font-weight: 500;
  color: #666;
  margin-left: auto;
  display: flex;
  align-items: flex-start;
}

:deep(.hash-symbol) {
  margin-right: 2px;
  line-height: 1.1;
}

:deep(.label-text) {
  line-height: 1.1;
  display: flex;
  flex-direction: column;
}

/* 添加两列布局容器 */
:deep(.two-column-wrapper) {
  display: none;
  /* 隐藏两列布局 */
}

:deep(.two-column-row) {
  display: none;
  /* 隐藏两列行 */
}

:deep(.two-column-item) {
  display: none;
  /* 隐藏两列项 */
}

/* 添加单列布局容器 */
:deep(.single-column-wrapper) {
  display: flex;
  flex-direction: column;
  width: 100%;
  padding: 6px 0;
  /* 增加内边距 */
}

:deep(.single-column-wrapper .feature-item) {
  height: 36px;
  /* 增加高度 */
  width: 100%;
}

:deep(.feature-item) {
  margin-bottom: 0px;
}

/* 三区块布局样式 */
:deep(.suggestions-container) {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 2px;
  width: 100%;
  min-height: 80px;
}

/* 缩小两侧区域的间距 */
:deep(.suggestions-add-section) {
  flex: 1;
  min-width: 100%;
  max-width: 100%;
  border: none;
  border-radius: 0;
  padding: 0;
  background-color: transparent;
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  min-height: 100px;
}

:deep(.suggestions-Modify-section),
:deep(.suggestions-stroke-section) {
  flex: 1;
  width: 100%;
  min-width: 100%;
  max-width: 100%;
  min-height: 35px;
  border: none;
  border-radius: 0;
  padding: 0;
  background-color: transparent;
  display: flex;
  flex-direction: column;
}

:deep(.suggestions-right-section) {
  flex: 1;
  min-width: 48%;
  max-width: 50%;
  display: flex;
  flex-direction: column;
  height: 100%;
  position: relative;
  padding-left: 3px;
}

:deep(.suggestions-right-section::before) {
  background-color: rgba(0, 0, 0, 0.03);
  /* 更淡的分隔线 */
}

/* 调整标题样式 */
:deep(.suggestions-section-title) {
  font-size: 18px;
  /* 增大字体大小，从16px改为18px */
  font-weight: 700;
  color: #703710;
  margin-bottom: 4px;
  padding-bottom: 2px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  position: relative;
  line-height: 1.4;
}

:deep(.feature-item) {
  min-height: 26px;
  margin-bottom: 1px;
}

:deep(.suggestions-add-section .feature-item),
:deep(.suggestions-Modify-section .feature-item),
:deep(.suggestions-stroke-section .feature-item) {
  height: 36px;
  /* 增加高度 */
  display: flex;
  align-items: center;
  width: 100%;
  /* 确保占满整行 */
  box-sizing: border-box;
  /* 确保内边距不会增加元素总宽度 */
}

:deep(.suggestions-add-section .feature-tag),
:deep(.suggestions-Modify-section .feature-tag),
:deep(.suggestions-stroke-section .feature-tag) {
  height: 32px;
  display: flex;
  align-items: center;
  width: 100%;
  /* 确保占满整行 */
  font-size: 17px;
  /* 增大字体大小，从16px改为17px */
  font-weight: 400;
  /* 移除加粗效果 */
  color: #333;
  justify-content: space-between;
  /* 在两端对齐内容 */
  box-sizing: border-box;
  /* 确保内边距不会增加元素总宽度 */
}

:deep(.no-selection) {
  font-size: 15px;
  /* 增大字体大小 */
  color: #777;
  /* 使用更柔和的灰色 */
  min-height: 36px;
  /* 改为最小高度，允许内容增长 */
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 500;
  padding: 8px 12px;
  /* 增加内边距，确保文本有足够空间 */
  background-color: transparent;
  /* 移除背景色 */
  border-radius: 0;
  /* 移除圆角 */
  margin: 12px 0;
  /* 增加上下外边距，提供更好的空间 */
  border: none;
  /* 移除边框 */
  box-shadow: none;
  /* 移除阴影 */
  text-align: center;
  letter-spacing: 0.01em;
  /* 增加字母间距 */
  font-style: italic;
  /* 使用斜体 */
  width: 100%;
  /* 确保宽度足够 */
  box-sizing: border-box;
  /* 确保内边距不会增加元素总宽度 */
  overflow: visible;
  /* 允许内容溢出 */
  padding: 10px 12px !important;
  font-style: italic !important;
  color: #777 !important;
  background-color: rgba(0, 0, 0, 0.01) !important;
  text-align: center !important;
  width: 100% !important;
  box-sizing: border-box !important;
}

:deep(.no-selection span) {
  position: relative !important;
  display: inline-flex !important;
  align-items: center !important;
  flex-wrap: wrap !important;
  /* 允许文本换行 */
  justify-content: center !important;
  /* 居中对齐内容 */
  max-width: 100% !important;
  /* 限制最大宽度 */
  word-break: break-word !important;
  /* 允许在单词内换行 */
}

:deep(.no-selection span::before),
:deep(.no-selection span::after) {
  content: '' !important;
  display: none !important;
  /* 移除装饰线 */
}

/* 分隔线样式 */
:deep(.suggestions-right-section::before) {
  content: '';
  position: absolute;
  left: 0;
  top: 10%;
  height: 80%;
  width: 1px;
  background-color: rgba(0, 0, 0, 0.03);
}

/* 标题指示器 */
:deep(.suggestions-section-title::before) {
  content: '';
  position: absolute;
  left: 0;
  bottom: -1px;
  width: 20px;
  height: 1px;
  background-color: rgba(0, 0, 0, 0.08);
  border-radius: 0;
}

/* 预估显著性样式 */
:deep(.predicted-salience) {
  font-size: 15px !important;
  font-weight: 600 !important;
  color: #703710 !important;
  background-color: rgba(112, 55, 16, 0.05) !important;
  padding: 2px 8px !important;
  border-radius: 4px !important;
  margin-left: auto !important;
}

:deep(.feature-tag .predicted-salience) {
  position: relative;
  top: -1px;
  margin-left: auto;
  /* 确保在 feature-tag 内靠右对齐 */
}

/* 修改左右两侧区域的宽度比例 - All elements部分 */
:deep(.feature-column.positive.all-elements) {
  flex: 1;
  /* 增加Available effects */
}

:deep(.feature-column.negative.all-elements) {
  flex: 1.1;
}

/* 修改左右两侧区域的宽度比例 - Selected elements部分 */
:deep(.feature-column.positive.selected-elements) {
  flex: 1;
}

:deep(.feature-column.negative.selected-elements) {
  flex: 2;
}

.section-wrapper:first-child {
  flex: 1;
}

.section-wrapper:last-child {
  flex: 1.5;
}

:deep(.feature-tag) {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
}

:deep(.feature-name-container) {
  display: flex;
  align-items: center;
}

:deep(.predicted-salience) {
  margin-left: auto;
}

/* 添加共享容器样式 - 修改为纵向表格布局 */
:deep(.suggestions-shared-container) {
  display: flex !important;
  flex-direction: column !important;
  /* 从row改为column */
  justify-content: flex-start !important;
  width: 100% !important;
  gap: 0 !important;
  /* 移除间隙，表格形式更紧凑 */
  margin-top: 2px !important;
  border: 1px solid rgba(0, 0, 0, 0.08) !important;
  /* 添加表格边框 */
  border-radius: 4px !important;
  overflow: hidden !important;
  /* 确保圆角有效 */
}

/* 修改各区域样式，使它们在共享容器中纵向排列成表格形式 */
:deep(.suggestions-add-section),
:deep(.suggestions-Modify-section),
:deep(.suggestions-stroke-section) {
  flex: none !important;
  /* 不使用flex比例 */
  min-width: 100% !important;
  /* 占据全宽 */
  max-width: 100% !important;
  border: none !important;
  border-radius: 0 !important;
  /* 移除圆角，表格样式 */
  padding: 0 !important;
  /* 移除内边距 */
  background-color: transparent !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: visible !important;
  /* 允许内容溢出 */
  margin-bottom: 0 !important;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06) !important;
  /* 添加底部边框分隔表格行 */
}

/* 最后一个区域不需要底部边框 */
:deep(.suggestions-stroke-section) {
  border-bottom: none !important;
}

/* 表头样式 */
:deep(.suggestions-section-title) {
  font-size: 16px !important;
  font-weight: 600 !important;
  color: #333 !important;
  background-color: rgba(0, 0, 0, 0.03) !important;
  /* 表头背景色 */
  padding: 8px 12px !important;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08) !important;
  margin-bottom: 0 !important;
}

/* 特征项目样式，改为表格行样式 */
:deep(.feature-item) {
  min-height: 36px !important;
  border-bottom: 1px solid rgba(0, 0, 0, 0.03) !important;
  margin-bottom: 0 !important;
  padding: 6px 12px !important;
}

:deep(.feature-item:last-child) {
  border-bottom: none !important;
}

/* 让每个区域有独立的表格样式 */
:deep(.suggestions-add-section) {
  border-left: none !important;
}

:deep(.suggestions-Modify-section) {
  border-left: none !important;
  margin-top: 0 !important;
}

:deep(.suggestions-stroke-section) {
  border-left: none !important;
  margin-top: 0 !important;
}

/* 移除之前的垂直布局 */
:deep(.suggestions-container) {
  display: flex !important;
  flex-direction: column !important;
  width: 100% !important;
}

/* 添加共享容器样式 - 修改为表格样式 */
:deep(.suggestions-shared-container) {
  display: flex !important;
  flex-direction: column !important;
  justify-content: flex-start !important;
  width: 100% !important;
  gap: 0 !important;
  margin-top: 2px !important;
  border: 1px solid rgba(0, 0, 0, 0.08) !important;
  border-radius: 4px !important;
  overflow: hidden !important;
}

/* 表格行样式 */
:deep(.suggestions-table-row) {
  display: flex !important;
  flex-direction: row !important;
  /* 行内水平排列 */
  width: 100% !important;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06) !important;
}

:deep(.suggestions-table-row:last-child) {
  border-bottom: none !important;
}

/* 表头样式 - 放在左侧 */
:deep(.suggestions-section-title) {
  font-size: 16px !important;
  /* 从14px增加到16px */
  font-weight: 600 !important;
  color: #333 !important;
  background-color: rgba(0, 0, 0, 0.03) !important;
  padding: 8px 10px !important;
  /* 减小内边距 */
  width: 100px !important;
  /* 从120px减小到80px */
  min-width: 80px !important;
  /* 从120px减小到80px */
  border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
  margin-bottom: 0 !important;
  display: flex !important;
  align-items: center !important;
  flex-shrink: 0 !important;
  word-break: break-word !important;
  /* 允许单词换行 */
  line-height: 1.3 !important;
  /* 增加行高，使多行文本看起来更好 */
}

/* 内容单元格样式 - 放在右侧 */
:deep(.suggestions-content-cell) {
  flex: 1 !important;
  padding: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: visible !important;
}

/* 特征项目样式 */
:deep(.feature-item) {
  min-height: 36px !important;
  border-bottom: 1px solid rgba(0, 0, 0, 0.03) !important;
  margin-bottom: 0 !important;
  padding: 6px 12px !important;
}

:deep(.feature-item:last-child) {
  border-bottom: none !important;
}

/* 为表格行添加悬停效果 */
:deep(.suggestions-table-row:hover) {
  background-color: rgba(0, 0, 0, 0.01) !important;
}

/* 特征标签样式 */
:deep(.feature-tag) {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  background-color: transparent !important;
  border: none !important;
  width: 100% !important;
  height: 100% !important;
  padding: 0 !important;
}

/* 移除之前的区域样式 */
:deep(.suggestions-add-section),
:deep(.suggestions-Modify-section),
:deep(.suggestions-stroke-section) {
  display: none !important;
}

/* 为表格结构添加斑马纹 */
:deep(.suggestions-table-row:nth-child(even)) {
  background-color: rgba(0, 0, 0, 0.01) !important;
}

/* 添加共享容器样式 - 修改为表格样式 */
:deep(.suggestions-shared-container) {
  display: flex !important;
  flex-direction: column !important;
  justify-content: flex-start !important;
  width: 100% !important;
  gap: 0 !important;
  margin-top: 2px !important;
  border: 1px solid rgba(0, 0, 0, 0.08) !important;
  border-radius: 4px !important;
  overflow: hidden !important;
}

/* 添加表格标题样式 */
:deep(.suggestions-table-header) {
  font-size: 16px !important;
  font-weight: 600 !important;
  color: #333 !important;
  background-color: rgba(0, 0, 0, 0.04) !important;
  padding: 10px 12px !important;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08) !important;
  margin-bottom: 0 !important;
}

/* 添加复制提示样式 */
.selection-tooltip {
  position: fixed;
  background-color: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 14px;
  pointer-events: none;
  z-index: 1000;
  transform: translate(-50%, -100%);
  animation: fadeInOut 1.5s ease-in-out;
  white-space: nowrap;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

@keyframes fadeInOut {
  0% {
    opacity: 0;
    transform: translate(-50%, -90%);
  }

  20% {
    opacity: 1;
    transform: translate(-50%, -100%);
  }

  80% {
    opacity: 1;
    transform: translate(-50%, -100%);
  }

  100% {
    opacity: 0;
    transform: translate(-50%, -90%);
  }
}

/* 添加可复制值的样式 */
:deep(.feature-name-container .copyable-value) {
  cursor: pointer;
  transition: background-color 0.2s;
  padding: 2px 4px;
  border-radius: 3px;
}

:deep(.feature-name-container .copyable-value:hover) {
  background-color: rgba(144, 95, 41, 0.1);
}

/* 添加颜色预览块样式 */
:deep(.color-preview-inline) {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 2px;
  margin-right: 4px;
  border: 1px solid rgba(0, 0, 0, 0.1);
  cursor: pointer;
  vertical-align: middle;
  position: relative;
  top: -1px;
}

:deep(.color-preview-inline:hover) {
  border-color: rgba(0, 0, 0, 0.3);
  transform: scale(1.1);
}

/* 为注释标签添加特殊样式 */
:deep(.annotation-tag) {
  font-weight: 400 !important;
  /* 移除加粗效果 */
}

/* 修改多样性数值的样式 */
:deep(.value-count) {
  font-weight: 400 !important;
  /* 移除加粗效果 */
  margin-left: 4px;
}

/* 添加编辑工具提示样式 */
.edit-tooltip {
  position: fixed;
  background-color: rgb(25, 25, 25);
  color: white;
  padding: 12px;
  border-radius: 8px;
  z-index: 1000000 !important;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
  min-width: 200px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  pointer-events: auto !important;
  opacity: 1 !important;
  margin: 0;
  border: 1px solid rgb(45, 45, 45);
  backdrop-filter: blur(8px);
  transition: all 0.2s ease;
}

/* 根据位置设置不同的变换和箭头 */
.edit-tooltip[data-position="top"] {
  transform: translate(-50%, -100%);
}

.edit-tooltip[data-position="bottom"] {
  transform: translate(-50%, 10px);
}

.edit-tooltip[data-position="left"] {
  transform: translate(-100%, -50%);
}

.edit-tooltip[data-position="right"] {
  transform: translate(10px, -50%);
}

/* 箭头样式 */
.edit-tooltip::after {
  content: '';
  position: absolute;
  border: 8px solid transparent;
}

.edit-tooltip[data-position="top"]::after {
  bottom: -16px;
  left: 50%;
  transform: translateX(-50%);
  border-top-color: rgb(25, 25, 25);
}

.edit-tooltip[data-position="bottom"]::after {
  top: -16px;
  left: 50%;
  transform: translateX(-50%);
  border-bottom-color: rgb(25, 25, 25);
}

.edit-tooltip[data-position="left"]::after {
  right: -16px;
  top: 50%;
  transform: translateY(-50%);
  border-left-color: rgb(25, 25, 25);
}

.edit-tooltip[data-position="right"]::after {
  left: -16px;
  top: 50%;
  transform: translateY(-50%);
  border-right-color: rgb(25, 25, 25);
}

.edit-tooltip-header {
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 4px;
  color: #eee;
  width: 100%;
  text-align: center;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.edit-tooltip-footer {
  font-size: 12px;
  color: #aaa;
  margin-top: 4px;
  text-align: center;
  width: 100%;
  padding-top: 6px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.click-to-copy-hint {
  font-style: italic;
}

/* 自定义Element UI组件样式 */
:deep(.el-color-picker__trigger) {
  border-color: rgba(255, 255, 255, 0.2);
}

:deep(.el-input-number) {
  width: 120px;
  background: rgb(45, 45, 45);
  border-radius: 4px;
  border: 1px solid rgb(60, 60, 60);
}

:deep(.el-input-number .el-input__inner) {
  color: white;
  text-align: center;
  background: transparent;
}

:deep(.el-input-number .el-input-number__decrease),
:deep(.el-input-number .el-input-number__increase) {
  background: rgb(55, 55, 55);
  color: white;
  border-color: rgb(60, 60, 60);
}

:deep(.el-select) {
  width: 70px;
}

:deep(.el-select .el-input__inner) {
  color: white;
  background: rgb(45, 45, 45);
  border: 1px solid rgb(60, 60, 60);
  border-radius: 4px;
}

/* 确保下拉菜单在最顶层 */
:deep(.el-select__popper),
:deep(.el-color-dropdown),
:deep(.el-popper) {
  z-index: 1000001 !important;
}

/* 添加交互状态 */
:deep(.copyable-value) {
  position: relative;
  cursor: pointer;
  border-radius: 3px;
  transition: all 0.2s ease;
  padding: 2px 6px;
  margin: -2px -2px;
  user-select: none;
}

:deep(.copyable-value:hover) {
  background-color: rgb(144, 95, 41, 0.15);
  box-shadow: 0 0 0 1px rgb(144, 95, 41, 0.2);
}

:deep(.copyable-value.editing) {
  background-color: rgb(144, 95, 41, 0.25);
  box-shadow: 0 0 0 2px rgb(144, 95, 41, 0.4);
}

/* 添加调试样式 */
.debug-message {
  position: fixed;
  top: 10px;
  right: 10px;
  background-color: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 10px;
  border-radius: 5px;
  z-index: 10000;
  max-width: 300px;
  max-height: 200px;
  overflow: auto;
  font-family: monospace;
  font-size: 12px;
}

/* 全局样式，确保工具提示在最顶层 */
.el-popper,
.el-select__popper,
.el-color-dropdown,
.el-color-picker,
.el-color-picker__panel {
  z-index: 1000000 !important;
}

/* 确保颜色选择器在最顶层 */
.el-color-picker__panel {
  z-index: 1000002 !important;
}

/* 自定义颜色选择器样式 */
:deep(.el-color-picker) {
  border: none;
  height: 32px;
}

:deep(.el-color-picker__trigger) {
  border: 1px solid rgb(60, 60, 60);
  background: rgb(45, 45, 45);
  width: 60px;
  height: 32px;
}

:deep(.el-color-picker__color) {
  border: none;
}

:deep(.el-color-picker__panel) {
  border: 1px solid rgb(60, 60, 60);
  background: rgb(35, 35, 35);
}

:deep(.el-color-dropdown__link-btn) {
  color: rgb(255, 255, 255);
}

/* 颜色选择器弹出层样式 */
.color-picker-popper {
  z-index: 1000002 !important;
  position: fixed !important;
  margin-top: 8px !important;
}

/* 自定义颜色选择器样式 */
:deep(.el-color-picker) {
  border: none;
  height: 32px;
}

:deep(.el-color-picker__trigger) {
  border: 1px solid rgb(60, 60, 60);
  background: rgb(45, 45, 45);
  width: 60px;
  height: 32px;
  cursor: pointer;
}

:deep(.el-color-picker__color) {
  border: none;
}

:deep(.el-color-picker__panel) {
  border: 1px solid rgb(60, 60, 60);
  background: rgb(35, 35, 35);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

:deep(.el-color-dropdown__link-btn) {
  color: rgb(255, 255, 255);
}

/* 调整编辑工具提示中颜色编辑器的布局 */
.color-editor {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  justify-content: center;
  padding: 8px 0;
  position: relative;
}

.color-preview {
  width: 32px;
  height: 32px;
  border-radius: 4px;
  border: 1px solid rgb(60, 60, 60);
  background-color: var(--el-color-white);
}

/* 添加直接颜色选择器样式 */
.direct-color-picker {
  position: fixed;
  z-index: 1000001;
}

.direct-color-picker .el-color-picker {
  opacity: 0; /* 隐藏触发器，只显示下拉菜单 */
}

/* 调整颜色选择器下拉菜单样式 */
:deep(.el-color-dropdown) {
  z-index: 1000002 !important;
}

.direct-number-editor {
  position: fixed;
  z-index: 9999;
  display: flex;
  align-items: center;
  background-color: #ffffff;
  border-radius: 4px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
  padding: 6px;
}

.direct-number-editor .close-button {
  margin-left: 6px;
  cursor: pointer;
  color: #606266;
}

.direct-number-editor .close-button:hover {
  color: #409EFF;
}

/* 确保数值编辑器里的文本颜色为深色 */
.direct-number-editor :deep(.el-input-number) {
  background-color: #ffffff;
}

.direct-number-editor :deep(.el-input__inner) {
  color: #333333 !important;
  background-color: #ffffff !important;
}

.direct-number-editor :deep(.el-input-number__decrease),
.direct-number-editor :deep(.el-input-number__increase) {
  background-color: #f5f7fa;
  color: #606266;
  border-color: #dcdfe6;
}

.analysis-words-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
  color: var(--gray800);
  position: relative;
  overflow: hidden;
}
</style>