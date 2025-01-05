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
    
    <div class="analysis-content" @scroll="handleScroll" v-html="analysisContent">
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

// 数据源URL
const MAPPING_DATA_URL = "http://localhost:5000/average_equivalent_mapping";
const EQUIVALENT_WEIGHTS_URL = "http://localhost:5000/equivalent_weights_by_tag";

// 特征名称映射
const featureNameMap = {
    'tag_name': '元素名称',
    'tag': '标签类型',
    'opacity': '不透明度',
    'fill_h_cos': '填充色相(C)',
    'fill_h_sin': '填充色相(S)',
    'fill_s_n': '填充饱和度',
    'fill_l_n': '填充亮度',
    'stroke_h_cos': '描边色相(C)',
    'stroke_h_sin': '描边色相(S)',
    'stroke_s_n': '描边饱和度',
    'stroke_l_n': '描边亮度',
    'stroke_width': '描边宽度',
    'bbox_left_n': '左边界位置',
    'bbox_right_n': '右边界位置',
    'bbox_top_n': '上边界位置',
    'bbox_bottom_n': '下边界位置',
    'bbox_mds_1': '位置mds特征1',
    'bbox_mds_2': '位置mds特征2',
    'bbox_center_x_n': '中心X坐标',
    'bbox_center_y_n': '中心Y坐标',
    'bbox_width_n': '宽度',
    'bbox_height_n': '高度',
    'bbox_fill_area': '元素面积'
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

// 将 showDialog 改名为 showDrawer
const showDrawer = ref(false)

// 生成分析文字的函数
const generateAnalysis = (dataMapping, dataEquivalentWeights) => {
    if (!dataMapping || !dataEquivalentWeights) return '等待分析...';

    let analysis = '';
    const inputDimensions = dataMapping.input_dimensions;
    const outputDimensions = dataMapping.output_dimensions;
    const weights = dataMapping.weights;

    // 分析每个输出维度的主要特征
    outputDimensions.forEach((outDim, j) => {
        analysis += `<div class="dimension-analysis">【维度 ${j + 1} 的主要组成】 `;
        
        const dimensionWeights = weights[j];
        const weightEntries = dimensionWeights.map((w, i) => ({ weight: w, index: i }));
        weightEntries.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
        
        const topFeatures = weightEntries.slice(0, 3);
        
        let featureTexts = topFeatures.map(({ weight, index }) => {
            const featureName = inputDimensions[index];
            const displayName = featureNameMap[featureName] || featureName;
            const influence = weight > 0 ? '正相关' : '负相关';
            const strength = Math.abs(weight).toFixed(2);
            
            let strengthSymbol = '•';
            if (strength > 1) strengthSymbol = '★';
            else if (strength > 0.8) strengthSymbol = '☆';
            
            return `<span class="feature-tag">${displayName}</span> ${influence} ${strength} ${strengthSymbol}`;
        })
        
        analysis += featureTexts.join('   ') + '</div>';
    });

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

        // 生成分析文字
        analysisContent.value = generateAnalysis(responseMapping.data, responseEquivalentWeights.data);
    } catch (error) {
        console.error('获取数据失败:', error);
        analysisContent.value = '分析生成失败，请重试';
    }
};

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

.analysis-words-container:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  transform: translateY(-1px);
}

.analysis-header {
  margin-bottom: 16px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  padding-bottom: 16px;
  flex-shrink: 0;
}


.analysis-content {
  color: #424245;
  font-size: 13px;
  overflow-y: auto;
  line-height: 1.4;
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
</style>