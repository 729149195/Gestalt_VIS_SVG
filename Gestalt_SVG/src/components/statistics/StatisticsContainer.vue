<template>
  <div class="statistics-container">
    <span class="title">SVG Statistics</span>
    
    <!-- 添加排序按钮 -->
    <button class="sort-button" @click="toggleSortOrder">
      <span class="sort-icon">
        <svg v-if="sortAscending" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M7 14l5-5 5 5H7z" fill="currentColor"/>
        </svg>
        <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M7 10l5 5 5-5H7z" fill="currentColor"/>
        </svg>
      </span>
      <span>{{ sortAscending ? 'Diversity Asc' : 'Diversity Desc' }}</span>
    </button>
    
    <!-- 添加图例说明 -->
    <div class="legend-container" v-if="!isLoading">
      <div class="legend-item">
        <div class="legend-color blue"></div>
        <span class="legend-text">Pattern Statistics</span>
      </div>
      <div class="legend-item">
        <div class="legend-color gray"></div>
        <span class="legend-text">Other Statistics</span>
      </div>
    </div>
    
    <div v-if="isLoading" class="loading-indicator">
      <div class="loading-spinner"></div>
      <div class="loading-text">Data loading...</div>
    </div>
    <div v-else class="statistics-cards">
      <template v-for="(component, index) in sortedComponents" :key="index">
        <v-card v-if="component.hasData" class="position-card">
          <!-- <div class="variance-info">Diversity  {{ component.variance.toFixed(2) }}</div> -->
          <component :is="component.component" 
                    :position="component.props?.position" 
                    :title="component.props?.title" 
                    :key="componentKey + index" />
        </v-card>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch, markRaw } from 'vue';
import HisEleProportions from './ElementStatistics.vue';
import FillStatistician from './FillStatistics.vue';
import strokeStatistician from './StrokeStatistics.vue';
import PositionStatistics from './PositionStatistics.vue';

const props = defineProps({
  title: {
    type: String,
    default: 'SVG Statistics'
  },
  componentKey: {
    type: Number,
    default: 0
  }
});

const componentsData = ref([]);
const isLoading = ref(true); // 添加loading状态
const sortAscending = ref(true); // 添加排序状态，默认为正序（从小到大）

// 定义所有可能的组件配置
const allComponents = [
  { 
    id: 'top-position', 
    component: markRaw(PositionStatistics), 
    props: { position: 'top', title: 'Elementals Top Edge' },
    dataUrl: 'http://127.0.0.1:5000/top_position',
    hasData: false,
    variance: 0
  },
  { 
    id: 'bottom-position', 
    component: markRaw(PositionStatistics), 
    props: { position: 'bottom', title: 'Elementals Bottom Edge' },
    dataUrl: 'http://127.0.0.1:5000/bottom_position',
    hasData: false,
    variance: 0
  },
  { 
    id: 'right-position', 
    component: markRaw(PositionStatistics), 
    props: { position: 'right', title: 'Elementals Right Edge' },
    dataUrl: 'http://127.0.0.1:5000/right_position',
    hasData: false,
    variance: 0
  },
  { 
    id: 'left-position', 
    component: markRaw(PositionStatistics), 
    props: { position: 'left', title: 'Elementals Left Edge' },
    dataUrl: 'http://127.0.0.1:5000/left_position',
    hasData: false,
    variance: 0
  },
  { 
    id: 'fill-statistics', 
    component: markRaw(FillStatistician),
    dataUrl: 'http://127.0.0.1:5000/fill_num',
    hasData: false,
    variance: 0
  },
  { 
    id: 'stroke-statistics', 
    component: markRaw(strokeStatistician),
    dataUrl: 'http://127.0.0.1:5000/stroke_num',
    hasData: false,
    variance: 0
  },
  { 
    id: 'element-statistics', 
    component: markRaw(HisEleProportions),
    dataUrl: 'http://127.0.0.1:5000/ele_num_data',
    hasData: false,
    variance: 0
  }
];

// 计算方差的函数
const calculateVariance = (data) => {
  if (!data || data.length === 0) return 0;
  
  // 对于不同类型的数据结构进行处理
  let values = [];
  
  // 处理PositionStatistics组件的数据
  if (typeof data === 'object' && !Array.isArray(data)) {
    // 检查是否是PositionStatistics组件的数据格式
    if (Object.values(data).some(val => val.tags && Array.isArray(val.tags))) {
      // 提取所有range的tags数量
      Object.keys(data).forEach(range => {
        if (data[range].tags && data[range].tags.length) {
          values.push(data[range].tags.length);
        }
      });
    } else {
      // 处理FillStatistics和StrokeStatistics组件的数据
      values = Object.values(data);
    }
  } 
  // 处理ElementStatistics和AttributeStatistics组件的数据
  else if (Array.isArray(data)) {
    // 检查是否是ElementStatistics组件的数据（包含visible属性）
    if (data.some(item => 'visible' in item)) {
      // 对于ElementStatistics只统计visible为true的元素
      values = data
        .filter(item => item.visible === true)
        .map(item => item.num || 0);
    } else {
      // 对于其他数组类型数据
      values = data.map(item => item.num || 0);
    }
  }
  
  if (values.length === 0) return 0;
  
  // 计算平均值
  const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
  
  // 计算方差
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  
  return variance;
};

// 获取组件数据并计算方差
const fetchComponentData = async () => {
  isLoading.value = true; // 开始加载
  
  // 清空现有数据，以避免闪烁
  componentsData.value = [];
  
  const updatedComponents = [...allComponents];
  let hasErrors = false;
  
  // 创建所有获取请求的Promise
  const fetchPromises = updatedComponents.map(async (component) => {
    try {
      const response = await fetch(component.dataUrl);
      
      if (!response.ok) {
        console.error(`获取组件 ${component.id} 数据失败: ${response.status} ${response.statusText}`);
        component.hasData = false;
        component.variance = 0;
        return component;
      }
      
      const data = await response.json();      
      // 检查数据是否为空
      let isEmpty = false;
      
      // 特别处理 stroke-statistics 和 fill-statistics 组件（对象类型数据）
      if (component.id === 'stroke-statistics' || component.id === 'fill-statistics') {
        if (typeof data === 'object' && !Array.isArray(data)) {
          const keysCount = Object.keys(data).length;
          isEmpty = keysCount === 0;
          
          if (!isEmpty) {
            // 转换为数组格式，方便后续处理
            const values = Object.values(data);
            const variance = calculateVarianceFromArray(values);
            
            component.hasData = true;
            component.variance = variance;
            return component;
          }
        }
      }
      // 处理数组类型数据
      else if (Array.isArray(data)) {
        isEmpty = data.length === 0;
      } 
      // 处理PositionStatistics等其他对象类型数据
      else if (typeof data === 'object') {
        isEmpty = Object.keys(data).length === 0;
      } else {
        isEmpty = !data;
      }
      
      if (!isEmpty) {
        const variance = calculateVariance(data);
        
        component.hasData = true;
        component.variance = variance;
      } else {
        component.hasData = false;
        component.variance = 0;
      }
      
      return component;
    } catch (error) {
      console.error(`获取组件 ${component.id} 数据时出错:`, error);
      component.hasData = false;
      component.variance = 0;
      hasErrors = true;
      return component;
    }
  });
  
  // 等待所有请求完成
  try {
    const results = await Promise.all(fetchPromises);
    
    // 更新组件数据
    componentsData.value = results;
        
    // 添加延迟确保DOM更新
    setTimeout(() => {
      isLoading.value = false;
    }, 200);
  } catch (error) {
    console.error('获取组件数据时出现全局错误:', error);
    isLoading.value = false;
    componentsData.value = updatedComponents.filter(c => c.hasData);
  }
};

// 直接从数组计算方差的辅助函数
const calculateVarianceFromArray = (values) => {
  if (!values || values.length === 0) return 0;
  
  // 计算平均值
  const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
  
  // 计算方差
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  
  return variance;
};

// 根据方差排序组件
const sortedComponents = computed(() => {
  return [...componentsData.value]
    .filter(comp => comp.hasData)
    .sort((a, b) => sortAscending.value 
      ? a.variance - b.variance  // 正序：从小到大
      : b.variance - a.variance  // 倒序：从大到小
    );
});

// 切换排序顺序
const toggleSortOrder = () => {
  sortAscending.value = !sortAscending.value;
};

// 组件挂载时获取数据
onMounted(async () => {
  await fetchComponentData();
});

// 监听componentKey变化，重新获取数据
watch(() => props.componentKey, async () => {
  await fetchComponentData();
});
</script>

<style scoped>
.statistics-container {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  position: relative;
}

.statistics-cards {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  gap: 16px;
  padding: 10px;
  padding-top: 40px;
  height: 100%;
  overflow-y: auto;
}

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

/* 添加排序按钮样式 */
.sort-button {
  position: absolute;
  top: 7px;
  right: 250px; /* 位置在标题右侧 */
  height: 26px;
  padding: 0px 10px 0px 6px;
  font-size: 12px;
  border-radius: 13px;
  background-color: #f1f3f4;
  color: #202124;
  border: none;
  box-shadow: 0 1px 2px rgba(60, 64, 67, 0.1);
  cursor: pointer;
  z-index: 10;
  transition: all 0.15s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 3px;
  font-weight: 500;
  min-width: 105px;
  letter-spacing: 0.01em;
}

.sort-button span {
  display: flex;
  align-items: center;
  line-height: 1;
}

.sort-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #5f6368;
}

.sort-button:hover {
  background-color: #e8eaed;
  box-shadow: 0 1px 3px rgba(60, 64, 67, 0.2);
}

.sort-button:active {
  background-color: #dadce0;
  box-shadow: 0 1px 2px rgba(60, 64, 67, 0.15);
}

.position-card {
  flex: 1 1 calc(25% - 16px);
  min-width: 200px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  border: 1px solid rgba(200, 200, 200, 0.3);
  padding: 12px;
  height: 48%;
  position: relative;
}

.position-card:hover {
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  transform: translateY(-1px);
  border: 1px solid rgba(180, 180, 180, 0.4);
}

.variance-info {
  position: absolute;
  top: 2px;
  left: 12px;
  font-size: 11px;
  color: #666;
  z-index: 10;
}

.loading-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 100%;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(0, 0, 0, 0.1);
  border-radius: 50%;
  border-top-color: #885F35;
  animation: spin 1s ease-in-out infinite;
  margin-bottom: 10px;
}

.loading-text {
  font-size: 14px;
  color: #666;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.legend-container {
  display: flex;
  position: absolute;
  top: 12px;
  right: 16px;
  z-index: 10;
  gap: 12px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.legend-color {
  width: 16px;
  height: 16px;
  border-radius: 4px;
}

.legend-color.blue {
  background-color: #885F35;
}

.legend-color.gray {
  background-color: #e5e7eb;
}

.legend-text {
  font-size: 12px;
  color: #666;
}
</style> 