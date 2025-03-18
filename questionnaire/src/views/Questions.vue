<template>
  <div class="common-layout">
    <el-container class="full-height">
      <el-header class="header">
        <div class="header-content">
          <div class="left-content">
            <!-- <p class="id">分配ID：{{ formData.id }}</p> -->
            <a href="https://github.com/729149195/questionnaire" target="_blank">
              <img style="width: 30px;" src="/img/favicon.png" alt="Wechat QR Code">
            </a>
          </div>
        </div>
      </el-header>
      <el-main>
        <el-card class="main-card">
          <div style="display: flex;">
            <div class="left-two">
              <el-card class="top-card" shadow="never">
                <div v-html="Svg" class="svg-container"></div>
                <el-button class="top-title" disabled text bg>组合观察区域</el-button>
              </el-card>
              <el-card class="bottom-card" shadow="never">
                <div ref="chartContainer" class="chart-container" v-show="false"></div>
                <div v-html="Svg" class="svg-container2" ref="svgContainer2"></div>
                <el-button @click="toggleCropMode" class="Crop" :class="{ 'active-mode': isCropping }">
                  <el-icon><Crop /></el-icon>
                </el-button>
                <el-button @click="toggleTrackMode" class="track" :class="{ 'active-mode': isTracking }">
                  <el-icon><Pointer /></el-icon>
                </el-button>
                <el-button class="bottom-title" disabled text bg>选取交互区域</el-button>
              </el-card>
            </div>
            <el-card class="group-card" shadow="never">
              <div class="select-group">
                <el-select v-model="selectedGroup" placeholder="选择组合" @change="highlightGroup">
                  <el-option v-for="(group, index) in groupOptions" :key="index" :label="group" :value="group" />
                </el-select>
                <el-button @click="addNewGroup"><el-icon>
                    <Plus />
                  </el-icon></el-button>
                <el-button @click="deleteCurrentGroup"><el-icon>
                    <Delete />
                  </el-icon></el-button>
              </div>
              <div v-if="selectedGroup" class="group">
                <h3>{{ selectedGroup }}</h3>
                <el-scrollbar height="500px">
                  <div class="group-tags">
                    <el-tag v-for="node in currentGroupNodes" :key="node" closable
                      @close="removeFromGroup(selectedGroup, node)" @mousedown="highlightElement(node)"
                      @mouseup="resetHighlight">
                      {{ node }}
                    </el-tag>
                  </div>
                </el-scrollbar>
                <div v-if="ratings[selectedGroup]" ref="rateings" class="rate-container">
                  <el-tooltip content="评分越高表示这个组合越容易被注意到" placement="right">
                    <div class="rate-container2">
                      <div class="rate-text">显眼程度：</div>
                      <el-rate :icons="icons" :void-icon="Hide" :colors="['#409eff', '#67c23a', '#FF9900']" :max="3"
                        :texts="['低', '中', '高']" show-text v-model="ratings[selectedGroup].attention" class="rate"
                        @change="updateRating(selectedGroup, ratings[selectedGroup].attention, 'attention')" />
                    </div>
                  </el-tooltip>
                  <el-tooltip content="评分越高表示组内元素的关系越紧密" placement="right">
                    <div class="rate-container2">
                      <div class="rate-text">组内关联程度：</div>
                      <el-rate :icons="icons" :void-icon="Hide" :colors="['#409eff', '#67c23a', '#FF9900']" :max="3"
                        :texts="['低', '中', '高']" show-text v-model="ratings[selectedGroup].correlation_strength"
                        class="rate"
                        @change="updateRating(selectedGroup, ratings[selectedGroup].correlation_strength, 'correlation_strength')" />
                    </div>
                  </el-tooltip>
                  <el-tooltip content="评分越高表示组外元素越难被归入该组" placement="right">
                    <div class="rate-container2">
                      <div class="rate-text">组外排斥程度：</div>
                      <el-rate :icons="icons" :void-icon="Hide" :colors="['#409eff', '#67c23a', '#FF9900']" :max="3"
                        :texts="['低', '中', '高']" show-text v-model="ratings[selectedGroup].exclusionary_force"
                        class="rate"
                        @change="updateRating(selectedGroup, ratings[selectedGroup].exclusionary_force, 'exclusionary_force')" />
                    </div>
                  </el-tooltip>
                </div>
              </div>
            </el-card>
          </div>
        </el-card>
        <div class="steps-container">
          <el-button class="previous-button" @click="Previous"><el-icon>
              <CaretLeft />
            </el-icon></el-button>
          <el-steps :active="active" finish-status="success" class="steps">
            <el-step v-for="(step, index) in steps" :key="index" @click.native="goToStep(index)" />
          </el-steps>
          <el-button class="next-button" @click="next" type="primary" v-if="active != steps.length - 1"><el-icon>
              <CaretRight />
            </el-icon></el-button>
          <el-button 
            class="submit-button" 
            @click="submit" 
            type="success"
            :loading="submitLoading"
            v-if="active === steps.length - 1"
          >
            <el-icon><Select /></el-icon>
          </el-button>
          <el-button 
            class="export-button"
            @click="exportToJson" 
            type="warning"
            v-if="active === steps.length - 1"
          >
            <el-icon><Download /></el-icon>
          </el-button>
        </div>
      </el-main>

    </el-container>

    <el-dialog v-model="dialogVisible" title="提醒" width="700" align-center @close="handleDialogClose"
      :close-on-click-modal="false">
      <span>
        您已经做了15分钟了，以稍微闭眼休息一下哦~
      </span>
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="dialogVisible = false">我知道了</el-button>
        </div>
      </template>
    </el-dialog>

    <el-dialog 
      v-model="infoDialogVisible" 
      title="问卷说明" 
      width="800" 
      align-center
      :close-on-click-modal="false"
      class="info-dialog"
    >
      <div class="info-content">
        <h3 class="info-subtitle">开始问卷前，请了解以下要点：</h3>
        <ol class="info-list">
          <li>请根据您的直观感受，选出所有您认为应该归为一组的图形元素</li>
          <li>同一个图形元素可以同时属于多个不同的组合</li>
          <li>评分时请跟随第一印象，无需过度分析</li>
        </ol>
      </div>
    </el-dialog>
  </div>
  <el-card class="flow">
    <template #header>
      <div class="flow-header">
        <span class="flow-title">操作指南</span>
      </div>
    </template>
    <div class="flow-content">
      <div class="step-item">
        <span class="step-number">第1步</span>
        <el-card class="step-card" shadow="hover">
          <p>观察左上方区域中的图形，思考哪些图形可以归为一组</p>
        </el-card>
      </div>
      <div class="step-item">
        <span class="step-number">第2步</span>
        <el-card class="step-card" shadow="hover">
          <p>在下方区域中选择您认为属于同一组的图形</p>
          <ul class="step-list">
            <li>选择密集图形时可以使用：
              <el-button class="icon-btn" size="small">
                <el-icon><Crop /></el-icon>
              </el-button> 
              框选模式或
              <el-button class="icon-btn" size="small">
                <el-icon><Pointer /></el-icon>
              </el-button> 
              路径选择模式
            </li>
            <li>再次点击已选中的图形可取消选择</li>
            <li>普通模式下可用鼠标滚轮缩放和拖动图形</li>
          </ul>
        </el-card>
      </div>
      <div class="step-item">
        <span class="step-number">第3步</span>
        <el-card class="step-card" shadow="hover">
          <p>如需添加新的图形组合，点击右侧加号按钮创建新组</p>
        </el-card>
      </div>
      <div class="step-item">
        <span class="step-number">第4步</span>
        <el-card class="step-card" shadow="hover">
          <p>完成一组图形选择后，请不要忘记为该组进行评分哦~</p>
        </el-card>
      </div>
    </div>
  </el-card>
  <el-card class="flow2">
    <template #header>
      <div class="flow-header">
        <span class="flow-title">重要提示</span>
      </div>
    </template>
    <div class="tips-content">
      <ul class="tips-list">
        <li class="highlight-tip">
          <strong class="underline-text">请尽可能多地选出您感知到的图形组合</strong>
        </li>
        <li>一个图形可以属于多个不同的组合</li>
        <li>请根据直观感受进行选择和评分</li>
      </ul>
    </div>
  </el-card>
</template>

<script setup>
import { ref, computed, onMounted, nextTick, watch, onBeforeMount } from 'vue';
import { useStore } from 'vuex';
import { useRouter } from 'vue-router';
import * as d3 from 'd3';
import { Delete, Plus, Hide, View, CaretLeft, CaretRight, Select, Crop, Pointer, Download } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';
import { getSubmissionCount, incrementCount } from '../api/counter';
import emailjs from '@emailjs/browser';
import { saveAs } from 'file-saver';

const store = useStore();
const router = useRouter();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);
const allVisiableNodes = computed(() => store.state.AllVisiableNodes);
const steps = computed(() => store.state.steps);
const dialogVisible = ref(false);
const infoDialogVisible = ref(true);
const active = ref(0);
const icons = [View, View, View];
const svgContainer2 = ref(null);

const Svg = ref('');
const selectedGroup = ref('组合1');
const ratings = ref({});
let reminderTimerId = null;
const nodeEventHandlers = new Map();
const isCropping = ref(false);
const isTracking = ref(false);

// 存储当前的换状态
const currentTransform = ref(null);

// 添加ID检查函数
const checkUserId = () => {
  const userId = store.getters.getFormData?.id;
  if (!userId) {
    ElMessage.error('用户id失效，请重新进入');
    router.push('/');
    return false;
  }
  return true;
};

const goToStep = async (index) => {
  if (!checkUserId()) return;
  if (index !== active.value) {
    selectedGroup.value = '组合1';
    active.value = index;
    await fetchSvgContent(active.value + 1); // 加载对应步骤的SVG内容
    await fetchAndRenderTree(); // 加载对应步骤的树形结构
    ensureGroupInitialization(); // 确保组合初始化
    nextTick(() => {
      highlightGroup(); // 确保组合在初始加载时被高亮
    });
    isCropping.value = false;
    svgContainer2.value.classList.remove('crosshair-cursor');
    // await loadExampleData();
  }
};


const currentGroupNodes = computed(() => {
  if (!ratings.value[selectedGroup.value]) {
    ratings.value[selectedGroup.value] = { attention: 1, correlation_strength: 1, exclusionary_force: 1 };
  }
  return groups.value[selectedGroup.value] || [];
});

const updateRating = (group, rating, type) => {
  const step = active.value;
  store.commit('UPDATE_RATING', { step, group, rating, type });
};

const startTotalTimer = () => {
  setInterval(() => {
    store.commit('UPDATE_TOTAL_TIME_SPENT', store.state.totalTimeSpent + 1);
  }, 1000);
};

const startReminderTimer = () => {
  reminderTimerId = setTimeout(() => {
    dialogVisible.value = true;
  }, 15 * 60 * 1000);
};

const handleDialogClose = () => {
  dialogVisible.value = false;
  clearTimeout(reminderTimerId);
  startReminderTimer();
};

const fetchSvgContent = async (step) => {
  try {
    nodeEventHandlers.forEach((handler, node) => {
      node.removeEventListener('click', handler);
    });
    nodeEventHandlers.clear();

    const response = await fetch(`./Data4/${step}/${step}.svg`);
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    let svgContent = await response.text();
    
    // 确保SVG有正确的高度属性
    if (svgContent.includes('height="auto"')) {
      svgContent = svgContent.replace('height="auto"', 'height="100%"');
    }
    
    // 如果SVG没有高度属性，添加一个
    if (!svgContent.includes('height=')) {
      svgContent = svgContent.replace('<svg', '<svg height="100%"');
    }
    
    Svg.value = svgContent;
    
    nextTick(() => {
      // 确保所有SVG元素都有正确的高度设置
      const svgElements = document.querySelectorAll('svg');
      svgElements.forEach(svg => {
        if (svg.getAttribute('height') === 'auto') {
          svg.setAttribute('height', '100%');
        }
      });
      
      // 确保SVG完全加载后再添加缩放效果
      setTimeout(() => {
        addZoomEffectToSvg();
      }, 100);
      
      turnGrayVisibleNodes();
      addHoverEffectToVisibleNodes();
      addClickEffectToVisibleNodes();
      highlightGroup();
    });
  } catch (error) {
    console.error('Error loading SVG content:', error);
    Svg.value = '<svg height="100%"><text x="10" y="20" font-size="20">加载SVG时出错</text></svg>';
  }
};

const addZoomEffectToSvg = () => {
  const svgContainer = svgContainer2.value;
  if (!svgContainer) return;
  const svg = d3.select(svgContainer).select('svg');
  if (!svg.node()) return;  // 确保 SVG 元素存在

  // 创建一个包含实际SVG内容的组
  let g = svg.select('g.zoom-wrapper');
  if (g.empty()) {
    g = svg.append('g').attr('class', 'zoom-wrapper');
    // 将所有现有内容移动到新的组中（不再克隆，直接移动）
    const children = [...svg.node().childNodes];
    children.forEach(child => {
      if (child.nodeType === 1 && !child.classList.contains('zoom-wrapper')) {
        // 直接移动原始节点，保留事件监听器
        g.node().appendChild(child);
      }
    });
  }

  const zoom = d3.zoom()
    .scaleExtent([0.5, 10])
    .on('zoom', (event) => {
      if (!isCropping.value && g.node()) {  // 确保 g 元素存在
        g.attr('transform', event.transform);
      }
    });

  svg.call(zoom);

  // 获取参考 SVG 的位置和尺寸
  const referenceSvg = d3.select('.svg-container svg');
  if (!referenceSvg.node()) return;  // 确保参考 SVG 存在

  try {
    // 获取两个 SVG 的 viewBox
    const refViewBox = referenceSvg.node().viewBox.baseVal;
    const currentViewBox = svg.node().viewBox.baseVal;

    // 获取实际显示尺寸
    const refRect = referenceSvg.node().getBoundingClientRect();
    const currentRect = svg.node().getBoundingClientRect();

    // 检查所有值是否为有效数字
    if (isNaN(refRect.width) || isNaN(refRect.height) || 
        isNaN(currentRect.width) || isNaN(currentRect.height) ||
        !refViewBox || !currentViewBox) {
      return;  // 如果有无效值，直接返回
    }

    // 计算缩放比例
    const scaleX = (refRect.width / refViewBox.width) / (currentRect.width / currentViewBox.width);
    const scaleY = (refRect.height / refViewBox.height) / (currentRect.height / currentViewBox.height);
    const scale = Math.min(scaleX, scaleY);

    if (isNaN(scale) || scale <= 0) return;  // 确保缩放比例有效

    // 计算偏移量
    const translateX = (refViewBox.x - currentViewBox.x) * scale + 
                      (refRect.width - currentRect.width * scale) / 2;
    const translateY = (refViewBox.y - currentViewBox.y) * scale + 
                      (refRect.height - currentRect.height * scale) / 2;

    // 检查计算结果是否有效
    if (!isNaN(translateX) && !isNaN(translateY) && !isNaN(scale)) {
      const initialTransform = d3.zoomIdentity
        .translate(translateX, translateY)
        .scale(scale);

      svg.call(zoom.transform, initialTransform);
    }
  } catch (error) {
    console.error('Error in zoom calculation:', error);
  }
};

let isDrawing = false; // 标志是否正在绘制
let rectElement; // 矩形元素
let handleMouseClick, handleMouseMove, handleMouseUp; // 事件处理程序

const toggleCropMode = () => {
  isCropping.value = !isCropping.value;
  const svg = d3.select(svgContainer2.value).select('svg');
  
  if (isCropping.value) {
    nextTick(() => {
      svgContainer2.value.classList.add('crosshair-cursor');
    });
    if (isTracking.value) {
      isTracking.value = false;
      svgContainer2.value.classList.remove('copy-cursor');
      ElMessage.info('退出路径模式');
      disableTrackMode();
    }
    ElMessage.info('进入选框模式');
    enableCropSelection();
    
    // 保存当前变换状态
    const transform = d3.zoomTransform(svg.node());
    currentTransform.value = transform;
    
    svg.on('.zoom', null); // 禁用缩放事件
  } else {
    svgContainer2.value.classList.remove('crosshair-cursor');
    ElMessage.info('退出选框模式');
    disableCropSelection();
    
    // 重新启用缩放并恢复之的变换状态
    const zoom = d3.zoom()
      .scaleExtent([0.5, 10])
      .on('zoom', (event) => {
        if (!isCropping.value) {
          svg.select('g.zoom-wrapper').attr('transform', event.transform);
        }
      });
      
    svg.call(zoom);
    if (currentTransform.value) {
      svg.call(zoom.transform, currentTransform.value);
    }
  }
};



const enableCropSelection = () => {
  let startX, startY;
  const svg = svgContainer2.value.querySelector('svg');

  handleMouseClick = (event) => {
    if (!isDrawing) {
      isDrawing = true;
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      const svgPoint = point.matrixTransform(svg.getScreenCTM().inverse());

      startX = svgPoint.x;
      startY = svgPoint.y;

      rectElement = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rectElement.setAttribute('x', startX);
      rectElement.setAttribute('y', startY);
      rectElement.setAttribute('stroke', 'red');
      rectElement.setAttribute('stroke-width', '2');
      rectElement.setAttribute('fill', 'none');
      svg.appendChild(rectElement);

      svg.addEventListener('mousemove', handleMouseMove);
      svg.addEventListener('mouseup', handleMouseUp);
    }
  };

  handleMouseMove = (event) => {
    if (isDrawing) {
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      const svgPoint = point.matrixTransform(svg.getScreenCTM().inverse());

      const endX = svgPoint.x;
      const endY = svgPoint.y;
      const rectWidth = Math.abs(endX - startX);
      const rectHeight = Math.abs(endY - startY);
      const rectX = Math.min(startX, endX);
      const rectY = Math.min(startY, endY);

      rectElement.setAttribute('width', rectWidth);
      rectElement.setAttribute('height', rectHeight);
      rectElement.setAttribute('x', rectX);
      rectElement.setAttribute('y', rectY);
    }
  };

  const handleMouseUp = (event) => {
    if (isDrawing) {
      isDrawing = false;

      const rectX = parseFloat(rectElement.getAttribute('x'));
      const rectY = parseFloat(rectElement.getAttribute('y'));
      const rectWidth = parseFloat(rectElement.getAttribute('width'));
      const rectHeight = parseFloat(rectElement.getAttribute('height'));

      const svg = svgContainer2.value.querySelector('svg');
      svg.querySelectorAll('*').forEach(node => {
        if (typeof node.getBBox === 'function') {
          const bbox = node.getBBox();
          const isTouched =
            (bbox.x + bbox.width) >= rectX &&
            bbox.x <= (rectX + rectWidth) &&
            (bbox.y + bbox.height) >= rectY &&
            bbox.y <= (rectY + rectHeight);

          if (isTouched) {
            node.dispatchEvent(new Event('click')); // 模拟点击事件
          }
        }
      });

      rectElement.remove(); // 移除选框
      svg.removeEventListener('mousemove', handleMouseMove);
      svg.removeEventListener('mouseup', handleMouseUp);
    }
  };

  svg.addEventListener('mousedown', handleMouseClick);
};

const disableCropSelection = () => {
  const svg = svgContainer2.value.querySelector('svg');
  if (svg) {
    svg.removeEventListener('mousedown', handleMouseClick);
    svg.removeEventListener('mousemove', handleMouseMove);
    svg.removeEventListener('mouseup', handleMouseUp);
  }
};

const toggleTrackMode = () => {
  isTracking.value = !isTracking.value;
  const svg = d3.select(svgContainer2.value).select('svg');
  
  if (isTracking.value) {
    nextTick(() => {
      svgContainer2.value.classList.add('copy-cursor');
    });
    if (isCropping.value) {
      isCropping.value = false;
      svgContainer2.value.classList.remove('crosshair-cursor');
      ElMessage.info('退出选框模式');
      disableCropSelection();
    }
    ElMessage.info('进入路径模式');
    enableTrackMode();
    
    // 保存当前变换状态
    const transform = d3.zoomTransform(svg.node());
    currentTransform.value = transform;
    
    svg.on('.zoom', null); // 禁用缩放事件
  } else {
    svgContainer2.value.classList.remove('copy-cursor');
    ElMessage.info('退出路径模式');
    disableTrackMode();
    
    // 重新启用缩放并恢复之前的变换状态
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
  const svg = svgContainer2.value.querySelector('svg');

  const handleMouseDown = () => {
    isMouseDown = true;
    clickedElements.clear(); // 点元素集合
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
        clickedElements.add(node); // 记录已点击的元素
        node.dispatchEvent(new Event('click', { bubbles: true })); // 模拟点击事件
      }
    }
  };

  svg.addEventListener('mousedown', handleMouseDown);
  svg.addEventListener('mouseup', handleMouseUp);
  svg.addEventListener('mousemove', handleMouseMove);

  nodeEventHandlers.set(svg, { handleMouseDown, handleMouseUp, handleMouseMove });
};

const disableTrackMode = () => {
  const svg = svgContainer2.value.querySelector('svg');
  if (svg) {
    const handlers = nodeEventHandlers.get(svg);
    if (handlers) {
      svg.removeEventListener('mousedown', handlers.handleMouseDown);
      svg.removeEventListener('mouseup', handlers.handleMouseUp);
      svg.removeEventListener('mousemove', handlers.handleMouseMove);
    }
    nodeEventHandlers.delete(svg);
  }
};

const turnGrayVisibleNodes = () => {
  const svgContainer = svgContainer2.value;
  if (!svgContainer) return;
  const svg = svgContainer.querySelector('svg');
  if (!svg) return;

  svg.querySelectorAll('*').forEach(node => {
    if (allVisiableNodes.value.includes(node.id)) {
      node.style.opacity = '0.1';
      // if(isCropping.value === false && isTracking.value === false){
      // node.style.cursor = 'pointer';
      // }
      // node.style.cursor = 'pointer';
      node.style.transition = 'opacity 0.3s ease';
    }
  });
};
//isTracking.valueisCropping.value
const addHoverEffectToVisibleNodes = () => {
  const svgContainer = svgContainer2.value;
  if (!svgContainer) return;
  const svg = svgContainer.querySelector('svg');
  if (!svg) return;

  svg.querySelectorAll('*').forEach(node => {
    if (allVisiableNodes.value.includes(node.id)) {
      const handleMouseOver = () => {
        node.style.opacity = '1';
      };
      const handleMouseOut = () => {
        node.style.opacity = '0.1';
        node.style.transition = 'opacity 0.3s ease';
        highlightGroup();
      };

      node.removeEventListener('mouseover', handleMouseOver);
      node.removeEventListener('mouseout', handleMouseOut);

      node.addEventListener('mouseover', handleMouseOver);
      node.addEventListener('mouseout', handleMouseOut);
    }
  });
};

const addClickEffectToVisibleNodes = () => {
  const svgContainer = svgContainer2.value;
  if (!svgContainer) return;
  const svg = svgContainer.querySelector('svg');
  if (!svg) return;

  svg.querySelectorAll('*').forEach(node => {
    if (allVisiableNodes.value.includes(node.id)) {
      const oldHandler = nodeEventHandlers.get(node);

      if (oldHandler) {
        node.removeEventListener('click', oldHandler);
      }

      const handleNodeClick = () => {
        const groupNodes = store.state.groups[active.value]?.[selectedGroup.value] || [];
        if (groupNodes.includes(node.id)) {
          store.commit('REMOVE_NODE_FROM_GROUP', { step: active.value, group: selectedGroup.value, nodeId: node.id });
          console.log("REMOVE_NODE_FROM_GROUP", node.id);  // 调试用，检查节点移除
        } else {
          store.commit('ADD_NODE_TO_GROUP', { step: active.value, group: selectedGroup.value, nodeId: node.id });
          console.log("ADD_NODE_TO_GROUP", node.id);  // 调试用，检查节点添加
        }
        nextTick(() => {
          highlightGroup();
        });
      };

      nodeEventHandlers.set(node, handleNodeClick);

      node.addEventListener('click', handleNodeClick);
    }
  });
};

const highlightGroup = () => {
  const groupNodes = store.state.groups[active.value]?.[selectedGroup.value] || [];
  const svgContainer = svgContainer2.value;
  if (!svgContainer) return;
  const svg = svgContainer.querySelector('svg');
  if (!svg) return;

  svg.querySelectorAll('*').forEach(node => {
    if (groupNodes.includes(node.id)) {
      node.style.opacity = '1';
    } else if (allVisiableNodes.value.includes(node.id)) {
      node.style.opacity = '0.1';
      node.style.transition = 'opacity 0.3s ease';
    }
  });
};

const highlightElement = (nodeId) => {
  const svgContainer = svgContainer2.value;
  if (!svgContainer) return;
  const svg = svgContainer.querySelector('svg');
  if (!svg) return;
  nextTick(() => {
    svg.querySelectorAll('*').forEach(node => {
      if (node.id === nodeId) {
        node.style.opacity = '1';
      } else if (allVisiableNodes.value.includes(node.id)) {
        node.style.opacity = '0.1';
        node.style.transition = 'opacity 0.3s ease';
      }
    });
  });
};

const resetHighlight = () => {
  nextTick(() => {
    highlightGroup();
  });
};

const deleteCurrentGroup = () => {
  const step = active.value;
  store.commit('DELETE_GROUP', { step, group: selectedGroup.value });
  selectedGroup.value = '组合1';
  nextTick(() => {
    highlightGroup();
  });
};

const eleURL = computed(() => {
  const step = store.state.steps[active.value];
  return `./Data4/${step}/layer_data.json`;
});

const chartContainer = ref(null);

const next = async () => {
  if (!checkUserId()) return;
  
  // 检查当前步骤的组合情况
  const currentGroups = store.getters.getGroups(active.value);
  
  // 检查组数是否大于2
  if (Object.keys(currentGroups).length < 2) {
    ElMessage.error('请至少创建2个组合后再继续');
    return;
  }
  
  // 检查是否存在空组
  const hasEmptyGroup = Object.values(currentGroups).some(group => group.length === 0);
  if (hasEmptyGroup) {
    ElMessage.error('存在空组合，请确保每个组合都包含元素');
    return;
  }

  // 检查所有组合的评分情况
  const allRatingsAreLow = Object.keys(currentGroups).every(group => {
    // 直接从 ratings 中获取评分数据
    const groupRatings = ratings.value[group];
    return (
      groupRatings?.attention === 1 &&
      groupRatings?.correlation_strength === 1 &&
      groupRatings?.exclusionary_force === 1
    );
  });

  if (allRatingsAreLow) {
    ElMessage({
      type: 'warning',
      message: '目前所有组合的三个评分都为低，请确保已评分',
      duration: 5000,
      showClose: true
    });
    return;
  }

  const count = await getSubmissionCount();
  if (count >= 50) {
    router.push('/limit-reached');
    return;
  }

  if (steps.value && active.value < steps.value.length - 1) {
    selectedGroup.value = '组合1';
    active.value++;
    await fetchSvgContent(steps.value[active.value]);
    await fetchAndRenderTree();
    ensureGroupInitialization();
    nextTick(() => {
      highlightGroup();
    });
    // 关闭选框模式
    isCropping.value = false;
    // 关闭路径选择模式
    isTracking.value = false;
    svgContainer2.value.classList.remove('crosshair-cursor');
    svgContainer2.value.classList.remove('copy-cursor');
    // 禁用相应的事件处理
    disableCropSelection();
    disableTrackMode();
  }
};

const Previous = async () => {
  if (!checkUserId()) return;
  const count = await getSubmissionCount();
  if (count >= 50) {
    router.push('/limit-reached');
    return;
  }

  if (steps.value && active.value > 0) {
    selectedGroup.value = '组合1';
    active.value--;
    await fetchSvgContent(steps.value[active.value]);
    await fetchAndRenderTree();
    ensureGroupInitialization();
    nextTick(() => {
      highlightGroup();
    });
    // 关闭选框模式
    isCropping.value = false;
    // 关闭路径选择模式
    isTracking.value = false;
    svgContainer2.value.classList.remove('crosshair-cursor');
    svgContainer2.value.classList.remove('copy-cursor');
    // 禁用相应的事件处理
    disableCropSelection();
    disableTrackMode();
  }
};

const formatDate = (date) => {
  const d = new Date(date);
  const offset = d.getTimezoneOffset() * 60000;
  const localDate = new Date(d.getTime() + offset + 28800000); // Convert to UTC+8
  return localDate.toISOString().replace('T', ' ').substring(0, 19);
};

const generateJsonData = () => {
  const currentTime = new Date();
  const endTime = formatDate(currentTime);
  const duration = (currentTime - new Date(store.state.startTime)) / 1000; // in seconds

  const data = {
    formData: store.getters.getFormData,
    startTime: formatDate(store.state.startTime),
    endTime: endTime,
    duration: `${Math.floor(duration / 60)} minutes ${Math.floor(duration % 60)} seconds`,
    steps: []
  };

  steps.value.forEach((stepId, index) => {
    const stepData = {
      stepId,
      groups: []
    };
    const groups = store.getters.getGroups(index);
    for (const group in groups) {
      stepData.groups.push({
        group: group,
        nodes: groups[group],
        ratings: {
          attention: store.getters.getRating(index, group, 'attention'),
          correlation_strength: store.getters.getRating(index, group, 'correlation_strength'),
          exclusionary_force: store.getters.getRating(index, group, 'exclusionary_force')
        }
      });
    }
    data.steps.push(stepData);
  });

  return data;
};

const sendEmail = (data) => {
  const emailData = {
    form_id: store.getters.getFormData.id,
    to_email: 'zxx729149195@163.com',
    subject: `问卷+${store.getters.getFormData.id}`,
    message: JSON.stringify(data, null, 2)
  };

  return emailjs.send('service_e1fyicu', 'template_a753pml', emailData, 'V-soSEM_lhq-gts4J')
    .then((response) => {
      console.log('Email sent successfully!', response.status, response.text);
      ElMessage.success('数据文件已自动上传成功!');
    })
    .catch((error) => {
      console.error('Failed to send email:', error);
      ElMessage.error('数据文件上传失败。请导出一份问卷数据动发送给管理员😭');
      throw error; // 重新抛出错误以便上层处理
    });
};

// 添加 loading ref
const submitLoading = ref(false);

const submit = async () => {
  if (!checkUserId()) return;
  
  submitLoading.value = true;
  ElMessage.info('正在提交数据，请稍候...');

  try {
    const [count, data] = await Promise.all([
      getSubmissionCount(),
      Promise.resolve(generateJsonData()) // 同步操作包装成 Promise
    ]);

    if (count >= 50) {
      submitLoading.value = false;
      router.push('/limit-reached');
      return;
    }

    const formData = store.getters.getFormData;

    // 将数据保存到 Vuex store
    store.commit('SET_SUBMITTED_DATA', data);

    // 并行处理邮件发送和数据存储
    await Promise.all([
      sendEmail(data),
      Promise.all([
        localStorage.setItem('submitId', formData.id),
        incrementCount()
      ])
    ]);

    // 清除用户ID
    store.commit('CLEAR_FORM_DATA');
    
    submitLoading.value = false;
    router.push('/thanks');

  } catch (error) {
    console.error('Failed to submit:', error);
    submitLoading.value = false;
    ElMessage.error('提交失败，请重试');
  }
};

const fetchAndRenderTree = async () => {
  if (!chartContainer.value) return;
  try {
    const response = await fetch(eleURL.value);
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const data = await response.json();
    renderTree(data);
  } catch (error) {
    console.error('There has been a problem with your fetch operation:', error);
  }
};

const renderTree = (data) => {
  const width = 1300;
  const height = 400;

  d3.select(chartContainer.value).select('svg').remove();

  const svg = d3.select(chartContainer.value)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', '100%')
    .attr('height', '100%')
    .style('max-height', '800px'); // 添加最大高度限制

  const root = d3.treemap()
    .size([width, height])
    .padding(1)
    .round(true)
    (d3.hierarchy(data)
      .sum(d => d.value)
      .sort((a, b) => b.value - a.value));

  const leaf = svg.selectAll("g")
    .data(root.leaves())
    .join("g")
    .attr("transform", d => `translate(${d.x0},${d.y0})`);

  const nodeIds = [];
  leaf.each(d => {
    nodeIds.push(d.data.name.split("/").pop());
  });
  store.commit('UPDATE_ALL_VISIABLE_NODES', nodeIds);
  nextTick(() => {
    highlightGroup();
  });
};

const removeFromGroup = (group, nodeId) => {
  const step = active.value;
  store.commit('REMOVE_NODE_FROM_GROUP', { step, group, nodeId });
  nextTick(() => {
    highlightGroup();
  });
};

const addNewGroup = () => {
  const step = active.value;
  
  // 检查当前组的评分情况 
  if (selectedGroup.value && ratings.value[selectedGroup.value]) {
    const currentRatings = ratings.value[selectedGroup.value];
    if (currentRatings.attention === 1 && 
        currentRatings.correlation_strength === 1 && 
        currentRatings.exclusionary_force === 1) {
      ElMessage.warning({
        message: '请确定前一组合的三个评分都是低，如已确定，请忽略该提示',
        duration: 3000,  // 显示5秒
        showClose: true
      });
    }
  }
  
  const newGroup = `组合${Object.keys(groups.value).length + 1}`;
  store.commit('ADD_NEW_GROUP', { step, group: newGroup });
  selectedGroup.value = newGroup;
  ratings.value[newGroup] = { attention: 1, correlation_strength: 1, exclusionary_force: 1 };
  nextTick(() => {
    highlightGroup();
  });
};

const groups = computed(() => store.getters.getGroups(active.value));

const filteredGroups = computed(() => {
  const result = {};
  for (const group of Object.keys(groups.value)) {
    result[group] = groups.value[group];
    if (!ratings.value[group]) {
      ratings.value[group] = { attention: 1, correlation_strength: 1, exclusionary_force: 1 };
    }
  }
  return result;
});

const groupOptions = computed(() => Object.keys(groups.value));

const ensureGroupInitialization = () => {
  const step = active.value;
  if (!groups.value['组合1']) {
    store.commit('ADD_NEW_GROUP', { step, group: '组合1' });
    ratings.value['组合1'] = { attention: 1, correlation_strength: 1, exclusionary_force: 1 };
  }
};

const exportToJson = () => {
  const data = store.state.submittedData;
  if (!data) {
    ElMessage.error('请先提交问卷后再导出');
    return;
  }
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  saveAs(blob, `${store.getters.getFormData?.id || 'anonymous'}.json`);
};

onMounted(async () => {
  if (!checkUserId()) return;
  const count = await getSubmissionCount();
  if (count >= 50) {
    router.push('/limit-reached');
  }
  store.dispatch('initializeSteps');
  if (steps.value && steps.value.length > 0) {
    fetchSvgContent(steps.value[active.value]);
  }
  fetchAndRenderTree();
  ensureGroupInitialization();
  startReminderTimer();
  startTotalTimer();
});

watch([active, groups], () => {
  ratings.value = {};
  const stepRatings = store.state.ratings[active.value] || {};
  for (const group in groups.value) {
    ratings.value[group] = stepRatings[group] || { attention: 1, correlation_strength: 1, exclusionary_force: 1 };
  }
  nextTick(() => {
    highlightGroup();
  });
});

watch(steps, (newSteps) => {
  if (newSteps && newSteps.length > 0) {
    nextTick(() => {
      fetchSvgContent(newSteps[active.value]);
    });
  }
});

watch(active, async () => {
  await fetchSvgContent(store.state.steps[active.value]);
  await fetchAndRenderTree();
  ensureGroupInitialization();
  // 关闭选框模式
  isCropping.value = false;
  // 关闭路径选择模式
  isTracking.value = false;
  if (svgContainer2.value) {
    svgContainer2.value.classList.remove('crosshair-cursor');
    svgContainer2.value.classList.remove('copy-cursor');
  }
  // 禁用相应的事件处理
  disableCropSelection();
  disableTrackMode();
  nextTick(() => {
    highlightGroup();
  });
});

watch(selectedNodeIds, () => {
  nextTick(() => {
    highlightGroup();
  });
});

watch(allVisiableNodes, () => {
  turnGrayVisibleNodes();
  addHoverEffectToVisibleNodes();
  addClickEffectToVisibleNodes();
  highlightGroup();
});

onBeforeMount(() => {
  if (!checkUserId()) return;
});
</script>

<style scoped>
.common-layout {
  display: flex;
  flex-direction: column;
  height: 98vh;
  width: 70vw;
}

.header {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 10px;
  border-bottom: 1px solid #dcdcdc;
}

.header-content {
  display: flex;
  justify-content: space-between;
  width: 100%;
}

.left-content {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  color: #999;
}

.right-content {
  display: flex;
  align-items: center;
}

.id {
  font-size: 16px;
  font-weight: bold;
}

.main-card {
  width: 100%;

  .left-two {
    display: flex;
    flex-direction: column;
    width: 200%;
    margin-right: 10px;

    .top-card {
      margin-bottom: 10px;
      height: 100%;
    }

    .bottom-card {
      position: relative;
      height: 105%;

      .Crop {
        position: absolute;
        top: 10px;
        right: 10px;
      }

      .track {
        position: absolute;
        top: 10px;
        right: 65px;
      }

      .bottom-title {
        position: absolute;
        top: 5px;
        left: -15px;
      }
    }
  }

  .group-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;

    .select-group {
      display: flex;
      align-items: center;

      .el-select {
        margin-right: 10px;
        width: 200px;
      }
    }

    .group {
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 100%;
      margin-top: 10px;

      .group-tags-container {
        width: 100%;
        height: 100%;
      }

      .group-tags {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-start;
        width: 300px;

        .el-tag {
          margin: 5px;
          flex: 1 0 calc(33.33% - 10px);
          box-sizing: border-box;
          text-align: center;
          cursor: pointer;
        }
      }
    }
  }
}


.steps-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  margin: 25px 0;
}

.steps {
  flex-grow: 1;
  margin: 0 20px;
}

.top-card {
  position: relative;

  .top-title {
    position: absolute;
    top: 5px;
    left: -5px;
  }
}

.crosshair-cursor {
  cursor: crosshair !important;
}

.copy-cursor {
  cursor: copy !important;
}

.flow {
  position: absolute;
  left: 10px;
  top: 100px;
  width: 15vw;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.flow2 {
  position: absolute;
  right: 10px;
  top: 100px;
  width: 15vw;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.flow-header {
  padding: 0;
  margin: 0;
}

.flow-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.flow-content {
  padding: 10px 0;
}

.step-item {
  margin-bottom: 15px;
}

.step-number {
  display: block;
  font-size: 14px;
  color: #409EFF;
  margin-bottom: 8px;
  font-weight: 500;
}

.step-card {
  margin: 0;
  border: none;
  background-color: #f5f7fa;

  :deep(.el-card__body) {
    padding: 12px;
  }

  p {
    margin: 0;
    font-size: 14px;
    color: #606266;
    line-height: 1.6;
  }
}

.step-list {
  margin: 8px 0 0 0;
  padding-left: 20px;

  li {
    color: #606266;
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 4px;

    &:last-child {
      margin-bottom: 0;
    }
  }
}

.tips-content {
  padding: 5px 0;
}

.tips-list {
  margin: 0;
  padding-left: 20px;

  li {
    color: #606266;
    font-size: 14px;
    line-height: 1.8;
    margin-bottom: 8px;

    &:last-child {
      margin-bottom: 0;
    }
  }
}

.buzhou {
  font-size: 12px;
  color: #999;
}

.rate-container2 {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  margin: 10px 0;
}

.rate-text {
  text-align: left;
  min-width: 200px;
}

.tour-dialog :deep(.el-dialog__header) {
  padding: 20px;
  margin-right: 0;
  border-bottom: 1px solid #eee;
}

.tour-dialog :deep(.el-dialog__title) {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
}

.dialog-content {
  padding: 30px 20px;
}

.dialog-text {
  font-size: 16px;
  color: #303133;
  margin-bottom: 12px;
}

.dialog-subtext {
  font-size: 14px;
  color: #909399;
  margin: 0;
}

.dialog-footer {
  padding: 20px;
  border-top: 1px solid #eee;
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.skip-btn {
  min-width: 80px;
}

.start-btn {
  min-width: 80px;
}

.practice-step :deep(.el-tour-step__title) {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 16px;
}

.practice-content {
  padding: 0 10px;
}

.practice-text {
  font-size: 15px;
  color: #606266;
  line-height: 1.8;
  margin: 8px 0;
}

.practice-note {
  font-size: 14px;
  color: #909399;
  line-height: 1.6;
  margin: 12px 0 8px;
  font-style: italic;
}

.svg-container, .svg-container2 {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 300px;
}

.svg-container svg, .svg-container2 svg {
  width: 100%;
  height: 100%;
  display: block;
  min-height: inherit;
}

.active-mode {
  background-color: var(--el-button-hover-bg-color) !important;
  border-color: var(--el-button-hover-border-color) !important;
}

.icon-btn {
  padding: 4px 8px;
  margin: 0 4px;
  vertical-align: middle;
  min-width: 32px;
}

.icon-btn :deep(.el-icon) {
  margin: 0;
}

.tips-list .highlight-tip {
  font-size: 15px;
  color: #409EFF;
}

.tips-list .highlight-tip strong {
  font-weight: 600;
}

.underline-text {
  text-decoration: underline;
}

/* 禁止所有文本选择 */
* {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/* 特别处理 SVG text 元素 */
:deep(svg text) {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  pointer-events: auto;
}

.submit-button {
  order: 1;
}

.export-button {
  order: 2;
  margin-left: 10px;
}

.steps-container {
  justify-content: center;
}
</style>
