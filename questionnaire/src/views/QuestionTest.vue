<template>
  <div class="common-layout">
    <el-container class="full-height">
      <el-header class="header">
        <div class="header-content">
          <div ref="idandtime" class="left-content">
            <a href="https://github.com/729149195/questionnaire" target="_blank">
              <img style="width: 30px;" src="/img/favicon.png" alt="Wechat QR Code">
            </a>
          </div>
          <div class="right-content">
            <!-- <el-button ref="openDialogBtn" plain @click="infoDialogVisible = true">
              打开说明<el-icon style='margin-left:5px'>
                <WindPower />
              </el-icon>
            </el-button> -->
          </div>
        </div>
      </el-header>
      <el-main class="main-card">
        <el-card>
          <div style="display: flex;">
            <div class="left-two">
              <el-card ref="svg1" class="top-card" shadow="never">
                <div v-html="Svg" class="svg-container"></div>
                <el-button class="top-title" disabled text bg>组合观察区域</el-button>
              </el-card>
              <el-card ref="svg2" class="bottom-card" shadow="never">
                <div v-html="Svg" class="svg-container2" ref="svgContainer2"></div>
                <div ref="chartContainer" class="chart-container" v-show="false"></div>
                <el-button @click="toggleCropMode" class="Crop" ref="cropBtn" :class="{ 'active-mode': isCropping }"><el-icon>
                    <Crop />
                  </el-icon></el-button>
                <el-button @click="toggleTrackMode" class="track" ref="trackBtn" :class="{ 'active-mode': isTracking }"><el-icon>
                    <Pointer />
                  </el-icon></el-button>
                <el-button class="bottom-title" disabled text bg>选取交互区域</el-button>
              </el-card>
            </div>
            <el-card ref="groupCard" class="group-card" shadow="never">
              <div class="select-group">
                <el-select ref="groupSelector" v-model="selectedGroup" placeholder="选择组合" @change="highlightGroup">
                  <el-option v-for="(group, index) in groupOptions" :key="index" :label="group" :value="group" />
                </el-select>
                <el-button ref="addGroupBtn" @click="addNewGroup"><el-icon>
                    <Plus />
                  </el-icon></el-button>
                <el-button ref="deleteGroupBtn" @click="deleteCurrentGroup"><el-icon>
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
                  <el-tooltip class="box-item" effect="dark" content="越先被注意到的组合评分越高" placement="right">
                    <div class="rate-container2">
                      <div class="rate-text">显眼程度：</div>
                      <el-rate :icons="icons" :void-icon="Hide" :colors="['#409eff', '#67c23a', '#FF9900']" :max="3"
                        :texts="['低', '中', '高']" show-text v-model="ratings[selectedGroup].attention" class="rate"
                        @change="updateRating(selectedGroup, ratings[selectedGroup].attention, 'attention')" />
                    </div>
                  </el-tooltip>
                  <el-tooltip class="box-item" effect="dark" content="组合中不可缺少的元素占比越高评分越高" placement="right">
                    <div class="rate-container2">
                      <div class="rate-text">分组组内元素的关联强度：</div>
                      <el-rate :icons="icons" :void-icon="Hide" :colors="['#409eff', '#67c23a', '#FF9900']" :max="3"
                        :texts="['低', '中', '高']" show-text v-model="ratings[selectedGroup].correlation_strength"
                        class="rate"
                        @change="updateRating(selectedGroup, ratings[selectedGroup].correlation_strength, 'correlation_strength')" />
                    </div>
                  </el-tooltip>
                  <el-tooltip class="box-item" effect="dark" content="组外可以划分到该组的元素越少评分越高" placement="right">
                    <div class="rate-container2">
                      <div class="rate-text">分组对组外元素的排斥程度：</div>
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
          <el-button ref="previousBtn" class="previous-button" @click="Previous"><el-icon>
              <CaretLeft />
            </el-icon></el-button>
          <el-steps :active="active" finish-status="success" class="steps" ref="stepsContainer">
            <el-step v-for="(step, index) in steps" :key="index" @click.native="goToStep(index)" />
          </el-steps>
          <el-tooltip content="查看示例组合" placement="top-start" :hide-after="1000">
            <el-button ref="nextBtn" class="next-button" @click="next" type="primary" v-if="active != steps.length - 1">
              <el-icon><CaretRight /></el-icon>
            </el-button>
          </el-tooltip>
          <el-tooltip content="进入正式问卷" placement="top-start" :hide-after="1000">
            <el-button class="submit-button" @click="submit" type="success" v-if="active === steps.length - 1">
              <el-icon><DArrowRight /></el-icon>
            </el-button>
          </el-tooltip>
        </div>
      </el-main>

    </el-container>
    <el-dialog v-model="infoDialogVisible" title="问卷说明" width="800" align-center>
      <div class="info-content">
        <h3 class="info-subtitle">开始问卷前，请了解以下要点：</h3>
        <ol class="info-list">
          <li>请根据您的直观感受，选出所有您认为应该归为一组的图形元素</li>
          <li>同一个图形元素可以同时属于多个不同的组合</li>
          <li>评分时请跟随第一印象，无需过度分析</li>
        </ol>
      </div>
    </el-dialog>

    <el-dialog v-model="tourDialogVisible" title="使用引导" width="500" class="tour-dialog">
      <div class="dialog-content">
        <p class="dialog-text">是否需要查看操作指南？</p>
        <p class="dialog-subtext">如果您已熟悉系统操作，可以直接跳过</p>
      </div>
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="tourDialogVisible = false" class="skip-btn">跳过</el-button>
          <el-button @click="startTour" type="primary" class="start-btn">开始引导</el-button>
        </div>
      </template>
    </el-dialog>
    <el-tour v-model="openTour">
      <el-tour-step :target="svg1?.$el" title="组合观察区域" placement="right">您将在这里观察原图并进行图形组合的感知。</el-tour-step>
      <el-tour-step :target="svg2?.$el" placement="right" title="选取交互区域">
        在这里，您可以通过点击元素来添加或删除它们，以构建或修改当前的图形组合。您还可以使用鼠标滚轮进行缩放，以便更好地查看和选择细小的元素。<div v-html="getGifHtml('2.gif')"></div>
      </el-tour-step>
      <el-tour-step :target="cropBtn?.$el" placement="right" title="切换框选按钮">
        当遇到的图形组合元素较为细小时，可以点击进入选框组合进行元素框选，被框选的元素相当于被点击一下，未被选中的被框选到会被选中，已选中的被框选到会取消选中（再次点击即可退出选框组合，选框组合下也可进行当个元素的点击）。<div
          v-html="getGifHtml('3.gif')"></div></el-tour-step>
      <el-tour-step :target="trackBtn?.$el" placement="right" title="切换路径选择按钮">
        当遇到的图形组合中元素较为密集时，可以点击进入路径选择组合进行元素路径选择，被按住的鼠标经过的元素相当于被点击一下（再次点击即可退出路径选择组合，选框组合下也可进行当个元素的点击）。<div
          v-html="getGifHtml('12.gif')"></div></el-tour-step>
      <el-tour-step :target="groupCard?.$el" title="分组卡片"
        placement="left">显示选中组合中所包含标签，以及一些操作按钮，点击蓝色标签后可在选取交互区域定位到单一标签所对应的元素，也可通过取消蓝色标签来移除对应元素。
        <div v-html="getGifHtml('4.gif')"></div>
      </el-tour-step>
      <el-tour-step :target="groupSelector?.$el" title="分组选择器">在这里可以下拉选择已创建的组合。<div v-html="getGifHtml('5.gif')"></div>
      </el-tour-step>
      <el-tour-step :target="addGroupBtn?.$el" title="添加分组按钮">点击这里可以添加新的组合。<div v-html="getGifHtml('6.gif')"></div>
      </el-tour-step>
      <el-tour-step :target="deleteGroupBtn?.$el" title="删除分组按钮"> 点击这里可以删除当前组合及其内，后续的内容会往前覆盖同时继承被删除的组合编号。<div
          v-html="getGifHtml('7.gif')"></div></el-tour-step>
      <el-tour-step :target="rateings?.$el" title="组合评分">
        <p>显眼程度：越先被注意到的组合评分越高</p>
        <p>分组组内元素的关联强度：组合中不可缺少的元素占比越高评分越高</p>
        <p>分组对组外元素的排斥程度：组外可以划分到该组的元素越少评分越高</p>
        <p>请根据第一印象为每一个图形组合估计评分。</p>
        <div v-html="getGifHtml('8.gif')"></div>
      </el-tour-step>
      <el-tour-step :target="stepsContainer?.$el" title="问卷进度">这里显示了问卷的进度。已完成的示例节点会变绿。<div v-html="getGifHtml('9.gif')">
        </div></el-tour-step>
      <el-tour-step :target="previousBtn?.$el" title="上一个按钮">点击这里可以回到上一个示例节点。<div v-html="getGifHtml('10.gif')"></div>
      </el-tour-step>
      <el-tour-step :target="nextBtn?.$el" title="下一个按钮">点击这里可以前往下一个示例节点。到最后一个节点时该按钮会变为绿色的提交按钮，点击后获取ID并导出图形组合数据。<div
          v-html="getGifHtml('11.gif')"></div></el-tour-step>
      <el-tour-step title="尝试" class="practice-step">
        <div class="practice-content">
          <p class="practice-text">现在可以使用当前示例进行练习</p>
          <p class="practice-text">并对下一个示例已选好的组合进行浏览</p>
          <p class="practice-note">（示例仅供参考, 不用选出那么多组合，只用把自己感觉到的组合选出即可）</p>
        </div>
      </el-tour-step>
    </el-tour>
  </div>
  <el-card class="flow">
    <template #header>
      <div class="flow-header">
        <span class="flow-title">操作流程提示</span>
      </div>
    </template>
    <div class="flow-content">
      <div class="step-item">
        <span class="step-number">步骤1:</span>
        <el-card class="step-card" shadow="hover">
          <p>查看组合观察区域（左上角板块）并记下感知到的元素组合。</p>
        </el-card>
      </div>
      <div class="step-item">
        <span class="step-number">步骤2:</span>
        <el-card class="step-card" shadow="hover">
          <p>在选取交互区域选择您感知中可以组成一个组合的所有元素</p>
          <ul class="step-list">
            <li>组合元素较密集的时候，建议使用
              <el-button class="icon-btn" size="small">
                <el-icon><Crop /></el-icon>
              </el-button> 
              框选或
              <el-button class="icon-btn" size="small">
                <el-icon><Pointer /></el-icon>
              </el-button> 
              路径选择功能批量选区元素
            </li>
            <li>已经被选中的元素再次被选择后，会取消选中状态</li>
            <li>非路径/选框模式下，鼠标滚轮可以对交互图表放大缩小拖动</li>
          </ul>
        </el-card>
      </div>
      <div class="step-item">
        <span class="step-number">步骤3:</span>
        <el-card class="step-card" shadow="hover">
          <p>选取完一个组后，若还有其他组合未添加，点击组合板块的加号按钮创建新组。</p>
        </el-card>
      </div>
      <div class="step-item">
        <span class="step-number">步骤4:</span>
        <el-card class="step-card" shadow="hover">
          <p>组元素选完后不要忘记评分嗷~</p>
        </el-card>
      </div>
    </div>
  </el-card>
  <el-card class="flow2">
    <template #header>
      <div class="flow-header">
        <span class="flow-title">友情提示</span>
      </div>
    </template>
    <div class="tips-content">
      <ul class="tips-list">
        <li class="highlight-tip">
          <strong>请尽可能多地选出自己感知到的图形组合</strong>
        </li>
        <li>相同的元素在不同的组合中可以重复选择</li>
        <li>尽量遵循自己的第一印象</li>
      </ul>
    </div>
  </el-card>
</template>

<script setup>
import { ref, computed, onMounted, nextTick, watch, onBeforeMount } from 'vue';
import { useStore } from 'vuex';
import { useRouter } from 'vue-router';
import * as d3 from 'd3';
import { Delete, Plus, Hide, View, CaretLeft, CaretRight, DArrowRight, WindPower, Crop, Pointer } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';
import { getSubmissionCount } from '../api/counter';

const store = useStore();
const router = useRouter();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);
const allVisiableNodes = computed(() => store.state.AllVisiableNodes);
const dialogVisible = ref(false);
const tourDialogVisible = ref(false);
const infoDialogVisible = ref(false);
const openTour = ref(false);
const active = ref(0);
const steps = Array.from({ length: 2 });
const icons = [View, View, View];

const svgContainer2 = ref(null);

const Svg = ref('');
const selectedGroup = ref('组合1');
const ratings = ref({});

const idandtime = ref(null);
const svg1 = ref(null);
const svg2 = ref(null);
const groupCard = ref(null);
const stepsContainer = ref(null);
const rateings = ref(null);
const openDialogBtn = ref(null);
const groupSelector = ref(null);
const addGroupBtn = ref(null);
const deleteGroupBtn = ref(null);
const previousBtn = ref(null);
const nextBtn = ref(null);
const cropBtn = ref(null);
const trackBtn = ref(null);
const nodeEventHandlers = new Map();
const isCropping = ref(false);
const isTracking = ref(false);

// 添加状态存储
const currentTransform = ref(null);

const props = defineProps(['data']);
const emits = defineEmits(['change', 'prev', 'next']);

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
    ensureGroupInitialization(); // 确保合初始化
    nextTick(() => {
      highlightGroup(); // 确保合在初始加时被高亮
    });
    isCropping.value = false;
    svgContainer2.value.classList.remove('crosshair-cursor');
    await loadExampleData();
  }
};


const currentGroupNodes = computed(() => {
  if (!ratings.value[selectedGroup.value]) {
    ratings.value[selectedGroup.value] = { attention: 1, correlation_strength: 1, exclusionary_force: 1 };
  }
  return groups.value[selectedGroup.value] || [];
});

const getGifHtml = (filename) => {
  const gifPath = `./gif/${filename}`;
  return `<img src="${gifPath}" alt="GIF" style="max-width: 100%; height: auto;">`;
};

const updateRating = (group, rating, type) => {
  const step = active.value;
  store.commit('UPDATE_RATING', { step, group, rating, type });
};

// 加载并提交 example.json 数据
const loadExampleData = async () => {
  try {
    const step = active.value + 1;
    const response = await fetch(`./TestData/${step}/example.json`);

    if (!response.ok) {
      throw new Error('Failed to fetch example data');
    }

    const data = await response.json();
    
    // 清空当前步骤的评分数据
    store.commit('RESET_STEP_RATINGS', active.value);
    
    // 更新本地 ratings 对象
    ratings.value = {};

    // 动态更新基于当前步骤的数据
    data.groups.forEach((groupData) => {
      const groupName = groupData.group;
      
      // 添加组和节点
      store.commit('ADD_NEW_GROUP', { step: active.value, group: groupName });
      store.commit('ADD_OTHER_GROUP', { step: active.value, group: groupName, nodeIds: groupData.nodes });
      
      // 更新评分
      const groupRatings = {
        attention: groupData.ratings.attention,
        correlation_strength: groupData.ratings.correlation_strength,
        exclusionary_force: groupData.ratings.exclusionary_force
      };
      
      // 更新 store 中的评分
      store.commit('UPDATE_RATING', {
        step: active.value,
        group: groupName,
        rating: groupRatings.attention,
        type: 'attention'
      });
      store.commit('UPDATE_RATING', {
        step: active.value,
        group: groupName,
        rating: groupRatings.correlation_strength,
        type: 'correlation_strength'
      });
      store.commit('UPDATE_RATING', {
        step: active.value,
        group: groupName,
        rating: groupRatings.exclusionary_force,
        type: 'exclusionary_force'
      });
      
      // 更新本地 ratings 对象
      ratings.value[groupName] = groupRatings;
    });

    nextTick(() => {
      highlightGroup();
    });
  } catch (error) {
    console.error('Error loading example data:', error);
  }
};



const fetchSvgContent = async (step) => {
  try {
    nodeEventHandlers.forEach((handler, node) => {
      node.removeEventListener('click', handler);
    });
    nodeEventHandlers.clear();

    const response = await fetch(`./TestData/${step}/${step}.svg`);
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const svgContent = await response.text();
    Svg.value = svgContent;
    turnGrayVisibleNodes();
    addHoverEffectToVisibleNodes();
    addClickEffectToVisibleNodes();
    nextTick(() => {
      highlightGroup();
      addZoomEffectToSvg();
    });
  } catch (error) {
    console.error('Error loading SVG content:', error);
    Svg.value = '<svg><text x="10" y="20" font-size="20">加载SVG时出错</text></svg>';
  }
  await loadExampleData();
};

const addZoomEffectToSvg = () => {
  const svgContainer = svgContainer2.value;
  if (!svgContainer) return;
  const svg = d3.select(svgContainer).select('svg');
  if (!svg) return;

  // 创建一个包裹实际SVG内容的组
  let g = svg.select('g.zoom-wrapper');
  if (g.empty()) {
    g = svg.append('g').attr('class', 'zoom-wrapper');
    // 将所有现有内容移动到新的组中
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
      if (!isCropping.value) {
        g.attr('transform', event.transform);
      }
    });

  svg.call(zoom);

  // 获取参考 SVG 的位置和尺寸
  const referenceSvg = d3.select('.svg-container svg');
  if (referenceSvg.node()) {
    // 获取两个 SVG 的 viewBox
    const refViewBox = referenceSvg.node().viewBox.baseVal;
    const currentViewBox = svg.node().viewBox.baseVal;

    // 获取实际显示尺寸
    const refRect = referenceSvg.node().getBoundingClientRect();
    const currentRect = svg.node().getBoundingClientRect();

    // 计算缩放比例
    const scaleX = (refRect.width / refViewBox.width) / (currentRect.width / currentViewBox.width);
    const scaleY = (refRect.height / refViewBox.height) / (currentRect.height / currentViewBox.height);
    const scale = Math.min(scaleX, scaleY);

    // 计算偏移量，使两个 SVG 的内容对齐
    const refCenterX = refViewBox.x + refViewBox.width / 2;
    const refCenterY = refViewBox.y + refViewBox.height / 2;
    const currentCenterX = currentViewBox.x + currentViewBox.width / 2;
    const currentCenterY = currentViewBox.y + currentViewBox.height / 2;

    const translateX = (refCenterX - currentCenterX) * scale + (refRect.width - currentRect.width * scale) / 2;
    const translateY = (refCenterY - currentCenterY) * scale + (refRect.height - currentRect.height * scale) / 2;

    // 应用变换
    const initialTransform = d3.zoomIdentity
      .translate(translateX, translateY)
      .scale(scale);

    svg.call(zoom.transform, initialTransform);
  }
};


let isDrawing = false; // 标志是否正在绘制
let rectElement; // 矩元素
let handleMouseClick, handleMouseMove, handleMouseUp; // 件处理程序

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
    
    // 重新启用缩放并恢复之前的变换状态
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
      
      // 获取当前的变换矩阵
      const transform = d3.zoomTransform(svg);
      
      // 获取鼠标点击的SVG坐标
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      
      // 应用当前变换的逆矩阵来获取正确的起始点坐标
      const matrix = svg.getScreenCTM().inverse();
      const transformedPoint = point.matrixTransform(matrix);
      
      startX = transformedPoint.x;
      startY = transformedPoint.y;

      rectElement = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rectElement.setAttribute('x', startX);
      rectElement.setAttribute('y', startY);
      rectElement.setAttribute('stroke', 'red');
      rectElement.setAttribute('stroke-width', '2');
      rectElement.setAttribute('fill', 'none');
      
      // 将矩形添加到zoom-wrapper组内，这样它会跟随缩放
      const wrapper = svg.querySelector('.zoom-wrapper');
      wrapper.appendChild(rectElement);

      svg.addEventListener('mousemove', handleMouseMove);
      svg.addEventListener('mouseup', handleMouseUp);
    }
  };

  handleMouseMove = (event) => {
    if (isDrawing) {
      // 获取当前鼠标位置的实际SVG坐标
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      
      const matrix = svg.getScreenCTM().inverse();
      const transformedPoint = point.matrixTransform(matrix);
      
      const endX = transformedPoint.x;
      const endY = transformedPoint.y;
      
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

      // 获取当前的变换矩阵
      const transform = d3.zoomTransform(svg);

      svg.querySelectorAll('*').forEach(node => {
        if (typeof node.getBBox === 'function' && allVisiableNodes.value.includes(node.id)) {
          const bbox = node.getBBox();
          
          // 获取元素的实际变换矩阵
          const ctm = node.getCTM();
          if (!ctm) return;
          
          // 计算元素四个角的坐标
          const points = [
            {x: bbox.x, y: bbox.y},
            {x: bbox.x + bbox.width, y: bbox.y},
            {x: bbox.x, y: bbox.y + bbox.height},
            {x: bbox.x + bbox.width, y: bbox.y + bbox.height}
          ];
          
          // 检查任何一个角是否在选区内
          const isInside = points.some(point => {
            const transformedPoint = svg.createSVGPoint();
            transformedPoint.x = point.x;
            transformedPoint.y = point.y;
            const screenPoint = transformedPoint.matrixTransform(ctm);
            
            return screenPoint.x >= rectX &&
                   screenPoint.x <= (rectX + rectWidth) &&
                   screenPoint.y >= rectY &&
                   screenPoint.y <= (rectY + rectHeight);
          });

          if (isInside) {
            node.dispatchEvent(new Event('click'));
          }
        }
      });

      rectElement.remove();
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
    ElMessage.info('退出路径组模式');
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
    clickedElements.clear(); // 重置点击元素集合
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
  const svg = d3.select(svgContainer).select('svg');
  if (!svg) return;

  svg.selectAll('*').each(function() {
    const node = d3.select(this);
    if (allVisiableNodes.value.includes(this.id)) {
      node.style('opacity', '0.2');
      node.style.transition = 'opacity 0.3s ease';
    }
  });
};

const addHoverEffectToVisibleNodes = () => {
  const svgContainer = svgContainer2.value;
  if (!svgContainer) return;
  const svg = d3.select(svgContainer).select('svg');
  if (!svg) return;

  svg.selectAll('*').each(function() {
    const node = d3.select(this);
    if (allVisiableNodes.value.includes(this.id)) {
      const handleMouseOver = () => {
        node.style('opacity', '1');
      };
      const handleMouseOut = () => {
        node.style('opacity', '0.2');
        node.style.transition = 'opacity 0.3s ease';
        highlightGroup();
      };

      node
        .on('mouseover', handleMouseOver)
        .on('mouseout', handleMouseOut);
    }
  });
};

const addClickEffectToVisibleNodes = () => {
  const svgContainer = svgContainer2.value;
  if (!svgContainer) return;
  const svg = d3.select(svgContainer).select('svg');
  if (!svg) return;

  svg.selectAll('*').each(function() {
    const node = d3.select(this);
    if (allVisiableNodes.value.includes(this.id)) {
      const handleNodeClick = () => {
        const groupNodes = store.state.groups[active.value]?.[selectedGroup.value] || [];
        if (groupNodes.includes(this.id)) {
          store.commit('REMOVE_NODE_FROM_GROUP', { step: active.value, group: selectedGroup.value, nodeId: this.id });
        } else {
          store.commit('ADD_NODE_TO_GROUP', { step: active.value, group: selectedGroup.value, nodeId: this.id });
        }
        nextTick(() => {
          highlightGroup();
        });
      };

      node.on('click', handleNodeClick);
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
  const step = active.value + 1;
  return `./TestData/${step}/layer_data.json`;
});

const chartContainer = ref(null);

const next = async () => {
  if (!checkUserId()) return;
  if (active.value < steps.length - 1) {
    selectedGroup.value = '组合1';
    active.value++;
    await fetchSvgContent(active.value + 1);
    await fetchAndRenderTree();
    ensureGroupInitialization();
    nextTick(() => {
      highlightGroup();
    });
    isCropping.value = false;
    svgContainer2.value.classList.remove('crosshair-cursor');
    await loadExampleData();
  }
};

const Previous = async () => {
  if (!checkUserId()) return;
  if (active.value > 0) {
    selectedGroup.value = '组合1';
    active.value--;
    await fetchSvgContent(active.value + 1);
    await fetchAndRenderTree();
    ensureGroupInitialization();
    nextTick(() => {
      highlightGroup();
    });
    isCropping.value = false;
    svgContainer2.value.classList.remove('crosshair-cursor');
  }
};

const submit = () => {
  resetTrainingData();
  router.push('/questions');
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
  const height = 300;

  d3.select(chartContainer.value).select('svg').remove();

  const svg = d3.select(chartContainer.value)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .style('width', '100%')
    .style('height', 'auto');

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
};

const addToGroup = (nodeId) => {
  const step = active.value;
  store.commit('ADD_NODE_TO_GROUP', { step, group: selectedGroup.value, nodeId });
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

const resetTrainingData = () => {
  store.commit('RESET_TRAINING_DATA');
};

const startTour = () => {
  tourDialogVisible.value = false;
  openTour.value = true;
};

onMounted(async () => {
  tourDialogVisible.value = true; // Show the tour dialog on mount
  await fetchSvgContent(active.value + 1);
  await fetchAndRenderTree();
  ensureGroupInitialization();
  const count = await getSubmissionCount();
  if (count >= 50) {
    router.push('/limit-reached');
  }
});

onMounted(() => {
  const stepRatings = store.state.ratings[active.value] || {};
  for (const group in groups.value) {
    ratings.value[group] = stepRatings[group] || { attention: 1, correlation_strength: 1, exclusionary_force: 1 };
  }
  nextTick(() => {
    highlightGroup(); // Ensure the group is highlighted on initial load
  });
});

watch([active, groups], () => {
  // 获取当前步骤的评分
  const stepRatings = store.state.ratings[active.value] || {};
  
  // 重置本地 ratings 对象
  ratings.value = {};
  
  // 为每个组设置评分
  for (const group in groups.value) {
    if (stepRatings[group]) {
      // 如果存在评分则使用已有评分
      ratings.value[group] = stepRatings[group];
    } else {
      // 否则设置默认评分
      ratings.value[group] = { 
        attention: 1, 
        correlation_strength: 1, 
        exclusionary_force: 1 
      };
    }
  }
  
  nextTick(() => {
    highlightGroup();
  });
});

watch(active, async () => {
  await fetchSvgContent(active.value + 1);
  await fetchAndRenderTree();
  ensureGroupInitialization();
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
  margin: auto;
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
  height: auto;

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
    height: auto;

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
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.flow2 {
  position: absolute;
  right: 10px;
  top: 100px;
  width: 15vw;
  height: auto;
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

.rate {
  margin-left: auto;
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
}

.svg-container svg, .svg-container2 svg {
  width: 100%;
  height: 100%;
  display: block;
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
</style>
