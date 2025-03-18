<template>
    <div class="main">
        <!-- 添加系统介绍对话框 -->
        <el-dialog v-model="dialogVisible" width="600px" :show-close="true" :close-on-click-modal="false" :close-on-press-escape="true" class="intro-dialog">
            <div class="dialog-content">
                <div class="intro-header">
                    <div class="intro-icon">
                        <i class="el-icon-view"></i>
                    </div>
                    <h2 style="font-size: 1.8em; font-weight: 600; color: #905F29;">欢迎使用感知模拟可视化评估系统</h2>
                </div>

                <div class="intro-description">
                    <p>本系统可以模拟人类的视觉感知行为，识别不同的图形模式。它提供多样化的视觉感知结果，帮助可视化创作者（包括专业和非专业人士）发现潜在的图形模式。</p>
                    <p>此外，系统支持交互式、易理解的感知模拟机制，如：</p>

                </div>
                <div class="intro-features">
                    <ul>
                        <li><span class="feature-icon">✦</span> 图形模式高亮分析</li>
                        <li><span class="feature-icon">✦</span> 贡献视觉编码排名</li>
                        <li><span class="feature-icon">✦</span> 图形模式的量化感知概率</li>
                        <li><span class="feature-icon">✦</span> 可排序的差异统计</li>
                        <li><span class="feature-icon">✦</span> 支持迭代修改源代码和分析</li>
                    </ul>
                </div>
            </div>
            <template #footer>
                <div class="dialog-footer">
                    <el-button type="primary" @click="dialogVisible = false" color="#905F29" style="width: 100%;">开始使用</el-button>
                </div>
            </template>
        </el-dialog>

        <div class="left" :style="{ width: leftWidth + '%' }">
            <div class="left-top" :style="{ height: leftTopHeight + '%' }">
                <div class="fill-height">
                    <CodeToSvg />
                </div>
                <div class="resize-handle horizontal" @mousedown="startResizeLeftVertical"></div>
            </div>
            <div class="left-bottom" :style="{ height: (100 - leftTopHeight) + '%' }">
                <div class="svgZoom">
                    <SvgUploader @file-uploaded="handleFileUploaded" />
                </div>
            </div>
            <div class="resize-handle vertical" @mousedown="startResizeHorizontal"></div>
        </div>
        <div class="right" :style="{ width: (100 - leftWidth) + '%' }">
            <div class="fill-height datas">
                <div v-if="file" class="data-cards">
                    <div class="maxtistic">
                        <div class="subgroup-section" :style="{ height: subgroupHeight + '%' }">
                            <SubgroupVisualization v-if="file" :key="componentKey2" class="subgroup-visualization" />
                        </div>
                        <div class="resize-handle horizontal resize-right1" @mousedown="startResizeRightVertical1" 
                             :style="{ top: subgroupHeight + '%', width: '100%' }"></div>
                        <div class="statistics-section" :style="{ height: statisticsHeight + '%' }">
                            <StatisticsContainer :component-key="componentKey4" title="SVG Statistics" class="main-card" />
                        </div>
                        <div class="resize-handle horizontal resize-right2" @mousedown="startResizeRightVertical2"
                             :style="{ top: (subgroupHeight + statisticsHeight) + '%', width: '100%' }"></div>
                        <div class="analysis-section" :style="{ height: analysisHeight + '%' }">
                            <analysisWords title="Feature dimension mapping analysis" :update-key="componentKey2" class="analysis-words" />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import SubgroupVisualization from '../components/visualization/SubgroupVisualization.vue';
import SvgUploader from '../components/SvgUploader.vue';
import analysisWords from '@/components/statistics/analysisWords.vue';
import CodeToSvg from '@/components/visualization/CodeToSvg.vue';
import StatisticsContainer from '@/components/statistics/StatisticsContainer.vue';

// 添加 dialog 控制变量
const dialogVisible = ref(true)

const file = ref(null)
const componentKey = ref(0)
const componentKey2 = ref(1)
const componentKey4 = ref(2)
const componentKey3 = ref(3)

// 区域大小设置
const leftWidth = ref(50) // 左侧宽度百分比
const leftTopHeight = ref(45) // 左上区域高度百分比
const subgroupHeight = ref(30) // 右侧子组可视化区域高度百分比
const statisticsHeight = ref(41) // 统计区域高度百分比
const analysisHeight = ref(29) // 分析文字区域高度百分比

// 记录调整状态 - 改回普通变量，避免响应式问题
let isResizing = false;
let currentResize = '';
let startX = 0;
let startY = 0;
let startWidth = 0;
let startHeight = 0;

// 开始水平调整大小（左右分隔线）
const startResizeHorizontal = (e) => {
    isResizing = true;
    currentResize = 'horizontal';
    startX = e.clientX;
    startWidth = leftWidth.value;

    // 添加全局事件监听
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'ew-resize';
    e.preventDefault();
    e.stopPropagation();
}

// 开始左侧垂直调整大小（上下分隔线）
const startResizeLeftVertical = (e) => {
    isResizing = true;
    currentResize = 'leftVertical';
    startY = e.clientY;
    startHeight = leftTopHeight.value;

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'ns-resize';
    e.preventDefault();
    e.stopPropagation();
}

// 开始右侧第一个垂直调整（子组可视化和分析文字之间）
const startResizeRightVertical1 = (e) => {
    isResizing = true;
    currentResize = 'rightVertical1';
    startY = e.clientY;
    startHeight = subgroupHeight.value;

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'ns-resize';
    e.preventDefault();
    e.stopPropagation();
}

// 开始右侧第二个垂直调整（统计和分析文字之间）
const startResizeRightVertical2 = (e) => {
    isResizing = true;
    currentResize = 'rightVertical2';
    startY = e.clientY;
    startHeight = statisticsHeight.value;

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'ns-resize';
    e.preventDefault();
    e.stopPropagation();
}

// 处理鼠标移动
const handleMouseMove = (e) => {
    if (!isResizing) return;

    if (currentResize === 'horizontal') {
        // 计算左侧宽度百分比变化
        const container = document.querySelector('.main');
        const containerWidth = container.offsetWidth;
        const delta = e.clientX - startX;
        const newWidth = startWidth + (delta / containerWidth * 100);
        // 限制最小/最大宽度
        leftWidth.value = Math.max(20, Math.min(80, newWidth));
    }
    else if (currentResize === 'leftVertical') {
        // 计算左上区域高度百分比变化
        const container = document.querySelector('.left');
        const containerHeight = container.offsetHeight;
        const rect = container.getBoundingClientRect();
        const relativeY = e.clientY - rect.top;
        const newHeight = (relativeY / containerHeight) * 100;
        // 限制最小/最大高度
        leftTopHeight.value = Math.max(20, Math.min(80, newHeight));
    }
    else if (currentResize === 'rightVertical1') {
        // 调整子组可视化区域和统计区域之间的分割
        const container = document.querySelector('.maxtistic');
        const containerHeight = container.offsetHeight;
        const rect = container.getBoundingClientRect();
        const relativeY = e.clientY - rect.top;
        const newHeight = (relativeY / containerHeight) * 100;

        // 更新高度并保持总和为100%
        subgroupHeight.value = Math.max(20, Math.min(70, newHeight));
        // 计算剩余空间并分配给统计区域和分析区域
        const remaining = 100 - subgroupHeight.value;
        // 保持统计区域和分析区域的比例
        const ratio = statisticsHeight.value / (statisticsHeight.value + analysisHeight.value);
        statisticsHeight.value = Math.max(10, Math.min(40, remaining * ratio));
        analysisHeight.value = remaining - statisticsHeight.value;
    }
    else if (currentResize === 'rightVertical2') {
        // 调整统计区域和分析文字之间的分割
        const container = document.querySelector('.maxtistic');
        const containerHeight = container.offsetHeight;
        const rect = container.getBoundingClientRect();

        // 计算鼠标移动的相对距离
        const deltaY = e.clientY - startY;
        const deltaPercent = (deltaY / containerHeight) * 100;
        
        // 更新统计区域高度，考虑鼠标移动方向
        const newStatisticsHeight = startHeight + deltaPercent;
        
        // 限制统计区域的最小和最大高度
        statisticsHeight.value = Math.max(10, Math.min(60, newStatisticsHeight));
        
        // 确保三个区域的总高度为100%
        analysisHeight.value = 100 - subgroupHeight.value - statisticsHeight.value;
    }

    // 防止事件冒泡和默认行为
    e.stopPropagation();
    e.preventDefault();
}

// 处理鼠标松开
const handleMouseUp = (e) => {
    if (!isResizing) return;

    isResizing = false;
    currentResize = '';

    // 移除全局事件监听器
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'default';

    // 防止事件冒泡和默认行为
    if (e) {
        e.stopPropagation();
        e.preventDefault();
    }
}

const handleFileUploaded = () => {
    componentKey.value += 1;
    componentKey2.value += 1;
    componentKey3.value += 1;
    componentKey4.value += 1;
    file.value = true;
};

// 组件加载时清空 uploadSvg 目录
onMounted(async () => {
    try {
        const response = await fetch('http://127.0.0.1:5000/clear_upload_folder', {
            method: 'POST'
        });
        if (!response.ok) {
            console.error('Failed to empty upload folder');
        }
    } catch (error) {
        console.error('Error emptying upload folder:', error);
    }

    // 确保在组件卸载时清理任何可能的事件监听器
    window.addEventListener('beforeunload', cleanupEvents);
});

// 清理所有事件监听器
const cleanupEvents = () => {
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', handleMouseUp);
}

// 组件卸载前清理事件监听器
onBeforeUnmount(() => {
    cleanupEvents();
    window.removeEventListener('beforeunload', cleanupEvents);
});
</script>

<style scoped>
.main {
    display: flex;
    width: 100vw;
    height: 100vh;
    padding: 12px;
    box-sizing: border-box;
    gap: 12px;
    position: relative;
}

.main,
.main * {
    user-select: none;
}

.left {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    gap: 12px;
    position: relative;
}

.left-bottom {
    display: flex;
    gap: 12px;
    background-color: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.2);
    position: relative;
}

.left-bottom:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
}

.svgZoom {
    flex-grow: 1;
    padding: 12px;
}

.left-top {
    background-color: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.2);
    position: relative;
}

/* 调整大小的分隔线样式 */
.resize-handle {
    position: absolute;
    background-color: transparent;
    z-index: 100;
    transition: background-color 0.2s;
}

.resize-handle.vertical {
    cursor: ew-resize;
    width: 12px;
    height: 100%;
    top: 0;
    right: -6px;
    /* 调整到left容器的右侧边缘 */
    z-index: 1000;
}

.resize-handle.horizontal {
    cursor: ns-resize;
    height: 12px;
    width: 100%;
    left: 0;
    transform: translateY(-50%);
    z-index: 101;
}

.resize-right1, .resize-right2 {
    position: absolute;
    background-color: transparent;
    height: 12px;
    transform: translateY(-50%);
    cursor: ns-resize;
}

.resize-handle:hover,
.resize-handle:active {
    background-color: rgba(144, 95, 41, 0.3);
}

.resize-right1:hover, .resize-right2:hover,
.resize-right1:active, .resize-right2:active {
    background-color: rgba(144, 95, 41, 0.3);
}

.right {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    background-color: transparent;
    position: relative;
}

.datas {
    height: 100%;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.7) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    overflow: hidden;
    position: relative;
}

.datas:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08) !important;
    transform: translateY(-1px);
}

.title {
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: var(--el-text-color-primary);
    margin: 16px;
}

.fill-height {
    height: 100%;
    width: 100%;
    padding: 0;
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
    background-color: white;
    border-radius: 12px;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.03);
}

.svg-container svg {
    max-width: 80%;
    max-height: 80%;
    width: auto;
    height: auto;
    display: block;
}

.svg-container svg * {
    cursor: pointer;
}

.data-cards {
    height: 100%;
    width: 100%;
    display: flex;
    padding: 8px;
}

.maxtistic {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 4px;
    position: relative;
}

.maxtistic>* {
    border-radius: 12px;
}

/* 为了解决右侧面板的布局问题，添加子区域容器 */
.subgroup-section,
.analysis-section,
.statistics-section {
    position: relative;
    width: 100%;
    overflow: hidden;
}

.subgroup-visualization,
.analysis-words,
.main-card {
    height: 100%;
    width: 100%;
}

/* 只为 SubgroupVisualization 和 maxstic 添加悬浮效果 */
.maxtistic .subgroup-visualization:hover,
.maxtistic .maxstic:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
}

.maxtistic .analysis-words {
    background-color: transparent !important;
    box-shadow: none !important;
}

.maxtistic .analysis-words:hover {
    transform: none !important;
}

.main-card {
    width: 100%;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(200, 200, 200, 0.2);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
}

.position-card {
    flex: 1 1 calc(25% - 16px);
    min-width: 200px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(200, 200, 200, 0.3);
    padding: 12px;
}

.position-card:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
    border: 1px solid rgba(180, 180, 180, 0.4);
}

:deep(.v-card) {
    background-color: transparent !important;
    box-shadow: none !important;
}

:deep(.v-card-text) {
    padding: 0 !important;
}

/* 自定义滚动条样式 */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.3);
}

.title {
    position: absolute;
    top: 12px;
    left: 16px;
    font-size: 16px;
    font-weight: bold;
    color: #905F29;
    margin: 0;
    padding: 0;
    z-index: 10;
    letter-spacing: -0.01em;
    opacity: 0.8;
}

/* 添加对话框相关样式 */
.intro-dialog :deep(.el-dialog__header) {
    padding: 20px 24px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    margin-right: 0;
}

.intro-dialog :deep(.el-dialog__title) {
    font-size: 20px;
    font-weight: 600;
    color: #905F29;
}

.intro-dialog :deep(.el-dialog__body) {
    padding: 24px;
}

.intro-dialog :deep(.el-dialog__footer) {
    padding: 16px 24px;
    border-top: 1px solid rgba(0, 0, 0, 0.05);
}

.dialog-content {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
}

.intro-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 24px;
    text-align: center;
}

.intro-icon {
    font-size: 40px;
    color: #905F29;
    margin-bottom: 16px;
}

.intro-header h2 {
    font-size: 22px;
    color: #333;
    margin: 0;
}

.intro-description {
    margin-bottom: 10px;
    line-height: 1.6;
    font-size: 16px;
    color: #333;
    text-align: justify;
    text-indent: 2em;
    font-weight: 500;
}

.intro-description p {
    text-indent: 2em;
    margin: 0;
}

.intro-features {
    background: rgba(144, 95, 41, 0.05);
    border-radius: 12px;
    padding: 20px;
    font-size: 16px;
    margin-bottom: 10px;
    text-align: justify;
}

.intro-features h3 {
    margin-top: 0;
    margin-bottom: 16px;
    color: #905F29;
    font-size: 18px;
}

.intro-features ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.intro-features li {
    padding: 8px 0;
    display: flex;
    align-items: center;
}

.feature-icon {
    color: #905F29;
    font-size: 18px;
    margin-right: 10px;
    display: inline-block;
}

.intro-terminology {
    background: rgba(0, 0, 0, 0.02);
    border-radius: 12px;
    padding: 20px;
}

.intro-terminology h3 {
    margin-top: 0;
    margin-bottom: 16px;
    color: #333;
    font-size: 18px;
}

.term {
    margin-bottom: 16px;
}

.term h4 {
    margin: 0 0 8px 0;
    color: #905F29;
    font-size: 16px;
}

.term p {
    margin: 0;
    color: #666;
    line-height: 1.5;
    text-align: justify;
    text-indent: 2em;
}

.dialog-footer {
    display: flex;
    justify-content: flex-end;
}
</style>
