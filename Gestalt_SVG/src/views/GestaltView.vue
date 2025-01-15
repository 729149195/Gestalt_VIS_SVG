<template>
    <div class="main">
        <div class="left">
            <div class="left-top">
                <div class="svgZoom">
                    <SvgUploader @file-uploaded="handleFileUploaded" />
                </div>
            </div>
            <div class="left-bottom">
                <div class="fill-height">
                    <CodeToSvg />
                </div>
            </div>
        </div>
        <div class="right">
            <div class="fill-height datas">
                <div v-if="file" class="data-cards">
                    <div class="maxtistic">
                        <SubgroupVisualization v-if="file" :key="componentKey2" />
                        <analysisWords title="Feature dimension mapping analysis" :update-key="componentKey2"/>
                        <div class="main-card margin-right">
                            <v-card class="position-card card1">
                                <PositionStatistics position="top" title="Top Position" :key="componentKey4" />
                            </v-card>
                            <v-card class="position-card card2">
                                <PositionStatistics position="bottom" title="Bottom Position"
                                    :key="componentKey4 + 1" />
                            </v-card>
                            <v-card class="position-card card3">
                                <PositionStatistics position="right" title="Right Position" :key="componentKey4 + 2" />
                            </v-card>
                            <v-card class="position-card card4">
                                <PositionStatistics position="left" title="Left Position" :key="componentKey4 + 3" />
                            </v-card>
                        </div>
                        <div class="main-card margin-right">
                            <v-card class="position-card card2">
                                <FillStatistician :key="componentKey4 + 5" />
                            </v-card>
                            <v-card class="position-card card4">
                                <strokeStatistician :key="componentKey4 + 7" />
                            </v-card>
                            <v-card class="position-card card1">
                                <HisEleProportions :key="componentKey4 + 4" />
                            </v-card>
                            <v-card class="position-card card3">
                                <HisAttrProportionsVue :key="componentKey4 + 6" />
                            </v-card>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import HisEleProportions from '../components/statistics/ElementStatistics.vue';
import HisAttrProportionsVue from '../components/statistics/AttributeStatistics.vue';
import FillStatistician from '../components/statistics/FillStatistics.vue';
import strokeStatistician from '../components/statistics/StrokeStatistics.vue';
import PositionStatistics from '../components/statistics/PositionStatistics.vue';
import SubgroupVisualization from '../components/visualization/SubgroupVisualization.vue';
import SvgUploader from '../components/SvgUploader.vue';
import analysisWords from '@/components/statistics/analysisWords.vue';
import CodeToSvg from '@/components/visualization/CodeToSvg.vue';

const file = ref(null)
const componentKey = ref(0)
const componentKey2 = ref(1)
const componentKey4 = ref(2)
const componentKey3 = ref(3)

const handleFileUploaded = () => {
    componentKey.value += 1;
    componentKey2.value += 1;
    componentKey3.value += 1;
    componentKey4.value += 1;
    file.value = true;
};

// 组件加时清空 uploadSvg 目录
onMounted(async () => {
    try {
        const response = await fetch('http://127.0.0.1:5000/clear_upload_folder', {
            method: 'POST'
        });
        if (!response.ok) {
            console.error('清空上传文件夹失败');
        }
    } catch (error) {
        console.error('清空上传文件夹出错:', error);
    }
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
}

.main,
.main * {
    user-select: none;
}

.left {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    width: 100%;
    height: 100%;
    gap: 12px;
}

.left-top {
    display: flex;
    gap: 12px;
    height: 50%;
    background-color: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
}

.left-top:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
}

.svgZoom {
    flex-grow: 1;
    padding: 12px;
}

.left-bottom {
    height: 65%;
    background-color: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
}


.right {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    background-color: transparent;
}

.datas {
    height: 100%;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.7) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    transition: all 0.3s ease;
    overflow: hidden;
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
    transition: transform 0.3s ease;
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
    gap: 16px;
    padding: 4px;
}

.maxtistic>* {
    border-radius: 12px;
    transition: all 0.3s ease;
}

/* 只为 SubgroupVisualization 和 maxstic 添加悬浮效果 */
.maxtistic>SubgroupVisualization:hover,
.maxtistic>maxstic:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
}

.maxtistic>analysisWords {
    background-color: transparent !important;
    box-shadow: none !important;
}

.maxtistic>analysisWords:hover {
    transform: none !important;
}

.maxtistic>maxstic {
    height: 40%;
}

.maxtistic>SubgroupVisualization {
    height: 60%;
}

.main-card {
    width: 100%;
    display: flex;
    flex-direction: row;
    gap: 16px;
    height: 30%;
}

.position-card {
    flex: 1;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(200, 200, 200, 0.3);
    padding: 12px;
    transition: all 0.3s ease;
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
    transition: all 0.3s ease;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.3);
}
</style>
