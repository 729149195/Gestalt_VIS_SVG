<template>
    <div class="main">
        <div class="left">
            <div class="left-top">
                <div class="svgZoom">
                    <SvgUploader @file-uploaded="handleFileUploaded" />
                </div>
            </div>
            <div class="left-bottom">
                <v-card class="fill-height">
                    <span style="font-size:1.1em; font-weight: 700; padding-left: 8px">interactivity SVG and Hulls
                        result:
                    </span><v-btn density="compact" icon="mdi-refresh" variant="plain" @click="refresh"
                        v-if="file"></v-btn>
                    <CommunityDetectionMult v-if="file" :key="componentKey3" />
                </v-card>
            </div>
        </div>
        <div class="right">
            <v-card class="fill-height datas">
                <div style="display: flex; align-items: center;">
                    <span style="font-size:1.1em; font-weight: 700; padding-left: 8px">Parse SVG data:</span>
                    <div class="radio-group-horizontal" v-if="file">
                        <label>
                            <input type="radio" value="maxtistic" v-model="selectedView" />
                            Maxistic
                        </label>
                        <label>
                            <input type="radio" value="positionAproportions" v-model="selectedView" />
                            position and proportions
                        </label>
                        <label>
                            <input type="radio" value="layer" v-model="selectedView" />
                            Layer
                        </label>
                    </div>
                </div>
                <div v-if="file" class="data-cards">
                    <div class="position" v-if="selectedView === 'positionAproportions'">
                        <div class="main-card margin-right">
                            <v-card class="position-card card1">
                                <PositionStatistics position="top" title="Top Position" />
                            </v-card>
                            <v-card class="position-card card2">
                                <PositionStatistics position="bottom" title="Bottom Position" />
                            </v-card>
                            <v-card class="position-card card3">
                                <PositionStatistics position="right" title="Right Position" />
                            </v-card>
                            <v-card class="position-card card4">
                                <PositionStatistics position="left" title="Left Position" />
                            </v-card>
                        </div>
                        <div class="main-card margin-right">
                            <v-card class="position-card card1">
                                <HisEleProportions />
                            </v-card>
                            <v-card class="position-card card2">
                                <FillStatistician />
                            </v-card>
                            <v-card class="position-card card3">
                                <HisAttrProportionsVue />
                            </v-card>
                            <v-card class="position-card card4">
                                <strokeStatistician />
                            </v-card>
                        </div>
                    </div>
                    <div class="maxtistic" v-if="selectedView === 'maxtistic'">
                        <maxstic :key="componentKey" />
                        <SubgroupVisualization :key="componentKey2" />
                    </div>
                    <div class="layer" v-if="selectedView === 'layer'">
                        <layerStatistician />
                    </div>
                </div>
            </v-card>
        </div>
    </div>
</template>

<script setup>
import { ref, watch, computed, nextTick } from 'vue'
import { useStore } from 'vuex';
import CommunityDetectionMult from '../components/visualization/CommunityDetection.vue';
import HisEleProportions from '../components/statistics/ElementStatistics.vue';
import HisAttrProportionsVue from '../components/statistics/AttributeStatistics.vue';
import FillStatistician from '../components/statistics/FillStatistics.vue';
import strokeStatistician from '../components/statistics/StrokeStatistics.vue';
import PositionStatistics from '../components/statistics/PositionStatistics.vue';
import layerStatistician from '../components/visualization/LayerVisualization.vue';
import maxstic from './maxstic.vue';
import SubgroupVisualization from '../components/visualization/SubgroupVisualization.vue';
import SvgUploader from '../components/SvgUploader.vue';

const file = ref(null)
const componentKey = ref(0)
const componentKey2 = ref(1)
const componentKey4 = ref(2)
const componentKey3 = ref(3)
const store = useStore();
const selectedView = ref('maxtistic');

const handleFileUploaded = () => {
    componentKey.value += 1;
    componentKey2.value += 1;
    componentKey3.value += 1;
    componentKey4.value += 1;
    file.value = true;
}

const refresh = () => {
    componentKey.value += 1;
    componentKey2.value += 1;
    componentKey4.value += 1;
}

const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);
const allVisiableNodes = computed(() => store.state.AllVisiableNodes);
</script>

<style scoped>
.main {
    display: flex;
    width: 100vw;
    height: 100vh;
    padding: 8px;
    box-sizing: border-box;
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
    gap: 8px;
}

.left-top {
    display: flex;
    gap: 8px;
    height: 40%;
}

.svgZoom {
    flex-grow: 1;
}

.left-bottom {
    height: 65%;
}

.right {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding-left: 8px;
    box-sizing: border-box;
}

.title {
    font-size: 1.5rem;
    font-weight: bold;
    letter-spacing: 2px;
    text-align: left;
    color: black;
    margin-left: 8px;
    margin-bottom: 8px
}

.fill-height {
    height: 100%;
    width: 100%;
    padding: 8px;
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
}

.svg-container svg {
    max-width: 100%;
    max-height: 100%;
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
}

.position {
    height: 100%;
    width: 100%;
    display: flex;
    justify-content: center;
}

.position-card {
    width: 100%;
    height: 24.7%;
    padding: 8px;
}

.card1 {
    margin-bottom: 8px;
    margin-right: 8px;
}

.card2 {
    margin-bottom: 8px;
    margin-right: 8px;
}

.card3 {
    margin-bottom: 8px;
    margin-right: 8px;
}

.card4 {
    margin-bottom: 8px;
    margin-right: 8px;
}

.main-card {
    width: 100%;
}


.margin-right {
    margin-right: 8px
}

.radio-group-horizontal {
    display: flex;
    flex-direction: row;
    gap: 16px;
    margin-left: 16px;
}

.radio-group-horizontal label {
    display: flex;
    align-items: center;
    cursor: pointer;
}

.maxtistic {
    width: 100%;
}
</style>
