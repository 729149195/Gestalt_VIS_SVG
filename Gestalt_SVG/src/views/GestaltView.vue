<template>
    <div class="main">
        <div class="left">
            <div class="left-top">
                <div class="svgZoom">
                    <v-card class="fill-height">
                        <h2 class="title">
                            Parse SVG by Gestalt
                        </h2>
                        <v-file-input v-model="file" prepend-icon="mdi-paperclip" label="Select Svg File"
                            density="compact" accept=".svg" show-size @change="uploadFile"></v-file-input>
                        <div v-if="file" class="svg-container" v-html="processedSvgContent"></div>
                    </v-card>
                </div>
            </div>
            <div class="left-bottom">
                <v-card class="fill-height datas">
                    <div style="display: flex; align-items: center;">
                        <span style="font-size:1.1em; font-weight: 700; padding-left: 8px">Parse SVG data:</span>
                        <div class="radio-group-horizontal" v-if="file">
                            <label>
                                <input type="radio" value="maxtistic" v-model="selectedView" />
                                 Maxistic
                            </label>
                            <label>
                                <input type="radio" value="FFT" v-model="selectedView" />
                                 FFT
                            </label>
                            <label>
                                <input type="radio" value="position" v-model="selectedView" />
                                 Position
                            </label>
                            <label>
                                <input type="radio" value="layer" v-model="selectedView" />
                                 Layer
                            </label>
                            <label>
                                <input type="radio" value="proportions" v-model="selectedView" />
                                 Proportions
                            </label>
                        </div>
                    </div>
                    <div v-if="file" class="data-cards">
                        <div class="position" v-if="selectedView === 'position'">
                            <div class="main-card margin-right">
                                <v-card class="position-card card1"><topStatistician/></v-card>
                                <v-card class="position-card card2"><bottomStatistician/></v-card>
                            </div>
                            <div class="main-card">
                                <v-card class="position-card card3"><rightStatistician/></v-card>
                                <v-card class="position-card card4"><leftStatistician/></v-card>
                            </div>
                        </div>
                        <div class="maxtistic" v-if="selectedView === 'maxtistic'">
                            <maxsticStatician :key="componentKey"/>
                        </div>
                        <div class="maxtistic2" v-if="selectedView === 'FFT'">
                            <maxsticStaticianfly :key="componentKey2"/>
                        </div>
                        <div class="layer" v-if="selectedView === 'layer'">
                            <layerStatistician/>
                        </div>
                        <div class="position" v-if="selectedView === 'proportions'">
                            <div class="main-card margin-right">
                                <v-card class="position-card card1"><HisEleProportions/></v-card>
                                <v-card class="position-card card2"><FillStatistician/></v-card>
                            </div>
                            <div class="main-card">
                                <v-card class="position-card card3"><HisAttrProportionsVue/></v-card>
                                <v-card class="position-card card4"><strokeStatistician/></v-card>
                            </div>
                        </div>
                    </div>
                </v-card>
            </div>
        </div>
        <div class="right">
            <v-card class="fill-height">
                <span style="font-size:1.1em; font-weight: 700; padding-left: 8px">interactivity SVG and Hulls
                    result: </span><v-btn density="compact" icon="mdi-refresh" variant="plain" @click="refresh" v-if="file"></v-btn>
                <CommunityDetectionMult v-if="file" :key="componentKey3" />
            </v-card>
        </div>
    </div>
</template>

<script setup>
import { ref, watch, computed, nextTick } from 'vue'
import axios from 'axios'
import { useStore } from 'vuex';
import CommunityDetectionMult from './Community-Detection-Mult.vue';
import HisEleProportions from './His-EleProportions.vue';
import HisAttrProportionsVue from './His-AttrProportions.vue';
import FillStatistician from './Fill-Statistician.vue';
import strokeStatistician from './stroke-Statistician.vue';
import topStatistician from './top-Statistician.vue';
import bottomStatistician from './bottom-Statistician.vue';
import leftStatistician from './left-Statistician.vue';
import rightStatistician from './right-Statistician.vue';
import layerStatistician from './layer-Statistician.vue';
import maxsticStatician from './maxstic-Statician.vue';
import maxsticStaticianfly from './maxstic-Statician-fly.vue';

const file = ref(null)
const processedSvgContent = ref('')
const componentKey = ref(0)
const componentKey2 = ref(0)
const componentKey3 = ref(0)
const store = useStore();
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds);
const allVisiableNodes = computed(() => store.state.AllVisiableNodes);

const selectedView = ref('maxtistic');

const uploadFile = () => {
    if (!file.value) return

    const formData = new FormData()
    formData.append('file', file.value)

    axios.post('http://localhost:5000/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })
        .then(response => {
            // console.log('File uploaded successfully:', response.data)
            fetchProcessedSvg();
            componentKey.value += 1;
            componentKey2.value += 1;
            componentKey3.value += 1;
        })
        .catch(error => {
            console.error('Error uploading file:', error)
        })
}

const fetchProcessedSvg = () => {
    axios.get('http://localhost:5000/get_svg', { responseType: 'text' })
        .then(svgResponse => {
            processedSvgContent.value = svgResponse.data.replace(/height="auto"/g, 'height="340px"');
        })
        .catch(error => {
            console.error('Error fetching SVG:', error);
        });
}

watch(selectedNodeIds, () => {
    nextTick(() => {
        const svgContainer = document.querySelector('.svg-container');
        if (!svgContainer) return;
        if(!selectedNodeIds) return;

        const svg = svgContainer.querySelector('svg');
        if (!svg) return;

        svg.querySelectorAll('*').forEach(node => {
            node.style.opacity = '';
        });

        svg.querySelectorAll('*').forEach(node => {
            if (allVisiableNodes.value.includes(node.id) && !selectedNodeIds.value.includes(node.id)) {
                node.style.opacity = '0.05';
            }
        });
    });
});

const refresh = () => {
    store.commit('CLEAR_SELECTED_NODES');
    const svgContainer = document.querySelector('.svg-container');
    const svg = svgContainer.querySelector('svg');
    svg.querySelectorAll('*').forEach(node => {
        node.style.opacity = '1';
    })
}
</script>

<style scoped>
.main {
    display: flex;
    width: 100vw;
    height: 100vh;
    padding: 8px;
    box-sizing: border-box;
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
    height: 50%;
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
    max-width: 1200px;
    max-height: 400px;
    width: 100%;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
}

.svg-container svg {
    max-width: 100%;
    display: block;
}

.data-cards {
    height: 100%;
    width: 100%;
}

.position {
    height: 100%;
    width: 100%;
    display: flex;
}

.position-card {
    width: 100%;
    height: 50%;
    padding: 8px;
}

.card1 {
    margin-bottom: 8px;
}

.card3 {
    margin-bottom: 8px;
}

.main-card {
    width: 100%;
}

.datas {
    padding-bottom: 43px;
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
</style>
