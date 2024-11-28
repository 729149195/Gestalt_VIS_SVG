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
                                <topStatistician />
                            </v-card>
                            <v-card class="position-card card2">
                                <bottomStatistician />
                            </v-card>
                            <v-card class="position-card card3">
                                <rightStatistician />
                            </v-card>
                            <v-card class="position-card card4">
                                <leftStatistician />
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
                        <!-- <maxsticStaticiannormal :key="componentKey4" /> -->
                        <!-- <SankeyStatician :key="componentKey2"/> -->
                        <!-- <maxsticStatician :key="componentKey" /> -->
                        <maxstic :key="componentKey" />
                        <subgroup_test :key="componentKey2" />
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
import maxstic from './maxstic.vue';
import subgroup_test from './subgroup_test.vue';

const file = ref(null)
const processedSvgContent = ref('')
const componentKey = ref(0)
const componentKey2 = ref(1)
const componentKey4 = ref(2)
const componentKey3 = ref(3)
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
            componentKey4.value += 1;
        })
        .catch(error => {
            console.error('Error uploading file:', error)
        })
}

const fetchProcessedSvg = () => {
    axios.get('http://localhost:5000/get_svg', { responseType: 'text' })
        .then(svgResponse => {
            let svgContent = svgResponse.data;

            // 检查并添加 viewBox 属性，确保 SVG 可以适应容器大小
            if (!svgContent.includes('viewBox')) {
                const widthMatch = svgContent.match(/width="(\d+)"/);
                const heightMatch = svgContent.match(/height="(\d+)"/);

                if (widthMatch && heightMatch) {
                    const width = widthMatch[1];
                    const height = heightMatch[1];

                    svgContent = svgContent.replace(
                        '<svg',
                        `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet"`
                    );
                }
            }

            processedSvgContent.value = svgContent.replace(/height="auto"/g, 'height="340px"');

            nextTick(() => {
                const svgContainer = document.querySelector('.svg-container svg');
                if (svgContainer) {
                    svgContainer.removeEventListener('click', handleSvgClick); // 移除现有监听器
                    svgContainer.addEventListener('click', handleSvgClick);
                }
            });
        })
        .catch(error => {
            console.error('Error fetching SVG:', error);
        });
};

// 点击 SVG 节点的处理函数
const handleSvgClick = (event) => {
    const nodeId = event.target.id;

    if (!nodeId) return; // 忽略无 ID 的元素

    // 如果节点 ID 已存在于 selectedNodeIds 中，则移除；否则，添加
    if (selectedNodeIds.value.includes(nodeId)) {
        store.commit('REMOVE_SELECTED_NODE', nodeId);
    } else {
        store.commit('ADD_SELECTED_NODE', nodeId);
    }
};



watch(selectedNodeIds, () => {
    nextTick(() => {
        const svgContainer = document.querySelector('.svg-container');
        if (!svgContainer) return;
        if (!selectedNodeIds) return;

        const svg = svgContainer.querySelector('svg');
        if (!svg) return;

        if (selectedNodeIds.value.length === 0) {
            // 如果 selectedNodeIds 为空，恢复所有元素的透明度为 1
            svg.querySelectorAll('*').forEach(node => {
                node.style.opacity = '1';
            });
        } else {
            // 否则，应用现有的透明度逻辑
            svg.querySelectorAll('*').forEach(node => {
                node.style.opacity = '';
            });

            svg.querySelectorAll('*').forEach(node => {
                if (allVisiableNodes.value.includes(node.id) && !selectedNodeIds.value.includes(node.id)) {
                    node.style.opacity = '0.1';
                }
            });
        }
    });
});


const refresh = () => {
    componentKey.value += 1;
    componentKey2.value += 1;
    componentKey4.value += 1;
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
    max-width: 1000px;
    margin: auto;
    overflow: hidden;
}

.svg-container svg {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    display: block;
    margin: auto;
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
