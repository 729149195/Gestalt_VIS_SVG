<template>
    <div class="common-layout">
        <el-container class="full-height">
            <el-header class="header">
                <div class="header-content">
                    <div class="left-content">
                        <el-select v-model="questionnaireId" placeholder="选择问卷ID" @change="handleQuestionnaireChange">
                            <el-option
                                v-for="id in questionnaireIds"
                                :key="id"
                                :label="`问卷ID：${id} (${getStepsLength(id)}步)`"
                                :value="id"
                            />
                        </el-select>
                    </div>
                    <div class="questionnaire-info" v-if="questionnaireData">
                        <el-descriptions :column="2" border>
                            <el-descriptions-item label="学号">{{ questionnaireData.formData.studentid }}</el-descriptions-item>
                            <el-descriptions-item label="年龄">{{ questionnaireData.formData.age }}</el-descriptions-item>
                            <el-descriptions-item label="性别">{{ questionnaireData.formData.gender }}</el-descriptions-item>
                            <el-descriptions-item label="视觉障碍">{{ questionnaireData.formData.visualimpairment }}</el-descriptions-item>
                            <el-descriptions-item label="可视化经验">{{ questionnaireData.formData.visualizationExperience }}</el-descriptions-item>
                            <el-descriptions-item label="开始时间">{{ questionnaireData.startTime }}</el-descriptions-item>
                            <el-descriptions-item label="结束时间">{{ questionnaireData.endTime }}</el-descriptions-item>
                            <el-descriptions-item label="持续时间">{{ questionnaireData.duration }}</el-descriptions-item>
                        </el-descriptions>
                    </div>
                </div>
            </el-header>
            <el-main class="main-container">
                <el-card v-if="loading" class="loading-card">
                    <el-skeleton :rows="3" animated />
                </el-card>
                <el-card v-else class="main-card">
                    <div style="display: flex;">
                        <div class="left-two">
                            <el-card class="top-card" shadow="never">
                                <div v-html="Svg" class="svg-container"></div>
                                <el-button class="top-title" disabled text bg>组合观察区域</el-button>
                            </el-card>
                            <el-card class="bottom-card" shadow="never">
                                <div ref="chartContainer" class="chart-container" v-show="false"></div>
                                <div v-html="Svg" class="svg-container2" ref="svgContainer2"></div>
                                <el-button class="bottom-title" disabled text bg>选取交互区域</el-button>
                            </el-card>
                        </div>
                        <el-card class="group-card" shadow="never">
                            <div class="select-group">
                                <el-select v-model="selectedGroup" placeholder="选择组合" @change="highlightGroup">
                                    <el-option v-for="(group, index) in groupOptions" :key="index" :label="group"
                                        :value="group" />
                                </el-select>
                            </div>
                            <div v-if="selectedGroup" class="group">
                                <h3>{{ selectedGroup }}</h3>
                                <el-scrollbar height="500px">
                                    <div class="group-tags">
                                        <el-tag v-for="node in currentGroupNodes" :key="node"
                                            @mousedown="highlightElement(node)" @mouseup="resetHighlight">
                                            {{ node }}
                                        </el-tag>
                                    </div>
                                </el-scrollbar>
                                <div v-if="ratings[selectedGroup]" class="rate-container">
                                    <div class="rate-container2">
                                        <div class="rate-text">显著程度：</div>
                                        <el-rate disabled :model-value="ratings[selectedGroup].attention" :max="3"
                                            show-score />
                                    </div>
                                    <div class="rate-container2">
                                        <div class="rate-text">分组组内元素的关联强度：</div>
                                        <el-rate disabled :model-value="ratings[selectedGroup].correlation_strength"
                                            :max="3" show-score />
                                    </div>
                                    <div class="rate-container2">
                                        <div class="rate-text">分组对组外元素的排斥程度：</div>
                                        <el-rate disabled :model-value="ratings[selectedGroup].exclusionary_force"
                                            :max="3" show-score />
                                    </div>
                                </div>
                            </div>
                        </el-card>
                    </div>
                </el-card>
                <div class="steps-container">
                    <el-button class="previous-button" @click="Previous" :disabled="active <= 0">
                        <el-icon>
                            <CaretLeft />
                        </el-icon>
                    </el-button>
                    <el-steps :active="active" finish-status="success" class="steps">
                        <el-step v-for="(step, index) in steps" :key="index" :title="`步骤 ${step}`"
                            @click.native="goToStep(index)" />
                    </el-steps>
                    <el-button class="next-button" @click="next" :disabled="active >= steps.length - 1" type="primary">
                        <el-icon>
                            <CaretRight />
                        </el-icon>
                    </el-button>
                </div>
            </el-main>
        </el-container>
    </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import * as d3 from 'd3';
import { CaretLeft, CaretRight } from '@element-plus/icons-vue';

const route = useRoute();
const router = useRouter();
const questionnaireId = ref('');
const questionnaireIds = ref([]);
const questionnaireData = ref(null);
const active = ref(0);
const steps = ref([]);
const Svg = ref('');
const selectedGroup = ref(null);
const ratings = ref({});
const svgContainer2 = ref(null);
const allVisiableNodes = ref([]);

// 添加载状态
const loading = ref(true);
const chartContainer = ref(null);

// 获取所有问卷ID
const loadQuestionnaireIds = async () => {
    try {
        loading.value = true;
        const response = await fetch('http://localhost:3000/questionnaire/list', {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            mode: 'cors'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Received data:', data);
        
        if (!Array.isArray(data)) {
            throw new Error('Invalid response format');
        }
        
        questionnaireIds.value = data;
        
        // 如果有问卷ID，选择第一个
        if (questionnaireIds.value.length > 0) {
            questionnaireId.value = questionnaireIds.value[0];
            await loadQuestionnaireData();
        }
    } catch (error) {
        console.error('Error loading questionnaire list:', error);
        ElMessage.error(`加载问卷列表失败：${error.message}`);
    } finally {
        loading.value = false;
    }
};

// 处理问卷ID变化
const handleQuestionnaireChange = async (newId) => {
    questionnaireId.value = newId;
    await loadQuestionnaireData();
};

// 获取特定问卷的步骤数量
const getStepsLength = (id) => {
    if (id === questionnaireId.value && questionnaireData.value) {
        return questionnaireData.value.steps.length;
    }
    return '...';
};

// 获取问卷数据
const loadQuestionnaireData = async () => {
    try {
        loading.value = true;
        if (!questionnaireId.value) return;

        const response = await fetch(`http://localhost:3000/questionnaire/${questionnaireId.value}.json`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            mode: 'cors'
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        questionnaireData.value = await response.json();
        steps.value = questionnaireData.value.steps.map(step => step.stepId);

        if (steps.value.length > 0) {
            active.value = 0;
            await loadStep(0);
            
            setTimeout(async () => {
                await next();
                setTimeout(async () => {
                    await Previous();
                }, 0);
            }, 0);
        }
    } catch (error) {
        console.error('Error loading questionnaire:', error);
        ElMessage.error(`加载问卷数据失败：${error.message}`);
    } finally {
        loading.value = false;
    }
};

// 加载特定步骤的数据
const loadStep = async (stepIndex) => {
    if (!questionnaireData.value) return;

    const stepData = questionnaireData.value.steps[stepIndex];
    if (!stepData) return;

    selectedGroup.value = null;

    // 加载 SVG
    await fetchSvgContent(stepData.stepId);
    await fetchAndRenderTree(stepData.stepId);

    // 设置组合和评分
    ratings.value = {};
    stepData.groups.forEach(group => {
        ratings.value[group.group] = group.ratings;
    });

    // 设置选中的组合
    selectedGroup.value = stepData.groups[0]?.group || null;

    // 延迟执行高亮
    setTimeout(() => {
        if (selectedGroup.value) {
            highlightGroup();
        }
    }, 100);
};

const fetchSvgContent = async (stepId) => {
    try {
        const response = await fetch(`/questionnaire/Data4/${stepId}/${stepId}.svg`);
        if (!response.ok) throw new Error('Network response was not ok');
        Svg.value = await response.text();

        await nextTick();
        turnGrayVisibleNodes();
        addHoverEffectToVisibleNodes();
        addZoomEffectToSvg();
    } catch (error) {
        console.error('Error loading SVG:', error);
        Svg.value = '<svg><text x="10" y="20" font-size="20">加载SVG时出错</text></svg>';
    }
};

const fetchAndRenderTree = async (stepId) => {
    try {
        const response = await fetch(`/questionnaire/Data4/${stepId}/layer_data.json`);
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        renderTree(data);
    } catch (error) {
        console.error('Error loading tree data:', error);
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
    allVisiableNodes.value = nodeIds;
};

const currentGroupNodes = computed(() => {
    if (!questionnaireData.value || !selectedGroup.value) return [];
    const currentStep = questionnaireData.value.steps[active.value];
    const group = currentStep.groups.find(g => g.group === selectedGroup.value);
    return group?.nodes || [];
});

const groupOptions = computed(() => {
    if (!questionnaireData.value) return [];
    const currentStep = questionnaireData.value.steps[active.value];
    return currentStep.groups.map(g => g.group);
});

const turnGrayVisibleNodes = () => {
    const svgContainer = svgContainer2.value;
    if (!svgContainer) return;
    const svg = svgContainer.querySelector('svg');
    if (!svg) return;

    // 只处理可见节点
    svg.querySelectorAll('*').forEach(node => {
        if (allVisiableNodes.value.includes(node.id)) {
            node.style.opacity = '0.05';
            // 添加过渡效果使颜色变化更平滑
            node.style.transition = 'opacity 0.3s ease';
        }
    });
};

const addHoverEffectToVisibleNodes = () => {
    const svgContainer = svgContainer2.value;
    if (!svgContainer) return;
    const svg = svgContainer.querySelector('svg');
    if (!svg) return;

    svg.querySelectorAll('*').forEach(node => {
        if (allVisiableNodes.value.includes(node.id)) {
            node.addEventListener('mouseover', () => {
                node.style.opacity = '1';
            });
            node.addEventListener('mouseout', () => {
                node.style.opacity = '0.05';
                highlightGroup();
            });
        }
    });
};

const highlightGroup = () => {
    if (!selectedGroup.value) return;

    const nodes = currentGroupNodes.value;
    const svgContainer = svgContainer2.value;
    if (!svgContainer) return;

    const svg = svgContainer.querySelector('svg');
    if (!svg) return;

    svg.querySelectorAll('*').forEach(node => {
        if (allVisiableNodes.value.includes(node.id)) {
            node.style.opacity = nodes.includes(node.id) ? '1' : '0.1';
            node.style.transition = 'opacity 0.3s ease';
        }
    });
};

const highlightElement = (nodeId) => {
    const svgContainer = svgContainer2.value;
    if (!svgContainer) return;
    const svg = svgContainer.querySelector('svg');
    if (!svg) return;

    svg.querySelectorAll('*').forEach(node => {
        if (node.id === nodeId) {
            node.style.opacity = '1';
        } else if (allVisiableNodes.value.includes(node.id)) {
            node.style.opacity = '0.2';
        }
    });
};

const resetHighlight = () => {
    nextTick(() => {
        highlightGroup();
    });
};

const addZoomEffectToSvg = () => {
    const svgContainer = svgContainer2.value;
    if (!svgContainer) return;
    const svg = d3.select(svgContainer).select('svg');
    if (!svg.empty()) {
        let g = svg.select('g.zoom-wrapper');
        if (g.empty()) {
            g = svg.append('g').attr('class', 'zoom-wrapper');
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
                g.attr('transform', event.transform);
            });

        svg.call(zoom);
    }
};

const next = async () => {
    if (active.value < steps.value.length - 1) {
        active.value++;
        await loadStep(active.value);
    }
};

const Previous = async () => {
    if (active.value > 0) {
        active.value--;
        await loadStep(active.value);
    }
};

const goToStep = async (index) => {
    if (index !== active.value) {
        active.value = index;
        await loadStep(active.value);
    }
};

onMounted(() => {
    loadQuestionnaireIds();
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
    align-items: flex-start;
    padding: 0 10px;
    border-bottom: 1px solid #dcdcdc;
    height: auto !important;
    min-height: 60px;
}

.header-content {
    display: flex;
    justify-content: space-between;
    width: 100%;
    padding: 10px;
}

.left-content {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    color: #999;

    .el-select {
        width: 300px;
    }
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

            .bottom-title {
                position: absolute;
                top: 5px;
                left: -5px;
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
                width: 100%;
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

.svg-container,
.svg-container2 {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.svg-container svg,
.svg-container2 svg {
    width: 100%;
    height: 100%;
    display: block;
}

:deep(.el-step__title) {
    font-size: 14px;
}

:deep(.el-rate__icon) {
    margin-right: 4px;
}

:deep(.el-select) {
    width: 100%;
}

.questionnaire-info {
    flex-grow: 1;
    margin-left: 20px;
    
    :deep(.el-descriptions) {
        padding: 10px;
    }
    
    :deep(.el-descriptions__label) {
        width: 100px;
        color: #666;
    }
    
    :deep(.el-descriptions__content) {
        color: #333;
    }
}
</style>