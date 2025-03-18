<template>
  <el-card shadow="always" >
    <div v-if="loading && files.length === 0">
      <el-skeleton :rows="15" animated />
    </div>
    <div 
      v-infinite-scroll="loadMoreFiles" 
      :infinite-scroll-disabled="loading || noMore" 
      infinite-scroll-distance="50" 
      class="grid-container" 
      style="height: auto; overflow-y: auto;"
    >
      <div v-for="file in files" :key="file" class="svg-container" @click="showSvg(file)">
        <img :src="`/questionnaire/newData5/${file}.svg`" alt="SVG Image" />
        <span class="svgid">{{file}}</span>
      </div>
    </div>
    <el-dialog v-model="dialogVisible" width="60%" :before-close="handleClose">
      <img :src="`/questionnaire/newData5/${selectedFile}.svg`" alt="SVG Image" class="large-svg" />
      <span>no.{{ selectedFile }}</span>
    </el-dialog>
  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const files = ref([]);
const dialogVisible = ref(false);
const selectedFile = ref('');
const maxFiles = 20;
const loading = ref(false);
const noMore = ref(false);  // 新增：标记是否还有更多数据
const batchSize = 10;  // 减小批量加载数量
let currentFileIndex = 0;

const loadMoreFiles = async () => {
  if (loading.value || noMore.value) return;
  
  loading.value = true;
  const loadedFiles = [];
  
  try {
    for (let i = 0; i < batchSize; i++) {
      const fileIndex = currentFileIndex + i + 1;
      if (fileIndex > maxFiles) {
        noMore.value = true;
        break;
      }

      const fileName = `${fileIndex}.svg`;
      const response = await fetch(`/questionnaire/newData5/${fileName}`);
      if (response.ok) {
        loadedFiles.push(fileIndex.toString());
      } else {
        noMore.value = true;
        break;
      }
    }

    files.value = [...files.value, ...loadedFiles];
    currentFileIndex += loadedFiles.length;
  } catch (error) {
    console.error('Error loading files:', error);
  } finally {
    loading.value = false;
  }
};

const showSvg = (file) => {
  selectedFile.value = file;
  dialogVisible.value = true;
};

const handleClose = () => {
  dialogVisible.value = false;
};

onMounted(() => {
  loadMoreFiles(); // 初次加载
});
</script>

<style scoped>
/* 滚动条整体宽度 */
.grid-container::-webkit-scrollbar {
  width: 8px;
}

/* 滚动条轨道 */
.grid-container::-webkit-scrollbar-track {
  background-color: #f1f1f1; /* 滚动条轨道的背景色 */
  border-radius: 10px; /* 圆角 */
}

/* 滚动条滑块 */
.grid-container::-webkit-scrollbar-thumb {
  background-color: #999; /* 滑块的颜色 */
  border-radius: 10px; /* 圆角 */
}

/* 滑块悬停状态 */
.grid-container::-webkit-scrollbar-thumb:hover {
  background-color: #666; /* 悬停时滑块的颜色 */
}

.grid-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 15px;
  padding: 20px;
  width: 90vw;
  box-sizing: border-box;
}

.svg-container {
  border: 1px solid #ccc;
  padding: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #fff;
  cursor: pointer;
  position: relative;
  aspect-ratio: 16 / 9;
  height: 180px;
}

.svgid {
  position: absolute;
  top: 0px;
  left: 5px;
  font-size: 0.7em;
}

.svg-container img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  width: 100%;
  height: 100%;
}

.el-dialog__body {
  display: flex;
  align-items: center;
  justify-content: center;
}

.large-svg {
  max-width: 100%;
  max-height: 100%;
}
</style>
