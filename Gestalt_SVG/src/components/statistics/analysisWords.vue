<template>
  <div class="analysis-words-container">
    <div class="analysis-header">
      <h3>{{ props.title }}</h3>
    </div>
    <div class="analysis-content" @scroll="handleScroll">
      <p>{{ props.content }}</p>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  title: {
    type: String,
    default: 'analysis'
  },
  content: {
    type: String,
    default: 'No analysis available'
  }
})

const emit = defineEmits(['scroll'])

const handleScroll = (event) => {
  emit('scroll', {
    scrollTop: event.target.scrollTop,
    scrollHeight: event.target.scrollHeight
  })
}
</script>

<style scoped>
.analysis-words-container {
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-radius: 16px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(200, 200, 200, 0.3);
  padding: 16px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  height: 400px;
  display: flex;
  flex-direction: column;
}

.analysis-words-container:hover {
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
  transform: translateY(-1px);
  border: 1px solid rgba(180, 180, 180, 0.4);
}

.analysis-header {
  margin-bottom: 12px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  padding-bottom: 12px;
  flex-shrink: 0;
}

.analysis-header h3 {
  margin: 0;
  color: var(--el-text-color-primary);
  font-size: 18px;
  font-weight: 500;
  letter-spacing: 0.5px;
}

.analysis-content {
  color: var(--el-text-color-regular);
  line-height: 1.8;
  font-size: 14px;
  overflow-y: auto;
  flex-grow: 1;
  padding: 0 12px 0 0;
}

.analysis-content::-webkit-scrollbar {
  width: 6px;
}

.analysis-content::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  transition: background 0.3s ease;
}

.analysis-content::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.3);
}

.analysis-content::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.05);
  border-radius: 3px;
}

.analysis-content p {
  margin: 0;
  padding: 4px 0;
  transition: color 0.3s ease;
}

.analysis-content p:hover {
  color: var(--el-text-color-primary);
}
</style>