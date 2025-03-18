<template>
  <div class="limit-container">
    <h1 class="title">问卷收集已达上限</h1>
    <div class="content">
      <el-icon size="60" color="#F56C6C"><CircleCloseFilled /></el-icon>
      <p class="message">非常抱歉，本次问卷调查已收集足够样本</p>
      <p class="sub-message">感谢您的关注与支持！</p>
    </div>
  </div>
</template>

<script setup>
import { CircleCloseFilled } from '@element-plus/icons-vue'
</script>

<style scoped>
.limit-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: 20px;
  text-align: center;
}

.title {
  font-size: 2.5rem;
  color: #F56C6C;
  margin-bottom: 2rem;
}

.content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
}

.message {
  font-size: 1.5rem;
  color: #606266;
  margin: 1rem 0;
}

.sub-message {
  font-size: 1.2rem;
  color: #909399;
}
</style> 