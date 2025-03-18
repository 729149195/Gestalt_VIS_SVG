<script setup>
import { RouterLink, RouterView } from 'vue-router'
import { onMounted } from 'vue'

onMounted(() => {
  // 强制移除任何可能的深色模式类名
  document.documentElement.classList.remove('dark')
  document.documentElement.style.colorScheme = 'light'
})
</script>

<template>
  <RouterView />
</template>

<style>
/* 确保整个应用使用浅色主题 */
#app {
  background-color: #ffffff !important;
  color: #000000 !important;
}
</style>
