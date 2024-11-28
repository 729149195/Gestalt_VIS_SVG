import { createRouter, createWebHistory } from 'vue-router'
import GestaltView from '../views/GestaltView.vue'
import subgroup_test from '@/views/subgroup_test.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'GestaltView',
      component: GestaltView
    },
    {
      path: '/subgroup',
      name: 'subgroup_test',
      component: subgroup_test
    }
  ]
})

export default router
