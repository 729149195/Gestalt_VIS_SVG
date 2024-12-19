import { createRouter, createWebHistory } from 'vue-router'
import GestaltView from '../views/GestaltView.vue'


const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'GestaltView',
      component: GestaltView
    }
  ]
})

export default router
