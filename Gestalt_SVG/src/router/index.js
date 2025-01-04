import { createRouter, createWebHistory } from 'vue-router'
import GestaltView from '../views/GestaltView.vue'
import maxstic from '../components/visualization/maxstic.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'GestaltView',
      component: GestaltView
    },
    {
      path: '/maxstic',
      name: 'maxstic',
      component: maxstic
    }
  ]
})

export default router
