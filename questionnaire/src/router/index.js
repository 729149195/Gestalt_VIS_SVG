import { createRouter, createWebHistory } from 'vue-router'
import QuestionnaireView from '../views/QuestionnaireView.vue'
import Questions from '../views/Questions.vue'
import Thanks from '../views/Thanks.vue'
import QuestionTest from '../views/QuestionTest.vue'
import QuestionsShow from '../views/QuestionsShow.vue'
import showData from '../views/showData.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'Questionnaire',
      component: QuestionnaireView
    },
    {
      path: '/questionstest',
      name: 'Questionstest',
      component: QuestionTest
    },
    {
      path: '/questions',
      name: 'Questions',
      component: Questions
    },
    {
      path: '/thanks',
      name: 'Thanks',
      component: Thanks
    },
    {
      path: '/limit-reached',
      name: 'LimitReached',
      component: () => import('../views/LimitReached.vue')
    },
    {
      path: '/questions-show',
      name: 'QuestionsShow',
      component: QuestionsShow
    },
    {
      path: '/showData',
      name: 'showData',
      component: showData
    } 
  ]
})

export default router
