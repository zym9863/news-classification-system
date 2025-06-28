import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Classification from '../views/Classification.vue'
import Statistics from '../views/Statistics.vue'
import History from '../views/History.vue'
import AITools from '../views/AITools.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/classification',
    name: 'Classification',
    component: Classification
  },
  {
    path: '/statistics',
    name: 'Statistics',
    component: Statistics
  },
  {
    path: '/history',
    name: 'History',
    component: History
  },
  {
    path: '/ai-tools',
    name: 'AITools',
    component: AITools
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
