import './assets/main.css'

import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

// Vuetify
import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'

// Components
import App from './App.vue'
import router from './router'
import store from './store'; // 引入 store

const vuetify = createVuetify({
    components,
    directives,
  })

const app = createApp(App)

app.use(router)
app.use(ElementPlus)
app.use(vuetify)
app.use(store)

app.mount('#app')
