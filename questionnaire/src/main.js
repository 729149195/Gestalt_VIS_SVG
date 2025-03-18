import './assets/main.css'

import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'dayjs/locale/zh-cn' // Chinese locale for dayjs
import zhCn from 'element-plus/es/locale/lang/zh-cn' // Chinese locale for Element Plus
import App from './App.vue'
import store from './store'
import router from './router'
import { loadSlim } from "@tsparticles/slim"
import Particles from "@tsparticles/vue3"
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

const app = createApp(App)

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
}

app.use(router)
app.use(store)
app.use(Particles, {
    init: async engine => {
        await loadSlim(engine)
    }
})
app.use(ElementPlus, { locale: zhCn })

app.mount('#app')
