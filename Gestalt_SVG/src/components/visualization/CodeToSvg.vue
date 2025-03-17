<template>
  <div class="code-to-svg-container">
    <div class="editors-container">
      <!-- 声明式语法编辑器 -->
      <div v-show="isDeclarativeMode" class="editor-section editor-transition">
        <div class="section-header">
          <div class="left-tools">
            <span class="title">SVG Editor</span>
          </div>
          
          <!-- 声明式模式下的文件上传器 -->
          <div class="header-upload-container" @click="triggerFileInput" @dragover.prevent @drop.prevent="handleDrop">
            <input type="file" ref="fileInput" accept=".svg" class="hidden-input" @change="handleFileChange">
            <div class="header-upload-content">
              <v-icon size="24" class="upload-icon">mdi-cloud-upload-outline</v-icon>
              <span class="upload-text">Select SVG file here</span>
              <div v-if="file" class="file-info">
                <span class="file-name">{{ file.name }}</span>
                <span class="file-size">{{ formatFileSize(file.size) }}</span>
              </div>
            </div>
          </div>
          
          <div class="side-mode-switch">
            <el-select v-model="selectedSyntax" placeholder="选择生成式语法" class="syntax-selector" popper-class="syntax-selector-dropdown">
              <el-option v-for="item in syntaxOptions" :key="item.value" :label="item.label" :value="item.value">
                <div class="syntax-option">
                  <span class="syntax-label">{{ item.label }}</span>
                  <span class="syntax-description">{{ item.description }}</span>
                </div>
              </el-option>
            </el-select>
            <el-button type="primary" @click="generateAndUpload" class="larger-text-btn">Upload</el-button>
            <el-button @click="copyCode" class="larger-text-btn">Copy</el-button>
            <el-button @click="downloadSvg" class="larger-text-btn">Download</el-button>
            
            <div class="mode-tabs">
              <div class="mode-tab larger-text-tab" :class="{ active: isDeclarativeMode }" @click="isDeclarativeMode = true">
                Syntax
              </div>
              <div class="mode-tab larger-text-tab" :class="{ active: !isDeclarativeMode }" @click="isDeclarativeMode = false">
                SVG
              </div>
            </div>
          </div>
        </div>
        
        <!-- 进度提示卡片 - 声明式模式 -->
        <div v-if="analyzing" class="progress-card">
          <div class="progress-label">{{ currentStep }}</div>
          <el-progress :percentage="progress" :show-text="false" class="upload-progress"></el-progress>
        </div>
        
        <div class="code-editor">
          <div class="editor-wrapper" ref="declarativeEditorContainer"></div>
        </div>
      </div>

      <!-- SVG编辑器 -->
      <div v-show="!isDeclarativeMode" class="editor-section editor-transition">
        <div class="section-header">
          <div class="left-tools">
            <span class="title">SVG Editor</span>
          </div>
          
          <!-- SVG模式下的文件上传器 -->
          <div class="header-upload-container" @click="triggerFileInput" @dragover.prevent @drop.prevent="handleDrop">
            <input type="file" ref="fileInput" accept=".svg" class="hidden-input" @change="handleFileChange">
            <div class="header-upload-content">
              <v-icon size="24" class="upload-icon">mdi-cloud-upload-outline</v-icon>
              <span class="upload-text">Select SVG file here</span>
              <div v-if="file" class="file-info">
                <span class="file-name">{{ file.name }}</span>
                <span class="file-size">{{ formatFileSize(file.size) }}</span>
              </div>
            </div>
          </div>
          
          <div class="side-mode-switch">
            <el-button type="primary" @click="generateAndUpload" class="larger-text-btn">Upload</el-button>
            <el-button @click="copyCode" class="larger-text-btn">Copy</el-button>
            <el-button @click="downloadSvg" class="larger-text-btn">Download</el-button>
            <div class="mode-tabs">
              <div class="mode-tab larger-text-tab" :class="{ active: isDeclarativeMode }" @click="isDeclarativeMode = true">
                Syntax
              </div>
              <div class="mode-tab larger-text-tab" :class="{ active: !isDeclarativeMode }" @click="isDeclarativeMode = false">
                SVG
              </div>
            </div>
          </div>
        </div>
        
        <!-- 进度提示卡片 - SVG模式 -->
        <div v-if="analyzing" class="progress-card">
          <div class="progress-label">{{ currentStep }}</div>
          <el-progress :percentage="progress" :show-text="false" class="upload-progress"></el-progress>
        </div>
        
        <div class="code-editor">
          <div class="editor-wrapper" ref="svgEditorContainer"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, nextTick, onUnmounted, computed } from 'vue'
import * as monaco from 'monaco-editor'
import * as d3 from 'd3'
import { syntaxOptions, placeholders } from '@/config/codeToSvgConfig'
import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker'
import jsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker'
import tsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker'
import htmlWorker from 'monaco-editor/esm/vs/language/html/html.worker?worker'
import { useStore } from 'vuex'
import { Upload } from '@element-plus/icons-vue'
import axios from 'axios'

// 配置 Monaco Editor 的 Web Worker
window.MonacoEnvironment = {
  getWorker(_, label) {
    switch (label) {
      case 'json':
        return new jsonWorker()
      case 'javascript':
      case 'typescript':
        return new tsWorker()
      case 'html':
      case 'xml':
        return new htmlWorker()
      default:
        return new editorWorker()
    }
  }
}

// 状态定义
const code = ref('')
const svgCode = ref('')
const svgOutput = ref('')
const selectedSyntax = ref('d3')
const isDeclarativeMode = ref(true)
const declarativeEditorContainer = ref(null)
const svgEditorContainer = ref(null)
const previewContainer = ref(null)
let declarativeEditor = null
let svgEditor = null

// 从Vuex获取selectedNodes
const store = useStore()
const selectedNodeIds = computed(() => store.state.selectedNodes.nodeIds)

// 文件上传相关变量
const file = ref(null)
const fileInput = ref(null)
const analyzing = ref(false)
const progress = ref(0)
const currentStep = ref('')

// Monaco Editor配置
const declarativeEditorOptions = {
  theme: 'gestaltTheme',
  fontSize: 14,
  lineNumbers: 'on',
  roundedSelection: false,
  scrollBeyondLastLine: false,
  readOnly: false,
  minimap: {
    enabled: true,
    renderCharacters: true,
    maxColumn: 120,
    scale: 1
  },
  automaticLayout: true,
  wordWrap: 'on',
  scrollbar: {
    vertical: 'visible',
    horizontal: 'auto'
  },
  lineHeight: 20,
  padding: {
    top: 10,
    bottom: 10
  },
  formatOnPaste: true,
  formatOnType: true
}

const svgEditorOptions = {
  ...declarativeEditorOptions,
  language: 'xml',
  formatOnPaste: true,
  formatOnType: true
}

// 根据选择的语法设置编辑器语言
const getEditorLanguage = () => {
  switch (selectedSyntax.value) {
    case 'd3':
      return 'javascript'
    case 'echarts':
    case 'highcharts':
    case 'vega':
    case 'vega-lite':
      return 'json'
    default:
      return 'plaintext'
  }
}

// 初始化编辑器
const initEditors = () => {
  // 自定义编辑器主题
  monaco.editor.defineTheme('gestaltTheme', {
    base: 'vs',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '6A9955' },
      { token: 'keyword', foreground: '905F29', fontStyle: 'bold' },
      { token: 'string', foreground: 'A67C4A' },  // 调整字符串颜色为更亮的棕色
      { token: 'number', foreground: '8A6B47' },  // 保持数字颜色
      { token: 'attribute.name', foreground: '5D4126' }, // 调整属性名颜色为更深的棕色
      { token: 'tag', foreground: '7D5A32' },     // 调整标签颜色为中等棕色
      { token: 'delimiter', foreground: '905F29' }, // 添加分隔符颜色
      { token: 'delimiter.bracket', foreground: '905F29' }, // 添加括号颜色
      { token: 'operator', foreground: '905F29' }, // 添加操作符颜色
      { token: 'variable', foreground: '7F5427' }, // 添加变量颜色
      { token: 'type', foreground: '7D5A32' }, // 添加类型颜色
      { token: 'function', foreground: '7F5427' }, // 添加函数颜色
      { token: 'constant', foreground: '8A6B47' }, // 添加常量颜色
      { token: 'regexp', foreground: 'A67C4A' }, // 添加正则表达式颜色
      { token: 'metatag', foreground: '7D5A32' } // 添加元标签颜色
    ],
    colors: {
      'editor.foreground': '#000000',
      'editor.background': '#FFFFFF',
      'editor.selectionBackground': 'rgba(144, 95, 41, 0.35)',
      'editor.lineHighlightBackground': 'rgba(144, 95, 41, 0.05)',
      'editorCursor.foreground': '#905F29',
      'editorLineNumber.foreground': '#AAAAAA',
      'editorLineNumber.activeForeground': '#905F29',
      'editorIndentGuide.background': '#EEEEEE',
      'editorIndentGuide.activeBackground': 'rgba(144, 95, 41, 0.3)',
      'editor.selectionHighlightBorder': 'rgba(144, 95, 41, 0.3)',
      'editorLink.activeForeground': '#7F5427',    // 保持链接高亮颜色
      'editorBracketMatch.background': 'rgba(144, 95, 41, 0.2)', // 保持括号匹配高亮
      'editorBracketMatch.border': 'rgba(144, 95, 41, 0.4)',     // 保持括号匹配边框
      'minimap.background': '#FFFFFF',
      'minimap.selectionHighlight': 'rgba(144, 95, 41, 0.5)',
      'minimap.errorHighlight': '#905F29',
      'minimap.warningHighlight': 'rgba(144, 95, 41, 0.7)',
      'minimap.findMatchHighlight': 'rgba(144, 95, 41, 0.6)',
      'minimapSlider.background': 'rgba(144, 95, 41, 0.2)',
      'minimapSlider.hoverBackground': 'rgba(144, 95, 41, 0.4)',
      'minimapSlider.activeBackground': 'rgba(144, 95, 41, 0.5)',
      'minimapGutter.addedBackground': '#905F29',
      'minimapGutter.modifiedBackground': '#905F29',
      'minimapGutter.deletedBackground': '#905F29'
    }
  });

  // 初始化声明式编辑器
  if (declarativeEditorContainer.value && !declarativeEditor) {
    declarativeEditor = monaco.editor.create(declarativeEditorContainer.value, {
      ...declarativeEditorOptions,
      value: code.value,
      language: getEditorLanguage(),
      theme: 'gestaltTheme'
    })

    declarativeEditor.onDidChangeModelContent(() => {
      code.value = declarativeEditor.getValue()
    })

    // 添加失去焦点时自动格式化的事件监听
    declarativeEditor.onDidBlurEditorWidget(() => {
      formatDeclarativeEditor()
    })

    // 初始化后延迟格式化文档
    setTimeout(() => {
      formatDeclarativeEditor()
      // 确保编辑器已布局
      declarativeEditor.layout()
    }, 300)
  }

  // 初始化SVG编辑器
  if (svgEditorContainer.value && !svgEditor) {
    svgEditor = monaco.editor.create(svgEditorContainer.value, {
      ...svgEditorOptions,
      value: svgCode.value,
      theme: 'gestaltTheme'
    })

    svgEditor.onDidChangeModelContent(() => {
      svgCode.value = svgEditor.getValue()
    })

    // 添加失去焦点时自动格式化的事件监听
    svgEditor.onDidBlurEditorWidget(() => {
      formatSvgEditor()
    })

    // 初始化后延迟格式化文档
    setTimeout(() => {
      formatSvgEditor()
      // 确保编辑器已布局
      svgEditor.layout()
      
      // 强制更新编辑器配置
      svgEditor.updateOptions({
        wordWrap: 'on',
        scrollBeyondLastLine: false,
        scrollbar: {
          vertical: 'visible',
          horizontal: 'hidden' // 完全隐藏横向滚动条
        }
      })
    }, 300)
  }
}

// 格式化声明式编辑器文档
const formatDeclarativeEditor = () => {
  if (declarativeEditor) {
    try {
      declarativeEditor.getAction('editor.action.formatDocument')?.run()
    } catch (error) {
      console.error('Formatting declarative code editor fails:', error)
    }
  }
}

// 格式化SVG编辑器文档
const formatSvgEditor = () => {
  if (svgEditor) {
    try {
      // 尝试直接运行格式化命令
      const formatAction = svgEditor.getAction('editor.action.formatDocument');
      if (formatAction) {
        formatAction.run();
      } else {
        // 如果直接格式化命令不可用，使用我们的自定义SVG格式化方法
        const currentValue = svgEditor.getValue();
        const formattedValue = formatSvgCode(currentValue);
        // 只有当格式化后的值与当前值不同时才设置
        if (formattedValue !== currentValue) {
          const position = svgEditor.getPosition();
          svgEditor.setValue(formattedValue);
          // 恢复光标位置
          if (position) {
            svgEditor.setPosition(position);
          }
        }
      }
    } catch (error) {
      console.error('Formatting SVG editor fails:', error);
      // 出错时尝试使用我们的自定义SVG格式化
      try {
        const currentValue = svgEditor.getValue();
        const formattedValue = formatSvgCode(currentValue);
        if (formattedValue !== currentValue) {
          const position = svgEditor.getPosition();
          svgEditor.setValue(formattedValue);
          if (position) {
            svgEditor.setPosition(position);
          }
        }
      } catch (e) {
        console.error('Alternate formatting method also fails:', e);
      }
    }
  }
}

// 监听语法变化
watch(selectedSyntax, (newValue) => {
  if (declarativeEditor) {
    const model = declarativeEditor.getModel()
    monaco.editor.setModelLanguage(model, getEditorLanguage())
    code.value = placeholders[newValue]
    declarativeEditor.setValue(code.value)
    
    // 语法变化后格式化文档
    setTimeout(() => {
      formatDeclarativeEditor()
    }, 300)
  }
})

// 防抖函数
const debounce = (fn, delay) => {
  let timer = null
  return function (...args) {
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      fn.apply(this, args)
    }, delay)
  }
}

// 格式化SVG代码
const formatSvgCode = (svgString) => {
  try {
    // 创建一个DOMParser实例
    const parser = new DOMParser()
    // 解析SVG字符串
    const doc = parser.parseFromString(svgString, 'image/svg+xml')

    // 格式化函数
    const format = (node, level) => {
      const indent = '  '.repeat(level)
      let formatted = ''

      // 处理元素节点
      if (node.nodeType === 1) { // 元素节点
        const tagName = node.tagName.toLowerCase()
        const attributes = Array.from(node.attributes)
          .map(attr => `${attr.name}="${attr.value}"`)
          .join(' ')

        if (node.children.length === 0 && !node.textContent.trim()) {
          // 自闭合标签
          formatted = `${indent}<${tagName}${attributes ? ' ' + attributes : ''}/>\n`
        } else {
          // 开始标签
          formatted = `${indent}<${tagName}${attributes ? ' ' + attributes : ''}>\n`

          // 处理子节点
          for (const child of node.childNodes) {
            formatted += format(child, level + 1)
          }

          // 结束标签
          formatted += `${indent}</${tagName}>\n`
        }
      } else if (node.nodeType === 3 && node.textContent.trim()) { // 文本节点
        formatted = `${indent}${node.textContent.trim()}\n`
      }

      return formatted
    }

    // 获取格式化后的SVG代码
    const formattedSvg = format(doc.documentElement, 0)
    return formattedSvg.trim()
  } catch (error) {
    console.error('Error formatting SVG code:', error)
    return svgString // 如果格式化失败，返回原始字符串
  }
}

// 更新预览
const updatePreview = () => {
  nextTick(() => {
    // 格式化SVG代码
    const formattedSvg = formatSvgCode(svgCode.value)
    svgCode.value = formattedSvg
    svgOutput.value = formattedSvg

    // 确保SVG编辑器内容同步
    if (svgEditor && svgEditor.getValue() !== formattedSvg) {
      svgEditor.setValue(formattedSvg)
      // 更新后格式化
      setTimeout(() => {
        formatSvgEditor()
      }, 300)
    }
    setupZoomAndPan()
  })
}

// 生成SVG的主函数
const generateSvg = async () => {
  if (!code.value.trim()) return

  try {
    switch (selectedSyntax.value) {
      case 'd3':
        await handleD3Code()
        break
      case 'echarts':
        await handleEChartsCode()
        break
      case 'vega':
        await handleVegaCode()
        break
      case 'vega-lite':
        await handleVegaLiteCode()
        break
      case 'highcharts':
        await handleHighchartsCode()
        break
    }
    // 更新SVG编辑器的内容
    if (svgEditor) {
      svgEditor.setValue(svgCode.value)
    }
  } catch (error) {
    console.error('generate an error:', error)
    ElMessage.error('An error occurred during generation')
    const errorSvg = `<svg width="200" height="50">
      <text x="10" y="20" fill="red">generate an error: ${error.message}</text>
    </svg>`
    svgCode.value = errorSvg
    svgOutput.value = errorSvg
    // 更新错误信息到SVG编辑器
    if (svgEditor) {
      svgEditor.setValue(errorSvg)
    }
  }
}

// 处理D3代码
const handleD3Code = async () => {
  const container = document.createElement('div')
  const func = new Function('d3', 'container', code.value)
  func(d3, container)
  svgCode.value = container.innerHTML
  svgOutput.value = container.innerHTML
}

// 处理ECharts代码
const handleEChartsCode = async () => {
  try {
    const echarts = await import('echarts')
    const container = document.createElement('div')
    container.style.width = '800px'
    container.style.height = '600px'
    document.body.appendChild(container)

    const chart = echarts.init(container, null, {
      renderer: 'svg',
      ssr: false,
      width: 800,
      height: 600
    })

    let option
    try {
      option = typeof code.value === 'string' ? JSON.parse(code.value) : code.value
    } catch (e) {
      throw new Error('JSON parsing error: ' + e.message)
    }

    // 确保必要的配置项存在并处理默认值
    option = {
      animation: false,
      backgroundColor: 'transparent',
      grid: {
        containLabel: true,
        left: '10%',
        right: '10%',
        top: '10%',
        bottom: '10%'
      },
      ...option,
      // 确保某些配置不被覆盖
      textStyle: {
        fontFamily: 'Arial, sans-serif',
        ...(option.textStyle || {})
      }
    }

    // 设置图表配置
    try {
      chart.setOption(option)
    } catch (e) {
      throw new Error('Failed to set up chart configuration: ' + e.message)
    }

    // 等待图表渲染完成
    await new Promise(resolve => setTimeout(resolve, 200))

    // 获取SVG内容
    const svgElement = container.querySelector('svg')
    if (!svgElement) {
      throw new Error('Unable to fetch SVG elements')
    }

    try {
      // 优化SVG属性
      const width = svgElement.getAttribute('width') || '800'
      const height = svgElement.getAttribute('height') || '600'

      // 创建新的SVG元素以确保属性正确
      const newSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')

      // 复制所有原始属性
      Array.from(svgElement.attributes).forEach(attr => {
        newSvg.setAttribute(attr.name, attr.value)
      })

      // 设置基本属性
      newSvg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
      newSvg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink')
      newSvg.setAttribute('version', '1.1')
      newSvg.setAttribute('viewBox', `0 0 ${width} ${height}`)
      newSvg.setAttribute('width', '100%')
      newSvg.setAttribute('height', '100%')
      newSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet')

      // 复制内容
      newSvg.innerHTML = svgElement.innerHTML

      // 获取处理后的SVG代码
      const svg = newSvg.outerHTML

      // 清理资源
      chart.dispose()
      document.body.removeChild(container)

      // 更新输出
      svgCode.value = svg
      svgOutput.value = svg
    } catch (e) {
      throw new Error('SVG Processing Errors: ' + e.message)
    }
  } catch (error) {
    console.error('ECharts error:', error)
    throw new Error(`ECharts error: ${error.message}`)
  }
}

// 处理Vega代码
const handleVegaCode = async () => {
  try {
    const vegaImport = await import('vega')

    const container = document.createElement('div')
    container.style.width = '800px'
    container.style.height = '600px'
    document.body.appendChild(container)

    const spec = JSON.parse(code.value)
    const view = new vegaImport.View(vegaImport.parse(spec))
      .initialize(container)
      .renderer('svg')
      .width(800)
      .height(600)
      .run()

    await new Promise(resolve => setTimeout(resolve, 100))
    const svg = await view.toSVG()

    view.finalize()
    document.body.removeChild(container)

    const parser = new DOMParser()
    const svgDoc = parser.parseFromString(svg, 'image/svg+xml')
    const svgElement = svgDoc.querySelector('svg')

    if (!svgElement.hasAttribute('viewBox')) {
      svgElement.setAttribute('viewBox', '0 0 800 600')
    }
    svgElement.setAttribute('width', '100%')
    svgElement.setAttribute('height', '100%')

    svgCode.value = svgElement.outerHTML
    svgOutput.value = svgElement.outerHTML
  } catch (error) {
    throw new Error(`Vega错误: ${error.message}`)
  }
}

// 处理Vega-Lite代码
const handleVegaLiteCode = async () => {
  try {
    const vegaImport = await import('vega')
    const vegaLiteImport = await import('vega-lite')

    const container = document.createElement('div')
    container.style.width = '800px'
    container.style.height = '600px'
    document.body.appendChild(container)

    const spec = JSON.parse(code.value)
    const vegaSpec = vegaLiteImport.compile(spec).spec

    const view = new vegaImport.View(vegaImport.parse(vegaSpec))
      .initialize(container)
      .renderer('svg')
      .width(800)
      .height(600)
      .run()

    await new Promise(resolve => setTimeout(resolve, 100))
    const svg = await view.toSVG()

    view.finalize()
    document.body.removeChild(container)

    const parser = new DOMParser()
    const svgDoc = parser.parseFromString(svg, 'image/svg+xml')
    const svgElement = svgDoc.querySelector('svg')

    if (!svgElement.hasAttribute('viewBox')) {
      svgElement.setAttribute('viewBox', '0 0 800 600')
    }
    svgElement.setAttribute('width', '100%')
    svgElement.setAttribute('height', '100%')

    svgCode.value = svgElement.outerHTML
    svgOutput.value = svgElement.outerHTML
  } catch (error) {
    throw new Error(`Vega-Lite错误: ${error.message}`)
  }
}

// 处理Highcharts代码
const handleHighchartsCode = async () => {
  try {
    const Highcharts = await import('highcharts')
    await import('highcharts/modules/exporting')
    await import('highcharts/modules/accessibility')

    const container = document.createElement('div')
    container.style.width = '800px'
    container.style.height = '600px'
    document.body.appendChild(container)

    // 解析用户配置
    const userConfig = JSON.parse(code.value)

    // 基础配置
    const baseConfig = {
      chart: {
        type: 'column',
        forExport: true,
        width: 800,
        height: 600,
        renderTo: container,
        animation: false,
        renderer: 'svg',
        style: {
          fontFamily: 'Arial, sans-serif'
        }
      },
      title: {
        style: {
          fontSize: '18px',
          fontWeight: 'bold'
        }
      },
      credits: {
        enabled: false
      },
      accessibility: {
        enabled: true,
        description: 'Chart Description'
      },
      plotOptions: {
        series: {
          animation: false,
          borderRadius: 3,
          states: {
            hover: {
              brightness: 0.1
            }
          }
        }
      }
    }

    // 深度合并配置
    const mergeConfigs = (target, source) => {
      Object.keys(source).forEach(key => {
        if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
          target[key] = target[key] || {}
          mergeConfigs(target[key], source[key])
        } else {
          target[key] = source[key]
        }
      })
      return target
    }

    // 合并配置
    const config = mergeConfigs(baseConfig, userConfig)

    // 创建图表
    const chart = Highcharts.chart(config)
    await new Promise(resolve => setTimeout(resolve, 200))

    // 获取SVG
    const svg = chart.getSVG()
    const parser = new DOMParser()
    const svgDoc = parser.parseFromString(svg, 'image/svg+xml')
    const svgElement = svgDoc.querySelector('svg')

    // 设置SVG属性
    svgElement.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    svgElement.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink')
    svgElement.setAttribute('version', '1.1')
    svgElement.setAttribute('viewBox', '0 0 800 600')
    svgElement.setAttribute('width', '100%')
    svgElement.setAttribute('height', '100%')
    svgElement.setAttribute('preserveAspectRatio', 'xMidYMid meet')

    // 清理资源
    chart.destroy()
    document.body.removeChild(container)

    // 更新输出
    svgCode.value = svgElement.outerHTML
    svgOutput.value = svgElement.outerHTML
  } catch (error) {
    console.error('Highcharts error:', error)
    throw new Error(`Highcharts error: ${error.message}`)
  }
}

// 复制SVG代码
const copyCode = () => {
  navigator.clipboard.writeText(svgCode.value)
    .then(() => ElMessage({
      message: 'SVG code has been copied to the clipboard',
      type: 'success',
      position: 'top-right',
      customClass: 'custom-message'
    }))
    .catch(() => ElMessage({
      message: 'Reproduction Failure',
      type: 'error',
      position: 'top-right',
      customClass: 'custom-message'
    }))
}

// 下载SVG文件
const downloadSvg = () => {
  const blob = new Blob([svgCode.value], { type: 'image/svg+xml' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `generated-${selectedSyntax.value}.svg`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// 添加缩放和拖拽功能
const setupZoomAndPan = () => {
  const container = previewContainer.value
  if (!container) return

  const svgElement = container.querySelector('svg')
  if (!svgElement) return

  try {
    // 确保SVG有正确的属性
    svgElement.setAttribute('width', '100%')
    svgElement.setAttribute('height', '100%')

    // 获取容器和SVG的尺寸
    const containerRect = container.getBoundingClientRect()
    const containerWidth = containerRect.width
    const containerHeight = containerRect.height

    // 获取SVG的原始尺寸
    const bbox = svgElement.getBBox()
    const svgWidth = bbox.width
    const svgHeight = bbox.height

    // 计算适当的缩放比例
    const scaleX = (containerWidth - 80) / svgWidth // 增加边距
    const scaleY = (containerHeight - 80) / svgHeight
    const scale = Math.min(scaleX, scaleY) // 移除最大缩放限制，允许放大

    // 计算居中位置
    const translateX = (containerWidth - svgWidth * scale) / 2
    const translateY = (containerHeight - svgHeight * scale) / 2

    // 设置viewBox
    svgElement.setAttribute('viewBox', `${bbox.x} ${bbox.y} ${svgWidth} ${svgHeight}`)
    svgElement.setAttribute('preserveAspectRatio', 'xMidYMid meet')

    // 创建或获取包装组
    const svg = d3.select(svgElement)
    let g = svg.select('g.zoom-wrapper')
    if (g.empty()) {
      g = svg.append('g').attr('class', 'zoom-wrapper')
      const children = [...svgElement.childNodes]
      children.forEach(child => {
        if (child.nodeType === 1 && !child.classList.contains('zoom-wrapper')) {
          g.node().appendChild(child)
        }
      })
    }

    // 创建缩放行为
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4]) // 允许更大范围的缩放
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom)

    // 设置初始变换以适应容器
    const initialTransform = d3.zoomIdentity
      .translate(translateX, translateY)
      .scale(scale)

    svg.call(zoom.transform, initialTransform)
  } catch (error) {
    console.error('设置缩放时出错:', error)
  }
}

// 监听SVG内容变化
watch(svgOutput, async () => {
  await nextTick()
  setupZoomAndPan()
})

// 组件挂载时初始化
onMounted(() => {
  // 配置Monaco Editor对HTML/XML的格式化支持
  monaco.languages.registerDocumentFormattingEditProvider('xml', {
    provideDocumentFormattingEdits: (model) => {
      const text = model.getValue();
      const formattedText = formatSvgCode(text);
      return [
        {
          range: model.getFullModelRange(),
          text: formattedText
        }
      ];
    }
  });
  
  selectedSyntax.value = 'vega'
  code.value = placeholders[selectedSyntax.value]
  nextTick(() => {
    initEditors()
  })

  if (svgOutput.value) {
    setupZoomAndPan()
  }

  // 添加事件监听器
  window.addEventListener('svg-content-updated', handleSvgContentUpdated)
  
  // 添加窗口大小改变事件监听
  window.addEventListener('resize', handleWindowResize)
})

// 添加窗口大小变化处理函数
const handleWindowResize = debounce(() => {
  nextTick(() => {
    if (declarativeEditor) {
      declarativeEditor.layout()
      
      // 强制更新配置
      declarativeEditor.updateOptions({
        wordWrap: 'on',
        scrollbar: {
          horizontal: 'hidden'
        }
      })
    }
    if (svgEditor) {
      svgEditor.layout()
      
      // 强制更新配置
      svgEditor.updateOptions({
        wordWrap: 'on',
        scrollbar: {
          horizontal: 'hidden'
        }
      })
    }
  })
}, 100)

// 组件卸载时清理
onUnmounted(() => {
  declarativeEditor?.dispose()
  svgEditor?.dispose()
  window.removeEventListener('svg-content-updated', handleSvgContentUpdated)
  window.removeEventListener('resize', handleWindowResize)
})

// 处理从SvgUploader接收到的新SVG内容
const handleSvgContentUpdated = async (event) => {
  try {
    const response = await fetch('http://127.0.0.1:5000/get_svg', {
      responseType: 'text',
      headers: {
        'Accept': 'image/svg+xml'
      }
    })

    const svgContent = await response.text()

    // 格式化SVG代码
    const formattedSvg = formatSvgCode(svgContent)

    // 更新SVG代码和预览
    svgCode.value = formattedSvg
    svgOutput.value = formattedSvg

    // 如果是从SvgUploader上传的SVG，自动切换到SVG模式
    if (event.detail.filename && event.detail.filename.startsWith('uploaded_')) {
      isDeclarativeMode.value = false
    }

    // 更新SVG编辑器内容
    if (svgEditor) {
      svgEditor.setValue(formattedSvg)
      // 内容更新后格式化
      setTimeout(() => {
        formatSvgEditor()
      }, 300)
    }

    // 更新预览区域
    await nextTick()
    setupZoomAndPan()

    // 根据更新类型显示不同的消息
    if (event.detail.type === 'analysis') {
      ElMessage({
        message: 'SVG perception results have been updated',
        type: 'success',
        position: 'top-right',
        customClass: 'custom-message'
      })
    } else {
      ElMessage({
        message: 'SVG content has been updated',
        type: 'success',
        position: 'top-right',
        customClass: 'custom-message'
      })
    }
  } catch (error) {
    console.error('Error while updating SVG content:', error)
    ElMessage({
      message: 'Failed to update SVG content: ' + error.message,
      type: 'error',
      position: 'top-right',
      customClass: 'custom-message'
    })
  }
}

// 生成SVG并上传到分析器
const generateAndUpload = async () => {
  try {
    // 根据当前模式获取要上传的内容
    let contentToUpload = ''

    if (isDeclarativeMode.value) {
      // 如果是代码模式，先生成SVG
      await generateSvg()
      contentToUpload = svgCode.value
    } else {
      // 如果是SVG模式，直接使用SVG编辑器的内容
      contentToUpload = svgEditor.getValue()
    }

    if (!contentToUpload) {
      ElMessage({
        message: 'No uploadable SVG content',
        type: 'warning',
        position: 'top-right',
        customClass: 'custom-message'
      })
      return
    }

    // 创建Blob对象
    const blob = new Blob([contentToUpload], { type: 'image/svg+xml' })

    
    const file = new File([blob], `generated_width_id.svg`, { type: 'image/svg+xml' })

    // 创建FormData
    const formData = new FormData()
    formData.append('file', file)

    // 发送上传请求
    const response = await fetch('http://127.0.0.1:5000/upload', {
      method: 'POST',
      body: formData
    })

    const result = await response.json()

    if (result.success) {
      ElMessage({
        message: 'SVG has been successfully uploaded to the analyser',
        type: 'success',
        position: 'top-right',
        customClass: 'custom-message'
      })
      // 触发全局事件，通知SvgUploader组件刷新
      window.dispatchEvent(new CustomEvent('svg-uploaded', { detail: { filename: file.name } }))
    } else {
      throw new Error(result.error || 'Upload Failed')
    }
  } catch (error) {
    console.error('Generation or upload errors:', error)
    ElMessage({
      message: 'failure of an operation: ' + error.message,
      type: 'error',
      position: 'top-right',
      customClass: 'custom-message'
    })
  }
}

// 监听模式切换
watch(isDeclarativeMode, (newValue) => {
  nextTick(() => {
    if (!declarativeEditor || !svgEditor) {
      initEditors();
    }
    if (newValue) {
      declarativeEditor?.layout();
    } else {
      svgEditor?.layout();
    }
    
    // 强制调整编辑器大小以适应容器
    setTimeout(() => {
      if (newValue) {
        declarativeEditor?.layout();
      } else {
        svgEditor?.layout();
      }
    }, 50);
  });
});

// 添加监听selectedNodes变化的函数，当变化时高亮代码
watch(selectedNodeIds, (newSelectedNodeIds) => {
  // 确保编辑器已初始化
  if (!declarativeEditor) return;
  
  // 清除旧的装饰
  const oldDecorations = declarativeEditor.getModel()?.getAllDecorations() || [];
  declarativeEditor.deltaDecorations(
    oldDecorations.map(d => d.id),
    []
  );
  
  if (!newSelectedNodeIds.length) return;
  
  // 在代码中查找这些ID并高亮显示
  const model = declarativeEditor.getModel();
  if (!model) return;
  
  const decorations = [];
  const content = model.getValue();
  const lines = content.split('\n');
  
  // 检查每一行是否包含任何选中的节点ID
  lines.forEach((line, index) => {
    // 对于每个选中的节点ID，检查当前行是否包含它
    for (const nodeId of newSelectedNodeIds) {
      // 提取不包含路径前缀的纯ID
      const pureId = nodeId.includes('/') ? nodeId.split('/').pop() : nodeId;
      
      // 搜索id="nodeId"或者id='nodeId'这样的模式
      if (line.includes(`id="${pureId}"`) || line.includes(`id='${pureId}'`)) {
        // 创建一个高亮装饰
        decorations.push({
          range: new monaco.Range(index + 1, 1, index + 1, line.length + 1),
          options: {
            isWholeLine: true,
            className: 'highlighted-line',
            inlineClassName: 'highlighted-text'
          }
        });
        // 找到一个匹配后就跳出当前节点ID的循环
        break;
      }
    }
  });
  
  // 应用高亮装饰
  if (decorations.length > 0) {
    declarativeEditor.deltaDecorations([], decorations);
    
    // 滚动到第一个高亮行
    if (decorations.length > 0) {
      declarativeEditor.revealLineInCenter(decorations[0].range.startLineNumber);
    }
  }
}, { immediate: true });

// 同样监听SVG编辑器
watch(selectedNodeIds, (newSelectedNodeIds) => {
  // 确保编辑器已初始化
  if (!svgEditor) return;
  
  // 清除旧的装饰
  const oldDecorations = svgEditor.getModel()?.getAllDecorations() || [];
  svgEditor.deltaDecorations(
    oldDecorations.map(d => d.id),
    []
  );
  
  if (!newSelectedNodeIds.length) return;
  
  // 在SVG代码中查找这些ID并高亮显示
  const model = svgEditor.getModel();
  if (!model) return;
  
  const decorations = [];
  const content = model.getValue();
  const lines = content.split('\n');
  
  // 检查每一行是否包含任何选中的节点ID
  lines.forEach((line, index) => {
    // 对于每个选中的节点ID，检查当前行是否包含它
    for (const nodeId of newSelectedNodeIds) {
      // 提取不包含路径前缀的纯ID
      const pureId = nodeId.includes('/') ? nodeId.split('/').pop() : nodeId;
      
      // 搜索id="nodeId"或者id='nodeId'这样的模式
      if (line.includes(`id="${pureId}"`) || line.includes(`id='${pureId}'`)) {
        // 创建一个高亮装饰
        decorations.push({
          range: new monaco.Range(index + 1, 1, index + 1, line.length + 1),
          options: {
            isWholeLine: true,
            className: 'highlighted-line',
            inlineClassName: 'highlighted-text'
          }
        });
        // 找到一个匹配后就跳出当前节点ID的循环
        break;
      }
    }
  });
  
  // 应用高亮装饰
  if (decorations.length > 0) {
    svgEditor.deltaDecorations([], decorations);
    
    // 滚动到第一个高亮行
    if (decorations.length > 0) {
      svgEditor.revealLineInCenter(decorations[0].range.startLineNumber);
    }
  }
}, { immediate: true });

// 处理文件输入触发
const triggerFileInput = () => {
  fileInput.value.click()
}

// 处理文件选择变化
const handleFileChange = (event) => {
  const selectedFile = event.target.files[0]
  if (selectedFile) {
    file.value = selectedFile
    uploadFile()
  }
}

// 处理文件拖放
const handleDrop = (event) => {
  event.preventDefault()
  const droppedFile = event.dataTransfer.files[0]
  if (droppedFile && droppedFile.type === 'image/svg+xml') {
    file.value = droppedFile
    uploadFile()
  }
}

// 格式化文件大小
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// 上传文件
const uploadFile = () => {
  if (!file.value) return
  const formData = new FormData()

  // 创建新的File对象，添加uploaded_前缀
  const newFile = new File([file.value], `uploaded_${file.value.name}`, { type: file.value.type })
  formData.append('file', newFile)

  // 显示分析状态
  analyzing.value = true
  progress.value = 0
  currentStep.value = '准备处理上传文件...'

  // 清除选中的节点
  store.dispatch('clearSelectedNodes')

  // 连接进度事件源
  const eventSource = new EventSource('http://127.0.0.1:5000/progress')
  
  // 监听进度更新
  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data)
    progress.value = data.progress
    currentStep.value = data.step
  }

  // 监听错误
  eventSource.onerror = () => {
    eventSource.close()
  }

  // 上传文件
  axios.post('http://127.0.0.1:5000/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
    .then(response => {
      if (response.data.success) {
        // 获取SVG内容
        const svgContent = response.data.svgContent || ''
        
        // 自动切换到SVG模式
        isDeclarativeMode.value = false
        
        // 使用Promise确保SVG内容加载完成
        const loadSvgContent = async () => {
          // 更新SVG代码
          svgCode.value = svgContent
          
          // 等待DOM更新
          await nextTick()
          
          // 确保SVG编辑器存在并设置内容
          if (svgEditor) {
            // 设置编辑器内容
            svgEditor.setValue(svgContent)
            
            // 等待编辑器内容设置完成
            await new Promise(resolve => {
              // 检查编辑器内容是否已加载
              const checkContent = () => {
                const editorContent = svgEditor.getValue()
                if (editorContent && editorContent.trim() !== '') {
                  resolve()
                } else {
                  // 如果内容还未加载，继续等待
                  setTimeout(checkContent, 100)
                }
              }
              
              // 开始检查
              checkContent()
            })
            
            // 等待额外的时间确保内容完全渲染
            await new Promise(resolve => setTimeout(resolve, 500))
            
            // 执行upload操作
            generateAndUpload()
          }
        }
        
        // 执行加载过程
        loadSvgContent().catch(error => {
          console.error('Error loading SVG content:', error)
        })
        
        // 触发事件通知其他组件
        window.dispatchEvent(new CustomEvent('svg-content-updated', {
          detail: { filename: newFile.name }
        }))
      }
    })
    .catch(error => {
      console.error('Error in upload process:', error)
    })
    .finally(() => {
      analyzing.value = false
      eventSource.close()
    })
}
</script>

<style scoped>
.code-to-svg-container {
  display: flex;
  height: 100%;
  width: 100%;
  min-width: 0;
  overflow: hidden;
}

.editors-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100%;
  position: relative;
  min-width: 0;
  width: 100%;
}

.editor-section {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  min-width: 0;
  width: 100%;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(200, 200, 200, 0.3);
  position: relative;
}

.editor-section:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  transform: translateY(-1px);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-radius: 12px 12px 0 0;
  flex-shrink: 0;
  padding: 12px;
  border-bottom: 1px solid rgba(200, 200, 200, 0.3);
  flex-wrap: wrap;
  gap: 8px;
}

.left-tools {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

@media (max-width: 768px) {
  .section-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .left-tools {
    margin-bottom: 8px;
    width: 100%;
    justify-content: space-between;
  }
  
  .side-mode-switch {
    width: 100%;
    justify-content: flex-end;
    height: 35px !important;
  }

  :deep(.el-button) {
    padding: 8px 12px;
    font-size: 13px;
  }
  
  .title {
    font-size: 1.5em;
    margin-left: 0;
  }
}

@media (max-width: 480px) {
  .left-tools {
    gap: 6px;
  }
  
  :deep(.el-button) {
    padding: 6px 10px;
    font-size: 12px;
  }
  
  .syntax-selector {
    width: 100px;
  }
}

.syntax-selector {
  width: 120px;
  height: 35px !important;
}

:deep(.syntax-selector-dropdown) {
  min-width: 400px !important;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(200, 200, 200, 0.3);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}

.syntax-option {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 50px;
  padding: 8px;
}

.syntax-label {
  font-weight: 500;
  color: #1d1d1f;
  font-size: 14px;
  line-height: 1.4;
}

.syntax-description {
  font-size: 12px;
  color: #86868b;
  line-height: 1.4;
  word-break: break-word;
  max-width: 260px;
}

.code-editor {
  flex: 1;
  border-radius: 0 0 12px 12px;
  overflow: hidden;
  display: flex;
  min-height: 0;
  background-color: #ffffff;
  margin: 0;
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.05);
  width: 100%;
}

.editor-wrapper {
  width: 100%;
  height: 100%;
  min-height: 200px;
  overflow: hidden;
}

:deep(.el-button) {
  margin-left: 0;
  border-radius: 8px;
  background: #905F29 !important;
  border-color: #905F29;
  color: white;
  font-weight: 500;
  box-shadow: 0 2px 8px rgba(136, 95, 53, 0.2);
  height: 35px !important;
  line-height: 20px !important;
  padding: 6px 16px !important;
}

:deep(.el-button:hover) {
  background: #7F5427 !important;
  border-color: #7F5427;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(144, 95, 41, 0.3);
}

:deep(.el-button:active) {
  transform: translateY(1px);
  box-shadow: 0 2px 6px rgba(30, 144, 255, 0.2);
}

:deep(.el-select) {
  .el-input__wrapper {
    border-radius: 8px;
    background: rgba(240, 240, 240, 0.6);
    border: 1px solid rgba(200, 200, 200, 0.3);
    box-shadow: none;
    height: 35px !important;
    line-height: 35px !important;
  }

  .el-input__inner {
    height: 35px !important;
    line-height: 35px !important;
  }

  .el-input__wrapper:hover {
    background: rgba(235, 235, 235, 0.8);
  }

  .el-input__wrapper.is-focus {
    border-color: #1E90FF;
    box-shadow: 0 0 0 2px rgba(30, 144, 255, 0.2);
  }
}

.editor-fade-enter-from,
.editor-fade-leave-to {
  opacity: 0;
  transform: scale(0.98);
}

.side-mode-switch {
  display: flex;
  flex-direction: row;
  gap: 10px;
  align-items: center;
}

.mode-tabs {
  display: flex;
  gap: 1px;
  padding: 2px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(200, 200, 200, 0.3);
}

.mode-tab {
  padding: 4px 12px;
  font-size: 13px;
  color: #666;
  cursor: pointer;
  border-radius: 6px;
  font-weight: 500;
  height: 35px;
  line-height: 27px;
  display: flex;
  align-items: center;
}

.mode-tab:hover {
  color: #333;
  background: rgba(144, 95, 41, 0.1);
}

.mode-tab.active {
  background: #905F29;
  color: white;
}

/* 添加放大按钮文字的样式 */
.larger-text-btn {
  font-size: 15px !important;
  font-weight: 600 !important;
  padding: 6px 16px !important;
  height: 35px !important;
  line-height: 20px !important;
}

/* 添加放大模式切换标签文字的样式 */
.larger-text-tab {
  font-size: 15px !important;
  font-weight: 600 !important;
  padding: 6px 16px !important;
  height: 35px !important;
  line-height: 20px !important;
}

/* 全局样式，不能使用 scoped */
:global(.custom-message) {
  min-width: 380px !important;
  padding: 14px 26px 14px 13px !important;
  height: 40px !important;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1) !important;
  background: #fff !important;
  border-radius: 8px !important;
  border: 1px solid #e4e7ed !important;
  display: flex !important;
  align-items: center !important;
}

:global(.custom-message .el-message__content) {
  font-size: 14px !important;
  color: #333 !important;
  line-height: 1 !important;
  padding-left: 8px !important;
}

:global(.custom-message.el-message--success .el-message__icon) {
  color: #905F29 !important;
  font-size: 16px !important;
}

:global(.custom-message.el-message--error .el-message__icon) {
  color: #ff4d4f !important;
  font-size: 16px !important;
}

:global(.custom-message.el-message--warning .el-message__icon) {
  color: #faad14 !important;
  font-size: 16px !important;
}

.title {
  margin: 0 0 0 10px;
  font-size: 1.8em;
  font-weight: bold;
  color: #1d1d1f;
  letter-spacing: -0.01em;
  opacity: 0.8;
}

/* 添加高亮样式 */
:deep(.highlighted-line) {
  background-color: rgba(144, 95, 41, 0.15);
  border-left: 3px solid #905F29;
}

:deep(.highlighted-text) {
  font-weight: bold;
  color: #905F29;
}

/* 自定义选中文本的样式 */
:deep(.monaco-editor .selected-text) {
  background-color: rgba(144, 95, 41, 0.35) !important;
}

:deep(.monaco-editor .cursor) {
  background-color: #905F29 !important;
  border-color: #905F29 !important;
}

/* 覆盖所有可能的红色高亮 */
:deep(.monaco-editor .line-delete),
:deep(.monaco-editor .line-insert),
:deep(.monaco-editor .monaco-diff-editor .line-insert),
:deep(.monaco-editor .monaco-diff-editor .line-delete),
:deep(.monaco-editor .monaco-diff-editor .char-insert),
:deep(.monaco-editor .monaco-diff-editor .char-delete),
:deep(.monaco-editor .current-line-highlight),
:deep(.monaco-editor .current-line),
:deep(.monaco-editor .view-overlays .current-line),
:deep(.monaco-editor .margin-view-overlays .current-line-margin),
:deep(.monaco-editor .view-overlays .line-highlight) {
  background-color: rgba(144, 95, 41, 0.15) !important;
  border-left-color: #905F29 !important;
}

/* 确保所有高亮文本使用棕色 */
:deep(.monaco-editor .findMatch),
:deep(.monaco-editor .currentFindMatch),
:deep(.monaco-editor .selectionHighlight),
:deep(.monaco-editor .wordHighlight),
:deep(.monaco-editor .wordHighlightStrong) {
  background-color: rgba(144, 95, 41, 0.35) !important;
  border-color: rgba(144, 95, 41, 0.5) !important;
}

/* 修改行号前后的红色标记为棕色 */
:deep(.monaco-editor .margin .margin-view-overlays .line-numbers),
:deep(.monaco-editor .margin-view-overlays .line-numbers) {
  color: rgba(144, 95, 41, 0.8) !important;
}

:deep(.monaco-editor .margin .margin-view-overlays .cldr.folding),
:deep(.monaco-editor .margin .margin-view-overlays .cldr) {
  color: #905F29 !important;
}

:deep(.monaco-editor .glyph-margin .cgmr) {
  background-color: #905F29 !important;
}

:deep(.monaco-editor .margin .margin-view-overlays .line-numbers.active-line-number) {
  color: #905F29 !important;
  font-weight: bold;
}

/* 修改断点和错误标记颜色 */
:deep(.monaco-editor .breakpoint-glyph-margin),
:deep(.monaco-editor .breakpoint) {
  background-color: #905F29 !important;
}

:deep(.monaco-editor .error-glyph-margin),
:deep(.monaco-editor .error) {
  background-color: rgba(144, 95, 41, 0.7) !important;
}

.editor-transition {
  transition: opacity 0.3s, transform 0.3s;
}

:deep(.monaco-editor .monaco-scrollable-element) {
  overflow: auto !important;
}

:deep(.monaco-editor .scrollbar) {
  opacity: 0.6;
  transition: opacity 0.2s;
}

:deep(.monaco-editor .scrollbar:hover) {
  opacity: 1;
}

/* 隐藏不必要的水平滚动条 */
:deep(.monaco-editor .scrollbar.horizontal) {
  display: none !important;
}

/* 确保长行自动换行 */
:deep(.monaco-editor .view-line) {
  word-wrap: break-word !important;
  white-space: pre-wrap !important;
}

:deep(.monaco-editor) {
  width: 100% !important;
  overflow: hidden !important;
}

/* 全局样式覆盖，确保没有红色高亮 */
:global(.monaco-editor .line-delete),
:global(.monaco-editor .line-insert),
:global(.monaco-editor .current-line),
:global(.monaco-editor .view-overlays .current-line),
:global(.monaco-editor .margin-view-overlays .current-line-margin),
:global(.monaco-editor .view-overlays .line-highlight) {
  background-color: rgba(144, 95, 41, 0.15) !important;
  border-left-color: #905F29 !important;
}

:global(.monaco-editor .findMatch),
:global(.monaco-editor .currentFindMatch),
:global(.monaco-editor .selectionHighlight),
:global(.monaco-editor .wordHighlight),
:global(.monaco-editor .wordHighlightStrong) {
  background-color: rgba(144, 95, 41, 0.35) !important;
  border-color: rgba(144, 95, 41, 0.5) !important;
}

:global(.monaco-editor .selected-text) {
  background-color: rgba(144, 95, 41, 0.35) !important;
}

/* 全局覆盖行号前后的红色标记 */
:global(.monaco-editor .margin .margin-view-overlays .line-numbers),
:global(.monaco-editor .margin-view-overlays .line-numbers) {
  color: rgba(144, 95, 41, 0.8) !important;
}

:global(.monaco-editor .margin .margin-view-overlays .cldr.folding),
:global(.monaco-editor .margin .margin-view-overlays .cldr) {
  color: #905F29 !important;
}

:global(.monaco-editor .glyph-margin .cgmr) {
  background-color: #905F29 !important;
}

:global(.monaco-editor .margin .margin-view-overlays .line-numbers.active-line-number) {
  color: #905F29 !important;
  font-weight: bold;
}

:global(.monaco-editor .breakpoint-glyph-margin),
:global(.monaco-editor .breakpoint) {
  background-color: #905F29 !important;
}

:global(.monaco-editor .error-glyph-margin),
:global(.monaco-editor .error) {
  background-color: rgba(144, 95, 41, 0.7) !important;
}

/* 修改行装饰器和标记颜色 */
:global(.monaco-editor .decorationsOverviewRuler .decorationsOverviewRuler-red),
:global(.monaco-editor .decorationsOverviewRuler .decorationsOverviewRuler-error),
:global(.monaco-editor .decorationsOverviewRuler .decorationsOverviewRuler-warning) {
  background-color: #905F29 !important;
}

:global(.monaco-editor .contentWidgets .codicon-error),
:global(.monaco-editor .contentWidgets .codicon-warning) {
  color: #905F29 !important;
}

:global(.monaco-editor .squiggly-error),
:global(.monaco-editor .squiggly-warning),
:global(.monaco-editor .squiggly-info) {
  background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 6 3' enable-background='new 0 0 6 3' height='3' width='6'%3E%3Cg fill='%23905F29'%3E%3Cpolygon points='5.5,0 2.5,3 1.1,3 4.1,0'/%3E%3Cpolygon points='4,0 6,2 6,0.6 5.4,0'/%3E%3Cpolygon points='0,2 1,3 2.4,3 0,0.6'/%3E%3C/g%3E%3C/svg%3E") repeat-x bottom left !important;
}

/* 修改行号前的红色标记 */
:global(.monaco-editor .margin-view-overlays .cgmr.codicon-circle-filled),
:global(.monaco-editor .margin-view-overlays .cgmr.codicon-error),
:global(.monaco-editor .margin-view-overlays .cgmr.codicon-warning),
:global(.monaco-editor .margin-view-overlays .cgmr.codicon-info) {
  color: #905F29 !important;
}

/* 修改行内装饰器 */
:global(.monaco-editor .inline-decoration),
:global(.monaco-editor .inline-decoration.error),
:global(.monaco-editor .inline-decoration.warning),
:global(.monaco-editor .inline-decoration.info) {
  color: #905F29 !important;
  border-color: #905F29 !important;
}

/* 修改行号前的红色圆点 */
:global(.monaco-editor .margin-view-overlays .cgmr),
:global(.monaco-editor .margin-view-overlays .cgmr.codicon-circle-filled),
:global(.monaco-editor .margin-view-overlays .cgmr.codicon-circle-outline) {
  color: #905F29 !important;
  background-color: transparent !important;
}

/* 修改行号前的红色箭头 */
:global(.monaco-editor .margin-view-overlays .cgmr.codicon-chevron-right),
:global(.monaco-editor .margin-view-overlays .cgmr.codicon-chevron-down) {
  color: #905F29 !important;
}

/* 修改右侧缩略图中的颜色 */
:global(.monaco-editor .minimap .minimap-slider),
:global(.monaco-editor .minimap .minimap-slider .minimap-slider-horizontal) {
  background: rgba(144, 95, 41, 0.3) !important;
}

:global(.monaco-editor .minimap .minimap-slider:hover),
:global(.monaco-editor .minimap .minimap-slider:active) {
  background: rgba(144, 95, 41, 0.5) !important;
}

:global(.monaco-editor .minimap-shadow-hidden),
:global(.monaco-editor .minimap-shadow-visible) {
  box-shadow: inset -6px 0 6px -6px rgba(144, 95, 41, 0.25) !important;
}

/* 修改缩略图中的高亮颜色 */
:global(.monaco-editor .minimap-decorations-layer .minimap-decoration) {
  background-color: rgba(144, 95, 41, 0.6) !important;
}

:global(.monaco-editor .minimap .minimap-decorations-layer .current-line) {
  background-color: rgba(144, 95, 41, 0.4) !important;
}

:global(.monaco-editor .minimap .minimap-decorations-layer .selection-highlight) {
  background-color: rgba(144, 95, 41, 0.5) !important;
}

/* 修改缩略图中的错误和警告标记 */
:global(.monaco-editor .minimap .minimap-decorations-layer .error-decoration),
:global(.monaco-editor .minimap .minimap-decorations-layer .warning-decoration) {
  background-color: #905F29 !important;
}

/* 修改缩略图中的代码颜色 - 使用CSS过滤器 */
:global(.monaco-editor .minimap) {
  filter: sepia(0.6) hue-rotate(320deg) saturate(1.2) !important;
}

/* 确保滑块不受过滤器影响 */
:global(.monaco-editor .minimap .minimap-slider) {
  filter: none !important;
  background: rgba(144, 95, 41, 0.3) !important;
}

:global(.monaco-editor .minimap .minimap-slider:hover),
:global(.monaco-editor .minimap .minimap-slider:active) {
  filter: none !important;
  background: rgba(144, 95, 41, 0.5) !important;
}

/* 添加XML/HTML属性名和值的特殊样式 */
:deep(.monaco-editor .mtk5),  /* 通常是属性名 */
:deep(.monaco-editor .mtk12), /* 通常是属性值 */
:deep(.monaco-editor .mtk13) {
  color: #5D4126 !important; /* 属性名颜色 - 深棕色 */
}

:deep(.monaco-editor .mtk4),  /* 通常是字符串/属性值 */
:deep(.monaco-editor .mtk6),
:deep(.monaco-editor .mtk7) {
  color: #A67C4A !important; /* 属性值颜色 - 浅棕色 */
}

/* 确保标签名称使用中等棕色 */
:deep(.monaco-editor .mtk3),
:deep(.monaco-editor .mtk10) {
  color: #7D5A32 !important; /* 标签名颜色 - 中等棕色 */
}

/* 全局覆盖XML/HTML属性名和值的样式 */
:global(.monaco-editor .mtk5),
:global(.monaco-editor .mtk12),
:global(.monaco-editor .mtk13) {
  color: #5D4126 !important; /* 深棕色 */
}

:global(.monaco-editor .mtk4),
:global(.monaco-editor .mtk6),
:global(.monaco-editor .mtk7) {
  color: #A67C4A !important; /* 浅棕色 */
}

:global(.monaco-editor .mtk3),
:global(.monaco-editor .mtk10) {
  color: #7D5A32 !important; /* 中等棕色 */
}

/* 文件上传区样式 */
.upload-section {
  margin-bottom: 16px;
  width: 100%;
  position: relative;
}

.mac-upload-zone {
  position: relative;
  z-index: 10;
  width: 100%;
}

.mac-upload-container {
  background: rgba(255, 255, 255, 0.95);
  border: 1px dashed rgba(144, 95, 41, 0.3);
  border-radius: 8px;
  padding: 8px 12px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.mac-upload-container:hover {
  border-color: rgba(144, 95, 41, 0.6);
  background: rgba(255, 255, 255, 0.98);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.hidden-input {
  display: none;
}

.upload-content {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
}

.upload-icon {
  color: #aa7134;
  font-size: 24px;
}

.upload-text {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
}

.primary-text {
  font-size: 16px;
  font-weight: 500;
  color: #1d1d1f;
}

.file-info {
  margin-left: auto;
  padding: 4px 8px;
  background: rgba(144, 95, 41, 0.1);
  border-radius: 4px;
  display: flex;
  gap: 6px;
  align-items: center;
}

.file-name {
  font-size: 14px;
  font-weight: 500;
  color: #aa7134;
  margin-right: 8px;
}

.file-size {
  color: #86868b;
  font-size: 12px;
}

.progress-card {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  z-index: 100;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 8px;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(200, 200, 200, 0.3);
  padding: 12px 16px;
  margin-top: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
}

.progress-label {
  font-size: 13px;
  font-weight: 500;
  color: #1d1d1f;
  margin-bottom: 8px;
}

.upload-progress {
  height: 6px;
}

/* 添加SVG编辑器和上传区的并排布局 */
.svg-editor-container {
  display: flex;
  height: 100%;
  width: 100%;
}

.code-editor {
  flex: 3;
  border-radius: 0 0 12px 12px;
  overflow: hidden;
  display: flex;
  min-height: 0;
  background-color: #ffffff;
  margin: 0;
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* 调整上传区样式以适应右侧 */
.upload-section {
  flex: 1;
  min-width: 250px;
  padding: 12px;
  border-left: 1px solid rgba(200, 200, 200, 0.3);
  display: flex;
  flex-direction: column;
  justify-content: center;
  position: relative;
}

.mac-upload-container {
  background: rgba(255, 255, 255, 0.95);
  border: 1px dashed rgba(144, 95, 41, 0.3);
  border-radius: 8px;
  padding: 12px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  height: 100%;
}

.mac-upload-container:hover {
  border-color: rgba(144, 95, 41, 0.6);
  background: rgba(255, 255, 255, 0.98);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  width: 100%;
}

.upload-icon {
  color: #aa7134;
  font-size: 32px;
}

.upload-text {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  text-align: center;
}

.primary-text {
  font-size: 16px;
  font-weight: 500;
  color: #1d1d1f;
}

.file-info {
  width: 100%;
  padding: 8px;
  background: rgba(144, 95, 41, 0.1);
  border-radius: 4px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: center;
}

.file-name {
  font-size: 14px;
  font-weight: 500;
  color: #aa7134;
  word-break: break-all;
}

.file-size {
  color: #86868b;
  font-size: 12px;
}

.progress-card {
  position: absolute;
  bottom: 12px;
  left: 12px;
  right: 12px;
  z-index: 100;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 8px;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(200, 200, 200, 0.3);
  padding: 12px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .svg-editor-container {
    flex-direction: column;
  }
  
  .upload-section {
    flex: none;
    min-height: 200px;
    border-left: none;
    border-top: 1px solid rgba(200, 200, 200, 0.3);
  }
}

/* 标题栏中间的上传区样式 */
.header-upload-container {
  flex: 1;
  margin: 0 16px;
  background: rgba(255, 255, 255, 0.8);
  border: 1px dashed rgba(144, 95, 41, 0.3);
  border-radius: 8px;
  padding: 6px 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  min-width: 250px;
}

.header-upload-container:hover {
  border-color: rgba(144, 95, 41, 0.6);
  background: rgba(255, 255, 255, 0.95);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.header-upload-content {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
}

.upload-icon {
  color: #aa7134;
}

.upload-text {
  color: #1d1d1f;
  font-size: 1.1em;
  font-weight: 400;
  white-space: nowrap;
}

.file-info {
  padding: 3px 8px;
  background: rgba(144, 95, 41, 0.1);
  border-radius: 4px;
  display: flex;
  gap: 6px;
  align-items: center;
  justify-content: space-between;
  flex-direction: row;
}

.file-name {
  font-size: 13px;
  font-weight: 500;
  color: #aa7134;
  max-width: 180px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file-size {
  color: #86868b;
  font-size: 12px;
}

.progress-card {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 8px;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(200, 200, 200, 0.3);
  padding: 12px 16px;
  margin: 8px 16px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
}

.progress-label {
  font-size: 13px;
  font-weight: 500;
  color: #1d1d1f;
  margin-bottom: 8px;
}

.upload-progress {
  height: 6px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-radius: 12px 12px 0 0;
  flex-shrink: 0;
  padding: 12px;
  border-bottom: 1px solid rgba(200, 200, 200, 0.3);
  flex-wrap: wrap;
  gap: 8px;
}

.left-tools {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

/* 响应式调整 */
@media (max-width: 1200px) {
  .section-header {
    flex-wrap: wrap;
  }
  
  .left-tools {
    width: auto;
  }
  
  .header-upload-container {
    order: 1;
    margin: 8px 0;
    width: 100%;
    flex: none;
  }
  
  .side-mode-switch {
    order: 2;
    width: 100%;
    justify-content: flex-end;
  }
}

@media (max-width: 768px) {
  .header-upload-content {
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
  }
  
  .file-info {
    margin-left: 0;
    width: 100%;
  }
}
</style>
