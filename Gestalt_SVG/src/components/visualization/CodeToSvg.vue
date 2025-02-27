<template>
  <div class="code-to-svg-container">
    <div class="editors-container">
      <!-- 声明式语法编辑器 -->
      <div v-show="isDeclarativeMode" class="editor-section editor-transition">
        <div class="section-header">
          <div class="left-tools">
            <span class="title">SVG Editor</span>
            <el-button type="primary" @click="generateAndUpload">Generate</el-button>
            <el-button @click="copyCode">Copy</el-button>
            <el-button @click="downloadSvg">Download</el-button>
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
            <div class="mode-tabs">
              <div class="mode-tab" :class="{ active: isDeclarativeMode }" @click="isDeclarativeMode = true">
                Code
              </div>
              <div class="mode-tab" :class="{ active: !isDeclarativeMode }" @click="isDeclarativeMode = false">
                SVG
              </div>
            </div>
          </div>
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
            <el-button type="primary" @click="generateAndUpload">Generate</el-button>
            <el-button @click="copyCode">Copy</el-button>
            <el-button @click="downloadSvg">Download</el-button>
          </div>
          <div class="side-mode-switch">
            <div class="mode-tabs">
              <div class="mode-tab" :class="{ active: isDeclarativeMode }" @click="isDeclarativeMode = true">
                Code
              </div>
              <div class="mode-tab" :class="{ active: !isDeclarativeMode }" @click="isDeclarativeMode = false">
                SVG
              </div>
            </div>
          </div>
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

// Monaco Editor配置
const declarativeEditorOptions = {
  theme: 'vs',
  fontSize: 14,
  lineNumbers: 'on',
  roundedSelection: false,
  scrollBeyondLastLine: false,
  readOnly: false,
  minimap: {
    enabled: true
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
  // 初始化声明式编辑器
  if (declarativeEditorContainer.value && !declarativeEditor) {
    declarativeEditor = monaco.editor.create(declarativeEditorContainer.value, {
      ...declarativeEditorOptions,
      value: code.value,
      language: getEditorLanguage()
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
      value: svgCode.value
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
      console.error('格式化声明式代码编辑器失败:', error)
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
    console.error('生成错误:', error)
    ElMessage.error('生成过程中发生错误')
    const errorSvg = `<svg width="200" height="50">
      <text x="10" y="20" fill="red">生成错误: ${error.message}</text>
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
      throw new Error('JSON解析错误: ' + e.message)
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
      throw new Error('设置图表配置失败: ' + e.message)
    }

    // 等待图表渲染完成
    await new Promise(resolve => setTimeout(resolve, 200))

    // 获取SVG内容
    const svgElement = container.querySelector('svg')
    if (!svgElement) {
      throw new Error('无法获取SVG元素')
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
      throw new Error('SVG处理错误: ' + e.message)
    }
  } catch (error) {
    console.error('ECharts错误:', error)
    throw new Error(`ECharts错误: ${error.message}`)
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
        description: '图表描述'
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
    console.error('Highcharts错误:', error)
    throw new Error(`Highcharts错误: ${error.message}`)
  }
}

// 处理Matplotlib代码
const handleMatplotlibCode = async () => {
  try {
    // 这里需要与后端API交互来执行Python代码
    const response = await fetch('http://127.0.0.1:5000/api/matplotlib', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        code: code.value,
        // 添加额外信息帮助后端正确处理
        options: {
          format: 'svg',
          dpi: 100,
          bbox_inches: 'tight',
          transparent: true
        }
      })
    })

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Matplotlib API响应错误:', errorText);
      throw new Error(`后端API调用失败: ${response.status} ${response.statusText}`);
    }

    // 先将响应读取为文本，只读取一次
    const responseText = await response.text();
    
    // 尝试将文本解析为JSON
    try {
      const result = JSON.parse(responseText);
      if (result.error) {
        throw new Error(result.error);
      }
      
      if (result.svg) {
        // 处理SVG数据
        svgCode.value = result.svg;
        svgOutput.value = result.svg;
      } else {
        throw new Error('返回的数据中没有SVG内容');
      }
    } catch (jsonError) {
      // 如果不是有效的JSON，检查是否是有效的SVG
      if (responseText.trim().startsWith('<svg') || responseText.trim().startsWith('<?xml')) {
        svgCode.value = responseText;
        svgOutput.value = responseText;
      } else {
        console.error('无法解析响应:', responseText.substring(0, 100) + '...');
        throw new Error('返回的不是有效的SVG或JSON内容');
      }
    }
  } catch (error) {
    console.error('Matplotlib处理错误:', error);
    throw new Error(`Matplotlib错误: ${error.message}`);
  }
}

// 复制SVG代码
const copyCode = () => {
  navigator.clipboard.writeText(svgCode.value)
    .then(() => ElMessage({
      message: 'SVG代码已复制到剪贴板',
      type: 'success',
      position: 'top-right',
      customClass: 'custom-message'
    }))
    .catch(() => ElMessage({
      message: '复制失败',
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
        message: '没有可上传的SVG内容',
        type: 'warning',
        position: 'top-right',
        customClass: 'custom-message'
      })
      return
    }

    // 创建Blob对象
    const blob = new Blob([contentToUpload], { type: 'image/svg+xml' })

    // 创建File对象，使用时间戳确保文件名唯一
    const timestamp = new Date().getTime()
    const file = new File([blob], `generated_${timestamp}.svg`, { type: 'image/svg+xml' })

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
        message: 'SVG已成功上传到分析器',
        type: 'success',
        position: 'top-right',
        customClass: 'custom-message'
      })
      // 触发全局事件，通知SvgUploader组件刷新
      window.dispatchEvent(new CustomEvent('svg-uploaded', { detail: { filename: file.name } }))
    } else {
      throw new Error(result.error || '上传失败')
    }
  } catch (error) {
    console.error('生成或上传错误:', error)
    ElMessage({
      message: '操作失败: ' + error.message,
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
  }

  :deep(.el-button) {
    padding: 8px 12px;
    font-size: 13px;
  }
  
  .title {
    font-size: 14px;
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
  background: #1E90FF !important;
  border-color: #1E90FF;
  color: white;
  font-weight: 500;
  box-shadow: 0 2px 8px rgba(30, 144, 255, 0.2);
}

:deep(.el-button:hover) {
  background: #1A7FE5 !important;
  border-color: #1A7FE5;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(30, 144, 255, 0.3);
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
}

.mode-tab:hover {
  color: #333;
  background: rgba(30, 144, 255, 0.1);
}

.mode-tab.active {
  background: #1E90FF;
  color: white;
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
  color: #1E90FF !important;
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
  font-size: 16px;
  font-weight: bold;
  color: #1d1d1f;
  letter-spacing: -0.01em;
  opacity: 0.8;
}

/* 添加高亮样式 */
:deep(.highlighted-line) {
  background-color: rgba(30, 144, 255, 0.15);
  border-left: 3px solid #1E90FF;
}

:deep(.highlighted-text) {
  font-weight: bold;
  color: #0066CC;
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
</style>
