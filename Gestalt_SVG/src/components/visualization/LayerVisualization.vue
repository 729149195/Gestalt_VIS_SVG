<template>
  <div ref="chartContainer" style="width: 1230px; height: 1350px;"></div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';
const store = useStore();

const eleURL = "http://192.168.107.209:5000/layer_data";
const chartContainer = ref(null);

const customColorMap = {
    "circle": "#FFE119", // 鲜黄
    "rect": "#E6194B", // 猩红
    "line": "#4363D8", // 宝蓝
    "polyline": "#911EB4", // 紫色
    "polygon": "#F58231", // 橙色
    "path": "#3CB44B", // 明绿
    "text": "#46F0F0", // 青色
    "ellipse": "#F032E6", // 紫罗兰
    "image": "#BCF60C", // 酸橙
    "use": "#FFD700", // 金色
    "defs": "#FF4500", // 橙红色
    "linearGradient": "#1E90FF", // 道奇蓝
    "radialGradient": "#FF6347", // 番茄
    "stop": "#4682B4", // 钢蓝
    "symbol": "#D2691E", // 巧克力
    "clipPath": "#FABEBE", // 粉红
    "mask": "#8B008B", // 深紫罗兰红色
    "pattern": "#A52A2A", // 棕色
    "filter": "#5F9EA0", // 冰蓝
    "feGaussianBlur": "#D8BFD8", // 紫丁香
    "feOffset": "#FFDAB9", // 桃色
    "feBlend": "#32CD32", // 酸橙绿
    "feFlood": "#FFD700", // 金色
    "feImage": "#FF6347", // 番茄
    "feComposite": "#FF4500", // 橙红色
    "feColorMatrix": "#1E90FF", // 道奇蓝
    "feMerge": "#FF1493", // 深粉色
    "feMorphology": "#00FA9A", // 中春绿色
    "feTurbulence": "#8B008B", // 深紫罗兰红色
    "feDisplacementMap": "#FFD700", // 金色
    "unknown": "#696969" // 暗灰色
};

onMounted(async () => {
  if (!chartContainer.value) return;
  try {
    const response = await fetch(eleURL);
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const data = await response.json();
    renderTree(data);
  } catch (error) {
    console.error('There has been a problem with your fetch operation:', error);
  }
});

const renderTree = (data) => {
  const width = 1230;
  const height = 1350;

  // 清除先前的 SVG 元素（如果存在）
  d3.select(chartContainer.value).select('svg').remove();

  // 创建 SVG 容器
  const svg = d3.select(chartContainer.value)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .style('width', '100%')
    .style('height', '100%');

  // 配置色彩比例尺
  const color = d3.scaleOrdinal(data.children.map(d => d.name), d3.schemeTableau10);

  // 计算布局
  const root = d3.treemap()
    .size([width, height])
    .padding(1)
    .round(true)
    (d3.hierarchy(data)
      .sum(d => d.value)
      .sort((a, b) => b.value - a.value));

  // 添加节点
  const leaf = svg.selectAll("g")
    .data(root.leaves())
    .join("g")
    .attr("transform", d => `translate(${d.x0},${d.y0})`);

  leaf.append("title")
    .text(d => {
      // 仅提取以 "/" 分隔的最后一部分名称
      const lastName = d.data.name.split("/").pop();
      return `${lastName}`;
    });

  leaf.append("rect")
    .attr("fill", d => {
      const lastName = d.data.name.split("/").pop(); // 获取以 / 分隔的最后一部分字符串
      const nameWithoutNumber = lastName.replace(/_.*$/, ''); // 去掉 _ 及其后的数字
      return customColorMap[nameWithoutNumber] || "#000"; // 使用 customColorMap 查找颜色，未找到则默认为黑色
    })
    .attr("fill-opacity", 0.6)
    .attr("width", d => d.x1 - d.x0)
    .attr("height", d => d.y1 - d.y0)
    .attr('stroke-width', 10)
    .attr("style", "cursor: pointer;")

    .on("click", function(event, d) {
        // 这里的 d 就是点击的那个节点的数据
        const nodeName = d.data.name.split("/").pop(); // 处理节点name，取最后一部分
        // console.log("Clicked node name:", nodeName); 
        store.commit('UPDATE_SELECTED_NODES', { nodeIds: [nodeName], group: null });
    });

  // 添加文本

  leaf.append("text")
    .attr("x", 3)
    .attr("pointer-events", "none") // 阻止文本元素的所有指针事件
    .attr("y", "1em") 
    .text(d => abbreviateText(d.data.name.split("/").pop(), d.x1 - d.x0, 10)) 
    .append("title") 
    .text(d => d.data.name.split("/").pop());

  function abbreviateText(text, maxWidth, fontSize) {
    // 估算每个字符的平均宽度。注意：这个估算取决于字体的具体类型和大小
    const avgCharWidth = fontSize;
    const maxChars = Math.floor(maxWidth / avgCharWidth);

    if (text.length > maxChars) {
      return text.substr(0, maxChars - 1) + "…"; // 缩略并添加省略号
    }
    return text;
  }
}
</script>

<style scoped>
* {
  font-size: 0.8em;
  font-weight: bold;
}
</style>
