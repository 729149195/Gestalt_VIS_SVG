// 语法选项配置
export const syntaxOptions = [
  {
    label: "Vega",
    value: "vega",
    description: "底层声明式语法，提供完整的可视化控制能力",
  },
  {
    label: "ECharts",
    value: "echarts",
    description: "复杂图表、高度定制，适合创建专业级数据可视化",
  },
  {
    label: "Vega-Lite",
    value: "vega-lite",
    description: "高级声明式语法，适合快速数据探索和原型设计",
  },
  {
    label: "D3.js",
    value: "d3",
    description: "复杂图表、高度定制，适合创建自定义可视化",
  },
  {
    label: "Highcharts",
    value: "highcharts",
    description: "商业级别图表，适合创建专业级数据可视化",
  },
  {
    label: "Matplotlib",
    value: "matplotlib",
    description: "科学绘图、需要较高自定义，适合创建科学图表",
  },
];

// 示例代码配置
export const placeholders = {
  // ECharts 示例
  echarts: `{
  "title": {
    "text": "销售数据分析",
    "subtext": "虚拟数据"
  },
  "tooltip": {
    "trigger": "axis"
  },
  "legend": {
    "data": ["销售额", "利润"]
  },
  "xAxis": {
    "type": "category",
    "data": ["一月", "二月", "三月", "四月", "五月", "六月"]
  },
  "yAxis": [
    {
      "type": "value",
      "name": "销售额"
    },
    {
      "type": "value",
      "name": "利润",
      "position": "right"
    }
  ],
  "series": [
    {
      "name": "销售额",
      "type": "line",
      "data": [2900, 3500, 3200, 4100, 4800, 5300]
    },
    {
      "name": "利润",
      "type": "bar",
      "yAxisIndex": 1,
      "data": [5000, 6800, 5900, 8100, 9600, 11000]
    }
  ]
}`,

  // Vega 示例
  vega: `{
  "$schema": "https://vega.github.io/schema/vega/v5.json",
  "width": 400,
  "height": 300,
  "padding": 5,

  "data": [
    {
      "name": "table",
      "values": [
        {"x": "一月", "y": 28},
        {"x": "二月", "y": 55},
        {"x": "三月", "y": 43},
        {"x": "四月", "y": 91},
        {"x": "五月", "y": 81},
        {"x": "六月", "y": 53}
      ]
    }
  ],

  "signals": [
    {
      "name": "tooltip",
      "value": {},
      "on": [
        {"events": "rect:mouseover", "update": "datum"},
        {"events": "rect:mouseout", "update": "{}"}
      ]
    }
  ],

  "scales": [
    {
      "name": "xscale",
      "type": "band",
      "domain": {"data": "table", "field": "x"},
      "range": "width",
      "padding": 0.05,
      "round": true
    },
    {
      "name": "yscale",
      "domain": {"data": "table", "field": "y"},
      "nice": true,
      "range": "height"
    }
  ],

  "axes": [
    { "orient": "bottom", "scale": "xscale", "title": "月份" },
    { "orient": "left", "scale": "yscale", "title": "销售额" }
  ],

  "marks": [
    {
      "type": "rect",
      "from": {"data": "table"},
      "encode": {
        "enter": {
          "x": {"scale": "xscale", "field": "x"},
          "width": {"scale": "xscale", "band": 1},
          "y": {"scale": "yscale", "field": "y"},
          "y2": {"scale": "yscale", "value": 0}
        },
        "update": {
          "fill": {"value": "#4CAF50"}
        },
        "hover": {
          "fill": {"value": "#81C784"}
        }
      }
    },
    {
      "type": "text",
      "encode": {
        "enter": {
          "align": {"value": "center"},
          "baseline": {"value": "bottom"},
          "fill": {"value": "#333"}
        },
        "update": {
          "x": {"scale": "xscale", "signal": "tooltip.x", "band": 0.5},
          "y": {"scale": "yscale", "signal": "tooltip.y", "offset": -2},
          "text": {"signal": "tooltip.y"},
          "fillOpacity": [
            {"test": "datum === tooltip", "value": 0},
            {"value": 1}
          ]
        }
      }
    }
  ],

  "title": {
    "text": "月度销售数据",
    "anchor": "middle",
    "fontSize": 16,
    "frame": "group"
  }
}`,

  // Vega-Lite 示例
  "vega-lite": `{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "description": "多维度销售数据分析",
  "width": 400,
  "height": 300,
  "data": {
    "values": [
      {"月份": "一月", "销售额": 2800, "利润": 1400, "类型": "A类"},
      {"月份": "二月", "销售额": 3500, "利润": 1800, "类型": "A类"},
      {"月份": "三月", "销售额": 3200, "利润": 1600, "类型": "B类"},
      {"月份": "四月", "销售额": 4100, "利润": 2100, "类型": "B类"},
      {"月份": "五月", "销售额": 4800, "利润": 2400, "类型": "A类"},
      {"月份": "六月", "销售额": 5300, "利润": 2700, "类型": "B类"}
    ]
  },
  "encoding": {
    "x": {
      "field": "月份",
      "type": "nominal",
      "axis": {"title": "月份"}
    }
  },
  "layer": [
    {
      "mark": {
        "type": "bar",
        "color": "#4CAF50",
        "opacity": 0.7
      },
      "encoding": {
        "y": {
          "field": "销售额",
          "type": "quantitative",
          "axis": {"title": "销售额"}
        }
      }
    },
    {
      "mark": {
        "type": "line",
        "color": "#2196F3",
        "point": true
      },
      "encoding": {
        "y": {
          "field": "利润",
          "type": "quantitative",
          "axis": {"title": "利润", "titleColor": "#2196F3"}
        }
      }
    }
  ],
  "resolve": {
    "scale": {"y": "independent"}
  },
  "config": {
    "axis": {
      "titleFontSize": 14,
      "labelFontSize": 12
    },
    "title": {
      "fontSize": 16,
      "anchor": "middle"
    }
  }
}`,
  // D3.js 示例
  d3: `const width = 400;
const height = 300;
const margin = { top: 20, right: 20, bottom: 30, left: 40 };

const svg = d3.select(container)
  .append('svg')
  .attr('width', width)
  .attr('height', height);

// 创建数据
const data = [
  { name: 'A', value: 30 },
  { name: 'B', value: 45 },
  { name: 'C', value: 25 },
  { name: 'D', value: 60 },
  { name: 'E', value: 35 }
];

// 创建比例尺
const x = d3.scaleBand()
  .domain(data.map(d => d.name))
  .range([margin.left, width - margin.right])
  .padding(0.1);

const y = d3.scaleLinear()
  .domain([0, d3.max(data, d => d.value)])
  .nice()
  .range([height - margin.bottom, margin.top]);

// 添加坐标轴
svg.append('g')
  .attr('transform', \`translate(0,\${height - margin.bottom})\`)
  .call(d3.axisBottom(x));

svg.append('g')
  .attr('transform', \`translate(\${margin.left},0)\`)
  .call(d3.axisLeft(y));

// 添加柱状图
svg.selectAll('rect')
  .data(data)
  .join('rect')
  .attr('x', d => x(d.name))
  .attr('y', d => y(d.value))
  .attr('width', x.bandwidth())
  .attr('height', d => y(0) - y(d.value))
  .attr('fill', '#4CAF50');`,

  // Highcharts 示例
  highcharts: `{
  "chart": {
    "type": "column"
  },
  "title": {
    "text": "年度销售业绩"
  },
  "xAxis": {
    "categories": ["一季度", "二季度", "三季度", "四季度"]
  },
  "yAxis": [
    {
      "title": {
        "text": "销售额 (万元)"
      }
    },
    {
      "title": {
        "text": "利润率 (%)"
      },
      "opposite": true
    }
  ],
  "series": [
    {
      "name": "销售额",
      "data": [49.9, 71.5, 106.4, 129.2]
    },
    {
      "name": "利润率",
      "type": "spline",
      "yAxis": 1,
      "data": [15.9, 17.2, 19.7, 20.5]
    }
  ]
}`,

  // Matplotlib 示例
  matplotlib: `import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', color='#4CAF50')
plt.plot(x, y2, label='cos(x)', color='#2196F3')
plt.fill_between(x, y1, alpha=0.3, color='#4CAF50')
plt.fill_between(x, y2, alpha=0.3, color='#2196F3')

# 添加标签和标题
plt.xlabel('X 轴')
plt.ylabel('Y 轴')
plt.title('三角函数图表')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 保存为SVG
plt.savefig('output.svg', format='svg', bbox_inches='tight')`,
};
