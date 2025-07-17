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
  }
  // {
  //   label: "Matplotlib",
  //   value: "matplotlib",
  //   description: "科学绘图、需要较高自定义，适合创建科学图表",
  // },
];

// 示例代码配置
export const placeholders = {
  // ECharts example
  echarts: `{
  "title": {
    "text": "Sales Data Analysis",
    "subtext": "Virtual Data"
  },
  "tooltip": {
    "trigger": "axis"
  },
  "legend": {
    "data": ["Sales", "Profit"]
  },
  "xAxis": {
    "type": "category",
    "data": ["January", "February", "March", "April", "May", "June"]
  },
  "yAxis": [
    {
      "type": "value",
      "name": "Sales"
    },
    {
      "type": "value",
      "name": "Profit",
      "position": "right"
    }
  ],
  "series": [
    {
      "name": "Sales",
      "type": "line",
      "data": [2900, 3500, 3200, 4100, 4800, 5300]
    },
    {
      "name": "Profit",
      "type": "bar",
      "yAxisIndex": 1,
      "data": [5000, 6800, 5900, 8100, 9600, 11000]
    }
  ]
}`,

  // Vega example
  vega: `{
  "$schema": "https://vega.github.io/schema/vega/v5.json",
  "width": 400,
  "height": 300,
  "padding": 5,

  "data": [
    {
      "name": "table",
      "values": [
        {"x": "January", "y": 28},
        {"x": "February", "y": 55},
        {"x": "March", "y": 43},
        {"x": "April", "y": 91},
        {"x": "May", "y": 81},
        {"x": "June", "y": 53}
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
    { "orient": "bottom", "scale": "xscale", "title": "Month" },
    { "orient": "left", "scale": "yscale", "title": "Sales" }
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
    "text": "Monthly Sales Data",
    "anchor": "middle",
    "fontSize": 16,
    "frame": "group"
  }
}`,

  // Vega-Lite example
  "vega-lite": `{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "description": "Multi-dimensional Sales Data Analysis",
  "width": 400,
  "height": 300,
  "data": {
    "values": [
      {"Month": "January", "Sales": 2800, "Profit": 1400, "Type": "Type A"},
      {"Month": "February", "Sales": 3500, "Profit": 1800, "Type": "Type A"},
      {"Month": "March", "Sales": 3200, "Profit": 1600, "Type": "Type B"},
      {"Month": "April", "Sales": 4100, "Profit": 2100, "Type": "Type B"},
      {"Month": "May", "Sales": 4800, "Profit": 2400, "Type": "Type A"},
      {"Month": "June", "Sales": 5300, "Profit": 2700, "Type": "Type B"}
    ]
  },
  "encoding": {
    "x": {
      "field": "Month",
      "type": "nominal",
      "axis": {"title": "Month"}
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
          "field": "Sales",
          "type": "quantitative",
          "axis": {"title": "Sales"}
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
          "field": "Profit",
          "type": "quantitative",
          "axis": {"title": "Profit", "titleColor": "#2196F3"}
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
  // D3.js example
  d3: `const width = 400;
const height = 300;
const margin = { top: 20, right: 20, bottom: 30, left: 40 };

const svg = d3.select(container)
  .append('svg')
  .attr('width', width)
  .attr('height', height);

// Create data
const data = [
  { name: 'A', value: 30 },
  { name: 'B', value: 45 },
  { name: 'C', value: 25 },
  { name: 'D', value: 60 },
  { name: 'E', value: 35 }
];

// Create scales
const x = d3.scaleBand()
  .domain(data.map(d => d.name))
  .range([margin.left, width - margin.right])
  .padding(0.1);

const y = d3.scaleLinear()
  .domain([0, d3.max(data, d => d.value)])
  .nice()
  .range([height - margin.bottom, margin.top]);

// Add axes
svg.append('g')
  .attr('transform', \`translate(0,\${height - margin.bottom})\`)
  .call(d3.axisBottom(x));

svg.append('g')
  .attr('transform', \`translate(\${margin.left},0)\`)
  .call(d3.axisLeft(y));

// Add bars
svg.selectAll('rect')
  .data(data)
  .join('rect')
  .attr('x', d => x(d.name))
  .attr('y', d => y(d.value))
  .attr('width', x.bandwidth())
  .attr('height', d => y(0) - y(d.value))
  .attr('fill', '#4CAF50');`,

  // Highcharts example
  highcharts: `{
  "chart": {
    "type": "column"
  },
  "title": {
    "text": "Annual Sales Performance"
  },
  "xAxis": {
    "categories": ["Q1", "Q2", "Q3", "Q4"]
  },
  "yAxis": [
    {
      "title": {
        "text": "Sales (10,000)"
      }
    },
    {
      "title": {
        "text": "Profit Rate (%)"
      },
      "opposite": true
    }
  ],
  "series": [
    {
      "name": "Sales",
      "data": [49.9, 71.5, 106.4, 129.2]
    },
    {
      "name": "Profit Rate",
      "type": "spline",
      "yAxis": 1,
      "data": [15.9, 17.2, 19.7, 20.5]
    }
  ]
}`,

//   // Matplotlib example
//   matplotlib: `import matplotlib.pyplot as plt
// import numpy as np
// from io import BytesIO

// # 设置渲染后端为Agg（非交互式后端，适合生成图像文件）
// plt.switch_backend('Agg')

// # 创建数据
// x = np.linspace(0, 10, 100)
// y1 = np.sin(x)
// y2 = np.cos(x)

// # 创建图表
// fig, ax = plt.subplots(figsize=(10, 6))
// ax.plot(x, y1, label='sin(x)', color='#4CAF50')
// ax.plot(x, y2, label='cos(x)', color='#2196F3')
// ax.fill_between(x, y1, alpha=0.3, color='#4CAF50')
// ax.fill_between(x, y2, alpha=0.3, color='#2196F3')

// # 添加标签和标题
// ax.set_xlabel('X Axis')
// ax.set_ylabel('Y Axis')
// ax.set_title('Trigonometric Functions Chart')
// ax.legend()
// ax.grid(True, linestyle='--', alpha=0.7)

// # 使用BytesIO而不是直接保存文件
// buf = BytesIO()
// fig.savefig(buf, format='svg', bbox_inches='tight')
// plt.close(fig)  # 确保关闭图表释放资源

// # 返回SVG内容（在后端API中使用）
// buf.seek(0)
// svg_content = buf.getvalue().decode('utf-8')
// # 不需要直接返回，后端API会处理这个结果`,
};
