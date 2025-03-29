// 使用D3.js绘制评估结果柱状图
document.addEventListener("DOMContentLoaded", () => {
  // 直接嵌入JSON数据而不是从文件加载
  const data = {
    individual_results: {
      "1.svg": {
        PCR: 0.9456700742820797,
        EGA: 0.8951689168969462,
        AC: 0.9264928147795337,
        Cohesion: 0.11533344868206,
        Separation: 0.507303216999401,
        final_score: 0.920899083181341,
        quality_level: "优秀",
      },
      "2.svg": {
        PCR: 0.8339260802291321,
        EGA: 0.9862589846809501,
        AC: 0.979743750813846,
        Cohesion: 0.263703734434834,
        Separation: 0.5147146892482621,
        final_score: 0.9411951349036125,
        quality_level: "优秀",
      },
      "3.svg": {
        PCR: 0.6787856796629401,
        EGA: 0.8805649011504161,
        AC: 0.7869444088392475,
        Cohesion: 0.039360282336227154,
        Separation: 0.5290617652858879,
        final_score: 0.7894271369787904,
        quality_level: "良好",
      },
      "4.svg": {
        PCR: 0.9082156168489076,
        EGA: 0.9817981513044693,
        AC: 0.8449595328308758,
        Cohesion: 0.04358534409973167,
        Separation: 0.4761975446964377,
        final_score: 0.9105647528216825,
        quality_level: "优秀",
      },
      "5.svg": {
        PCR: 0.8747350457424351,
        EGA: 0.8609993762115198,
        AC: 0.8656574779155173,
        Cohesion: 0.36486772486772484,
        Separation: 0.6796886446886448,
        final_score: 0.8665688613106551,
        quality_level: "良好",
      },
      "6.svg": {
        PCR: 0.8086478212618047,
        EGA: 0.7933451176818017,
        AC: 0.8622014177210046,
        Cohesion: 0.16606572049610024,
        Separation: 0.6677863636363637,
        final_score: 0.8231067056987076,
        quality_level: "良好",
      },
      "7.svg": {
        PCR: 0.8256814940860167,
        EGA: 0.6487468386924012,
        AC: 0.678420639591396,
        Cohesion: 0.015215837054453797,
        Separation: 0.6611781773046345,
        final_score: 0.7092678485352416,
        quality_level: "一般",
      },
      "8.svg": {
        PCR: 0.9253002051618054,
        EGA: 0.8091726830100615,
        AC: 0.7895258672360226,
        Cohesion: 0.06794163416861627,
        Separation: 0.6025327402064707,
        final_score: 0.8344190673761553,
        quality_level: "良好",
      },
      "9.svg": {
        PCR: 0.8362290478248342,
        EGA: 0.6625769138562989,
        AC: 0.8001550624315252,
        Cohesion: 0.06527777777777778,
        Separation: 0.7109090909090908,
        final_score: 0.7621034263403226,
        quality_level: "良好",
      },
      "10.svg": {
        PCR: 0.9362286665241353,
        EGA: 1.0,
        AC: 0.9381712310753847,
        Cohesion: 0.18333866240489075,
        Separation: 0.37755938543287165,
        final_score: 0.9592673821246502,
        quality_level: "优秀",
      },
      "11.svg": {
        PCR: 0.8617398743911948,
        EGA: 0.8166932853980872,
        AC: 0.6146930822120534,
        Cohesion: 0.009377289377289376,
        Separation: 0.7208216378859236,
        final_score: 0.7545662551373249,
        quality_level: "良好",
      },
      "12.svg": {
        PCR: 0.9732930402930402,
        EGA: 0.8405941583792008,
        AC: 0.6184368274379635,
        Cohesion: 0.14517816666509475,
        Separation: 0.6052124204249841,
        final_score: 0.7955516328668181,
        quality_level: "良好",
      },
      "13.svg": {
        PCR: 0.8500387514854425,
        EGA: 1.0,
        AC: 0.9219854214552682,
        Cohesion: 0.22923482227115444,
        Separation: 0.4973349334462368,
        final_score: 0.9291454563543732,
        quality_level: "优秀",
      },
      "14.svg": {
        PCR: 0.9445172371488162,
        EGA: 0.85009627433037,
        AC: 0.6238710642935683,
        Cohesion: 0.11429672344124023,
        Separation: 0.5543743846583791,
        final_score: 0.7928308162059183,
        quality_level: "良好",
      },
      "15.svg": {
        PCR: 0.9350305461232623,
        EGA: 0.8281355339475701,
        AC: 0.7474580438505236,
        Cohesion: 0.08804226679956836,
        Separation: 0.585375708058806,
        final_score: 0.8282154660208567,
        quality_level: "良好",
      },
      "16.svg": {
        PCR: 0.90498791963466,
        EGA: 0.9303547554493874,
        AC: 0.7061389911502808,
        Cohesion: 0.024444900134051976,
        Separation: 0.49011296213141986,
        final_score: 0.8402922086305943,
        quality_level: "良好",
      },
      "17.svg": {
        PCR: 0.9278616624244097,
        EGA: 0.7604894909954205,
        AC: 0.6046517297422526,
        Cohesion: 0.04774746686989923,
        Separation: 0.6054875491384912,
        final_score: 0.7496937273318653,
        quality_level: "一般",
      },
      "18.svg": {
        PCR: 0.3736957075502258,
        EGA: 1.0,
        AC: 0.7623326897501712,
        Cohesion: 0.01460624262509101,
        Separation: 0.3506060606060606,
        final_score: 0.7366978933216266,
        quality_level: "一般",
      },
      "19.svg": {
        PCR: 0.764304092132124,
        EGA: 0.8204669242989869,
        AC: 0.7893588355457839,
        Cohesion: 0.09380238310986233,
        Separation: 0.5961323368057532,
        final_score: 0.7932313384535801,
        quality_level: "良好",
      },
      "20.svg": {
        PCR: 0.8995573949567179,
        EGA: 0.9917450439755459,
        AC: 0.611475417527085,
        Cohesion: 0.05535595577673673,
        Separation: 0.4792491000459933,
        final_score: 0.8252327404643435,
        quality_level: "良好",
      },
    },
    statistics: {
      file_count: 20,
      metrics_summary: {
        PCR: {
          mean: 0.8504222978881992,
          std: 0.12953255214929357,
          min: 0.3736957075502258,
          max: 0.9732930402930402,
        },
        EGA: {
          mean: 0.8678603675129717,
          std: 0.1043476119630639,
          min: 0.6487468386924012,
          max: 1.0,
        },
        AC: {
          mean: 0.7736337153099652,
          std: 0.11806135591363608,
          min: 0.6046517297422526,
          max: 0.979743750813846,
        },
        Cohesion: {
          mean: 0.10733881916962025,
          std: 0.09186086244485156,
          min: 0.009377289377289376,
          max: 0.36486772486772484,
        },
        Separation: {
          mean: 0.5605819355805056,
          std: 0.0994947389752427,
          min: 0.3506060606060606,
          max: 0.7208216378859236,
        },
        final_score: {
          mean: 0.828113846702923,
          std: 0.07096468574108426,
          min: 0.7092678485352416,
          max: 0.9592673821246502,
        },
      },
      quality_distribution: {
        一般: 3,
        优秀: 5,
        良好: 12,
      },
    },
  };

  // 颜色配置 - 交换了PCR1和PCR的颜色
  const colors = {
    overall: {
      EGA: "#FBB03B", // 原来PCR1的颜色
      PCR: "#29ABE2", // 原来PCR的颜色
      AC: "#39B54A",
    },
    individual: {
      EGA: "#FFDF8A", // 原来PCR1的颜色
      PCR: "#AAE0FF", // 原来PCR的颜色
      AC: "#ABE594",
    },
  };

  // 图表尺寸
  const margin = { top: 50, right: 50, bottom: 120, left: 80 }; // 增加底部边距以容纳图像
  const width = 1200 - margin.left - margin.right;
  const height = 500 - margin.top - margin.bottom;

  // 创建SVG容器
  const svg = d3
    .select("#chart-container")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // 数据准备 - 交换了PCR1和PCR的顺序
  const metrics = ["EGA", "PCR", "AC"]; // 交换顺序
  const files = Object.keys(data.individual_results);

  // 构建完整数据集
  const chartData = [];

  // 添加总体数据（第一列）
  metrics.forEach((metric) => {
    chartData.push({
      file: "Overall",
      metric: metric,
      value: data.statistics.metrics_summary[metric].mean,
      std: data.statistics.metrics_summary[metric].std,
      isOverall: true,
    });
  });

  // 添加个体数据
  files.forEach((file) => {
    metrics.forEach((metric) => {
      chartData.push({
        file: file.replace(".svg", ""),
        metric: metric,
        value: data.individual_results[file][metric],
        std: 0,
        isOverall: false,
      });
    });
  });

  // 设置整体布局，让X轴占满并增加间隔
  // X轴比例尺
  const xScale = d3
    .scaleBand()
    .domain(["Overall", ...files.map((f) => f.replace(".svg", ""))])
    .range([0, width])
    .padding(0.4); // 增加间距

  // 特别处理第一列和后续列的间距
  const xPositions = {};
  let currentPosition = 20; // 减小第一列与y轴的距离

  // 为Overall分配位置
  xPositions["Overall"] = currentPosition;
  currentPosition += xScale.bandwidth() * 1.5; // 减小第一列与后续列的间距

  // 为其他文件分配间隔更大的位置
  files
    .map((f) => f.replace(".svg", ""))
    .forEach((file, i) => {
      xPositions[file] = currentPosition;
      currentPosition += xScale.bandwidth() * 1.1; // 减小后续列之间的间距
    });

  // 重新调整所有位置，使X轴占满
  const scaleFactor = width / currentPosition;
  Object.keys(xPositions).forEach((key) => {
    xPositions[key] *= scaleFactor;
  });

  // 创建自定义的X比例尺函数
  const customXScale = (file) => {
    return xPositions[file] || 0;
  };

  // 为每个指标创建子组
  const xSubScale = d3
    .scaleBand()
    .domain(metrics)
    .range([0, xScale.bandwidth()])
    .padding(0.40); // 减少padding

  // Y轴比例尺 - 范围从0到1
  const yScale = d3
    .scaleLinear()
    .domain([0, 1]) // 修改为0到1
    .range([height, 0]);

  // 创建X轴和Y轴的容器，但先不绘制线条
  const xAxis = svg
    .append("g")
    .attr("transform", `translate(0,${height})`)
    .attr("class", "x-axis");

  const yAxis = svg.append("g").attr("class", "y-axis");

  // 添加Overall标签
  xAxis
    .append("text")
    .attr("x", customXScale("Overall") + xScale.bandwidth() / 2)
    .attr("y", 30)
    .attr("text-anchor", "middle")
    .style("font-weight", "bold")
    .text("Overall");

  // 添加PNG图像替代数字标签
  files
    .map((f) => f.replace(".svg", ""))
    .forEach((file) => {
      // 使用base64编码嵌入图片
      const img = new Image();
      img.crossOrigin = "Anonymous"; // 设置跨域访问
      img.src = `png/${file}.png`;
      img.onload = function() {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const dataURL = canvas.toDataURL('image/png');
        
        xAxis
          .append("image")
          .attr("xlink:href", dataURL) // 使用base64编码的图片数据
          .attr("x", customXScale(file) + xScale.bandwidth() / 2 - 20)
          .attr("y", 10)
          .attr("width", 40)
          .attr("height", 40)
          .attr("preserveAspectRatio", "xMidYMid meet");
      };
    });

  // 修改Y轴刻度生成逻辑
  const yTicks = d3.range(0, 1.1, 0.1); // 生成0到1，步长为0.1的数组
  yTicks.forEach((tick) => {
    yAxis
      .append("line")
      .attr("x1", -5)
      .attr("y1", yScale(tick))
      .attr("x2", 0)
      .attr("y2", yScale(tick))
      .attr("stroke", "black")
      .attr("stroke-width", 1);

    yAxis
      .append("text")
      .attr("x", -10)
      .attr("y", yScale(tick))
      .attr("text-anchor", "end")
      .attr("dominant-baseline", "middle")
      .text(tick.toFixed(1)); // 保留一位小数
  });

  // Y轴标签
  svg
    .append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", -50) // 调整位置
    .attr("x", -height / 2)
    .attr("text-anchor", "middle")
    .text("Score");

  // 添加网格线
  svg
    .append("g")
    .attr("class", "grid")
    .call(d3.axisLeft(yScale).tickSize(-width).tickFormat(""))
    .selectAll("line")
    .style("stroke-dasharray", "3,3")
    .style("stroke", "#cccccc") // 使用浅灰色替代透明度
    .style("stroke-width", "0.5"); // 减小线宽使其看起来更轻

  // 误差条宽度
  const errorBarWidth = 6;

  // 添加柱状图
  svg
    .selectAll(".bar")
    .data(chartData)
    .enter()
    .append("rect")
    .attr("class", "bar")
    .attr("x", (d) => {
      // 获取基础位置
      const basePosition = customXScale(d.file) + xSubScale(d.metric);

      // 根据指标调整位置以增加间距
      if (d.metric === "EGA") {
        // 第一个柱子（左）
        return basePosition - xSubScale.bandwidth() * 0.5; // 大幅向左移动
      } else if (d.metric === "AC") {
        // 最后一个柱子（右）
        return basePosition + xSubScale.bandwidth() * 0.5; // 大幅向右移动
      }
      return basePosition; // 中间柱子保持原位
    })
    .attr("y", (d) => yScale(d.value))
    .attr("width", xSubScale.bandwidth() * 1.8) // 保持宽度系数
    .attr("height", (d) => height - yScale(d.value))
    .attr("fill", (d) => {
      if (d.isOverall) {
        return colors.overall[d.metric];
      } else {
        return colors.individual[d.metric];
      }
    });

  // 更新误差条的位置，与柱子的调整保持一致
  svg
    .selectAll(".error-bar")
    .data(chartData.filter((d) => d.isOverall))
    .enter()
    .append("line")
    .attr("class", "error-bar")
    .attr("x1", (d) => {
      const basePosition = customXScale(d.file) + xSubScale(d.metric);
      const barWidth = xSubScale.bandwidth() * 1.8;

      // 根据指标调整位置
      if (d.metric === "EGA") {
        return basePosition - xSubScale.bandwidth() * 0.5 + barWidth / 2; // 使用与柱子相同的偏移
      } else if (d.metric === "AC") {
        return basePosition + xSubScale.bandwidth() * 0.5 + barWidth / 2; // 使用与柱子相同的偏移
      }
      return basePosition + barWidth / 2;
    })
    .attr("y1", (d) => yScale(d.value - d.std / 2))
    .attr("x2", (d) => {
      const basePosition = customXScale(d.file) + xSubScale(d.metric);
      const barWidth = xSubScale.bandwidth() * 1.8;

      // 根据指标调整位置
      if (d.metric === "EGA") {
        return basePosition - xSubScale.bandwidth() * 0.5 + barWidth / 2;
      } else if (d.metric === "AC") {
        return basePosition + xSubScale.bandwidth() * 0.5 + barWidth / 2;
      }
      return basePosition + barWidth / 2;
    })
    .attr("y2", (d) => yScale(d.value + d.std / 2))
    .attr("stroke", "black")
    .attr("stroke-width", 2);

  // 更新误差条顶部和底部横线的位置
  svg
    .selectAll(".error-bar-top")
    .data(chartData.filter((d) => d.isOverall))
    .enter()
    .append("line")
    .attr("class", "error-bar-top")
    .attr("x1", (d) => {
      const basePosition = customXScale(d.file) + xSubScale(d.metric);
      const barWidth = xSubScale.bandwidth() * 1.8;

      // 根据指标调整位置
      if (d.metric === "EGA") {
        return (
          basePosition -
          xSubScale.bandwidth() * 0.5 +
          barWidth / 2 -
          errorBarWidth / 2
        );
      } else if (d.metric === "AC") {
        return (
          basePosition +
          xSubScale.bandwidth() * 0.5 +
          barWidth / 2 -
          errorBarWidth / 2
        );
      }
      return basePosition + barWidth / 2 - errorBarWidth / 2;
    })
    .attr("y1", (d) => yScale(d.value + d.std / 2))
    .attr("x2", (d) => {
      const basePosition = customXScale(d.file) + xSubScale(d.metric);
      const barWidth = xSubScale.bandwidth() * 1.8;

      // 根据指标调整位置
      if (d.metric === "EGA") {
        return (
          basePosition -
          xSubScale.bandwidth() * 0.5 +
          barWidth / 2 +
          errorBarWidth / 2
        );
      } else if (d.metric === "AC") {
        return (
          basePosition +
          xSubScale.bandwidth() * 0.5 +
          barWidth / 2 +
          errorBarWidth / 2
        );
      }
      return basePosition + barWidth / 2 + errorBarWidth / 2;
    })
    .attr("y2", (d) => yScale(d.value + d.std / 2))
    .attr("stroke", "black")
    .attr("stroke-width", 2);

  svg
    .selectAll(".error-bar-bottom")
    .data(chartData.filter((d) => d.isOverall))
    .enter()
    .append("line")
    .attr("class", "error-bar-bottom")
    .attr("x1", (d) => {
      const basePosition = customXScale(d.file) + xSubScale(d.metric);
      const barWidth = xSubScale.bandwidth() * 1.8;

      // 根据指标调整位置
      if (d.metric === "EGA") {
        return (
          basePosition -
          xSubScale.bandwidth() * 0.5 +
          barWidth / 2 -
          errorBarWidth / 2
        );
      } else if (d.metric === "AC") {
        return (
          basePosition +
          xSubScale.bandwidth() * 0.5 +
          barWidth / 2 -
          errorBarWidth / 2
        );
      }
      return basePosition + barWidth / 2 - errorBarWidth / 2;
    })
    .attr("y1", (d) => yScale(d.value - d.std / 2))
    .attr("x2", (d) => {
      const basePosition = customXScale(d.file) + xSubScale(d.metric);
      const barWidth = xSubScale.bandwidth() * 1.8;

      // 根据指标调整位置
      if (d.metric === "EGA") {
        return (
          basePosition -
          xSubScale.bandwidth() * 0.5 +
          barWidth / 2 +
          errorBarWidth / 2
        );
      } else if (d.metric === "AC") {
        return (
          basePosition +
          xSubScale.bandwidth() * 0.5 +
          barWidth / 2 +
          errorBarWidth / 2
        );
      }
      return basePosition + barWidth / 2 + errorBarWidth / 2;
    })
    .attr("y2", (d) => yScale(d.value - d.std / 2))
    .attr("stroke", "black")
    .attr("stroke-width", 2);

  // 添加图例 - 修改为横向排布在右上角
  const legendData = [
    { name: "EGA", color: colors.individual.EGA },
    { name: "PCR", color: colors.individual.PCR },
    { name: "AC", color: colors.individual.AC },
  ];

  const legend = svg
    .append("g")
    .attr("class", "legend")
    .attr("transform", `translate(${width - 250}, -40)`); // 放置在图表右上角

  // 横向添加图例项
  legendData.forEach((d, i) => {
    const legendItem = legend
      .append("g")
      .attr("transform", `translate(${i * 100}, 0)`); // 水平间隔100像素

    legendItem
      .append("rect")
      .attr("width", 20)
      .attr("height", 20)
      .attr("fill", d.color);

    legendItem.append("text").attr("x", 25).attr("y", 15).text(d.name);
  });

  // 在所有元素绘制完成后，最后绘制X轴和Y轴的实线及箭头，确保它们显示在最上层
  // 绘制X轴基线
  xAxis
    .append("line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", width)
    .attr("y2", 0)
    .attr("stroke", "black")
    .attr("stroke-width", 1.5);

  // 添加X轴刻度线
  const uniqueFiles = ["Overall", ...files.map(f => f.replace(".svg", ""))];
  uniqueFiles.forEach(file => {
    xAxis
      .append("line")
      .attr("x1", customXScale(file) + xScale.bandwidth()/2)
      .attr("y1", 0)
      .attr("x2", customXScale(file) + xScale.bandwidth()/2)
      .attr("y2", 5)
      .attr("stroke", "black")
      .attr("stroke-width", 1);
  });

  // 添加X轴箭头
  xAxis
    .append("polygon")
    .attr("points", `${width},0 ${width - 8},-4 ${width - 8},4`)
    .attr("fill", "black");

  // 绘制Y轴基线
  yAxis
    .append("line")
    .attr("x1", 0)
    .attr("y1", height)
    .attr("x2", 0)
    .attr("y2", 0)
    .attr("stroke", "black")
    .attr("stroke-width", 1.5);

  // 调整Y轴箭头位置，使其底部贴住Y轴顶端
  yAxis
    .append("polygon")
    .attr("points", "0,-8 -4,0 4,0") // 调整箭头位置，再向上移动4个像素
    .attr("fill", "black");

  // 确保X轴显示在最顶层
  xAxis.raise();
});
